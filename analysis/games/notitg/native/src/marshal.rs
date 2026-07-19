//! Marshal the Python AST (`expr/ast.py` dataclasses) into the Rust `ast`
//! mirror, ONCE at body-compile time. We dispatch on the Python class name
//! (`type(node).__name__`) and pull the documented fields; an unknown class ->
//! `Unparsed` (the same fall-back the tree-walk floor uses). No evaluation here,
//! pure structural copy - the language piece boundary.

use std::rc::Rc;

use pyo3::prelude::*;
use pyo3::types::{PyString, PyTuple};

use crate::ast::{Expr, Stmt};

/// Marshal a Python list/tuple of statement nodes into a Rust `[Stmt]`.
pub fn marshal_body(py_stmts: &Bound<'_, PyAny>) -> PyResult<Rc<[Stmt]>> {
    let mut out = Vec::new();
    for item in py_stmts.try_iter()? {
        out.push(marshal_stmt(&item?)?);
    }
    Ok(out.into())
}

fn class_name(node: &Bound<'_, PyAny>) -> PyResult<String> {
    let ty = node.get_type();
    Ok(ty.name()?.to_string())
}

fn attr<'py>(node: &Bound<'py, PyAny>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    node.getattr(name)
}

fn attr_str(node: &Bound<'_, PyAny>, name: &str) -> PyResult<Rc<str>> {
    let v = node.getattr(name)?;
    let s: String = v.extract()?;
    Ok(Rc::from(s.as_str()))
}

/// A tuple/list attribute of nodes -> Vec<Expr>.
fn marshal_expr_seq(node: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<Expr>> {
    let seq = node.getattr(name)?;
    let mut out = Vec::new();
    for item in seq.try_iter()? {
        out.push(marshal_expr(&item?)?);
    }
    Ok(out)
}

fn marshal_str_seq(node: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<Rc<str>>> {
    let seq = node.getattr(name)?;
    let mut out = Vec::new();
    for item in seq.try_iter()? {
        let s: String = item?.extract()?;
        out.push(Rc::from(s.as_str()));
    }
    Ok(out)
}

pub fn marshal_expr(node: &Bound<'_, PyAny>) -> PyResult<Expr> {
    let cls = class_name(node)?;
    let expr = match cls.as_str() {
        "Num" => Expr::Num(attr(node, "value")?.extract()?),
        "Str" => Expr::Str(attr_str(node, "value")?),
        "Bool" => Expr::Bool(attr(node, "value")?.extract()?),
        "Nil" => Expr::Nil,
        "Sym" => Expr::Sym(attr_str(node, "name")?),
        "Index" => Expr::Index {
            base: Rc::new(marshal_expr(&attr(node, "base")?)?),
            key: Rc::new(marshal_expr(&attr(node, "key")?)?),
        },
        "Field" => Expr::Field {
            base: Rc::new(marshal_expr(&attr(node, "base")?)?),
            name: attr_str(node, "name")?,
        },
        "Unary" => Expr::Unary {
            op: attr_str(node, "op")?,
            operand: Rc::new(marshal_expr(&attr(node, "operand")?)?),
        },
        "Binary" => Expr::Binary {
            op: attr_str(node, "op")?,
            left: Rc::new(marshal_expr(&attr(node, "left")?)?),
            right: Rc::new(marshal_expr(&attr(node, "right")?)?),
        },
        "Call" => Expr::Call {
            fn_: Rc::new(marshal_expr(&attr(node, "fn")?)?),
            args: marshal_expr_seq(node, "args")?,
        },
        "Method" => Expr::Method {
            recv: Rc::new(marshal_expr(&attr(node, "recv")?)?),
            name: attr_str(node, "name")?,
            args: marshal_expr_seq(node, "args")?,
        },
        "Table" => Expr::Table {
            array: marshal_expr_seq(node, "array")?,
            fields: marshal_fields(node)?,
        },
        "FuncExpr" => Expr::FuncExpr {
            params: marshal_str_seq(node, "params")?,
            body: marshal_body(&attr(node, "body")?)?,
        },
        _ => Expr::Unparsed,
    };
    Ok(expr)
}

fn marshal_fields(node: &Bound<'_, PyAny>) -> PyResult<Vec<(Rc<str>, Expr)>> {
    let seq = node.getattr("fields")?;
    let mut out = Vec::new();
    for item in seq.try_iter()? {
        let pair = item?;
        let tup = pair.cast::<PyTuple>()?;
        let key: String = tup.get_item(0)?.cast::<PyString>()?.extract()?;
        let value = marshal_expr(&tup.get_item(1)?)?;
        out.push((Rc::from(key.as_str()), value));
    }
    Ok(out)
}

pub fn marshal_stmt(node: &Bound<'_, PyAny>) -> PyResult<Stmt> {
    let cls = class_name(node)?;
    let stmt = match cls.as_str() {
        "Assign" => Stmt::Assign {
            targets: marshal_expr_seq(node, "targets")?,
            values: marshal_expr_seq(node, "values")?,
        },
        "Local" => Stmt::Local {
            names: marshal_str_seq(node, "names")?,
            values: marshal_expr_seq(node, "values")?,
        },
        "If" => Stmt::If {
            cond: marshal_expr(&attr(node, "cond")?)?,
            body: marshal_body(&attr(node, "body")?)?,
            elifs: marshal_elifs(node)?,
            orelse: marshal_body(&attr(node, "orelse")?)?,
        },
        "NumericFor" => Stmt::NumericFor {
            var: attr_str(node, "var")?,
            start: marshal_expr(&attr(node, "start")?)?,
            stop: marshal_expr(&attr(node, "stop")?)?,
            step: marshal_opt_expr(node, "step")?,
            body: marshal_body(&attr(node, "body")?)?,
        },
        "GenericFor" => Stmt::GenericFor {
            names: marshal_str_seq(node, "names")?,
            exprs: marshal_expr_seq(node, "exprs")?,
            body: marshal_body(&attr(node, "body")?)?,
        },
        "While" => Stmt::While {
            cond: marshal_expr(&attr(node, "cond")?)?,
            body: marshal_body(&attr(node, "body")?)?,
        },
        "FuncDef" => Stmt::FuncDef {
            name: attr_str(node, "name")?,
            params: marshal_str_seq(node, "params")?,
            body: marshal_body(&attr(node, "body")?)?,
        },
        "Return" => Stmt::Return {
            values: marshal_expr_seq(node, "values")?,
        },
        "ExprStmt" => Stmt::ExprStmt(marshal_expr(&attr(node, "expr")?)?),
        _ => Stmt::Unparsed,
    };
    Ok(stmt)
}

fn marshal_opt_expr(node: &Bound<'_, PyAny>, name: &str) -> PyResult<Option<Expr>> {
    let v = node.getattr(name)?;
    if v.is_none() {
        Ok(None)
    } else {
        Ok(Some(marshal_expr(&v)?))
    }
}

fn marshal_elifs(node: &Bound<'_, PyAny>) -> PyResult<Vec<(Expr, Rc<[Stmt]>)>> {
    let seq = node.getattr("elifs")?;
    let mut out = Vec::new();
    for item in seq.try_iter()? {
        let pair = item?;
        let tup = pair.cast::<PyTuple>()?;
        let cond = marshal_expr(&tup.get_item(0)?)?;
        let body = marshal_body(&tup.get_item(1)?)?;
        out.push((cond, body));
    }
    Ok(out)
}
