//! The AST the core lowers - a Rust mirror of `expr/ast.py`'s node set. This is
//! the LANGUAGE PIECE's output: the core does NOT parse (no CFG parser); it
//! receives the already-parsed Python AST and marshals it into these owned
//! structs ONCE (at body-compile time), then evaluates them natively per tick
//! with no Python round-trip.
//!
//! Node kinds match ast.py exactly (20 forms). `Unparsed` is a construct outside
//! the modeled subset - the core skips it (falls back), never guesses.

use std::rc::Rc;

#[derive(Clone)]
pub enum Expr {
    Num(f64),
    Str(Rc<str>),
    Bool(bool),
    Nil,
    Sym(Rc<str>),
    Index { base: Rc<Expr>, key: Rc<Expr> },
    Field { base: Rc<Expr>, name: Rc<str> },
    Unary { op: Rc<str>, operand: Rc<Expr> },
    Binary { op: Rc<str>, left: Rc<Expr>, right: Rc<Expr> },
    Call { fn_: Rc<Expr>, args: Vec<Expr> },
    Method { recv: Rc<Expr>, name: Rc<str>, args: Vec<Expr> },
    Table { array: Vec<Expr>, fields: Vec<(Rc<str>, Expr)> },
    FuncExpr { params: Vec<Rc<str>>, body: Rc<[Stmt]> },
    /// A construct outside the modeled subset - evaluates to UNRESOLVED.
    Unparsed,
}

#[derive(Clone)]
pub enum Stmt {
    Assign { targets: Vec<Expr>, values: Vec<Expr> },
    Local { names: Vec<Rc<str>>, values: Vec<Expr> },
    If {
        cond: Expr,
        body: Rc<[Stmt]>,
        elifs: Vec<(Expr, Rc<[Stmt]>)>,
        orelse: Rc<[Stmt]>,
    },
    NumericFor {
        var: Rc<str>,
        start: Expr,
        stop: Expr,
        step: Option<Expr>,
        body: Rc<[Stmt]>,
    },
    GenericFor {
        names: Vec<Rc<str>>,
        exprs: Vec<Expr>,
        body: Rc<[Stmt]>,
    },
    While { cond: Expr, body: Rc<[Stmt]> },
    FuncDef {
        name: Rc<str>,
        params: Vec<Rc<str>>,
        body: Rc<[Stmt]>,
    },
    Return { values: Vec<Expr> },
    ExprStmt(Expr),
    /// Outside the subset - skipped (the tree-walk floor's "fall back" arm).
    Unparsed,
}
