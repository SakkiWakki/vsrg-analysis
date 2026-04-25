//! Pattern parsing for memory signature scans.
//!
//! Input format: whitespace-separated hex bytes with ``??`` wildcards.
//!   "F8 01 74 04 ?? 65 8B"
//!
//! Output: a ``Pattern`` of (byte, is_wildcard) pairs, directly
//! consumable by the linear scanner. Kept intentionally dumb ; we
//! don't bother with Boyer-Moore or vectorization yet. For the
//! single-digit number of signatures we resolve per startup, a
//! straight scan over a few tens of MB is sub-100ms and fine.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum PatternError {
    #[error("empty pattern")]
    Empty,
    #[error("bad token '{0}' (expected two hex digits or ??)")]
    BadToken(String),
}

pub struct Pattern {
    // Each entry is (value, is_wildcard). Wildcard entries match any byte.
    bytes: Vec<(u8, bool)>,
}

impl Pattern {
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// True if ``window`` (same length as the pattern) matches every
    /// non-wildcard byte.
    pub fn matches(&self, window: &[u8]) -> bool {
        if window.len() != self.bytes.len() {
            return false;
        }
        for (i, &(val, wild)) in self.bytes.iter().enumerate() {
            if !wild && window[i] != val {
                return false;
            }
        }
        true
    }

    /// First non-wildcard byte in the pattern. Intended as a scanner
    /// fast-path (memchr-and-verify); unused right now because our
    /// patterns are short enough that the naive loop in `scan` wins.
    /// Keeping it because the next scanner rewrite will want it.
    #[allow(dead_code)]
    pub fn first_literal(&self) -> Option<(usize, u8)> {
        self.bytes
            .iter()
            .enumerate()
            .find_map(|(i, &(v, w))| if w { None } else { Some((i, v)) })
    }
}

pub fn parse(input: &str) -> Result<Pattern, PatternError> {
    let mut bytes = Vec::new();
    for tok in input.split_ascii_whitespace() {
        if tok == "??" || tok == "?" {
            bytes.push((0u8, true));
            continue;
        }
        if tok.len() != 2 {
            return Err(PatternError::BadToken(tok.to_string()));
        }
        let val =
            u8::from_str_radix(tok, 16).map_err(|_| PatternError::BadToken(tok.to_string()))?;
        bytes.push((val, false));
    }
    if bytes.is_empty() {
        return Err(PatternError::Empty);
    }
    Ok(Pattern { bytes })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_plain() {
        let p = parse("F8 01 74").unwrap();
        assert_eq!(p.len(), 3);
        assert!(p.matches(&[0xF8, 0x01, 0x74]));
        assert!(!p.matches(&[0xF8, 0x01, 0x75]));
    }

    #[test]
    fn parse_with_wildcards() {
        let p = parse("F8 ?? 74").unwrap();
        assert!(p.matches(&[0xF8, 0xAA, 0x74]));
        assert!(p.matches(&[0xF8, 0x00, 0x74]));
        assert!(!p.matches(&[0xF9, 0xAA, 0x74]));
    }

    #[test]
    fn empty_rejected() {
        assert!(matches!(parse(""), Err(PatternError::Empty)));
        assert!(matches!(parse("   "), Err(PatternError::Empty)));
    }

    #[test]
    fn bad_token_rejected() {
        assert!(matches!(parse("GG"), Err(PatternError::BadToken(_))));
        assert!(matches!(parse("F"), Err(PatternError::BadToken(_))));
    }

    #[test]
    fn first_literal_skips_wildcards() {
        let p = parse("?? ?? F8 01").unwrap();
        assert_eq!(p.first_literal(), Some((2, 0xF8)));
    }
}
