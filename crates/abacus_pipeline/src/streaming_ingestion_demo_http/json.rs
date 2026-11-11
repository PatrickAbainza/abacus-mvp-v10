//! `web::json` — minimal JSON value model + serializer/parser.
//!
//! Std-only by golden source (§0 H1: no third-party libraries). The surface is
//! deliberately tiny: build response objects, parse small request bodies (the
//! streaming_ingestion_demo start form), and escape strings correctly. No float precision beyond
//! what `f64` formatting gives (amounts are never serialized, so this never
//! carries monetary values — task §6).

use std::fmt::Write as _;

/// A JSON value (object/array/string/number/bool/null).
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    /// Null.
    Null,
    /// Boolean.
    Bool(bool),
    /// Number, stored as its source text (lossless round-trip for ints).
    Num(f64),
    /// Integer (serialized without a decimal point).
    Int(i64),
    /// String.
    Str(String),
    /// Array.
    Array(Vec<Json>),
    /// Object (insertion-ordered key/value pairs).
    Object(Vec<(String, Json)>),
}

impl Json {
    /// An empty object.
    #[must_use]
    pub fn object() -> Self {
        Json::Object(Vec::new())
    }

    /// Insert a key/value into an object (no-op if not an object).
    pub fn set(&mut self, key: &str, val: Json) -> &mut Self {
        if let Json::Object(m) = self {
            m.push((key.to_string(), val));
        }
        self
    }

    /// Shorthand: string value.
    #[must_use]
    pub fn str(s: impl Into<String>) -> Self {
        Json::Str(s.into())
    }

    /// Shorthand: integer value.
    #[must_use]
    pub fn int(n: i64) -> Self {
        Json::Int(n)
    }

    /// Shorthand: unsigned integer widened to `i64`.
    #[must_use]
    pub fn uint(n: u64) -> Self {
        Json::Int(n as i64)
    }

    /// Shorthand: float value.
    #[must_use]
    pub fn float(n: f64) -> Self {
        Json::Num(n)
    }

    /// Shorthand: bool value.
    #[must_use]
    pub fn bool(b: bool) -> Self {
        Json::Bool(b)
    }

    /// Serialize to a compact JSON string.
    #[must_use]
    pub fn to_json_string(&self) -> String {
        let mut out = String::new();
        self.write(&mut out);
        out
    }

    /// Recursive compact writer.
    fn write(&self, out: &mut String) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Int(n) => {
                let _ = write!(out, "{n}");
            }
            Json::Num(n) => {
                // Integers print without a trailing `.0`; NaN/Inf are not
                // produced by any field this layer serializes.
                if n.fract() == 0.0 && n.is_finite() && n.abs() < 9.007e15 {
                    let _ = write!(out, "{}", *n as i64);
                } else {
                    let _ = write!(out, "{n}");
                }
            }
            Json::Str(s) => write_json_string(out, s),
            Json::Array(a) => {
                out.push('[');
                for (i, v) in a.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    v.write(out);
                }
                out.push(']');
            }
            Json::Object(m) => {
                out.push('{');
                for (i, (k, v)) in m.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_json_string(out, k);
                    out.push(':');
                    v.write(out);
                }
                out.push('}');
            }
        }
    }

    /// Look up an object field by key.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(m) => m.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// Coerce to `u64` if this is a non-negative integer/float.
    #[must_use]
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Json::Int(n) if *n >= 0 => Some(*n as u64),
            Json::Num(n) if *n >= 0.0 => Some(*n as u64),
            _ => None,
        }
    }
}

impl std::fmt::Display for Json {
    /// Compact JSON serialization; also provides `to_string()` via `Display`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_json_string())
    }
}

/// Write a JSON-escaped string (with surrounding quotes).
fn write_json_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// A tiny recursive-descent JSON parser sufficient for the small request
/// bodies this server accepts (the streaming_ingestion_demo start form `{"tps":N,"chunk":M}`).
/// Rejects malformed input; unknown trailing content is an error.
///
/// # Errors
/// `Err(String)` describing the first parse failure.
pub fn parse(input: &str) -> Result<Json, String> {
    let bytes = input.as_bytes();
    let mut p = Parser { b: bytes, i: 0 };
    p.skip_ws();
    let v = p.value()?;
    p.skip_ws();
    if p.i != p.b.len() {
        return Err("trailing content after JSON value".into());
    }
    Ok(v)
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn skip_ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.b.get(self.i).copied()
    }

    fn value(&mut self) -> Result<Json, String> {
        self.skip_ws();
        match self.peek() {
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') => self.literal("true", Json::Bool(true)),
            Some(b'f') => self.literal("false", Json::Bool(false)),
            Some(b'n') => self.literal("null", Json::Null),
            Some(c) if c == b'-' || c.is_ascii_digit() => self.number(),
            _ => Err("unexpected token".into()),
        }
    }

    fn literal(&mut self, word: &str, val: Json) -> Result<Json, String> {
        if self.b[self.i..].starts_with(word.as_bytes()) {
            self.i += word.len();
            Ok(val)
        } else {
            Err(format!("expected `{word}`"))
        }
    }

    fn object(&mut self) -> Result<Json, String> {
        self.i += 1; // '{'
        let mut members = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.i += 1;
            return Ok(Json::Object(members));
        }
        loop {
            self.skip_ws();
            if self.peek() != Some(b'"') {
                return Err("expected object key".into());
            }
            let key = self.string()?;
            self.skip_ws();
            if self.peek() != Some(b':') {
                return Err("expected `:`".into());
            }
            self.i += 1;
            let val = self.value()?;
            members.push((key, val));
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.i += 1;
                }
                Some(b'}') => {
                    self.i += 1;
                    break;
                }
                _ => return Err("expected `,` or `}`".into()),
            }
        }
        Ok(Json::Object(members))
    }

    fn array(&mut self) -> Result<Json, String> {
        self.i += 1; // '['
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.i += 1;
            return Ok(Json::Array(items));
        }
        loop {
            let val = self.value()?;
            items.push(val);
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.i += 1;
                }
                Some(b']') => {
                    self.i += 1;
                    break;
                }
                _ => return Err("expected `,` or `]`".into()),
            }
        }
        Ok(Json::Array(items))
    }

    fn string(&mut self) -> Result<String, String> {
        self.i += 1; // opening quote
        let mut s = String::new();
        loop {
            match self.peek() {
                None => return Err("unterminated string".into()),
                Some(b'"') => {
                    self.i += 1;
                    break;
                }
                Some(b'\\') => {
                    self.i += 1;
                    match self.peek() {
                        Some(b'"') => s.push('"'),
                        Some(b'\\') => s.push('\\'),
                        Some(b'/') => s.push('/'),
                        Some(b'n') => s.push('\n'),
                        Some(b't') => s.push('\t'),
                        Some(b'r') => s.push('\r'),
                        Some(b'b') => s.push('\u{8}'),
                        Some(b'f') => s.push('\u{c}'),
                        Some(b'u') => {
                            let hex =
                                self.b.get(self.i + 1..self.i + 5).ok_or("bad \\u escape")?;
                            let code = u32::from_str_radix(
                                std::str::from_utf8(hex).map_err(|_| "bad \\u hex")?,
                                16,
                            )
                            .map_err(|_| "bad \\u hex")?;
                            self.i += 4;
                            s.push(char::from_u32(code).unwrap_or('\u{fffd}'));
                        }
                        _ => return Err("bad escape".into()),
                    }
                    self.i += 1;
                }
                Some(_) => {
                    // Copy one UTF-8 code point verbatim.
                    let start = self.i;
                    self.i += 1;
                    while self.i < self.b.len() && (self.b[self.i] & 0xc0) == 0x80 {
                        self.i += 1;
                    }
                    s.push_str(std::str::from_utf8(&self.b[start..self.i]).map_err(|_| "bad utf8")?);
                }
            }
        }
        Ok(s)
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.i;
        if self.peek() == Some(b'-') {
            self.i += 1;
        }
        let mut is_float = false;
        while let Some(c) = self.peek() {
            match c {
                b'0'..=b'9' => self.i += 1,
                b'.' | b'e' | b'E' | b'+' | b'-' => {
                    is_float = true;
                    self.i += 1;
                }
                _ => break,
            }
        }
        let text = std::str::from_utf8(&self.b[start..self.i]).map_err(|_| "bad number")?;
        if is_float {
            text.parse::<f64>().map(Json::Num).map_err(|_| "bad number".into())
        } else {
            match text.parse::<i64>() {
                Ok(n) => Ok(Json::Int(n)),
                Err(_) => text
                    .parse::<f64>()
                    .map(Json::Num)
                    .map_err(|_| "number out of range".into()),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_object() {
        let mut o = Json::object();
        o.set("a", Json::int(1));
        o.set("b", Json::str("x\"y"));
        o.set("c", Json::Array(vec![Json::bool(true), Json::Null]));
        let s = o.to_string();
        assert_eq!(s, r#"{"a":1,"b":"x\"y","c":[true,null]}"#);
        let back = parse(&s).unwrap();
        assert_eq!(back.get("a"), Some(&Json::int(1)));
        assert_eq!(back.get("b"), Some(&Json::str("x\"y")));
    }

    #[test]
    fn parse_rejects_garbage() {
        assert!(parse("{").is_err());
        assert!(parse("{\"a\":}").is_err());
        assert!(parse("1 2").is_err());
    }
}

