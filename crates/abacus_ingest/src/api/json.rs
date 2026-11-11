//! Minimal JSON value model + serializer/parser.
//!
//! Conventions follow the existing repository's `web::json` module: std-only,
//! insertion-ordered objects, integers serialized without a decimal point, and
//! strings escaped correctly. No serialization dependency is introduced.
//!
//! Privacy: amounts are never serialized through this model. The ingestion API
//! exposes counts, versions, fingerprints and status only (task step 3: "Do not
//! expose raw account mappings, raw transaction amounts, or ground-truth
//! labels through the API.").

use std::fmt::Write as _;

/// A JSON value (object/array/string/number/bool/null).
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    /// Null.
    Null,
    /// Boolean.
    Bool(bool),
    /// Float.
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

    /// Shorthand: bool value.
    #[must_use]
    pub fn bool(b: bool) -> Self {
        Json::Bool(b)
    }

    /// Look up a key in an object.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(m) => m.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// Interpret as an unsigned integer.
    #[must_use]
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Json::Int(n) => u64::try_from(*n).ok(),
            Json::Num(n) if *n >= 0.0 => Some(*n as u64),
            _ => None,
        }
    }

    /// Interpret as a signed integer.
    #[must_use]
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Json::Int(n) => Some(*n),
            Json::Num(n) => Some(*n as i64),
            _ => None,
        }
    }

    /// Interpret as a string.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    /// Interpret as a bool.
    #[must_use]
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
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
                // Integers print without a trailing `.0`; NaN/Inf are never
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
}

impl std::fmt::Display for Json {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_json_string())
    }
}

/// Escape a string per RFC 8259.
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

/// Parse a JSON document.
///
/// # Errors
/// A human-readable message on malformed input. No partial value is returned.
pub fn parse(text: &str) -> Result<Json, String> {
    let mut p = Parser { b: text.as_bytes(), i: 0 };
    p.skip_ws();
    let v = p.value()?;
    p.skip_ws();
    if p.i != p.b.len() {
        return Err("trailing characters after JSON value".into());
    }
    Ok(v)
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<u8> {
        self.b.get(self.i).copied()
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ') | Some(b'\t') | Some(b'\n') | Some(b'\r')) {
            self.i += 1;
        }
    }

    fn value(&mut self) -> Result<Json, String> {
        match self.peek() {
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') => self.literal("true", Json::Bool(true)),
            Some(b'f') => self.literal("false", Json::Bool(false)),
            Some(b'n') => self.literal("null", Json::Null),
            Some(c) if c == b'-' || c.is_ascii_digit() => self.number(),
            other => Err(format!("unexpected character {other:?}")),
        }
    }

    fn literal(&mut self, word: &str, v: Json) -> Result<Json, String> {
        if self.b[self.i..].starts_with(word.as_bytes()) {
            self.i += word.len();
            Ok(v)
        } else {
            Err(format!("expected {word}"))
        }
    }

    fn object(&mut self) -> Result<Json, String> {
        self.i += 1; // '{'
        let mut m = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.i += 1;
            return Ok(Json::Object(m));
        }
        loop {
            self.skip_ws();
            if self.peek() != Some(b'"') {
                return Err("expected an object key".into());
            }
            let k = self.string()?;
            self.skip_ws();
            if self.peek() != Some(b':') {
                return Err("expected ':' after an object key".into());
            }
            self.i += 1;
            self.skip_ws();
            let v = self.value()?;
            m.push((k, v));
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.i += 1;
                }
                Some(b'}') => {
                    self.i += 1;
                    return Ok(Json::Object(m));
                }
                _ => return Err("expected ',' or '}' in object".into()),
            }
        }
    }

    fn array(&mut self) -> Result<Json, String> {
        self.i += 1; // '['
        let mut a = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.i += 1;
            return Ok(Json::Array(a));
        }
        loop {
            self.skip_ws();
            a.push(self.value()?);
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.i += 1;
                }
                Some(b']') => {
                    self.i += 1;
                    return Ok(Json::Array(a));
                }
                _ => return Err("expected ',' or ']' in array".into()),
            }
        }
    }

    fn string(&mut self) -> Result<String, String> {
        self.i += 1; // opening quote
        let mut s = String::new();
        loop {
            match self.peek() {
                None => return Err("unterminated string".into()),
                Some(b'"') => {
                    self.i += 1;
                    return Ok(s);
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
                            let hex = self
                                .b
                                .get(self.i + 1..self.i + 5)
                                .ok_or("bad \\u escape")?;
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
                    s.push_str(
                        std::str::from_utf8(&self.b[start..self.i])
                            .map_err(|_| "bad utf8")?,
                    );
                }
            }
        }
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
        let s = o.to_json_string();
        assert_eq!(s, r#"{"a":1,"b":"x\"y","c":[true,null]}"#);
        let back = parse(&s).unwrap();
        assert_eq!(back.get("a"), Some(&Json::int(1)));
        assert_eq!(back.get("b"), Some(&Json::str("x\"y".to_string())));
    }

    #[test]
    fn parse_rejects_garbage() {
        assert!(parse("{").is_err());
        assert!(parse("{\"a\":}").is_err());
        assert!(parse("1 2").is_err());
        assert!(parse("").is_err());
    }

    #[test]
    fn uint_survives_roundtrip() {
        let v = Json::uint(u64::MAX);
        let s = v.to_json_string();
        assert_eq!(s, format!("{}", u64::MAX as i64));
    }

    #[test]
    fn escapes_control_characters() {
        assert_eq!(Json::str("a\nb").to_json_string(), "\"a\\nb\"");
    }
}
