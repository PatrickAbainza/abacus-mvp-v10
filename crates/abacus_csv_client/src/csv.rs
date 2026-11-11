//! A bounded, allocation-explicit CSV reader.
//!
//! Golden Source §4.1: "Use a streaming CSV parser... Do not require the
//! complete file or complete normalized dataset to reside in browser memory."
//! and §18: "The implementation must use bounded processing and explicit
//! resource limits."
//!
//! So this reader is a **single forward pass** over the file's bytes that
//! produces one row at a time; the caller decides how many to buffer. The
//! convenience [`parse_table`] collects a bounded sample for header inspection
//! and the normalization preview (§12 step 2: "Inspect the header and a bounded
//! sample"), which is what a UI needs, and it enforces its own row bound rather
//! than growing without limit.
//!
//! ## Quoting (RFC 4180 subset, and nothing more)
//!
//! - A field may be wrapped in `"`; inside it, `""` is a literal quote and a
//!   newline is data.
//! - A quote in the middle of an unquoted field is **data**, not a quoting
//!   error, because that is what real exports contain.
//! - Both `LF` and `CRLF` terminate a record; a lone `CR` is data (some
//!   exporters emit it inside fields).
//! - Nothing is coerced: a row with the wrong field count is returned as-is and
//!   the mapping layer rejects it, so the error names the offending source line
//!   instead of silently padding or truncating (§19: "never silently clamp,
//!   saturate, truncate").
//!
//! No CSV is written anywhere by this crate. CSV is an *input* format here, and
//! the workspace rule that keeps CSV out of active pipelines is about outputs
//! and internal transport — the upload payload is the binary `ABACSEG1` segment.

use abacus_ingest::error::{ErrorCode, IngestError, IngestResult};

/// Reader options. All of them are explicit; nothing is guessed.
///
/// §9 Stage 1: "Do not guess missing configuration when the guess changes event
/// meaning." A delimiter or header choice changes which column a value comes
/// from, so it must be stated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CsvOptions {
    /// Field delimiter. `,` by convention; `;` and tab are common in exports.
    pub delimiter: u8,
    /// Quote character, `"` by convention.
    pub quote: u8,
    /// Whether the first record is a header row to be mapped by name.
    pub has_header: bool,
    /// Maximum rows to materialize in a [`CsvTable`] (bounded preview).
    pub max_preview_rows: usize,
    /// Maximum bytes of one field. Bounds a hostile single field before it is
    /// allocated (§18).
    pub max_field_bytes: usize,
}

impl Default for CsvOptions {
    /// Comma-delimited, double-quoted, with a header row.
    fn default() -> Self {
        Self {
            delimiter: b',',
            quote: b'"',
            has_header: true,
            max_preview_rows: 100,
            max_field_bytes: 64 * 1024,
        }
    }
}

/// One parsed data row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CsvRow {
    /// 1-based **source line number** of the record's first physical line.
    ///
    /// This is the identifier the UI shows for a problem row (§12: "Use
    /// source-row numbers or opaque client-side references to identify problem
    /// rows"). It is the physical line, so a quoted embedded newline still
    /// reports where the record started.
    pub line: usize,
    /// The row's fields, in source order, unescaped.
    pub fields: Vec<String>,
}

/// A bounded CSV document: the header (if declared) plus a bounded sample of
/// data rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CsvTable {
    /// Header field names in source order, empty when `has_header` is false.
    pub header: Vec<String>,
    /// The parsed rows that fit inside [`CsvOptions::max_preview_rows`].
    pub rows: Vec<CsvRow>,
    /// True iff more rows existed than `max_preview_rows` allowed.
    ///
    /// Surfaced rather than hidden: a preview that silently stops is
    /// indistinguishable from a file that ended (§19 fail-closed spirit).
    pub truncated: bool,
    /// Total data rows actually parsed (including any beyond the sample bound).
    pub total_rows: usize,
}
/// The single-pass parser state machine.
///
/// Kept separate from [`CsvTable`] so a caller that wants true streaming can
/// pull rows without materializing a table; [`iter_rows`] exposes that. Fields
/// accumulate as raw bytes and are validated as UTF-8 once per field, so an
/// invalid sequence is a parse error rather than a silent replacement.
struct RowParser<'a> {
    bytes: &'a [u8],
    pos: usize,
    /// 1-based physical line counter, incremented on every newline seen.
    line: usize,
    opts: &'a CsvOptions,
}

/// Iterate every data row of `bytes` under `opts`, without a row bound.
///
/// The closure returns `false` to stop early, which is how a caller bounds its
/// own memory: the parser never allocates the whole file.
///
/// # Errors
/// [`ErrorCode::FileOrCsvParse`] for an unterminated quoted field, a field that
/// exceeds [`CsvOptions::max_field_bytes`], or non-UTF-8 field bytes.
pub fn iter_rows<F>(bytes: &[u8], opts: &CsvOptions, mut on_row: F) -> IngestResult<()>
where
    F: FnMut(CsvRow) -> bool,
{
    let mut p = RowParser {
        bytes,
        pos: 0,
        line: 1,
        opts,
    };
    if opts.has_header && p.pos < p.bytes.len() {
        // Consume and discard the header record; its line is not a data row.
        p.next_record()?;
    }
    while p.pos < p.bytes.len() {
        let start_line = p.line;
        let fields = p.next_record()?;
        // A completely empty final record is not a row. This is the one
        // tolerance applied, and only to a trailing newline: an empty line
        // *between* records is a real one-field row and is rejected downstream
        // by the mapping, because silently dropping interior blank rows would
        // change the record count.
        if fields.len() == 1 && fields[0].is_empty() && p.pos >= p.bytes.len() {
            break;
        }
        if !on_row(CsvRow {
            line: start_line,
            fields,
        }) {
            break;
        }
    }
    Ok(())
}

impl<'a> RowParser<'a> {
    /// Parse one record: a vector of unescaped fields.
    fn next_record(&mut self) -> IngestResult<Vec<String>> {
        let mut fields: Vec<String> = Vec::with_capacity(8);
        let mut field: Vec<u8> = Vec::new();
        loop {
            let Some(&b) = self.bytes.get(self.pos) else {
                // EOF ends the final record.
                fields.push(self.finish_field(field)?);
                return Ok(fields);
            };
            match b {
                b if b == self.opts.quote && field.is_empty() => {
                    // Opening quote: read until the closing quote.
                    self.pos += 1;
                    self.read_quoted(&mut field)?;
                }
                b if b == self.opts.delimiter => {
                    fields.push(self.finish_field(std::mem::take(&mut field))?);
                    self.pos += 1;
                }
                b'\n' => {
                    fields.push(self.finish_field(field)?);
                    self.pos += 1;
                    self.line += 1;
                    return Ok(fields);
                }
                b'\r' if self.bytes.get(self.pos + 1) == Some(&b'\n') => {
                    fields.push(self.finish_field(field)?);
                    self.pos += 2;
                    self.line += 1;
                    return Ok(fields);
                }
                _ => {
                    self.field_push(&mut field, b)?;
                    self.pos += 1;
                }
            }
        }
    }

    /// Read the body of a quoted field. The opening quote is already consumed;
    /// this returns just after the closing quote.
    fn read_quoted(&mut self, field: &mut Vec<u8>) -> IngestResult<()> {
        loop {
            let Some(&b) = self.bytes.get(self.pos) else {
                return Err(IngestError::new(
                    ErrorCode::FileOrCsvParse,
                    format!("unterminated quoted field at source line {}", self.line),
                ));
            };
            match b {
                b if b == self.opts.quote => {
                    if self.bytes.get(self.pos + 1) == Some(&self.opts.quote) {
                        // `""` inside a quoted field is one literal quote.
                        field.push(self.opts.quote);
                        self.pos += 2;
                    } else {
                        // Closing quote. Bytes before the next delimiter or
                        // newline are tolerated as trailing data on this field
                        // rather than failing the whole file, but they cannot
                        // re-open quoting.
                        self.pos += 1;
                        while let Some(&c) = self.bytes.get(self.pos) {
                            if c == self.opts.delimiter || c == b'\n' {
                                break;
                            }
                            if c == b'\r' && self.bytes.get(self.pos + 1) == Some(&b'\n') {
                                break;
                            }
                            self.field_push(field, c)?;
                            self.pos += 1;
                        }
                        return Ok(());
                    }
                }
                b'\n' => {
                    field.push(b'\n');
                    self.pos += 1;
                    self.line += 1;
                }
                b'\r' if self.bytes.get(self.pos + 1) == Some(&b'\n') => {
                    field.push(b'\r');
                    field.push(b'\n');
                    self.pos += 2;
                    self.line += 1;
                }
                _ => {
                    self.field_push(field, b)?;
                    self.pos += 1;
                }
            }
        }
    }

    /// Append one raw byte to a field, enforcing the field-size bound.
    fn field_push(&self, field: &mut Vec<u8>, b: u8) -> IngestResult<()> {
        if field.len() >= self.opts.max_field_bytes {
            return Err(IngestError::new(
                ErrorCode::FileOrCsvParse,
                format!(
                    "a field exceeds the {}-byte limit at source line {}",
                    self.opts.max_field_bytes, self.line
                ),
            ));
        }
        field.push(b);
        Ok(())
    }

    /// Validate a finished field as UTF-8 and hand it over as a `String`.
    fn finish_field(&self, field: Vec<u8>) -> IngestResult<String> {
        String::from_utf8(field).map_err(|_| {
            IngestError::new(
                ErrorCode::FileOrCsvParse,
                format!("non-UTF-8 bytes in a field at source line {}", self.line),
            )
        })
    }
}

/// Parse a bounded table: header plus up to `opts.max_preview_rows` data rows.
///
/// Used by the preview UI (§12 step 2/step 5) and by tests.
///
/// # Errors
/// As [`iter_rows`].
pub fn parse_table(bytes: &[u8], opts: &CsvOptions) -> IngestResult<CsvTable> {
    let mut header: Vec<String> = Vec::new();
    let mut rows: Vec<CsvRow> = Vec::new();
    let mut truncated = false;
    let mut total_rows = 0usize;

    if opts.has_header {
        // Read the first record through the same machine so quoting rules
        // cannot differ between header and data.
        let mut p = RowParser {
            bytes,
            pos: 0,
            line: 1,
            opts,
        };
        if p.pos < p.bytes.len() {
            header = p.next_record()?;
        }
    }
    iter_rows(bytes, opts, |row| {
        total_rows += 1;
        if rows.len() < opts.max_preview_rows {
            rows.push(row);
        } else {
            truncated = true;
        }
        true
    })?;
    Ok(CsvTable {
        header,
        rows,
        truncated,
        total_rows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts() -> CsvOptions {
        CsvOptions::default()
    }

    #[test]
    fn parses_header_and_rows_with_line_numbers() {
        let t = parse_table(b"a,b,c\n1,2,3\n4,5,6\n", &opts()).unwrap();
        assert_eq!(t.header, vec!["a", "b", "c"]);
        assert_eq!(t.total_rows, 2);
        assert_eq!(t.rows[0].line, 2);
        assert_eq!(t.rows[0].fields, vec!["1", "2", "3"]);
        assert_eq!(t.rows[1].line, 3);
        assert!(!t.truncated);
    }

    #[test]
    fn quoted_field_with_comma_newline_and_escaped_quote() {
        let t = parse_table(b"x,y\n\"a,b\",\"l1\nl2\"\n\"say \"\"hi\"\"\",2\n", &opts()).unwrap();
        assert_eq!(t.rows[0].fields, vec!["a,b", "l1\nl2"]);
        // The record started on physical line 2 even though it spans two lines.
        assert_eq!(t.rows[0].line, 2);
        assert_eq!(t.rows[1].line, 4);
        assert_eq!(t.rows[1].fields[0], "say \"hi\"");
    }

    #[test]
    fn crlf_records_are_handled() {
        let t = parse_table(b"a,b\r\n1,2\r\n", &opts()).unwrap();
        assert_eq!(t.rows.len(), 1);
        assert_eq!(t.rows[0].fields, vec!["1", "2"]);
    }

    #[test]
    fn ragged_rows_are_returned_not_padded() {
        // A short row and a long row both survive parsing; rejection is the
        // mapping layer's job, with the line number attached.
        let t = parse_table(b"a,b,c\n1,2\n1,2,3,4\n", &opts()).unwrap();
        assert_eq!(t.rows[0].fields.len(), 2);
        assert_eq!(t.rows[1].fields.len(), 4);
    }

    #[test]
    fn trailing_newline_does_not_create_a_phantom_row() {
        let t = parse_table(b"a\n1\n2\n", &opts()).unwrap();
        assert_eq!(t.total_rows, 2);
    }

    #[test]
    fn interior_blank_line_is_a_real_row() {
        let t = parse_table(b"a,b\n1,2\n\n3,4\n", &opts()).unwrap();
        assert_eq!(t.total_rows, 3, "an interior blank row must still count");
        assert_eq!(t.rows[1].fields, vec![""]);
        assert_eq!(t.rows[1].line, 3);
    }

    #[test]
    fn preview_is_bounded_and_says_so() {
        let mut csv = String::from("a\n");
        for i in 0..50 {
            csv.push_str(&format!("{i}\n"));
        }
        let o = CsvOptions {
            max_preview_rows: 10,
            ..opts()
        };
        let t = parse_table(csv.as_bytes(), &o).unwrap();
        assert_eq!(t.rows.len(), 10);
        assert_eq!(t.total_rows, 50);
        assert!(t.truncated);
    }

    #[test]
    fn unterminated_quote_is_rejected() {
        let e = parse_table(b"a,b\n\"unclosed,2\n", &opts()).unwrap_err();
        assert_eq!(e.code, ErrorCode::FileOrCsvParse);
    }

    #[test]
    fn oversized_field_is_rejected_before_allocation_grows() {
        let o = CsvOptions {
            max_field_bytes: 8,
            ..opts()
        };
        let e = parse_table(b"a\n0123456789\n", &o).unwrap_err();
        assert_eq!(e.code, ErrorCode::FileOrCsvParse);
    }

    #[test]
    fn delimiter_option_is_load_bearing_not_guessed() {
        let o = CsvOptions {
            delimiter: b';',
            ..opts()
        };
        let t = parse_table(b"a;b\n1;2\n", &o).unwrap();
        assert_eq!(t.rows[0].fields, vec!["1", "2"]);
        // With the default comma the same bytes are ONE field.
        let t2 = parse_table(b"a;b\n1;2\n", &opts()).unwrap();
        assert_eq!(t2.rows[0].fields, vec!["1;2"]);
    }

    #[test]
    fn leading_zeros_survive_parsing() {
        // §6: "Preserve leading zeros". The parser must not numeric-coerce.
        let t = parse_table(b"acct\n000123\n", &opts()).unwrap();
        assert_eq!(t.rows[0].fields[0], "000123");
    }

    #[test]
    fn utf8_fields_round_trip_exactly() {
        let t = parse_table("name\nJosé\n".as_bytes(), &opts()).unwrap();
        assert_eq!(t.rows[0].fields[0], "José");
    }

    #[test]
    fn non_utf8_field_is_rejected() {
        let e = parse_table(b"a\n\xff\xfe\n", &opts()).unwrap_err();
        assert_eq!(e.code, ErrorCode::FileOrCsvParse);
    }

    #[test]
    fn headerless_mode_treats_the_first_record_as_data() {
        let o = CsvOptions {
            has_header: false,
            ..opts()
        };
        let t = parse_table(b"1,2\n3,4\n", &o).unwrap();
        assert!(t.header.is_empty());
        assert_eq!(t.total_rows, 2);
        assert_eq!(t.rows[0].line, 1);
    }

    #[test]
    fn iter_rows_can_stop_early_to_bound_memory() {
        let mut csv = String::from("a\n");
        for i in 0..1000 {
            csv.push_str(&format!("{i}\n"));
        }
        let mut seen = 0usize;
        iter_rows(csv.as_bytes(), &opts(), |_row| {
            seen += 1;
            seen < 5
        })
        .unwrap();
        assert_eq!(seen, 5, "early stop must not parse the whole file");
    }
}
