//! COO layer: rows 10–14 (§B.2) — raw and canonical coordinate lists, the
//! account dictionary, the CSV build and the remaining COO operations.
//!
//! ```text
//! row 10  direct-sum decode / coordinate evaluation -> decode::eval_at (settle via tensor::decode_sum)
//! row 11  canonicalize + canonicity checks          -> encode::{canonicalize, is_canonical, is_sorted}
//! row 12  account dictionary construction/lookup    -> builder::AccountDict (a11 over prec_dict)
//! row 13  civil date -> epoch seconds               -> composites::epoch_seconds (d_hinnant)
//! row 14  dataset build from CSV                    -> builder::build_from_reader
//! ```

pub mod builder;
pub mod decode;
pub mod encode;
pub mod entry;

pub use builder::{build_from_csv, build_from_reader, AccountDict, CanonicalBuild};
pub use decode::{decode_to_tensor, eval_at, try_eval_at};
pub use encode::{canonicalize, is_canonical, is_sorted};
pub use entry::{RawEntry, TxEntry};
