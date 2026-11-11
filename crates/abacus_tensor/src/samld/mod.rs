//! SAMLD ordered-binary subsystem (addendum `golden_source_samld_binary_addendum.md`).
//!
//! ```text
//! convert  CSV -> ordered core.bin + eval sidecar (row-14/24 shape)
//! direct   core.bin -> direct CSF, no COO step (P4 in memory, §S7)
//! pack64   64-bit coordinate-key report (key vs weights, §S8)
//! sets     exact set-op selection + correspondence (set-operation rule)
//! diag     P11 diagnostic math (gap U-SAMLD-1, §S5)
//! ```
//!
//! Injection preserved: this module is a new ingestion representation
//! feeding the existing tensor/CSF/ingest objects; it merges with none.

pub mod convert;
pub mod core_bin;
pub mod core_decode;
pub mod core_pack;
pub mod diag;
pub mod direct;
pub mod eval_codec;
pub mod eval_codec2;
pub mod eval_codec3;
pub mod eval_sidecar;
pub mod pack64;
pub mod sets;
pub mod stage;

pub use convert::{convert_csv, convert_reader, Conversion};
pub use core_bin::{serialize_core, CoreImage, CoreRecord};
pub use core_decode::deserialize_core;
pub use core_pack::{deserialize_core_v2, serialize_core_v2};
pub use direct::{direct_csf, project_entries};
pub use eval_codec::{deserialize_eval, serialize_eval};
pub use eval_codec2::{deserialize_eval_v2, serialize_eval_v2};
pub use eval_codec3::{deserialize_eval_v3, serialize_eval_v3};
pub use eval_sidecar::EvalRecord;
pub use pack64::{bit_report, pack_dense, pack_offset, unpack_dense, unpack_offset};
pub use sets::{index_set, tx_correspondence};
