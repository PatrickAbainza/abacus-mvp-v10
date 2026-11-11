//! CSF layer: rows 11, 15, 16 (§B.2) — P6, the lossless compressed fiber form.
//!
//! ```text
//! row 11  compress (+ canonical CSF invariants)   -> compress
//! row 15  CSF decode / decode-to-tensor           -> decode
//! row 16  time slice, point lookup, sender/receiver at time -> query
//! ```

pub mod compress;
pub mod compress_v2;
pub mod decode;
pub mod query;
pub mod serialize;

pub use compress::{compress, no_self_aggregation, CsfTensor};
pub use compress_v2::{deserialize_csf_v2, serialize_csf_v2};
pub use decode::{decode_csf, decode_to_tensor};
pub use query::{lookup, receiver_at_time, sender_at_time, time_slice, TimeSlice};
pub use serialize::{checksum, deserialize, serialize, Header, Image};
