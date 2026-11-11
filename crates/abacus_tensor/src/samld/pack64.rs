//! 64-bit coordinate-key packing (addendum §S8).
//!
//! The key packs ONLY the coordinate `(t,s,r)`; weights / amount / tx_id
//! are values, never key bits. Two feasible layouts for the 90-day
//! cardinalities (|T|=2147219 → 22b dense / 23b offset,
//! |S|=121869 → 17b, |R|=410242 → 19b):
//!
//! ```text
//! dense:  key = (t_idx << 36) | (s << 19) | r     58 bits, order-preserving
//! offset: key = ((t-t_min) << 36) | (s << 19) | r 59 bits, order-preserving
//! raw epoch: t 31b + s 17b + r 19b = 67b > 64     NOT feasible (not forced)
//! ```
//!
//! Bit widths as `u32` constants; total functions returning `Option`
//! (fail-closed on out-of-range, `R_5`).

/// Dense-rank widths.
pub const T_DENSE_BITS: u32 = 22;
/// Sender code width.
pub const S_BITS: u32 = 17;
/// Receiver code width.
pub const R_BITS: u32 = 19;
/// Sender shift.
pub const S_SHIFT: u32 = R_BITS;
/// Time shift (dense and offset share s/r widths).
pub const T_SHIFT: u32 = S_BITS + R_BITS;
/// Epoch-offset time width (`t - t_min` spans 7 737 879 s → 23 bits).
pub const T_OFF_BITS: u32 = 23;

/// Maximum dense time index (`|T| - 1` bound for the 90-day population).
pub const T_DENSE_MAX: u64 = 2147219;
/// Maximum sender code.
pub const S_MAX: u32 = 121869;
/// Maximum receiver code.
pub const R_MAX: u32 = 410242;

/// Pack dense ranks `(t_idx, s, r)` into one `u64` (`None` if out of range).
#[must_use]
pub fn pack_dense(t_idx: u64, s: u32, r: u32) -> Option<u64> {
    if t_idx >= (1u64 << T_DENSE_BITS) || u64::from(s) >= (1u64 << S_BITS) || u64::from(r) >= (1u64 << R_BITS) {
        return None;
    }
    Some((t_idx << T_SHIFT) | (u64::from(s) << S_SHIFT) | u64::from(r))
}

/// Unpack a dense key.
#[must_use]
pub fn unpack_dense(key: u64) -> (u64, u32, u32) {
    (
        key >> T_SHIFT,
        ((key >> S_SHIFT) & ((1u64 << S_BITS) - 1)) as u32,
        (key & ((1u64 << R_BITS) - 1)) as u32,
    )
}

/// Pack `(t - t_min, s, r)` epoch-offset key (`None` if out of range).
#[must_use]
pub fn pack_offset(t_off: u64, s: u32, r: u32) -> Option<u64> {
    if t_off >= (1u64 << T_OFF_BITS) || u64::from(s) >= (1u64 << S_BITS) || u64::from(r) >= (1u64 << R_BITS) {
        return None;
    }
    Some((t_off << T_SHIFT) | (u64::from(s) << S_SHIFT) | u64::from(r))
}

/// Unpack an offset key.
#[must_use]
pub fn unpack_offset(key: u64) -> (u64, u32, u32) {
    unpack_dense(key)
}

/// Required bit allocation report for the 90-day dataset.
#[must_use]
pub fn bit_report() -> String {
    "dense: t_idx 22b + s 17b + r 19b = 58b <= 64 FEASIBLE; \
     offset: (t-t_min) 23b + s 17b + r 19b = 59b <= 64 FEASIBLE; \
     raw: t 31b + s 17b + r 19b = 67b > 64 NOT FEASIBLE"
        .to_string()
}
