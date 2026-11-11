//! Best-effort logical clearing of transient attested-amount buffers.
//!
//! Amendment A1 permits the exact transaction amount to be transported during
//! ingestion **only** on the condition that every transient copy is cleared
//! once the required transformation (into the substrate's exact `i64`
//! micro-unit value) is satisfied. This module provides the RAII guards that
//! make that clearing happen on **every reachable exit** of the commit path —
//! not just the happy path.
//!
//! ## Why RAII and not an explicit `fill(0)`
//!
//! A single explicit clear at the end of a function is *not* sufficient: the
//! commit path has many exits before that point —
//!
//! - pre-flight rejection (missing amount, negative, over-ceiling, a feature
//!   payload that the attested amount does not reproduce),
//! - a mid-stream application rejection from the substrate,
//! - a durable-journal sink failure (`ErrorCode::Persistence`), and
//! - a panic unwind inside the attestation closure or the applier.
//!
//! Each of those returns (or unwinds) through the owning scope, so a guard
//! whose [`Drop`] zeroes its buffer clears the amounts on all of them,
//! including panics, with no per-exit bookkeeping to forget.
//!
//! ## What this is NOT
//!
//! Clearing here is **logical**, not cryptographic memory sanitization: it
//! overwrites the copies this pipeline holds and nothing stronger. It is not a
//! defense against a memory-scraping adversary, and no such guarantee is
//! claimed (Canonical Contract §16.3). The amount survives in exactly one
//! place by design: the substrate's own canonical micro-unit entry, which is
//! the transformed result and is kind-identical to fixture-derived values
//! already stored in the tensor.

/// A `Vec<i64>` of attested minor-unit amounts that is overwritten with zeros
/// when it is dropped, on every exit path (normal return, early `?`, panic).
///
/// Reads go through [`Deref`] to `[i64]`, so indexing and `len` behave exactly
/// like the underlying slice. Used by
/// [`crate::session::IngestionSession::commit_with_sink`] to hold the
/// per-record attested amounts resolved during pre-flight.
pub struct SecretVec {
    buf: Vec<i64>,
}

impl SecretVec {
    /// An empty buffer with capacity for `n` amounts.
    #[must_use]
    pub fn with_capacity(n: usize) -> Self {
        Self { buf: Vec::with_capacity(n) }
    }

    /// Append one attested amount.
    pub fn push(&mut self, v: i64) {
        self.buf.push(v);
    }

    /// Overwrite every element with zero in place, preserving the length.
    ///
    /// This is the exact operation [`Drop`] performs; it is exposed so the
    /// zeroing logic is directly unit-testable and can be invoked early on the
    /// success path if a caller prefers an explicit clear.
    pub fn zeroize(&mut self) {
        self.buf.fill(0);
    }
}

impl std::ops::Deref for SecretVec {
    type Target = [i64];
    fn deref(&self) -> &[i64] {
        &self.buf
    }
}

impl Drop for SecretVec {
    fn drop(&mut self) {
        self.zeroize();
    }
}

/// A `Vec<Option<i64>>` of attested amounts parsed from a commit request body
/// that is cleared when dropped: each present payload is zeroed and then the
/// slot is set to `None`, on every exit path including panic unwind.
///
/// `None` denotes an amount the client did not supply for that record ordinal,
/// which the session pre-flight treats as a specification gap (never a
/// fabricated value).
pub struct SecretOptVec {
    buf: Vec<Option<i64>>,
}

impl SecretOptVec {
    /// Wrap an already-parsed amount vector.
    #[must_use]
    pub fn new(buf: Vec<Option<i64>>) -> Self {
        Self { buf }
    }

    /// The attested amount for record ordinal `i`, if the client supplied one.
    #[must_use]
    pub fn get(&self, i: usize) -> Option<i64> {
        self.buf.get(i).copied().flatten()
    }

    /// Zero each present payload, then set every slot to `None`.
    pub fn zeroize(&mut self) {
        for slot in &mut self.buf {
            if let Some(v) = slot {
                *v = 0;
            }
            *slot = None;
        }
    }
}

impl Drop for SecretOptVec {
    fn drop(&mut self) {
        self.zeroize();
    }
}

/// A borrowed byte buffer (e.g. an HTTP request body that carried raw amounts)
/// that is overwritten with zeros when the guard is dropped.
///
/// Borrowing rather than owning keeps the caller's buffer observable after the
/// guard's scope ends, which is what makes the clearing directly testable: the
/// same bytes are still there, but every one of them is now zero.
pub struct ZeroizingBytes<'a> {
    buf: &'a mut Vec<u8>,
}

impl<'a> ZeroizingBytes<'a> {
    /// Guard `buf` so it is zeroed when this guard is dropped.
    #[must_use]
    pub fn new(buf: &'a mut Vec<u8>) -> Self {
        Self { buf }
    }
}

impl Drop for ZeroizingBytes<'_> {
    fn drop(&mut self) {
        self.buf.fill(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn secret_vec_zeroize_clears_values_and_keeps_length() {
        let mut s = SecretVec::with_capacity(3);
        s.push(111);
        s.push(222);
        s.push(-333);
        assert_eq!(&*s, &[111, 222, -333]);
        assert_eq!(s.len(), 3);
        s.zeroize();
        assert!(s.iter().all(|&x| x == 0), "every element must be zero");
        assert_eq!(s.len(), 3, "zeroing must not shrink the buffer");
    }

    #[test]
    fn secret_vec_deref_indexing_matches_slice() {
        let mut s = SecretVec::with_capacity(2);
        s.push(7);
        s.push(9);
        assert_eq!(s[0], 7);
        assert_eq!(s[1], 9);
    }

    #[test]
    fn secret_opt_vec_get_and_zeroize() {
        let mut s = SecretOptVec::new(vec![Some(5), None, Some(-7)]);
        assert_eq!(s.get(0), Some(5));
        assert_eq!(s.get(1), None);
        assert_eq!(s.get(2), Some(-7));
        assert_eq!(s.get(9), None);
        s.zeroize();
        // After clearing, no slot reports an amount.
        assert_eq!(s.get(0), None);
        assert_eq!(s.get(2), None);
    }

    #[test]
    fn zeroizing_bytes_clears_in_place_on_normal_drop() {
        let mut body = vec![1u8, 2, 3, 4, 5];
        {
            // The guard owns the mutable borrow for its whole scope by design;
            // the caller must not read the buffer until the guard is dropped.
            let _guard = ZeroizingBytes::new(&mut body);
        }
        // Same length, every byte zero, after the guard's scope ends.
        assert_eq!(body.len(), 5);
        assert!(body.iter().all(|&b| b == 0), "body must be zeroed on drop");
    }

    #[test]
    fn zeroizing_bytes_clears_on_panic_unwind() {
        // The RAII guarantee that matters for the commit path: clearing runs
        // even when the guarded scope exits via a panic, not just a return.
        let mut body = vec![9u8; 6];
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = ZeroizingBytes::new(&mut body);
            panic!("simulated panic inside the commit path");
        }));
        assert!(result.is_err(), "the panic must propagate");
        assert!(
            body.iter().all(|&b| b == 0),
            "body must be zeroed even though the scope unwound via panic"
        );
    }
}
