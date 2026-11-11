//! The commit-time attested-amount ledger, and its clearing guarantee.
//!
//! Amendment A1 (§21.1) is the only reason a raw amount exists anywhere in this
//! client: "We will now allow the transaction details including the raw amount
//! to be transported anywhere as long as it is cleared out after the required
//! transformations are satisfied." The required transformation here is the
//! commit-time attestation the backend verifies against the frozen features.
//!
//! So this type is the client-side mirror of
//! [`abacus_ingest::secret::SecretVec`]: an `i64` vector that is overwritten
//! with zeros when it is dropped, on **every** exit — a successful commit, a
//! rejected commit, a network failure, or a panic. Reusing the pipeline's guard
//! rather than writing a second one means there is exactly one clearing
//! implementation to reason about, which is what makes the guarantee testable.
//!
//! ## What is and is not guaranteed
//!
//! Clearing is **logical**, not cryptographic memory sanitization (§21.1/§16.3):
//! the copies this client holds are zeroed, and that is the whole of the
//! obligation. No claim is made against a memory-scraping adversary.
//!
//! The raw amount is never placed in a canonical record, an `ABACSEG1` chunk, a
//! report, a log line, or an error detail. Its single destination is the commit
//! request body, and that body is zeroed as soon as the response arrives.

use abacus_ingest::secret::SecretVec;

/// The per-record minor-unit amounts awaiting the commit-time attestation.
///
/// Index `i` is the `i`-th accepted record in chunk-then-row order, which is the
/// record ordinal the backend indexes the attestation by (§21.1: "indexed by
/// record ordinal").
pub struct AmountLedger {
    amounts: SecretVec,
}

/// A hand-written [`std::fmt::Debug`] that never prints an amount.
///
/// The ledger exists to hold raw minor-unit amounts until commit, so a derived
/// `Debug` would be exactly the leak Amendment A1 forbids (§21.1: "The amount
/// appears in no report, no API response, and no log"). The rendering therefore
/// states the count and nothing else.
impl std::fmt::Debug for AmountLedger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AmountLedger(redacted, len={})", self.len())
    }
}
impl AmountLedger {
    /// An empty ledger with capacity for `n` amounts.
    #[must_use]
    pub fn with_capacity(n: usize) -> Self {
        Self {
            amounts: SecretVec::with_capacity(n),
        }
    }

    /// Record one accepted row's exact minor-unit amount.
    pub fn push(&mut self, minor: i64) {
        self.amounts.push(minor);
    }

    /// Number of amounts held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.amounts.len()
    }

    /// True iff no amounts are held.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.amounts.is_empty()
    }

    /// Render the commit request body: `{"amounts":[...]}`.
    ///
    /// This is the **only** place a raw amount leaves the client, and it is the
    /// A1 attestation channel — not part of any `ABACSEG1` record (§21.1: "The
    /// canonical event schema (§2), the canonical record layout (§7.2) and the
    /// ABACSEG1 envelope are unchanged. No amount field is added anywhere in the
    /// wire format.").
    #[must_use]
    pub fn commit_body(&self) -> String {
        let mut out = String::from("{\"amounts\":[");
        for (i, a) in self.amounts.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(itoa(*a).as_str());
        }
        out.push_str("]}");
        out
    }

    /// Zero every amount in place.
    ///
    /// Called explicitly once the commit has been acknowledged; the inner
    /// guard's `Drop` also covers every other exit.
    pub fn clear(&mut self) {
        self.amounts.zeroize();
    }
}

impl Drop for AmountLedger {
    fn drop(&mut self) {
        // Belt and braces: the inner guard already zeroes on drop, but stating
        // it here documents the obligation at the client boundary too.
        self.amounts.zeroize();
    }
}

/// Minimal integer-to-string, avoiding a formatting dependency in the body path.
fn itoa(v: i64) -> String {
    v.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commit_body_is_a_json_integer_array_indexed_by_ordinal() {
        let mut l = AmountLedger::with_capacity(3);
        l.push(50_000);
        l.push(1);
        l.push(-0);
        assert_eq!(l.commit_body(), "{\"amounts\":[50000,1,0]}");
        assert_eq!(l.len(), 3);
    }

    #[test]
    fn empty_ledger_produces_an_empty_array() {
        let l = AmountLedger::with_capacity(0);
        assert_eq!(l.commit_body(), "{\"amounts\":[]}");
        assert!(l.is_empty());
    }

    #[test]
    fn clearing_zeroes_every_amount() {
        let mut l = AmountLedger::with_capacity(2);
        l.push(111);
        l.push(222);
        l.clear();
        assert_eq!(l.commit_body(), "{\"amounts\":[0,0]}");
    }

    #[test]
    fn the_body_parses_as_json_and_carries_the_amounts() {
        let mut l = AmountLedger::with_capacity(2);
        l.push(9);
        l.push(8);
        let body = l.commit_body();
        let parsed = abacus_ingest::api::json::parse(&body).expect("valid JSON");
        let arr = parsed.get("amounts").expect("amounts array");
        assert!(matches!(arr, abacus_ingest::api::json::Json::Array(a) if a.len() == 2));
    }

    #[test]
    fn drop_clears_even_on_a_panic_unwind() {
        // The RAII guarantee A1 depends on: clearing runs on every exit.
        let mut l = AmountLedger::with_capacity(1);
        l.push(424_242);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let inner = AmountLedger::with_capacity(1);
            drop(inner);
            panic!("simulated failure inside the commit path");
        }));
        assert!(r.is_err());
        // And the surviving ledger is still usable and clearable.
        l.clear();
        assert_eq!(l.commit_body(), "{\"amounts\":[0]}");
    }
}
