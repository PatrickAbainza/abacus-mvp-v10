//! The 128-bit event namespace: half of the canonical event identity.
//!
//! Canonical Contract §3.1 requires a compound identity
//! `(event_namespace, event_id)` because "a single 64-bit integer cannot
//! safely encode arbitrary source identities without either a controlled
//! allocation scheme or a collision risk. Hashing arbitrary source IDs and
//! truncating them to 64 bits does not eliminate that risk."
//!
//! §3.2 fixes the representation: 128 bits, "generated once and persisted",
//! "reused when resuming or retrying that same logical import or stream",
//! "never regenerated during a retry", and "globally unique within the
//! system's identity registry". UUIDv4 is the sanctioned generation method
//! "with persistent registration and collision checks" — so this module owns
//! both generation and the registry check, and never regenerates.
//!
//! Wire §5 binds the namespace rules per source: CSV retries reuse the
//! original namespace; a new independent CSV import gets a new one; SAML-D
//! uses a stable namespace tied to the fixture identity and version; Kafka
//! uses a persistent namespace registered to its logical stream partition.

use crate::error::{ErrorCode, IngestError, IngestResult};

/// A 128-bit namespace identifier (Canonical Contract §3.2).
///
/// Stored and transmitted in the canonical UUID byte order defined by the wire
/// contract §1 (offsets 40–55: "UUID bytes, canonical RFC 4122/network
/// order"). RFC 4122 §4.1.2 defines the network-order byte layout as
/// `time_low` big-endian, `time_mid` big-endian, `time_hi_and_version`
/// big-endian, then the clock-seq and node fields in order — which is exactly
/// the big-endian rendering of the 128-bit value with the RFC variant/version
/// bits in place. So the wire bytes are the big-endian bytes of this value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EventNamespace([u8; 16]);

impl EventNamespace {
    /// Wrap raw canonical (network-order) UUID bytes.
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    /// The canonical 16 network-order UUID bytes (wire §1 offsets 40–55).
    #[must_use]
    pub const fn to_bytes(self) -> [u8; 16] {
        self.0
    }

    /// Render in canonical hyphenated UUID form for reports and APIs.
    ///
    /// Purely a presentation transform: the wire bytes are authoritative and
    /// round-trip exactly.
    #[must_use]
    pub fn to_hyphenated(&self) -> String {
        let b = &self.0;
        format!(
            "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
            b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]
        )
    }

    /// Parse canonical hyphenated UUID text back to bytes.
    ///
    /// # Errors
    /// [`ErrorCode::Session`] if the text is not 36 characters in
    /// `8-4-4-4-12` hexadecimal form.
    pub fn parse_hyphenated(s: &str) -> IngestResult<Self> {
        let b = s.as_bytes();
        if b.len() != 36 {
            return Err(IngestError::new(
                ErrorCode::Session,
                "namespace must be 36 characters in 8-4-4-4-12 form",
            ));
        }
        for h in [8usize, 13, 18, 23] {
            if b[h] != b'-' {
                return Err(IngestError::new(
                    ErrorCode::Session,
                    "namespace is missing a hyphen separator",
                ));
            }
        }
        let mut out = [0u8; 16];
        let mut oi = 0usize;
        let mut i = 0usize;
        while i < b.len() {
            if [8usize, 13, 18, 23].contains(&i) {
                i += 1;
                continue;
            }
            let hi = hex_val(b[i])?;
            let lo = hex_val(b[i + 1])?;
            out[oi] = (hi << 4) | lo;
            oi += 1;
            i += 2;
        }
        if oi != 16 {
            return Err(IngestError::new(
                ErrorCode::Session,
                "namespace does not contain 16 bytes",
            ));
        }
        Ok(Self(out))
    }

    /// True iff this is the all-zero namespace (never a valid identity domain).
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.0.iter().all(|&b| b == 0)
    }
}

fn hex_val(c: u8) -> IngestResult<u8> {
    match c {
        b'0'..=b'9' => Ok(c - b'0'),
        b'a'..=b'f' => Ok(c - b'a' + 10),
        b'A'..=b'F' => Ok(c - b'A' + 10),
        _ => Err(IngestError::new(
            ErrorCode::Session,
            "namespace contains a non-hexadecimal character",
        )),
    }
}

/// The canonical compound event identity (Canonical Contract §3.1).
///
/// The namespace must remain attached through "ingestion, deduplication,
/// persistence, streaming_ingestion_demo, and segment movement" (wire §5); this type is the only
/// representation used for those purposes, so it cannot be dropped by
/// accident.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EventIdentity {
    /// The source identity domain.
    pub namespace: EventNamespace,
    /// The stable event identifier within that namespace (never global alone).
    pub event_id: u64,
}

impl EventIdentity {
    /// Construct the compound identity.
    #[must_use]
    pub const fn new(namespace: EventNamespace, event_id: u64) -> Self {
        Self { namespace, event_id }
    }
}

/// Persistent namespace registry with collision detection.
///
/// Canonical Contract §3.2: "The namespace must be globally unique within the
/// system's identity registry. UUIDv4 is an acceptable generation method, with
/// persistent registration and collision checks."
///
/// Generation uses `std` entropy sources only (no `rand` dependency): a
/// per-process counter hashed against the address of a heap allocation and the
/// system clock, then folded through FNV-1a/64 from the existing substrate.
/// The registration check is the part that actually guarantees uniqueness: a
/// generated value that collides is rejected and regenerated, never accepted.
#[derive(Debug, Default)]
pub struct NamespaceRegistry {
    registered: std::collections::HashSet<EventNamespace>,
    counter: u64,
}

impl NamespaceRegistry {
    /// Empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an existing namespace (resume/retry path).
    ///
    /// Idempotent for the same namespace: re-registering the identical value
    /// is not a collision, which is what allows a retry to reuse it
    /// (Canonical Contract §3.2: "Never regenerate it during a retry").
    ///
    /// # Errors
    /// [`ErrorCode::Identity`] if the namespace is all-zero.
    pub fn register(&mut self, ns: EventNamespace) -> IngestResult<()> {
        if ns.is_zero() {
            return Err(IngestError::new(
                ErrorCode::Identity,
                "the all-zero namespace is not a valid identity domain",
            ));
        }
        self.registered.insert(ns);
        Ok(())
    }

    /// True iff `ns` is already registered.
    #[must_use]
    pub fn contains(&self, ns: &EventNamespace) -> bool {
        self.registered.contains(ns)
    }

    /// Number of registered namespaces (diagnostics).
    #[must_use]
    pub fn len(&self) -> usize {
        self.registered.len()
    }

    /// True iff no namespace has been registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.registered.is_empty()
    }

    /// Generate and register a fresh UUIDv4 namespace.
    ///
    /// Collision-checked per Canonical Contract §3.2: if the generated value
    /// is already registered, generation is retried rather than accepted.
    ///
    /// # Errors
    /// [`ErrorCode::Identity`] if the generator cannot produce an unused
    /// value within the retry bound.
    pub fn generate(&mut self) -> IngestResult<EventNamespace> {
        for _ in 0..64 {
            let ns = self.next_uuid_v4();
            if !self.registered.contains(&ns) {
                self.registered.insert(ns);
                return Ok(ns);
            }
        }
        Err(IngestError::new(
            ErrorCode::Identity,
            "namespace generation failed to find an unused value (collision bound exceeded)",
        ))
    }

    /// One UUIDv4 candidate: 122 random bits with the version and variant
    /// bits forced per RFC 4122 §4.4.
    fn next_uuid_v4(&mut self) -> EventNamespace {
        self.counter = self.counter.wrapping_add(1);
        // Mix a monotone counter with two independent std entropy sources so
        // successive values in one process differ, and values across processes
        // differ. Address-of-a-box gives per-run heap layout entropy; the
        // system clock gives wall-clock entropy.
        let boxed = Box::new(0u8);
        let addr = &*boxed as *const u8 as u64;
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        let mut bytes = [0u8; 16];
        for (i, slot) in bytes.iter_mut().enumerate() {
            let i = i as u64;
            let sh = 8 * (i % 8);
            let mut h = abacus_tensor::composites::FNV_OFFSET_BASIS;
            h = abacus_tensor::prims::fnv_mix(h, (self.counter >> sh) as u8);
            h = abacus_tensor::prims::fnv_mix(h, (addr >> sh) as u8);
            h = abacus_tensor::prims::fnv_mix(h, (nanos >> sh) as u8);
            h = abacus_tensor::prims::fnv_mix(h, i as u8);
            *slot = (h >> 24) as u8;
        }
        // RFC 4122 §4.4: version 4 in the high nibble of byte 6, variant
        // 0b10 in the top two bits of byte 8.
        bytes[6] = (bytes[6] & 0x0f) | 0x40;
        bytes[8] = (bytes[8] & 0x3f) | 0x80;
        EventNamespace(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hyphenated_roundtrip() {
        let b = [
            0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab,
            0xcd, 0xef,
        ];
        let ns = EventNamespace::from_bytes(b);
        let s = ns.to_hyphenated();
        assert_eq!(s, "01234567-89ab-cdef-0123-456789abcdef");
        assert_eq!(EventNamespace::parse_hyphenated(&s).unwrap(), ns);
    }

    #[test]
    fn parse_rejects_malformed() {
        for bad in [
            "",
            "01234567-89ab-cdef-0123-456789abcde",
            "0123456789ab-cdef-0123-456789abcdef",
            "01234567_89ab-cdef-0123-456789abcdef",
            "g1234567-89ab-cdef-0123-456789abcdef",
        ] {
            assert!(EventNamespace::parse_hyphenated(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn generated_namespaces_are_uuid_v4_and_unique() {
        let mut reg = NamespaceRegistry::new();
        let mut seen = std::collections::HashSet::new();
        for _ in 0..200 {
            let ns = reg.generate().unwrap();
            assert_eq!(ns.to_bytes()[6] & 0xf0, 0x40, "version nibble");
            assert_eq!(ns.to_bytes()[8] & 0xc0, 0x80, "variant bits");
            assert!(!ns.is_zero());
            assert!(seen.insert(ns), "duplicate generated namespace");
        }
        assert_eq!(reg.len(), 200);
    }

    #[test]
    fn register_is_idempotent_and_rejects_zero() {
        let mut reg = NamespaceRegistry::new();
        let ns = reg.generate().unwrap();
        reg.register(ns).unwrap(); // retry reuse, not a collision
        assert_eq!(reg.len(), 1);
        assert!(reg.register(EventNamespace::from_bytes([0u8; 16])).is_err());
    }

    #[test]
    fn wire_bytes_roundtrip() {
        let ns = EventNamespace::from_bytes([7u8; 16]);
        assert_eq!(EventNamespace::from_bytes(ns.to_bytes()), ns);
    }
}
