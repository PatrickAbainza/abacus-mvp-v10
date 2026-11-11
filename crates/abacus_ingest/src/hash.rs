//! SHA-256 digests required by the ABACSEG1 wire contract.
//!
//! Wire §4 defines three digests, all SHA-256:
//! - `payload_checksum`  — over the exact payload bytes only (no header),
//! - `config_fingerprint` — over the exact configuration-manifest bytes,
//! - `header_checksum`   — over header bytes 0–143 inclusive.
//!
//! Wire §4 also states the security scope precisely: "Checksums detect
//! corruption; they do not authenticate the producer." These are integrity
//! checks, never signatures, and no code here may present them as
//! authentication.
//!
//! The existing substrate (`abacus_tensor`) is std-only and provides FNV-1a/64
//! for its own artifact provenance. FNV is not acceptable here: the wire
//! contract mandates SHA-256 by name, so this module is the single place the
//! digest is computed and it is the only added dependency.

use crate::config::SHA256_LEN;
use sha2::{Digest, Sha256};

/// SHA-256 of `data`, returned as a fixed 32-byte digest.
#[must_use]
pub fn sha256(data: &[u8]) -> [u8; SHA256_LEN] {
    let mut h = Sha256::new();
    h.update(data);
    let out = h.finalize();
    let mut b = [0u8; SHA256_LEN];
    b.copy_from_slice(&out);
    b
}

/// Constant-time-ish equality of two digests.
///
/// Digest comparison is not secret-dependent here (the bytes are public
/// integrity data, not keys), but a fixed-time compare costs nothing and
/// avoids encoding a timing-sensitive pattern in a validator that runs on
/// attacker-supplied input.
#[must_use]
pub fn digest_eq(a: &[u8; SHA256_LEN], b: &[u8; SHA256_LEN]) -> bool {
    let mut diff = 0u8;
    for i in 0..SHA256_LEN {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::hex_encode;

    #[test]
    fn sha256_of_empty_is_the_known_digest() {
        // FIPS 180-4 / NIST published value.
        assert_eq!(
            hex_encode(&sha256(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn sha256_of_abc_is_the_known_digest() {
        assert_eq!(
            hex_encode(&sha256(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn digest_eq_detects_single_bit_difference() {
        let a = sha256(b"x");
        let mut b = a;
        b[31] ^= 1;
        assert!(digest_eq(&a, &a));
        assert!(!digest_eq(&a, &b));
    }
}
