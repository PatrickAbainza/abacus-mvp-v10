//! Persistent, per-partition event-namespace registration.
//!
//! Canonical Contract §3.2 and Wire §5 require that a logical Kafka stream
//! partition be bound to a **persistent** `event_namespace` (a UUIDv4) that is
//! generated once, stored, and reused on every resume or retry — and *never*
//! regenerated. This module is that durable store.
//!
//! It is deliberately std-only (a single append-only file, no database, no
//! broker): the persistence *decision* of Gap B (where namespaces live) is
//! satisfied here with a self-contained, restart-durable file, while the
//! *broker-client* decision remains open behind [`crate::transport`].
//!
//! ## File format (`ABACNS1`)
//!
//! ```text
//! magic   : b"ABACNS1"            (7 bytes)
//! version : u8 = 1                (1 byte)
//! entry*  : key_len:u16 LE | key:key_len bytes (UTF-8) | namespace:16 bytes
//! ```
//!
//! Entries are append-only; a partition key appears at most once. On open the
//! whole file is parsed into an in-memory map, and the first request for an
//! unseen partition appends exactly one new entry. Re-opening after a restart
//! reloads the identical mapping, so a resumed consumer re-derives the same
//! compound identities.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use abacus_ingest::namespace::{EventNamespace, NamespaceRegistry};
use abacus_ingest::{ErrorCode, IngestError, IngestResult};

const MAGIC: &[u8; 7] = b"ABACNS1";
const VERSION: u8 = 1;

/// A durable `partition key -> event namespace` registry.
#[derive(Debug)]
pub struct PartitionNamespaceStore {
    path: PathBuf,
    entries: BTreeMap<String, EventNamespace>,
    /// Generation + collision checking (std-entropy only; no `rand`).
    registry: NamespaceRegistry,
}

impl PartitionNamespaceStore {
    /// Open (or create) the store rooted at `dir`.
    ///
    /// If `dir/partition_namespaces.bin` exists it is parsed; a missing file is
    /// treated as an empty store and created lazily on the first registration.
    ///
    /// # Errors
    /// - [`ErrorCode::Persistence`] on an I/O failure.
    /// - [`ErrorCode::Binary`] on a bad magic/version or a truncated entry.
    pub fn open(dir: &Path) -> IngestResult<Self> {
        let path = dir.join("partition_namespaces.bin");
        let mut entries = BTreeMap::new();
        let mut registry = NamespaceRegistry::new();
        if path.exists() {
            let bytes = std::fs::read(&path)
                .map_err(|e| persist(format!("cannot read {}: {e}", path.display())))?;
            parse(&bytes, &mut entries, &mut registry)?;
        }
        Ok(Self {
            path,
            entries,
            registry,
        })
    }

    /// The namespace already bound to `partition_key`, if any.
    #[must_use]
    pub fn get(&self, partition_key: &str) -> Option<EventNamespace> {
        self.entries.get(partition_key).copied()
    }

    /// Number of registered partitions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True iff no partition is registered yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The namespace for `partition_key`, generating and persisting one on first
    /// use and reusing the stored one thereafter.
    ///
    /// This is the load-bearing invariant of §3.2: the *same* key always maps to
    /// the *same* namespace, across calls and across restarts. A generated value
    /// is registered for collision safety before it is returned.
    ///
    /// # Errors
    /// - [`ErrorCode::Identity`] if generation fails to find a unique namespace.
    /// - [`ErrorCode::Persistence`] on an I/O failure while persisting a new
    ///   entry (the in-memory map is left unchanged on failure, so a retry
    ///   regenerates rather than trusting a half-written file).
    pub fn namespace_for(&mut self, partition_key: &str) -> IngestResult<EventNamespace> {
        if let Some(ns) = self.entries.get(partition_key) {
            return Ok(*ns);
        }
        let ns = self.registry.generate()?;
        // Persist first: a crash before this returns leaves the file without the
        // entry, and the next call simply generates a fresh one (still unique,
        // still collision-checked). A crash after it returns is covered because
        // the file already holds the mapping.
        append_entry(&self.path, partition_key, ns)?;
        self.entries.insert(partition_key.to_string(), ns);
        self.registry.register(ns)?;
        Ok(ns)
    }
}

fn persist(detail: String) -> IngestError {
    IngestError::new(ErrorCode::Persistence, detail)
}

fn binary(detail: String) -> IngestError {
    IngestError::new(ErrorCode::Binary, detail)
}

/// Write the header (if the file is new) and append one entry.
fn append_entry(path: &Path, key: &str, ns: EventNamespace) -> IngestResult<()> {
    use std::io::Write as _;
    let exists = path.exists();
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| persist(format!("cannot open {}: {e}", path.display())))?;
    if !exists {
        f.write_all(MAGIC).map_err(|e| persist(e.to_string()))?;
        f.write_all(&[VERSION])
            .map_err(|e| persist(e.to_string()))?;
    }
    let kb = key.as_bytes();
    let klen = u16::try_from(kb.len())
        .map_err(|_| binary("partition key exceeds u16 length".to_string()))?;
    f.write_all(&klen.to_le_bytes())
        .map_err(|e| persist(e.to_string()))?;
    f.write_all(kb).map_err(|e| persist(e.to_string()))?;
    f.write_all(&ns.to_bytes())
        .map_err(|e| persist(e.to_string()))?;
    f.flush().map_err(|e| persist(e.to_string()))?;
    Ok(())
}

/// Parse the whole file into `entries`, registering each namespace.
fn parse(
    bytes: &[u8],
    entries: &mut BTreeMap<String, EventNamespace>,
    registry: &mut NamespaceRegistry,
) -> IngestResult<()> {
    if bytes.len() < MAGIC.len() + 1 {
        return Err(binary(
            "namespace store is shorter than its header".to_string(),
        ));
    }
    if &bytes[..MAGIC.len()] != MAGIC {
        return Err(binary("namespace store has a bad magic".to_string()));
    }
    if bytes[MAGIC.len()] != VERSION {
        return Err(binary("unsupported namespace store version".to_string()));
    }
    let mut i = MAGIC.len() + 1;
    while i < bytes.len() {
        if i + 2 > bytes.len() {
            return Err(binary("truncated entry key length".to_string()));
        }
        let klen = u16::from_le_bytes([bytes[i], bytes[i + 1]]) as usize;
        i += 2;
        if i + klen + 16 > bytes.len() {
            return Err(binary("truncated entry body".to_string()));
        }
        let key = std::str::from_utf8(&bytes[i..i + klen])
            .map_err(|_| binary("partition key is not valid UTF-8".to_string()))?
            .to_string();
        i += klen;
        let mut nb = [0u8; 16];
        nb.copy_from_slice(&bytes[i..i + 16]);
        i += 16;
        let ns = EventNamespace::from_bytes(nb);
        if ns.is_zero() {
            return Err(binary(
                "a stored namespace is the reserved all-zero value".to_string(),
            ));
        }
        registry.register(ns)?;
        entries.insert(key, ns);
    }
    Ok(())
}
