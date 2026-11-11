//! Sparse Boolean vector (GraphBLAS pattern vector) over the vertex domain.
//!
//! Ported verbatim from v7 `graphblas/vector.rs` (itself verbatim from v5
//! `abacus_canonical_execution::vector`; no tensor dependency; pure
//! `a_{01}/a_{10}/a_{11}/a_{15}` index machinery under `R_3`). Storage:
//! `members` sorted ascending, distinct — the `prec_3`-compatible order
//! (golden Addendum O, O.1).

/// Sparse Boolean vector: nonzero pattern stored sorted, distinct.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Vector {
    members: Vec<u32>,
}

impl Vector {
    /// Zero vector (no nonzeros).
    #[must_use]
    pub fn zero() -> Self {
        Self {
            members: Vec::new(),
        }
    }

    /// Wrap an already-sorted distinct list (debug-checked).
    #[must_use]
    pub fn from_sorted(members: Vec<u32>) -> Self {
        debug_assert!(members.windows(2).all(|w| w[0] < w[1]));
        Self { members }
    }

    /// Unit vector with a single nonzero index.
    #[must_use]
    pub fn unit(i: u32) -> Self {
        Self { members: vec![i] }
    }

    /// Number of nonzeros.
    #[inline]
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.members.len()
    }

    /// True iff no nonzeros.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// Nonzero indices (sorted).
    #[inline]
    #[must_use]
    pub fn indices(&self) -> &[u32] {
        &self.members
    }

    /// `true` iff index `i` is in the pattern (binary search).
    #[must_use]
    pub fn contains(&self, i: u32) -> bool {
        self.members.binary_search(&i).is_ok()
    }

    /// Insert a single index into the pattern (keeps sorted distinct).
    pub fn insert(&mut self, i: u32) {
        match self.members.binary_search(&i) {
            Ok(_) => {}
            Err(pos) => self.members.insert(pos, i),
        }
    }

    /// Extend from an iterator of indices (sorts + dedups).
    pub fn extend_unsorted<I: IntoIterator<Item = u32>>(&mut self, it: I) {
        self.members.extend(it);
        self.members.sort_unstable();
        self.members.dedup();
    }

    /// `ewise_add_lor` (Boolean OR): `C = A OR B`, pattern union (v5 #6).
    #[must_use]
    pub fn ewise_add_lor(&self, b: &Vector) -> Vector {
        let mut out = Vec::with_capacity(self.nnz() + b.nnz());
        let (mut i, mut j) = (0usize, 0usize);
        while i < self.members.len() && j < b.members.len() {
            match self.members[i].cmp(&b.members[j]) {
                std::cmp::Ordering::Less => {
                    out.push(self.members[i]);
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    out.push(b.members[j]);
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    out.push(self.members[i]);
                    i += 1;
                    j += 1;
                }
            }
        }
        out.extend_from_slice(&self.members[i..]);
        out.extend_from_slice(&b.members[j..]);
        Vector { members: out }
    }

    /// Boolean difference: `C = A \ B` (complement-mask assign).
    #[must_use]
    pub fn difference(&self, b: &Vector) -> Vector {
        let mut out = Vec::with_capacity(self.nnz());
        for &v in &self.members {
            if b.members.binary_search(&v).is_err() {
                out.push(v);
            }
        }
        Vector { members: out }
    }
}

impl FromIterator<u32> for Vector {
    fn from_iter<I: IntoIterator<Item = u32>>(iter: I) -> Self {
        let mut v = Self::zero();
        v.extend_unsorted(iter);
        v
    }
}
