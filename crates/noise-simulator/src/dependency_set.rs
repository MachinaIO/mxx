#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SourceId(pub [u8; 32]);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DependencySet {
    Known(Vec<SourceId>),
    Unknown,
}

impl DependencySet {
    pub fn empty() -> Self {
        DependencySet::Known(Vec::new())
    }

    pub fn singleton(source_id: SourceId) -> Self {
        DependencySet::Known(vec![source_id])
    }

    pub fn known(mut source_ids: Vec<SourceId>) -> Self {
        source_ids.sort_unstable();
        source_ids.dedup();
        DependencySet::Known(source_ids)
    }

    pub fn is_disjoint(&self, other: &Self) -> bool {
        let (DependencySet::Known(lhs), DependencySet::Known(rhs)) = (self, other) else {
            return false;
        };
        let mut i = 0usize;
        let mut j = 0usize;
        while i < lhs.len() && j < rhs.len() {
            match lhs[i].cmp(&rhs[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => return false,
            }
        }
        true
    }

    pub fn union(&self, other: &Self) -> Self {
        let (DependencySet::Known(lhs), DependencySet::Known(rhs)) = (self, other) else {
            return DependencySet::Unknown;
        };
        let mut out = Vec::with_capacity(lhs.len() + rhs.len());
        let mut i = 0usize;
        let mut j = 0usize;
        while i < lhs.len() || j < rhs.len() {
            let next = match (lhs.get(i), rhs.get(j)) {
                (Some(l), Some(r)) => match l.cmp(r) {
                    std::cmp::Ordering::Less => {
                        i += 1;
                        *l
                    }
                    std::cmp::Ordering::Greater => {
                        j += 1;
                        *r
                    }
                    std::cmp::Ordering::Equal => {
                        i += 1;
                        j += 1;
                        *l
                    }
                },
                (Some(l), None) => {
                    i += 1;
                    *l
                }
                (None, Some(r)) => {
                    j += 1;
                    *r
                }
                (None, None) => break,
            };
            if out.last().copied() != Some(next) {
                out.push(next);
            }
        }
        DependencySet::Known(out)
    }
}
