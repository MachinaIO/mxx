use crate::{matrix::PolyMatrix, poly::Poly};

#[cfg(feature = "gpu")]
pub mod gpu;
pub mod hash;
pub mod trapdoor;
pub mod uniform;

#[derive(Debug, Clone, Copy)]
/// Enum representing different types of distributions for random sampling.
pub enum DistType {
    /// Distribution over a finite ring, typically samples elements from a ring in a uniform or
    /// near-uniform manner
    FinRingDist,
    /// discrete Gaussian distribution described in [[GPV08](https://eprint.iacr.org/2007/432),[BDJ+24](https://eprint.iacr.org/2024/1742)], where
    /// noise is drawn from a discrete Gaussian over a lattice Λ with parameter σ > 0.
    /// Each sample is drawn proportionally to exp(-π‖x‖² / σ²), restricted to x ∈ Λ.
    ///
    /// * `sigma` - The Gaussian parameter (standard deviation).
    GaussDist { sigma: f64 },
    /// Distribution that produces random bits (0 or 1).
    BitDist,
    /// Distribution that produces random bits (-1,0,1).
    TernaryDist,
}

/// Trait for sampling a polynomial based on a hash function.
pub trait PolyHashSampler<K: AsRef<[u8]>> {
    type M: PolyMatrix;

    fn new() -> Self;

    /// Samples a matrix of ring elements from a pseudorandom source defined by a hash function `H`
    /// Compute H(key || tag || i)
    ///
    /// and a distribution type specified by `dist`.
    fn sample_hash<B: AsRef<[u8]>>(
        &self,
        params: &<<Self::M as PolyMatrix>::P as Poly>::Params,
        key: [u8; 32],
        tag: B,
        nrow: usize,
        ncol: usize,
        dist: DistType,
    ) -> Self::M;

    /// Samples the conceptual matrix returned by `sample_hash(..., total_ncol, ...)`, then returns
    /// only the requested column window `[col_start, col_start + col_len)`.
    fn sample_hash_columns<B: AsRef<[u8]>>(
        &self,
        params: &<<Self::M as PolyMatrix>::P as Poly>::Params,
        key: [u8; 32],
        tag: B,
        nrow: usize,
        total_ncol: usize,
        col_start: usize,
        col_len: usize,
        dist: DistType,
    ) -> Self::M {
        let col_end =
            col_start.checked_add(col_len).expect("sample_hash_columns column range overflow");
        assert!(
            col_end <= total_ncol,
            "sample_hash_columns range out of bounds: start={}, len={}, total_ncol={}",
            col_start,
            col_len,
            total_ncol
        );
        self.sample_hash(params, key, tag, nrow, total_ncol, dist).slice_columns(col_start, col_end)
    }

    fn sample_hash_decomposed<B: AsRef<[u8]>>(
        &self,
        params: &<<Self::M as PolyMatrix>::P as Poly>::Params,
        key: [u8; 32],
        tag: B,
        nrow: usize,
        ncol: usize,
        dist: DistType,
    ) -> Self::M {
        self.sample_hash(params, key, tag, nrow, ncol, dist).decompose_owned()
    }

    fn sample_hash_decomposed_columns<B: AsRef<[u8]>>(
        &self,
        params: &<<Self::M as PolyMatrix>::P as Poly>::Params,
        key: [u8; 32],
        tag: B,
        nrow: usize,
        total_ncol: usize,
        col_start: usize,
        col_len: usize,
        dist: DistType,
    ) -> Self::M {
        self.sample_hash_columns(params, key, tag, nrow, total_ncol, col_start, col_len, dist)
            .decompose_owned()
    }

    fn sample_hash_small_decomposed<B: AsRef<[u8]>>(
        &self,
        params: &<<Self::M as PolyMatrix>::P as Poly>::Params,
        key: [u8; 32],
        tag: B,
        nrow: usize,
        ncol: usize,
        dist: DistType,
    ) -> Self::M {
        self.sample_hash(params, key, tag, nrow, ncol, dist).small_decompose_owned()
    }

    fn sample_hash_small_decomposed_columns<B: AsRef<[u8]>>(
        &self,
        params: &<<Self::M as PolyMatrix>::P as Poly>::Params,
        key: [u8; 32],
        tag: B,
        nrow: usize,
        total_ncol: usize,
        col_start: usize,
        col_len: usize,
        dist: DistType,
    ) -> Self::M {
        self.sample_hash_columns(params, key, tag, nrow, total_ncol, col_start, col_len, dist)
            .small_decompose_owned()
    }
}

pub trait PolyUniformSampler {
    type M: PolyMatrix;

    fn new() -> Self;

    fn sample_poly(
        &self,
        params: &<<Self::M as PolyMatrix>::P as Poly>::Params,
        dist: &DistType,
    ) -> <Self::M as PolyMatrix>::P;

    fn sample_uniform(
        &self,
        params: &<<Self::M as PolyMatrix>::P as Poly>::Params,
        nrow: usize,
        ncol: usize,
        dist: DistType,
    ) -> Self::M;
}

pub trait PolyTrapdoorSampler {
    type M: PolyMatrix;
    type Trapdoor: Send + Sync;

    fn new(params: &<<Self::M as PolyMatrix>::P as Poly>::Params, sigma: f64) -> Self;

    fn trapdoor(
        &self,
        params: &<<Self::M as PolyMatrix>::P as Poly>::Params,
        size: usize,
    ) -> (Self::Trapdoor, Self::M);

    fn trapdoor_to_bytes(trapdoor: &Self::Trapdoor) -> Vec<u8>;

    fn trapdoor_from_bytes(
        params: &<<Self::M as PolyMatrix>::P as Poly>::Params,
        bytes: &[u8],
    ) -> Option<Self::Trapdoor>;

    fn preimage(
        &self,
        params: &<<Self::M as PolyMatrix>::P as Poly>::Params,
        trapdoor: &Self::Trapdoor,
        public_matrix: &Self::M,
        target: &Self::M,
    ) -> Self::M;

    #[cfg(feature = "gpu")]
    fn preimage_batched_sharded<'a>(
        &self,
        requests: Vec<crate::sampler::trapdoor::GpuPreimageRequest<'a, Self::M, Self::Trapdoor>>,
    ) -> Vec<(usize, Self::M)>
    where
        Self::Trapdoor: Send + Sync + 'a,
        Self::M: 'a,
    {
        requests
            .into_iter()
            .map(|request| {
                let out = self.preimage(
                    request.params,
                    request.trapdoor,
                    request.public_matrix,
                    &request.target,
                );
                (request.entry_idx, out)
            })
            .collect()
    }

    // Given a trapdoor of B, an extension matrix C, a target matrix U, return a preimage D s.t.
    // [B,C]D = U.
    fn preimage_extend(
        &self,
        params: &<<Self::M as PolyMatrix>::P as Poly>::Params,
        trapdoor: &Self::Trapdoor,
        public_matrix: &Self::M,
        ext_matrix: &Self::M,
        target: &Self::M,
    ) -> Self::M;
}
