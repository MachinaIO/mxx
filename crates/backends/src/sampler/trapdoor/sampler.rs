use super::{DCRTTrapdoor, utils::split_int64_mat_alt_to_elems};
use crate::{
    matrix::{
        CpuSmallMatrix, PolyMatrix, PolyMatrixColumnSource, SmallMatrixError,
        dcrt_poly::DCRTPolyMatrix,
    },
    openfhe_guard::ensure_openfhe_warmup,
    parallel_iter,
    poly::{
        Poly, PolyParams,
        dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    },
    sampler::{
        DistType, PolyTrapdoorSampler, PolyUniformSampler,
        bounds::{
            gadget_sampler_sigma, matrix_within_coefficient_bound, minimum_preimage_sigma,
            rational_from_decimal, validate_preimage_width,
        },
        trapdoor::KARNEY_THRESHOLD,
        uniform::DCRTPolyUniformSampler,
    },
};
use mxx_ir_core::{Rational, types::TrapdoorInstantiation};
use num_bigint::BigUint;
use num_traits::{FromPrimitive, ToPrimitive};
use rayon::iter::ParallelIterator;
use std::{
    ops::Range,
    sync::{Mutex, OnceLock},
    time::Instant,
};
use tracing::debug;

static GAUSS_SAMP_GQ_ARB_BASE_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

/// The CPU trapdoor sampler (MP12 with the GM18 perturbation sampler, after OpenFHE).
///
/// `sigma` is the TrapGen width; the gadget sampler has width `c = (base + 1) sigma`. A preimage
/// is sampled with the width its caller requests, which must be at least
/// [`crate::sampler::bounds::minimum_preimage_sigma`] of the trapdoor.
#[derive(Debug, Clone)]
pub struct DCRTPolyTrapdoorSampler {
    sigma: f64,
    base: u32,
    c: f64,
    instantiation: TrapdoorInstantiation,
}

impl DCRTPolyTrapdoorSampler {
    /// The smallest preimage width this sampler accepts for a trapdoor of `public_rows` rows, as an
    /// exact rational.
    pub fn minimum_preimage_width(&self, params: &DCRTPolyParams, public_rows: usize) -> Rational {
        rational_from_decimal(
            &minimum_preimage_sigma(
                params.ring_dimension(),
                public_rows,
                params.modulus_digits(),
                self.base,
                self.sigma,
                self.instantiation,
            )
            .expect("valid trapdoor sampler parameters"),
        )
    }

    /// Checks `preimage_sigma` and `max_coefficient_bound` against the trapdoor of `public_matrix`
    /// and returns the width as a float.
    fn checked_preimage_sigma(
        &self,
        params: &DCRTPolyParams,
        public_matrix: &DCRTPolyMatrix,
        preimage_sigma: &Rational,
        max_coefficient_bound: &BigUint,
    ) -> Result<f64, SmallMatrixError> {
        if params.dropped_moduli() != 0 {
            return Err(SmallMatrixError::InvalidConfig);
        }
        validate_preimage_width(
            params.ring_dimension(),
            public_matrix.row_size(),
            params.modulus_digits(),
            self.base,
            self.sigma,
            self.instantiation,
            preimage_sigma,
            max_coefficient_bound,
        )?;
        rational_to_f64(preimage_sigma).ok_or(SmallMatrixError::InvalidConfig)
    }

    /// Redraws whole candidates until every coefficient is within `max_coefficient_bound`.
    fn bounded_preimage(
        &self,
        params: &DCRTPolyParams,
        trapdoor: &DCRTTrapdoor,
        public_matrix: &DCRTPolyMatrix,
        target: &DCRTPolyMatrix,
        preimage_sigma: f64,
        max_coefficient_bound: &BigUint,
    ) -> DCRTPolyMatrix {
        loop {
            let candidate = expanded_preimage_candidate(
                self,
                params,
                trapdoor,
                public_matrix,
                target,
                preimage_sigma,
            );
            if matrix_within_coefficient_bound(&candidate, max_coefficient_bound) {
                return candidate;
            }
        }
    }
}

impl PolyTrapdoorSampler for DCRTPolyTrapdoorSampler {
    type M = DCRTPolyMatrix;
    type Trapdoor = DCRTTrapdoor;

    fn new(
        params: &<<Self::M as PolyMatrix>::P as Poly>::Params,
        sigma: f64,
        instantiation: TrapdoorInstantiation,
    ) -> Self {
        assert_eq!(
            params.dropped_moduli(),
            0,
            "exact trapdoor sampling requires dropped_moduli = 0"
        );
        let base = 1 << params.base_bits();
        let c = gadget_sampler_sigma(base, sigma);
        Self { sigma, base, c, instantiation }
    }

    fn trapdoor(
        &self,
        params: &<<Self::M as PolyMatrix>::P as Poly>::Params,
        size: usize,
        preimage_sigma: Option<&Rational>,
    ) -> Result<(Self::Trapdoor, Self::M), SmallMatrixError> {
        let TrapdoorInstantiation::Computational = self.instantiation else {
            let width = match preimage_sigma {
                Some(width) => width.clone(),
                None => self.minimum_preimage_width(params, size),
            };
            let width = rational_to_f64(&width).ok_or(SmallMatrixError::InvalidConfig)?;
            return DCRTTrapdoor::sample_statistical(
                params,
                size,
                self.sigma,
                self.instantiation,
                width,
            );
        };
        let trapdoor = DCRTTrapdoor::new(params, size, self.sigma);
        debug!("{}", "trapdoor generated");
        let uniform_sampler = DCRTPolyUniformSampler::new();
        debug!("{}", "uniform sampler created");
        let a_bar = uniform_sampler.sample_uniform(params, size, size, DistType::FinRingDist);
        debug!("{}", "a_bar generated");
        let g = DCRTPolyMatrix::gadget_matrix(params, size, None);
        debug!("{}", "gadget matrix generated");
        let a0 = a_bar.concat_columns(&[&DCRTPolyMatrix::identity(params, size, None)]);
        debug!("{}", "a0 generated");
        let trapdoor_r = trapdoor.r_cpu();
        let trapdoor_e = trapdoor.e_cpu();
        let a1 = g - (a_bar * &trapdoor_r + &trapdoor_e);
        debug!("{}", "a1 generated");
        let a = a0.concat_columns(&[&a1]);
        debug!("{}", "public matrix generated");
        Ok((trapdoor, a))
    }

    fn trapdoor_to_bytes(trapdoor: &Self::Trapdoor) -> Vec<u8> {
        trapdoor.to_compact_bytes()
    }

    fn trapdoor_from_bytes(
        params: &<<Self::M as PolyMatrix>::P as Poly>::Params,
        bytes: &[u8],
    ) -> Option<Self::Trapdoor> {
        DCRTTrapdoor::from_compact_bytes(params, bytes)
    }

    fn preimage(
        &self,
        params: &<<Self::M as PolyMatrix>::P as crate::poly::Poly>::Params,
        trapdoor: &Self::Trapdoor,
        public_matrix: &Self::M,
        target: &dyn PolyMatrixColumnSource<Self::M>,
        preimage_sigma: &Rational,
        max_coefficient_bound: BigUint,
        _randomness_seed: [u8; 32],
    ) -> Result<CpuSmallMatrix<DCRTPolyMatrix>, SmallMatrixError> {
        let width = self.checked_preimage_sigma(
            params,
            public_matrix,
            preimage_sigma,
            &max_coefficient_bound,
        )?;
        check_trapdoor_width(trapdoor, width)?;
        let target = target.load_columns(0, target.col_size());
        let preimage = self.bounded_preimage(
            params,
            trapdoor,
            public_matrix,
            &target,
            width,
            &max_coefficient_bound,
        );
        CpuSmallMatrix::new(preimage, max_coefficient_bound)
    }

    // Algorithm 5 of https://eprint.iacr.org/2017/601.pdf
    fn preimage_extend(
        &self,
        params: &<<Self::M as PolyMatrix>::P as Poly>::Params,
        trapdoor: &Self::Trapdoor,
        public_matrix: &Self::M,
        ext_matrix: &Self::M,
        target: &Self::M,
        preimage_sigma: &Rational,
        max_coefficient_bound: BigUint,
    ) -> Result<Self::M, SmallMatrixError> {
        let width = self.checked_preimage_sigma(
            params,
            public_matrix,
            preimage_sigma,
            &max_coefficient_bound,
        )?;
        check_trapdoor_width(trapdoor, width)?;
        let dist = DistType::GaussDist {
            sigma: width,
            max_coefficient_bound: Some(max_coefficient_bound.clone()),
        };
        let uniform_sampler = DCRTPolyUniformSampler::new();
        let preimage_right =
            uniform_sampler.sample_uniform(params, ext_matrix.col_size(), target.col_size(), dist);
        let t = target - &(ext_matrix * &preimage_right);
        let preimage_left = self.bounded_preimage(
            params,
            trapdoor,
            public_matrix,
            &t,
            width,
            &max_coefficient_bound,
        );
        Ok(preimage_left.concat_rows(&[&preimage_right]))
    }
}

/// The binary64 value of a width, `None` unless finite.
fn rational_to_f64(width: &Rational) -> Option<f64> {
    let width = width.numerator().to_f64()? / width.denominator().to_f64()?;
    width.is_finite().then_some(width)
}

/// A statistical trapdoor samples only at the width it precomputed for.
fn check_trapdoor_width(trapdoor: &DCRTTrapdoor, width: f64) -> Result<(), SmallMatrixError> {
    let DCRTTrapdoor::Statistical { precomputation, .. } = trapdoor else {
        return Ok(());
    };
    if precomputation.preimage_sigma == width {
        return Ok(());
    }
    let decimal = |value: f64| bigdecimal::BigDecimal::from_f64(value).unwrap_or_default();
    Err(SmallMatrixError::PreimageWidthMismatch {
        requested: decimal(width),
        trapdoor: decimal(precomputation.preimage_sigma),
    })
}

/// One preimage candidate `[p + T' z; p_2 + z]` of width `s = preimage_sigma`: the perturbation
/// `p`, the gadget preimage `z` of `target - A p`, and the trapdoor's top block `T'`.
fn expanded_preimage_candidate(
    sampler: &DCRTPolyTrapdoorSampler,
    params: &DCRTPolyParams,
    trapdoor: &DCRTTrapdoor,
    public_matrix: &DCRTPolyMatrix,
    target: &DCRTPolyMatrix,
    s: f64,
) -> DCRTPolyMatrix {
    let preimage_start = Instant::now();
    let d = public_matrix.row_size();
    let target_cols = target.col_size();
    debug_assert_eq!(
        target.row_size(),
        d,
        "Target matrix should have the same number of rows as the public matrix"
    );

    let k = params.modulus_digits();
    let dgg_large_std = (s * s - sampler.c * sampler.c).sqrt();
    let peikert = dgg_large_std < KARNEY_THRESHOLD;
    let (dgg_large_mean, dgg_large_table) = peikert_table(dgg_large_std);
    let dgg_large_params =
        (dgg_large_mean, dgg_large_std, dgg_large_table.as_ref().map(|v| &v[..]));
    debug!("{}", "preimage parameters computed");
    let p_hat = trapdoor.sample_perturbation(
        params,
        s,
        sampler.c,
        sampler.sigma,
        dgg_large_params,
        peikert,
        target_cols,
    );
    debug!("{}", "p_hat generated");
    let perturbed_syndrome = target - &(public_matrix * &p_hat);
    debug!("{}", "perturbed_syndrome generated");
    let mut z_hat_mat = DCRTPolyMatrix::zero(params, d * k, target_cols);
    let f = |row_offsets: Range<usize>, col_offsets: Range<usize>| -> Vec<Vec<DCRTPoly>> {
        let nrow = row_offsets.len();
        let ncol = col_offsets.len();
        let perturbed_syndromes = perturbed_syndrome.block_entries(row_offsets, col_offsets);
        let decomposed_results = parallel_iter!(0..nrow)
            .map(|i| {
                let row_results: Vec<_> = parallel_iter!(0..ncol)
                    .map(|j| {
                        let decomposed = decompose_dcrt_gadget(
                            &perturbed_syndromes[i][j],
                            sampler.c,
                            params,
                            sampler.base,
                            sampler.sigma,
                        );
                        (i, j, decomposed)
                    })
                    .collect();
                row_results
            })
            .flatten()
            .collect::<Vec<_>>();

        let mut block_matrix = vec![vec![DCRTPoly::const_zero(params); ncol]; k * nrow];
        for (i, j, decomposed) in decomposed_results {
            debug_assert_eq!(decomposed[0].len(), 1);
            for (decomposed_idx, vec) in decomposed.iter().enumerate() {
                block_matrix[i * k + decomposed_idx][j] = vec[0].clone();
            }
        }
        block_matrix
    };
    z_hat_mat.replace_entries_with_expand(0..d, 0..target_cols, k, 1, f);
    debug!("{}", "z_hat_mat generated");
    drop(perturbed_syndrome);
    let out = trapdoor.combine_preimage(p_hat, &z_hat_mat);
    debug!("preimage total time: {:?}", preimage_start.elapsed());
    out
}

/// Peikert's inversion table `(m_a, cumulative)` of the discrete Gaussian of width `std`, or
/// `(None, None)` above [`KARNEY_THRESHOLD`], where Karney's method is used instead.
fn peikert_table(std: f64) -> (Option<f64>, Option<Vec<f64>>) {
    if std > KARNEY_THRESHOLD {
        return (None, None);
    }
    let acc: f64 = 5e-32;
    let m = (-2.0 * acc.ln()).sqrt();
    let fin = (std * m).ceil() as usize;
    let mut m_vals = Vec::with_capacity(fin);
    let variance = 2.0 * std * std;
    let mut cusum = 0.0f64;
    for i in 1..=fin {
        cusum += (-(i as f64 * i as f64) / variance).exp();
        m_vals.push(cusum);
    }
    let m_a = 1.0 / (2.0 * cusum + 1.0);
    for value in &mut m_vals {
        *value *= m_a;
    }
    (Some(m_a), Some(m_vals))
}

pub(crate) fn decompose_dcrt_gadget(
    syndrome: &DCRTPoly,
    c: f64,
    params: &DCRTPolyParams,
    base: u32,
    sigma: f64,
) -> Vec<Vec<DCRTPoly>> {
    let depth = params.crt_depth();
    let z_hat_bbi = parallel_iter!(0..depth)
        .flat_map(|tower_idx| gauss_samp_gq_arb_base(syndrome, c, params, base, sigma, tower_idx))
        .collect::<Vec<_>>();
    split_int64_mat_alt_to_elems(&z_hat_bbi, params)
}

// A function corresponding to lines 260-266 in trapdoor-dcrtpoly.cpp and the `GaussSampGqArbBase`
// function provided by OpenFHE.
pub(crate) fn gauss_samp_gq_arb_base(
    syndrome: &DCRTPoly,
    c: f64,
    params: &DCRTPolyParams,
    base: u32,
    sigma: f64,
    tower_idx: usize,
) -> Vec<Vec<i64>> {
    ensure_openfhe_warmup(params);
    let n = params.ring_dimension();
    let depth = params.crt_depth();
    let k_res_digits = params.modulus_digits() / depth;
    // OpenFHE's GaussSampGqArbBase can race across threads depending on backend state.
    // Keep this FFI call serialized for stability.
    let result = {
        let _guard = GAUSS_SAMP_GQ_ARB_BASE_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap();
        crate::poly::dcrt::native::ffi::exact_basis_gauss_gq(
            syndrome.get_poly(),
            c,
            k_res_digits,
            base as i64,
            sigma,
            tower_idx,
        )
        .expect("exact CRT gadget sampling failed")
    };
    debug_assert_eq!(result.len(), n as usize * k_res_digits);
    // let mut matrix = I64Matrix::new_empty(&I64MatrixParams, k_res, n as usize);
    parallel_iter!(0..k_res_digits)
        .map(|i| {
            parallel_iter!(0..n as usize).map(|j| result[i * n as usize + j]).collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        element::PolyElem,
        poly::PolyParams,
        sampler::{
            PolyUniformSampler,
            bounds::{default_preimage_cutoff, hard_cutoff_from_rational},
            uniform::DCRTPolyUniformSampler,
        },
    };
    use num_bigint::BigUint;
    use std::ops::Range;

    const SIGMA: f64 = 4.578;

    #[test]
    fn test_decompose_dcrt_gadget() {
        let params = DCRTPolyParams::default();
        let uniform_sampler = DCRTPolyUniformSampler::new();
        let target = uniform_sampler.sample_uniform(&params, 1, 1, DistType::FinRingDist);
        let decomposed = DCRTPolyMatrix::from_poly_vec(
            &params,
            decompose_dcrt_gadget(&target.entry(0, 0), 3.0 * SIGMA, &params, 2, SIGMA),
        );
        let gadget_vec = DCRTPolyMatrix::gadget_matrix(&params, 1, None);
        assert_eq!(gadget_vec * decomposed, target);
    }

    #[test]
    fn test_decompose_dcrt_gadget_base_8() {
        let params = DCRTPolyParams::new(4, 2, 17, 3, None, None);
        let uniform_sampler = DCRTPolyUniformSampler::new();
        let target = uniform_sampler.sample_uniform(&params, 1, 1, DistType::FinRingDist);
        let decomposed = DCRTPolyMatrix::from_poly_vec(
            &params,
            decompose_dcrt_gadget(&target.entry(0, 0), (8.0 + 1.0) * SIGMA, &params, 8, SIGMA),
        );
        let gadget_vec = DCRTPolyMatrix::gadget_matrix(&params, 1, None);
        assert_eq!(gadget_vec * decomposed, target);
    }

    #[test]
    #[serial_test::serial]
    fn test_trapdoor_generation() {
        let size: usize = 3;
        let params = DCRTPolyParams::default();
        let trapdoor_sampler =
            DCRTPolyTrapdoorSampler::new(&params, SIGMA, TrapdoorInstantiation::Computational);

        let (trapdoor, public_matrix) = trapdoor_sampler.trapdoor(&params, size, None).unwrap();

        let expected_rows = size;
        let expected_cols = (&params.modulus_digits() + 2) * size;

        assert_eq!(
            public_matrix.row_size(),
            expected_rows,
            "Public matrix should have the correct number of rows"
        );
        assert_eq!(
            public_matrix.col_size(),
            expected_cols,
            "Public matrix should have the correct number of columns"
        );

        // Verify that all entries in the matrix are valid DCRTPolys
        for i in 0..public_matrix.row_size() {
            for j in 0..public_matrix.col_size() {
                let poly = public_matrix.entry(i, j);
                assert!(!poly.get_poly().is_null(), "Matrix entry should be a valid DCRTPoly");
            }
        }

        let muled = {
            let k = params.modulus_digits();
            let identity = DCRTPolyMatrix::identity(&params, size * k, None);
            let trapdoor_r = trapdoor.r_cpu();
            let trapdoor_e = trapdoor.e_cpu();
            let trapdoor_matrix = trapdoor_r.concat_rows(&[&trapdoor_e, &identity]);
            public_matrix * trapdoor_matrix
        };
        let gadget_matrix = DCRTPolyMatrix::gadget_matrix(&params, size, None);
        assert_eq!(muled, gadget_matrix);
    }

    #[test]
    #[serial_test::serial]
    fn test_preimage_generation_small_moduli_reconstructs_target() {
        // These small primes are below typical perturbation magnitudes. Signed
        // Gaussian coefficients must still enter every CRT limb modulo q.
        for (dimension, bits, base_bits) in [(2, 6, 3), (4, 12, 4), (4, 6, 3)] {
            let depth = if dimension == 4 && bits == 6 { 2 } else { 1 };
            let params = DCRTPolyParams::new(dimension, depth, bits, base_bits, None, None);
            let sampler =
                DCRTPolyTrapdoorSampler::new(&params, SIGMA, TrapdoorInstantiation::Computational);
            let (trapdoor, public) = sampler.trapdoor(&params, 1, None).unwrap();
            let bound = default_preimage_cutoff(
                params.ring_dimension(),
                1,
                params.modulus_digits(),
                1u32 << params.base_bits(),
                SIGMA,
                TrapdoorInstantiation::Computational,
            )
            .unwrap();
            for _ in 0..3 {
                let target = DCRTPolyUniformSampler::new().sample_uniform(
                    &params,
                    1,
                    2,
                    DistType::FinRingDist,
                );
                let sampled = sampler
                    .preimage(
                        &params,
                        &trapdoor,
                        &public,
                        &crate::matrix::ResidentPolyMatrixColumnSource::new(target.clone()),
                        &sampler.minimum_preimage_width(&params, public.row_size()),
                        bound.clone(),
                        rand::random(),
                    )
                    .unwrap();
                assert_eq!(
                    &public * sampled.value(),
                    target,
                    "small-modulus preimage relation: dimension={dimension}, bits={bits}, depth={depth}"
                );
            }
        }
    }

    #[test]
    #[serial_test::serial]
    fn test_preimage_generation_exact_non_prefix_basis() {
        let (n, depth, bits, base) = crate::env::modulus_conversion_test_parameters();
        let original = DCRTPolyParams::new(n, depth, bits, base, None, None);
        let primes = original.to_crt().0;
        let params = DCRTPolyParams::new(n, 2, bits, base, Some(vec![primes[2], primes[0]]), None);
        let sampler =
            DCRTPolyTrapdoorSampler::new(&params, SIGMA, TrapdoorInstantiation::Computational);
        let (trapdoor, public_matrix) = sampler.trapdoor(&params, 1, None).unwrap();
        let target =
            DCRTPolyUniformSampler::new().sample_uniform(&params, 1, 2, DistType::FinRingDist);
        let preimage = sampler
            .preimage(
                &params,
                &trapdoor,
                &public_matrix,
                &crate::matrix::ResidentPolyMatrixColumnSource::new(target.clone()),
                &sampler.minimum_preimage_width(&params, public_matrix.row_size()),
                params.modulus().as_ref() / BigUint::from(2u8),
                rand::random(),
            )
            .expect("exact-basis preimage must satisfy the requested bound");
        assert_eq!(&public_matrix * preimage.value(), target);
        assert_eq!(target.entry(0, 0).coeffs()[0].modulus, params.modulus());
        let coefficients = target.entry(0, 0).coeffs();
        let reconstructed = DCRTPoly::from_coeffs(&params, &coefficients);
        assert_eq!(reconstructed, target.entry(0, 0));
        let gadget = DCRTPolyMatrix::gadget_matrix(&params, 1, None);
        let identity = DCRTPolyMatrix::identity(&params, params.modulus_digits(), None);
        let trapdoor_matrix = trapdoor.r_cpu().concat_rows(&[&trapdoor.e_cpu(), &identity]);
        assert_eq!(public_matrix * trapdoor_matrix, gadget);
    }

    #[test]
    #[serial_test::serial]
    fn test_trapdoor_round_trip_bytes() {
        let size: usize = 3;
        let params = DCRTPolyParams::default();
        let trapdoor_sampler =
            DCRTPolyTrapdoorSampler::new(&params, SIGMA, TrapdoorInstantiation::Computational);
        let (trapdoor, _public_matrix) = trapdoor_sampler.trapdoor(&params, size, None).unwrap();

        let bytes = <DCRTPolyTrapdoorSampler as PolyTrapdoorSampler>::trapdoor_to_bytes(&trapdoor);
        let decoded =
            <DCRTPolyTrapdoorSampler as PolyTrapdoorSampler>::trapdoor_from_bytes(&params, &bytes)
                .expect("trapdoor bytes should decode");

        assert_eq!(trapdoor, decoded);
    }

    #[test]
    #[serial_test::serial]
    fn test_preimage_generation_square() {
        let params = DCRTPolyParams::default();
        let size = 3;
        let k = params.modulus_digits();
        let trapdoor_sampler =
            DCRTPolyTrapdoorSampler::new(&params, SIGMA, TrapdoorInstantiation::Computational);
        let (trapdoor, public_matrix) = trapdoor_sampler.trapdoor(&params, size, None).unwrap();

        let uniform_sampler = DCRTPolyUniformSampler::new();
        let target = uniform_sampler.sample_uniform(&params, size, size, DistType::FinRingDist);

        let preimage = trapdoor_sampler
            .preimage(
                &params,
                &trapdoor,
                &public_matrix,
                &crate::matrix::ResidentPolyMatrixColumnSource::new(target.clone()),
                &trapdoor_sampler.minimum_preimage_width(&params, public_matrix.row_size()),
                params.modulus().as_ref() / BigUint::from(2u8),
                rand::random(),
            )
            .expect("permissive bound should accept a preimage");
        let preimage = preimage.value();

        let expected_rows = size * (k + 2);
        let expected_cols = size;

        assert_eq!(
            preimage.row_size(),
            expected_rows,
            "Preimage matrix should have the correct number of rows"
        );

        assert_eq!(
            preimage.col_size(),
            expected_cols,
            "Preimage matrix should have the correct number of columns"
        );

        // public_matrix * preimage should be equal to target
        let product = public_matrix * preimage;
        assert_eq!(product, target, "Product of public matrix and preimage should equal target");
    }

    #[test]
    #[serial_test::serial]
    fn test_preimage_generation_non_square_target_lt() {
        let params = DCRTPolyParams::default();
        let size = 4;
        let target_cols = 2;
        let k = params.modulus_digits();
        let trapdoor_sampler =
            DCRTPolyTrapdoorSampler::new(&params, SIGMA, TrapdoorInstantiation::Computational);
        let (trapdoor, public_matrix) = trapdoor_sampler.trapdoor(&params, size, None).unwrap();

        // Create a non-square target matrix (size x target_cols) such that target_cols < size
        let uniform_sampler = DCRTPolyUniformSampler::new();
        let target =
            uniform_sampler.sample_uniform(&params, size, target_cols, DistType::FinRingDist);

        let preimage = trapdoor_sampler
            .preimage(
                &params,
                &trapdoor,
                &public_matrix,
                &crate::matrix::ResidentPolyMatrixColumnSource::new(target.clone()),
                &trapdoor_sampler.minimum_preimage_width(&params, public_matrix.row_size()),
                params.modulus().as_ref() / BigUint::from(2u8),
                rand::random(),
            )
            .expect("permissive bound should accept a preimage");
        let preimage = preimage.value();

        let expected_rows = size * (k + 2);
        let expected_cols = target_cols; // Preimage should be sliced to match target columns

        assert_eq!(
            preimage.row_size(),
            expected_rows,
            "Preimage matrix should have the correct number of rows"
        );

        assert_eq!(
            preimage.col_size(),
            expected_cols,
            "Preimage matrix should have the correct number of columns (sliced to match target)"
        );

        // public_matrix * preimage should be equal to target
        let product = public_matrix * preimage;

        assert_eq!(product, target, "Product of public matrix and preimage should equal target");
    }

    #[test]
    #[serial_test::serial]
    fn test_preimage_generation_non_square_target_gt_multiple() {
        let params = DCRTPolyParams::default();
        let size = 4;
        let multiple = 2;
        let target_cols = size * multiple;
        let k = params.modulus_digits();
        let trapdoor_sampler =
            DCRTPolyTrapdoorSampler::new(&params, SIGMA, TrapdoorInstantiation::Computational);
        let (trapdoor, public_matrix) = trapdoor_sampler.trapdoor(&params, size, None).unwrap();

        // Create a non-square target matrix (size x target_cols) such that target_cols > size
        // target_cols is a multiple of size
        let uniform_sampler = DCRTPolyUniformSampler::new();
        let target =
            uniform_sampler.sample_uniform(&params, size, target_cols, DistType::FinRingDist);

        let preimage = trapdoor_sampler
            .preimage(
                &params,
                &trapdoor,
                &public_matrix,
                &crate::matrix::ResidentPolyMatrixColumnSource::new(target.clone()),
                &trapdoor_sampler.minimum_preimage_width(&params, public_matrix.row_size()),
                params.modulus().as_ref() / BigUint::from(2u8),
                rand::random(),
            )
            .expect("permissive bound should accept a preimage");
        let preimage = preimage.value();

        let expected_rows = size * (k + 2);
        let expected_cols = target_cols;

        assert_eq!(
            preimage.row_size(),
            expected_rows,
            "Preimage matrix should have the correct number of rows"
        );

        assert_eq!(
            preimage.col_size(),
            expected_cols,
            "Preimage matrix should have the correct number of columns (equal to target columns)"
        );

        // public_matrix * preimage should be equal to target
        let product = public_matrix * preimage;

        assert_eq!(product, target, "Product of public matrix and preimage should equal target");
    }

    #[test]
    #[serial_test::serial]
    fn test_preimage_generation_non_square_target_gt_non_multiple() {
        let params = DCRTPolyParams::default();
        let size = 4;
        let target_cols = 6;
        let k = params.modulus_digits();
        let trapdoor_sampler =
            DCRTPolyTrapdoorSampler::new(&params, SIGMA, TrapdoorInstantiation::Computational);
        let (trapdoor, public_matrix) = trapdoor_sampler.trapdoor(&params, size, None).unwrap();

        // Create a non-square target matrix (size x target_cols) such that target_cols > size
        // target_cols is not a multiple of size
        let uniform_sampler = DCRTPolyUniformSampler::new();
        let target =
            uniform_sampler.sample_uniform(&params, size, target_cols, DistType::FinRingDist);

        let preimage = trapdoor_sampler
            .preimage(
                &params,
                &trapdoor,
                &public_matrix,
                &crate::matrix::ResidentPolyMatrixColumnSource::new(target.clone()),
                &trapdoor_sampler.minimum_preimage_width(&params, public_matrix.row_size()),
                params.modulus().as_ref() / BigUint::from(2u8),
                rand::random(),
            )
            .expect("permissive bound should accept a preimage");
        let preimage = preimage.value();

        let expected_rows = size * (k + 2);
        let expected_cols = target_cols;

        assert_eq!(
            preimage.row_size(),
            expected_rows,
            "Preimage matrix should have the correct number of rows"
        );

        assert_eq!(
            preimage.col_size(),
            expected_cols,
            "Preimage matrix should have the correct number of columns (equal to target columns)"
        );

        // public_matrix * preimage should be equal to target
        let product = public_matrix * preimage;

        assert_eq!(product, target, "Product of public matrix and preimage should equal target");
    }

    #[test]
    #[serial_test::serial]
    fn test_preimage_generation_base_8() {
        let params = DCRTPolyParams::new(4, 2, 17, 3, None, None);
        let size = 4;
        let target_cols = 6;
        let k = params.modulus_digits();
        let trapdoor_sampler =
            DCRTPolyTrapdoorSampler::new(&params, SIGMA, TrapdoorInstantiation::Computational);
        let (trapdoor, public_matrix) = trapdoor_sampler.trapdoor(&params, size, None).unwrap();

        // Create a non-square target matrix (size x target_cols) such that target_cols > size
        // target_cols is not a multiple of size
        let uniform_sampler = DCRTPolyUniformSampler::new();
        let target =
            uniform_sampler.sample_uniform(&params, size, target_cols, DistType::FinRingDist);

        let preimage = trapdoor_sampler
            .preimage(
                &params,
                &trapdoor,
                &public_matrix,
                &crate::matrix::ResidentPolyMatrixColumnSource::new(target.clone()),
                &trapdoor_sampler.minimum_preimage_width(&params, public_matrix.row_size()),
                params.modulus().as_ref() / BigUint::from(2u8),
                rand::random(),
            )
            .expect("permissive bound should accept a preimage");
        let preimage = preimage.value();

        let expected_rows = size * (k + 2);
        let expected_cols = target_cols;

        assert_eq!(
            preimage.row_size(),
            expected_rows,
            "Preimage matrix should have the correct number of rows"
        );

        assert_eq!(
            preimage.col_size(),
            expected_cols,
            "Preimage matrix should have the correct number of columns (equal to target columns)"
        );

        // public_matrix * preimage should be equal to target
        let product = public_matrix * preimage;

        assert_eq!(product, target, "Product of public matrix and preimage should equal target");
    }

    #[test]
    #[serial_test::serial]
    fn test_preimage_generation_base_1024() {
        let params = DCRTPolyParams::new(4, 2, 20, 10, None, None);
        let size = 4;
        let target_cols = 6;
        let k = params.modulus_digits();
        let trapdoor_sampler =
            DCRTPolyTrapdoorSampler::new(&params, SIGMA, TrapdoorInstantiation::Computational);
        let (trapdoor, public_matrix) = trapdoor_sampler.trapdoor(&params, size, None).unwrap();

        // Create a non-square target matrix (size x target_cols) such that target_cols > size
        // target_cols is not a multiple of size
        let uniform_sampler = DCRTPolyUniformSampler::new();
        let target =
            uniform_sampler.sample_uniform(&params, size, target_cols, DistType::FinRingDist);

        let preimage = trapdoor_sampler
            .preimage(
                &params,
                &trapdoor,
                &public_matrix,
                &crate::matrix::ResidentPolyMatrixColumnSource::new(target.clone()),
                &trapdoor_sampler.minimum_preimage_width(&params, public_matrix.row_size()),
                params.modulus().as_ref() / BigUint::from(2u8),
                rand::random(),
            )
            .expect("permissive bound should accept a preimage");
        let preimage = preimage.value();

        let expected_rows = size * (k + 2);
        let expected_cols = target_cols;

        assert_eq!(
            preimage.row_size(),
            expected_rows,
            "Preimage matrix should have the correct number of rows"
        );

        assert_eq!(
            preimage.col_size(),
            expected_cols,
            "Preimage matrix should have the correct number of columns (equal to target columns)"
        );

        // public_matrix * preimage should be equal to target
        let product = public_matrix * preimage;

        assert_eq!(product, target, "Product of public matrix and preimage should equal target");
    }

    #[test]
    #[serial_test::serial]
    fn test_preimage_sampler_parameters_follow_instance_sigma() {
        let params = DCRTPolyParams::new(1 << 10, 5, 51, 17, None, None);
        let base = 1u32 << params.base_bits();
        let computational = TrapdoorInstantiation::Computational;
        let default_sampler = DCRTPolyTrapdoorSampler::new(&params, SIGMA, computational);
        let larger_sigma = SIGMA * 1.5;
        let larger_sampler = DCRTPolyTrapdoorSampler::new(&params, larger_sigma, computational);
        assert_eq!(default_sampler.c, gadget_sampler_sigma(base, SIGMA));
        assert_eq!(larger_sampler.c, gadget_sampler_sigma(base, larger_sigma));
        assert!(larger_sampler.c > default_sampler.c);
        let (larger, default) = (
            larger_sampler.minimum_preimage_width(&params, 2),
            default_sampler.minimum_preimage_width(&params, 2),
        );
        assert!(
            larger.numerator() * default.denominator() > default.numerator() * larger.denominator()
        );
    }

    #[test]
    #[serial_test::serial]
    fn test_preimage_rejects_width_or_bound_below_the_minimum_before_sampling() {
        let params = DCRTPolyParams::new(4, 2, 17, 2, None, None);
        let size = 2usize;
        let sampler =
            DCRTPolyTrapdoorSampler::new(&params, SIGMA, TrapdoorInstantiation::Computational);
        let (trapdoor, public_matrix) = sampler.trapdoor(&params, size, None).unwrap();
        let target =
            DCRTPolyUniformSampler::new().sample_uniform(&params, size, 1, DistType::FinRingDist);
        let width = sampler.minimum_preimage_width(&params, size);
        let minimum = hard_cutoff_from_rational(&width).unwrap();
        assert_eq!(
            Some(minimum.clone()),
            default_preimage_cutoff(
                params.ring_dimension(),
                size,
                params.modulus_digits(),
                1u32 << params.base_bits(),
                SIGMA,
                TrapdoorInstantiation::Computational,
            )
        );
        let requested = &minimum - BigUint::from(1u8);
        let source = crate::matrix::ResidentPolyMatrixColumnSource::new(target.clone());
        assert_eq!(
            sampler.preimage(
                &params,
                &trapdoor,
                &public_matrix,
                &source,
                &width,
                requested.clone(),
                rand::random()
            ),
            Err(SmallMatrixError::PreimageBoundTooSmall { requested, minimum: minimum.clone() })
        );
        let narrower = Rational::new(width.numerator() - 1u8, width.denominator().clone()).unwrap();
        assert!(matches!(
            sampler.preimage(
                &params,
                &trapdoor,
                &public_matrix,
                &source,
                &narrower,
                minimum,
                rand::random()
            ),
            Err(SmallMatrixError::PreimageWidthTooSmall { .. })
        ));
    }

    fn assert_coefficients_within(matrix: &DCRTPolyMatrix, bound: &BigUint) {
        let modulus = matrix.params().modulus();
        for i in 0..matrix.row_size() {
            for j in 0..matrix.col_size() {
                for coeff in matrix.entry(i, j).coeffs() {
                    let value = coeff.value().clone();
                    let neg = modulus.as_ref() - &value;
                    assert!(value.min(neg) <= *bound, "coefficient at ({i}, {j}) exceeds {bound}");
                }
            }
        }
    }

    fn assert_preimage_reconstructs_target_and_respects_norm_bound(sigma: f64, widen: u32) {
        let size = 2usize;
        let params = DCRTPolyParams::new(1 << 10, 5, 51, 17, None, None);
        let trapdoor_sampler =
            DCRTPolyTrapdoorSampler::new(&params, sigma, TrapdoorInstantiation::Computational);
        let (trapdoor, public_matrix) = trapdoor_sampler.trapdoor(&params, size, None).unwrap();
        let uniform_sampler = DCRTPolyUniformSampler::new();
        let minimum = trapdoor_sampler.minimum_preimage_width(&params, size);
        let width =
            Rational::new(minimum.numerator() * widen, minimum.denominator().clone()).unwrap();
        let preimage_bound = hard_cutoff_from_rational(&width).unwrap();
        for _ in 0..4usize {
            let target = uniform_sampler.sample_uniform(&params, size, size, DistType::FinRingDist);
            let preimage = trapdoor_sampler
                .preimage(
                    &params,
                    &trapdoor,
                    &public_matrix,
                    &crate::matrix::ResidentPolyMatrixColumnSource::new(target.clone()),
                    &width,
                    preimage_bound.clone(),
                    rand::random(),
                )
                .expect("bounded sampler should return a valid preimage");
            assert_eq!(preimage.max_coefficient_bound(), &preimage_bound);
            let preimage = preimage.value();
            assert_eq!(&public_matrix * preimage, target);
            assert_coefficients_within(preimage, &preimage_bound);
        }
    }

    #[test]
    #[serial_test::serial]
    fn test_preimage_coefficients_below_the_minimum_width_cutoff() {
        assert_preimage_reconstructs_target_and_respects_norm_bound(SIGMA, 1);
    }

    #[test]
    #[serial_test::serial]
    fn test_preimage_coefficients_below_a_requested_wider_cutoff() {
        assert_preimage_reconstructs_target_and_respects_norm_bound(SIGMA * 1.25, 3);
    }

    #[test]
    #[serial_test::serial]
    fn test_p_hat_coefficients_below_the_minimum_width_cutoff() {
        let size = 2usize;
        let params = DCRTPolyParams::new(1 << 10, 5, 51, 17, None, None);
        let trapdoor_sampler =
            DCRTPolyTrapdoorSampler::new(&params, SIGMA, TrapdoorInstantiation::Computational);
        let (trapdoor, _public_matrix) = trapdoor_sampler.trapdoor(&params, size, None).unwrap();
        let width = trapdoor_sampler.minimum_preimage_width(&params, size);
        let preimage_bound = hard_cutoff_from_rational(&width).unwrap();
        let s = width.numerator().to_f64().unwrap() / width.denominator().to_f64().unwrap();
        let c = trapdoor_sampler.c;
        let dgg_large_std = (s * s - c * c).sqrt();
        let peikert = dgg_large_std < KARNEY_THRESHOLD;
        let (dgg_large_mean, dgg_large_table) = peikert_table(dgg_large_std);
        let dgg_large_params = (dgg_large_mean, dgg_large_std, dgg_large_table.as_deref());
        for _ in 0..4usize {
            let p_hat =
                trapdoor.sample_perturbation(&params, s, c, SIGMA, dgg_large_params, peikert, size);
            assert_coefficients_within(&p_hat, &preimage_bound);
        }
    }

    #[test]
    #[serial_test::serial]
    fn statistical_trapgen_publishes_a_bar_and_g_minus_a_bar_t() {
        let params = DCRTPolyParams::new(8, 2, 17, 9, None, None);
        let m_bar = 11;
        let instantiation = TrapdoorInstantiation::Statistical { m_bar };
        let sampler = DCRTPolyTrapdoorSampler::new(&params, SIGMA, instantiation);
        let (trapdoor, public) = sampler.trapdoor(&params, 1, None).unwrap();
        let digits = params.modulus_digits();
        assert_eq!(public.size(), (1, m_bar + digits));
        assert_eq!(trapdoor.instantiation(), instantiation);
        assert!(trapdoor.t_chunks().len() > 1, "T spans several row chunks");
        let t = trapdoor.t_chunks()[0]
            .concat_rows(&trapdoor.t_chunks()[1..].iter().collect::<Vec<_>>());
        assert_eq!(t.size(), (m_bar, digits));
        let trapdoor_matrix = t.concat_rows(&[&DCRTPolyMatrix::identity(&params, digits, None)]);
        assert_eq!(&public * &trapdoor_matrix, DCRTPolyMatrix::gadget_matrix(&params, 1, None));
        // T is an untruncated Gaussian, never uniform: its coefficients stay far below q.
        let bound = BigUint::from((40.0 * SIGMA) as u64);
        assert_coefficients_within(&t, &bound);
    }

    #[test]
    #[serial_test::serial]
    fn statistical_preimages_reconstruct_the_target_within_the_cutoff() {
        let params = DCRTPolyParams::new(16, 2, 30, 15, None, None);
        let instantiation = TrapdoorInstantiation::Statistical { m_bar: 10 };
        let sampler = DCRTPolyTrapdoorSampler::new(&params, SIGMA, instantiation);
        let (trapdoor, public) = sampler.trapdoor(&params, 1, None).unwrap();
        let width = sampler.minimum_preimage_width(&params, 1);
        let bound = hard_cutoff_from_rational(&width).unwrap();
        let target =
            DCRTPolyUniformSampler::new().sample_uniform(&params, 1, 3, DistType::FinRingDist);
        let preimage = sampler
            .preimage(
                &params,
                &trapdoor,
                &public,
                &crate::matrix::ResidentPolyMatrixColumnSource::new(target.clone()),
                &width,
                bound.clone(),
                rand::random(),
            )
            .unwrap();
        let preimage = preimage.value();
        assert_eq!(preimage.size(), (10 + params.modulus_digits(), 3));
        assert_eq!(&public * preimage, target);
        assert_coefficients_within(preimage, &bound);
        let extension =
            DCRTPolyUniformSampler::new().sample_uniform(&params, 1, 4, DistType::FinRingDist);
        let extended = sampler
            .preimage_extend(
                &params,
                &trapdoor,
                &public,
                &extension,
                &target,
                &width,
                bound.clone(),
            )
            .unwrap();
        assert_eq!(public.concat_columns(&[&extension]) * &extended, target);
        assert_coefficients_within(&extended, &bound);
    }

    /// The empirical variance of the centered coefficients of rows `rows` of `matrix`.
    pub(crate) fn coefficient_variance(matrix: &DCRTPolyMatrix, rows: Range<usize>) -> f64 {
        let modulus = matrix.params().modulus();
        let half = modulus.as_ref() / 2u8;
        let mut sum = 0.0;
        let mut count = 0usize;
        for row in rows {
            for column in 0..matrix.col_size() {
                for coeff in matrix.entry(row, column).coeffs() {
                    let value = coeff.value().clone();
                    let centered = if value > half {
                        -(modulus.as_ref() - &value).to_f64().unwrap()
                    } else {
                        value.to_f64().unwrap()
                    };
                    sum += centered * centered;
                    count += 1;
                }
            }
        }
        sum / count as f64
    }

    /// A preimage `p + [T z; z]` is spherical of width `s`: the perturbation's covariance
    /// `s² I - c² [T; I][T; I]*` plus the gadget part `c² [T; I][T; I]*`. Both blocks of
    /// coefficients have variance about `s²`.
    #[test]
    #[serial_test::serial]
    fn statistical_preimages_are_spherical_of_the_requested_width() {
        let params = DCRTPolyParams::new(32, 1, 40, 10, None, None);
        let m_bar = 11;
        let instantiation = TrapdoorInstantiation::Statistical { m_bar };
        let sampler = DCRTPolyTrapdoorSampler::new(&params, SIGMA, instantiation);
        let (trapdoor, public) = sampler.trapdoor(&params, 1, None).unwrap();
        let width = sampler.minimum_preimage_width(&params, 1);
        let s = width.numerator().to_f64().unwrap() / width.denominator().to_f64().unwrap();
        let target =
            DCRTPolyUniformSampler::new().sample_uniform(&params, 1, 64, DistType::FinRingDist);
        let preimage = sampler
            .preimage(
                &params,
                &trapdoor,
                &public,
                &crate::matrix::ResidentPolyMatrixColumnSource::new(target.clone()),
                &width,
                params.modulus().as_ref() / BigUint::from(4u8),
                rand::random(),
            )
            .unwrap();
        let preimage = preimage.value();
        assert_eq!(&public * preimage, target);
        let top = coefficient_variance(preimage, 0..m_bar) / (s * s);
        let bottom =
            coefficient_variance(preimage, m_bar..m_bar + params.modulus_digits()) / (s * s);
        assert!((top - 1.0).abs() < 0.1, "top variance {top} s²");
        assert!((bottom - 1.0).abs() < 0.1, "bottom variance {bottom} s²");
    }

    #[test]
    #[serial_test::serial]
    fn statistical_trapdoor_width_follows_its_estimate_and_needs_one_row() {
        let params = DCRTPolyParams::new(16, 2, 30, 15, None, None);
        let base = 1u32 << params.base_bits();
        let statistical = TrapdoorInstantiation::Statistical { m_bar: 2 };
        let s1 = crate::sampler::bounds::trapdoor_singular_value_estimate(
            16,
            1,
            params.modulus_digits(),
            SIGMA,
            statistical,
        )
        .unwrap()
        .to_f64()
        .unwrap();
        let width = crate::sampler::bounds::minimum_preimage_sigma(
            16,
            1,
            params.modulus_digits(),
            base,
            SIGMA,
            statistical,
        )
        .unwrap()
        .to_f64()
        .unwrap();
        // s² = c_G² (1 + ŝ_1²) + r², with the rounding width r = σ.
        let c = gadget_sampler_sigma(base, SIGMA);
        let expected = c * c * (1.0 + s1 * s1) + SIGMA * SIGMA;
        assert!((width * width - expected).abs() < 1e-9 * expected);
        assert!(
            crate::sampler::bounds::minimum_preimage_sigma(
                16,
                2,
                params.modulus_digits(),
                base,
                SIGMA,
                statistical
            )
            .is_none()
        );
    }

    #[test]
    #[serial_test::serial]
    fn trapdoor_encodings_round_trip_and_read_the_untagged_computational_form() {
        let params = DCRTPolyParams::new(8, 2, 17, 9, None, None);
        for instantiation in
            [TrapdoorInstantiation::Computational, TrapdoorInstantiation::Statistical { m_bar: 9 }]
        {
            let sampler = DCRTPolyTrapdoorSampler::new(&params, SIGMA, instantiation);
            let (trapdoor, _) = sampler.trapdoor(&params, 1, None).unwrap();
            let bytes = DCRTPolyTrapdoorSampler::trapdoor_to_bytes(&trapdoor);
            let decoded = DCRTPolyTrapdoorSampler::trapdoor_from_bytes(&params, &bytes).unwrap();
            assert_eq!(decoded, trapdoor);
            assert_eq!(decoded.instantiation(), instantiation);
        }
        let sampler =
            DCRTPolyTrapdoorSampler::new(&params, SIGMA, TrapdoorInstantiation::Computational);
        let (trapdoor, _) = sampler.trapdoor(&params, 1, None).unwrap();
        let mut legacy = Vec::new();
        for matrix in trapdoor.matrices() {
            let bytes = matrix.to_compact_bytes();
            legacy.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
            legacy.extend_from_slice(&bytes);
        }
        assert_eq!(DCRTPolyTrapdoorSampler::trapdoor_from_bytes(&params, &legacy), Some(trapdoor));
    }

    #[test]
    #[serial_test::serial]
    fn test_preimage_generation_extend() {
        let params = DCRTPolyParams::default();
        let size = 3;
        let k = params.modulus_digits();
        let trapdoor_sampler =
            DCRTPolyTrapdoorSampler::new(&params, SIGMA, TrapdoorInstantiation::Computational);
        let (trapdoor, public_matrix) = trapdoor_sampler.trapdoor(&params, size, None).unwrap();

        let uniform_sampler = DCRTPolyUniformSampler::new();
        let target = uniform_sampler.sample_uniform(&params, size, 1, DistType::FinRingDist);
        let m = size * params.modulus_digits();
        let extend = uniform_sampler.sample_uniform(&params, size, m, DistType::FinRingDist);

        let width = trapdoor_sampler.minimum_preimage_width(&params, size);
        let bound = hard_cutoff_from_rational(&width).unwrap();
        let preimage = trapdoor_sampler
            .preimage_extend(
                &params,
                &trapdoor,
                &public_matrix,
                &extend,
                &target,
                &width,
                bound.clone(),
            )
            .unwrap();
        // Both parts are sampled with the same width and bounded by the same cutoff.
        assert_coefficients_within(&preimage, &bound);
        let narrower = Rational::new(width.numerator() - 1u8, width.denominator().clone()).unwrap();
        assert!(matches!(
            trapdoor_sampler.preimage_extend(
                &params,
                &trapdoor,
                &public_matrix,
                &extend,
                &target,
                &narrower,
                bound,
            ),
            Err(SmallMatrixError::PreimageWidthTooSmall { .. })
        ));

        let expected_rows = size * (k + 2) + m;
        let expected_cols = 1;

        assert_eq!(
            preimage.row_size(),
            expected_rows,
            "Preimage matrix should have the correct number of rows"
        );

        assert_eq!(
            preimage.col_size(),
            expected_cols,
            "Preimage matrix should have the correct number of columns"
        );

        // public_matrix * preimage should be equal to target
        let product = public_matrix.concat_columns(&[&extend]) * &preimage;
        assert_eq!(product, target, "Product of public matrix and preimage should equal target");
    }
}
