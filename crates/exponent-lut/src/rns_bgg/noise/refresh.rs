//! Noise of the noise refresh (see [`crate::refresh`]).
//!
//! Column `j` of the decoder of prime `i` sees `Q_i y_j + M_ij + N_ij` with
//! `N_ij = e_b K_ij - E_ij - e^sm_ij` and `E_i = e_1 X_i - e_x Y_i + e_f Z_i`;
//! its rounding returns `y_j mod q_i` when `|M_ij + N_ij| < Q_i / 2`. Worst-case
//! coefficient bounds, term by term:
//!
//! - `e_1 X_i`: `X_i` has `2m` rows of polynomial digits with coefficients at most `B/2` in
//!   magnitude (balanced decomposition), so `2 m n (B/2) B_e` for the fresh encoding error bound
//!   `B_e` of the one encoding;
//! - `e_x Y_i`: `Y_i` has `w` rows of constant digits of the level-`(l+1)` gadget, at most `w
//!   floor(B_(l+1)/2)` in `l1` per column, times the target's simulated error;
//! - `e_f Z_i`: the exact balanced `l1` of `G_Q^-1(Q_i)` times the largest fresh carrier error
//!   after its ModDown to `Q` ([`RnsBggParameters::lut_mod_down_bound`] of the child's simulated
//!   raw coefficient output);
//! - `e_b K`: the unmask hint's error against the preimage, as for the decoder;
//! - `e^sm_ij`: column `j` of decoder `i`'s simulated smudge row (every column sums its own block
//!   of the same shape, so one column's sum bounds each).
//!
//! The fresh child's row has one simulated value for every column (its row `Sum` of equal
//! blocks).
//!
//! The parts are added coherently.
use super::*;
use crate::{api::NoiseBuild, refresh::RefreshDeclaration};

/// The noise of one refresh: per prime `q_i`, the largest decoder noise bound
/// `F_i` over its columns and the spacing `Q_i`; the smudge magnitude bound
/// `P^d / 2`; and the fresh noise bound `eta`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RnsBggRefreshNoise {
    /// The refresh specification commitment.
    pub refresh: [u8; 32],
    /// The worst-case `F_i >= ||N_ij||` for every column `j`, by prime.
    pub errors: Vec<BigUint>,
    /// With an average-case model, `F_i` at its tail (see
    /// [`simulate_refresh_noise`]), by prime; the accepted bound when present.
    pub average_errors: Option<Vec<BigUint>>,
    /// `Q_i = Q / q_i`, by prime.
    pub spacings: Vec<BigUint>,
    /// The smudge magnitude bound `P^d / 2`.
    pub smudge: BigUint,
    /// The fresh noise bound: `||e'||_inf <= eta`.
    pub fresh: BigUint,
}

impl RnsBggRefreshNoise {
    /// The accepted bounds: the average-case ones when computed, else the
    /// worst-case ones.
    pub fn accepted_errors(&self) -> &[BigUint] {
        self.average_errors.as_deref().unwrap_or(&self.errors)
    }

    /// Whether every decoder rounds exactly: `P^d / 2 + F_i < Q_i / 2`.
    pub fn fits(&self) -> bool {
        self.accepted_errors()
            .iter()
            .zip(&self.spacings)
            .all(|(error, spacing)| (&self.smudge + error) * 2u8 < *spacing)
    }

    /// Whether the smudge range `P^d` hides every decoder's noise by
    /// `P^d > F_i 2^bits`, `bits` per smudged value (see
    /// [`crate::decoder::decoding`]).
    pub fn hides(&self, statistical_bits: usize) -> bool {
        self.accepted_errors().iter().all(|error| &self.smudge * 2u8 > error << statistical_bits)
    }
}

/// The raw output values of `program` lowered on `parameters` with
/// `bindings`, its smudge members transformed when `decoder` is given.
fn lower_child(
    parameters: &RnsBggParameters,
    program: &crate::program::ExponentLutProgram,
    bindings: &RnsBggNoiseBindings,
    decoder: Option<&DecoderDeclaration>,
) -> Result<Vec<RnsBggNoiseValue>, RnsBggError> {
    let workload = RnsBggDerivedWorkload::from_program(program)?;
    let mut build = RnsNoiseBuild::new(parameters, program, &workload, bindings)?;
    if let Some(decoder) = decoder {
        build.bind_decoder_smudge(decoder)?;
    }
    let (evaluator, families, environment) =
        build.lowering_state().map_err(|error| RnsBggError::Graph(error.to_string()))?;
    Ok(crate::program::lower_program(program, evaluator, families, environment)
        .map_err(|error| RnsBggError::Graph(error.to_string()))?
        .into_iter()
        .map(|(_, value)| value)
        .collect())
}

/// The child noise contract of a PRF Program on `parameters`: the
/// accumulator start and the label are fresh monomial encodings, every
/// label member a separate encryption, and every RHS input a bit; the
/// selector families of `prf` keep their one-hot row contract
/// ([`RnsBggNoiseBindings::with_selector_families`]).
fn child_bindings(
    parameters: &RnsBggParameters,
    program: &crate::program::ExponentLutProgram,
    prf: &crate::program_gadgets::prf::SparseLwrPrf,
) -> Result<RnsBggNoiseBindings, RnsBggError> {
    let one = [num_bigint::BigInt::from(1u8)];
    let inputs = program
        .inputs()
        .iter()
        .map(|(wire, declaration)| {
            let value = RnsBggNoiseValue::sampled_encoding(
                parameters,
                format!("refresh-child-input-{wire:?}").as_bytes(),
                &one,
            );
            match declaration.shape() {
                InputBinding::Scalar(()) => value,
                InputBinding::Family(_) => value.independent_family_members(),
            }
        })
        .collect::<Vec<_>>();
    RnsBggNoiseBindings::from_inputs(
        program,
        &inputs,
        &vec![
            RnsBggNoiseRhs { payload_l1: BigUint::one(), payload_l2_squared: BigUint::one() };
            program.rhs_inputs().len()
        ],
    )?
    .with_selector_families(&prf.selector_families())
}

/// The moments `M` of a coefficient carrier after one ModDown by the scale
/// `p` of `level`: `M / p` plus the public-key and encoding roundings.
fn mod_down_moments(
    parameters: &RnsBggParameters,
    level: usize,
    moments: &NoiseMoments,
) -> Result<NoiseMoments, RnsBggError> {
    let p = parameters.scale(level)?.clone();
    let scaled = NoiseMoments {
        mean_numerator: moments.mean_numerator.clone(),
        mean_denominator: &moments.mean_denominator * &p,
        mean_radius: moments.mean_radius.scaled_ratio(&BigUint::one(), &p),
        variance: moments.variance.scaled_ratio(&BigUint::one(), &(&p * &p)),
    };
    let rounding = zero_mean_moments(parameters.lut_public_rounding_variance(level)?)
        .add_with_covariance(
            &NoiseMoments::discrete_rounding_residue(&p)
                .map_err(|error| RnsBggError::Graph(error.to_string()))?,
            NoiseCorrelation::Independent,
        );
    Ok(scaled.add_with_covariance(&rounding, NoiseCorrelation::Independent))
}

/// The image of `moments` under a constant action of coefficient `l1` norm
/// `l1`: the mean scales by `l1` and the variance coherently by `l1^2`.
fn constant_action(moments: &NoiseMoments, l1: &BigUint) -> NoiseMoments {
    NoiseMoments {
        mean_numerator: &moments.mean_numerator * BigInt::from(l1.clone()),
        mean_denominator: moments.mean_denominator.clone(),
        mean_radius: moments.mean_radius.scaled(l1),
        variance: moments.variance.scaled(&l1.pow(2u32)),
    }
}

/// `ceil(|mean| + radius + sqrt(z^2 Var))`: the average-case bound of a value
/// with `moments` at the tail multiplier `z^2`.
fn tail_bound(moments: &NoiseMoments, z_squared: &NonnegativeRational) -> BigUint {
    let mean = NonnegativeRational::new(
        moments.mean_numerator.magnitude().clone(),
        moments.mean_denominator.clone(),
    )
    .expect("positive mean denominator")
    .add(&moments.mean_radius);
    let spread = NonnegativeRational::new(
        &z_squared.numerator * &moments.variance.numerator,
        &z_squared.denominator * &moments.variance.denominator,
    )
    .expect("positive variance denominator");
    let root = (&spread.numerator * &spread.denominator).sqrt() / &spread.denominator + 1u8;
    mean.numerator / mean.denominator + 1u8 + root
}

/// Moments of `count` products of a value with `moments` by independent
/// uniform digits of variance `digit_variance` (zero mean): zero-mean, with
/// variance `count (Var + (|mean| + radius)^2) digit_variance`.
fn digit_action(
    moments: &NoiseMoments,
    count: &BigUint,
    digit_variance: &NonnegativeRational,
) -> NoiseMoments {
    let mean = NonnegativeRational::new(
        moments.mean_numerator.magnitude().clone(),
        moments.mean_denominator.clone(),
    )
    .expect("positive mean denominator")
    .add(&moments.mean_radius);
    let second = moments.variance.add(
        &NonnegativeRational::new(
            &mean.numerator * &mean.numerator,
            &mean.denominator * &mean.denominator,
        )
        .expect("positive denominator"),
    );
    zero_mean_moments(
        second.scaled(count).scaled_ratio(&digit_variance.numerator, &digit_variance.denominator),
    )
}

/// The noise of `declaration`'s refresh of output `target_index` of
/// `source` under `bindings`, with the fresh-noise child on `fresh`, a
/// top-refresh part's mask child on `mask`, the decoders' smudge children on
/// `smudges` (in decoder order), and the unmask trapdoor sigma
/// `unmask_sigma`.
///
/// The worst-case bounds are always computed. With `average` (accepting
/// average-case acceptance), every child's evaluation budget is audited in
/// the average-case model ([`RnsBggParameters::audit_evaluation`]) and
/// [`RnsBggRefreshNoise::average_errors`] bounds each decoder's noise by
/// `|mean| + z sigma` at the tail of the decoded coefficients. The digits of
/// `X_i` are uniform in `[-B/2, B/2)` (the decomposition of a uniform
/// matrix) and constant actions scale coherently. The fresh encryption
/// terms (the public one encoding's blocks and the hint) are independent
/// samples; the evaluated terms (the mask carrier, the target, the fresh
/// carriers and the smudge sum) share the source secret and their Program's
/// inputs, so they are added with unknown covariance, to each other and to
/// the fresh terms.
#[allow(clippy::too_many_arguments)]
pub fn simulate_refresh_noise(
    parameters: &RnsBggParameters,
    mask: Option<&RnsBggParameters>,
    fresh: &RnsBggParameters,
    smudges: &[&RnsBggParameters],
    source: &crate::program::ExponentLutProgram,
    bindings: &RnsBggNoiseBindings,
    declaration: &RefreshDeclaration,
    target_index: usize,
    unmask_sigma: f64,
    average: Option<&crate::decoder::simulation::AverageCaseConfig>,
) -> Result<RnsBggRefreshNoise, crate::api::ApiError> {
    use crate::api::ApiError;
    let level = parameters
        .validate_refresh_children(declaration, source, mask, fresh, smudges)
        .map_err(ApiError::scheme)?;
    if source.outputs().get(target_index) !=
        Some(&crate::program::Operand::Scalar(declaration.target_output()))
    {
        return Err(ApiError::scheme(RnsBggError::Decoder(DecoderError::TargetMismatch)));
    }
    let average = average.filter(|config| config.allow_average_acceptance);
    let workload = RnsBggDerivedWorkload::from_program(source).map_err(ApiError::scheme)?;
    let trace = crate::api::noise_simulate(
        source,
        RnsNoiseBuild::new(parameters, source, &workload, bindings).map_err(ApiError::scheme)?,
        |_, outputs| Ok(outputs),
    )?;
    let target = trace
        .outputs
        .get(target_index)
        .ok_or_else(|| ApiError::input("refresh target output is out of range"))?
        .clone();
    // A raw coefficient child's outputs, audited when average-case.
    let carriers = |chain: &RnsBggParameters, child: &crate::refresh::FreshNoiseProgram| {
        let program = child.program();
        let bindings =
            child_bindings(chain, program, child.sampler().prf()).map_err(ApiError::scheme)?;
        if let Some(config) = average {
            chain.audit_evaluation(program, &bindings, Some(config)).map_err(ApiError::scheme)?;
        }
        lower_child(chain, program, &bindings, None).map_err(ApiError::scheme)
    };
    // Fresh carriers: the raw coefficient outputs moved down once to `Q`.
    let fresh_level =
        RnsBggParameters::refresh_fresh_level(declaration).map_err(ApiError::scheme)?;
    let fresh_value = carriers(fresh, declaration.fresh())?
        .into_iter()
        .next()
        .ok_or_else(|| ApiError::input("the fresh Program has no output"))?;
    let mask_value = match (declaration.top(), mask) {
        (Some(part), Some(chain)) => {
            let value = carriers(chain, part.mask())?
                .into_iter()
                .next()
                .ok_or_else(|| ApiError::input("the mask Program has no output"))?;
            let level =
                RnsBggParameters::refresh_mask_level(declaration).map_err(ApiError::scheme)?;
            Some((chain, level, value))
        }
        _ => None,
    };
    // Smudge children of one Program share their noise: simulate each
    // distinct Program once.
    let mut smudge_values =
        BTreeMap::<crate::program::ExponentLutProgramId, RnsBggNoiseValue>::new();
    for (smudge, decoder) in smudges.iter().zip(declaration.decoders()) {
        let program = decoder.smudge().program();
        if smudge_values.contains_key(&program.id()) {
            continue;
        }
        let child =
            child_bindings(smudge, program, decoder.smudge().prf()).map_err(ApiError::scheme)?;
        if let Some(config) = average {
            smudge.audit_evaluation(program, &child, Some(config)).map_err(ApiError::scheme)?;
        }
        let value = lower_child(smudge, program, &child, Some(decoder))
            .map_err(ApiError::scheme)?
            .into_iter()
            .next()
            .ok_or_else(|| ApiError::input("the smudge Program has no output"))?;
        smudge_values.insert(program.id(), value);
    }
    let smudges = declaration
        .decoders()
        .iter()
        .map(|decoder| &smudge_values[&decoder.smudge().program().id()])
        .collect::<Vec<_>>();
    compose_refresh_noise(
        parameters,
        level,
        &target,
        mask_value.as_ref().map(|(chain, level, value)| (*chain, *level, value)),
        fresh,
        fresh_level,
        (&fresh_value, declaration.columns()),
        &smudges,
        &declaration.decoded_primes(),
        (*declaration.spec(), declaration.decoders()[0].smudge_bound(), declaration.eta()),
        unmask_sigma,
        average,
    )
}

/// The refresh noise of [`simulate_refresh_noise`] from its lowered parts:
/// the target's level-`level` value, a top-refresh part's mask carrier (its
/// chain, LUT level and raw value), the fresh carrier row's raw value (every
/// column's) on `fresh` at its LUT level `fresh_level` with its `outputs`
/// columns, and every decoder's smudge row value (every column's; in decoder
/// order, one per decoded prime), for the refresh `(spec, P^d / 2, eta)`.
#[allow(clippy::too_many_arguments)]
fn compose_refresh_noise(
    parameters: &RnsBggParameters,
    level: usize,
    target: &RnsBggNoiseValue,
    mask: Option<(&RnsBggParameters, usize, &RnsBggNoiseValue)>,
    fresh: &RnsBggParameters,
    fresh_level: usize,
    (fresh_value, outputs): (&RnsBggNoiseValue, usize),
    smudges: &[&RnsBggNoiseValue],
    decoded_primes: &[usize],
    (refresh, smudge_bound, eta): ([u8; 32], BigUint, u32),
    unmask_sigma: f64,
    average: Option<&crate::decoder::simulation::AverageCaseConfig>,
) -> Result<RnsBggRefreshNoise, crate::api::ApiError> {
    use crate::api::ApiError;
    let fresh_error = fresh
        .lut_mod_down_bound(fresh_level, &fresh_value.bound.error)
        .map_err(ApiError::scheme)?;
    let level_parameters = parameters.level(level).map_err(ApiError::scheme)?;
    let next = parameters.level(level + 1).map_err(ApiError::scheme)?;
    let n = BigUint::from(level_parameters.ring_dimension());
    let columns = level_parameters.modulus_digits();
    let level_primes = level_parameters.to_crt().0;
    if smudges.len() != decoded_primes.len() ||
        decoded_primes.iter().any(|prime| *prime >= level_primes.len())
    {
        return Err(ApiError::scheme(RnsBggError::Decoder(DecoderError::TargetMismatch)));
    }
    let half_base = |params: &DCRTPolyParams| BigUint::one() << (params.base_bits() - 1);
    // One block of `e_1 X_i` (`m` rows of polynomial digits): fresh encryption
    // error, or the mask carrier's error after its ModDown.
    let block = BigUint::from(columns) * &n * half_base(level_parameters);
    let fresh_block = &block * parameters.encoding_error_bound();
    let one_term = match &mask {
        None => &fresh_block * 2u8,
        Some((chain, mask_level, value)) => {
            &fresh_block +
                &block *
                    chain
                        .lut_mod_down_bound(*mask_level, &value.bound.error)
                        .map_err(ApiError::scheme)?
        }
    };
    let target_l1 = BigUint::from(next.modulus_digits()) * half_base(next);
    let target_term = &target_l1 * &target.bound.error;
    let (_, unmask_digits, preimage_bound) =
        parameters.unmask_gadget(level, unmask_sigma).map_err(ApiError::scheme)?;
    let unmask_terms =
        &n * BigUint::from(crate::decoder::unmask::UNMASK_ROWS * (unmask_digits + 2));
    let unmask_term = &unmask_terms * parameters.encoding_error_bound() * &preimage_bound;
    // Average case: the prime-independent moments.
    let z_squared = average
        .map(|config| {
            config
                .z_squared_for_events(&(BigUint::from(decoded_primes.len() * outputs) * &n))
                .map_err(|_| ApiError::input("invalid average-case event count"))
        })
        .transpose()?;
    let base = BigUint::one() << level_parameters.base_bits();
    let digit_variance = NonnegativeRational::new(&base * &base + 2u8, BigUint::from(12u8))
        .expect("positive denominator");
    let block_count = BigUint::from(columns) * &n;
    let fresh_block_moments = digit_action(
        &zero_mean_moments(parameters.encoding_error_variance()),
        &block_count,
        &digit_variance,
    );
    let unmask_moments = zero_mean_moments(
        parameters.encoding_error_variance().scaled(&(&unmask_terms * preimage_bound.pow(2u32))),
    );
    let moments = |value: &RnsBggNoiseValue| {
        value.moments.clone().ok_or_else(|| ApiError::input("average-case moments are missing"))
    };
    // The fresh terms are independent samples; the mask carrier is evaluated.
    let (fresh_terms, mask_moments) = match &mask {
        None => (
            fresh_block_moments
                .add_with_covariance(&fresh_block_moments, NoiseCorrelation::Independent)
                .add_with_covariance(&unmask_moments, NoiseCorrelation::Independent),
            None,
        ),
        Some((chain, mask_level, value)) if z_squared.is_some() => (
            fresh_block_moments.add_with_covariance(&unmask_moments, NoiseCorrelation::Independent),
            Some(digit_action(
                &mod_down_moments(chain, *mask_level, &moments(value)?)
                    .map_err(ApiError::scheme)?,
                &block_count,
                &digit_variance,
            )),
        ),
        Some(_) => (fresh_block_moments.clone(), None),
    };
    let modulus: BigUint = level_parameters.modulus().as_ref().clone();
    let mut errors = Vec::with_capacity(decoded_primes.len());
    let mut average_errors = Vec::with_capacity(decoded_primes.len());
    let mut spacings = Vec::with_capacity(decoded_primes.len());
    for (position, prime) in decoded_primes.iter().enumerate() {
        let cofactor = &modulus / level_primes[*prime];
        let (z_l1, _) =
            parameters.gadget_inverse_norms(level, &cofactor).map_err(ApiError::scheme)?;
        let common = &one_term + &target_term + &z_l1 * &fresh_error + &unmask_term;
        errors.push(common + &smudges[position].bound.error);
        if let Some(z_squared) = &z_squared {
            // Every column has the same moments: the target's, the fresh row's
            // and the smudge row's are those of each of their columns.
            let target = constant_action(&moments(target)?, &target_l1);
            let fresh = constant_action(
                &mod_down_moments(fresh, fresh_level, &moments(fresh_value)?)
                    .map_err(ApiError::scheme)?,
                &z_l1,
            );
            let mut evaluated = target
                .add_with_covariance(&fresh, NoiseCorrelation::Unknown)
                .add_with_covariance(&moments(smudges[position])?, NoiseCorrelation::Unknown);
            if let Some(mask) = &mask_moments {
                evaluated = evaluated.add_with_covariance(mask, NoiseCorrelation::Unknown);
            }
            let total = evaluated.add_with_covariance(&fresh_terms, NoiseCorrelation::Unknown);
            average_errors.push(tail_bound(&total, z_squared));
        }
        spacings.push(cofactor);
    }
    Ok(RnsBggRefreshNoise {
        refresh,
        errors,
        average_errors: z_squared.is_some().then_some(average_errors),
        spacings,
        smudge: smudge_bound,
        fresh: BigUint::from(eta),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rns_bgg::refresh::tests::{declaration, identity_source};

    #[test]
    fn refresh_noise_fits_iff_smudge_and_error_stay_below_every_half_spacing() {
        let noise = |errors: &[u32], smudge: u32| RnsBggRefreshNoise {
            refresh: [0; 32],
            errors: errors.iter().copied().map(BigUint::from).collect(),
            average_errors: None,
            spacings: vec![BigUint::from(40u8); errors.len()],
            smudge: BigUint::from(smudge),
            fresh: BigUint::one(),
        };
        assert!(noise(&[3, 2], 16).fits());
        assert!(!noise(&[3, 4], 16).fits(), "2 (16 + 4) = 40 is not below 40");
        assert!(noise(&[3, 3], 16).hides(3));
        assert!(!noise(&[3, 4], 16).hides(3), "32 does not exceed 4 * 2^3");
    }

    /// Noise-free encodings on two 30-bit primes (`m = 4`) with two primes
    /// per child drop: the smudge range of some digit count both hides every
    /// decoder's simulated noise by one bit and keeps it inside the decoding
    /// margin, and the bound grows with the digit count (the smudge sum has
    /// more members).
    #[test]
    fn a_noise_free_refresh_has_a_fitting_and_hiding_digit_count() {
        let pool = DCRTPolyParams::new(8, 12, 30, 15, None, None);
        let prefix = |depth: usize| {
            let modulus =
                pool.to_crt().0[..depth].iter().map(|prime| BigUint::from(*prime)).product();
            pool.select_modulus(&modulus).unwrap()
        };
        let parameters = RnsBggParameters::new(vec![prefix(2), prefix(1)], 0, 1).unwrap();
        let (source, input) = identity_source();
        let one = [num_bigint::BigInt::from(1u8)];
        let bindings = RnsBggNoiseBindings::from_inputs(
            &source,
            &[RnsBggNoiseValue::sampled_encoding(&parameters, b"refresh-target", &one)],
            &[],
        )
        .unwrap();
        let report = |digits: usize| {
            let (declaration, _, _) = declaration(&parameters, &source, input, digits);
            let fresh =
                parameters.refresh_fresh_parameters(&declaration, &source, &pool, 2).unwrap();
            let smudge = parameters
                .decoder_smudge_parameters(&declaration.decoders()[0], &source, &pool, 2)
                .unwrap();
            let smudges = vec![&smudge; declaration.decoders().len()];
            simulate_refresh_noise(
                &parameters,
                None,
                &fresh,
                &smudges,
                &source,
                &bindings,
                &declaration,
                0,
                4.578,
                None,
            )
            .unwrap()
        };
        let (small, large) = (report(1), report(14));
        assert!(small.errors[0] < large.errors[0]);
        assert!(large.fits() && large.hides(1), "{large:?}");
        assert!(!small.hides(1));
        assert_eq!(large.fresh, BigUint::one());
    }

    /// The multi-member PRF (root cell loops and per-member row sums) has the
    /// noise of the one-member PRF: the one-hot selector rows keep the
    /// payload support of each row sum at one, and its noise is simulated
    /// once per loop body, so a `nu = 1024` layout (34 buckets of about 90
    /// cells) lowers at once. Without the selector contract, every row sum
    /// multiplies the payload bound by the row width.
    #[test]
    fn multi_member_prf_noise_matches_one_member() {
        use crate::{
            ExpLayout,
            program::{ExponentLutProgramBuilder, LutOutputRepresentation, Operand},
            program_gadgets::{
                pbc::{PbcParameters, PbcRootSeed, generate_key_layout},
                prf::{PrfKeys, PrfOutput, SparseLwrPrf, SparseLwrPrfProfile},
            },
        };
        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(7);
        let support = rand::seq::index::sample(&mut rng, 1024, 31).into_vec();
        let root = PbcRootSeed(rand::Rng::random(&mut rng));
        // The members' rows have the layout's own width, its width limit: the same root seed
        // accepts the same layout under that limit.
        let unlimited = PbcParameters::paper_evaluation(1024, 31);
        let width =
            generate_key_layout(&unlimited, root, &support).unwrap().public_layout.bucket_width;
        let layout = generate_key_layout(
            &PbcParameters { bucket_width_limit: Some(width), ..unlimited },
            root,
            &support,
        )
        .unwrap()
        .public_layout;
        assert_eq!(layout.bucket_width, width);
        let profile =
            SparseLwrPrfProfile::new(ExpLayout { stride_log2: 0, width_log2: 7 }, 32, 1024)
                .unwrap();
        let noise = |members: usize, contract: bool| {
            let mut builder = ExponentLutProgramBuilder::new();
            let prf = SparseLwrPrf::build(
                &mut builder,
                profile,
                PrfKeys::new(&vec![layout.clone(); members]).unwrap(),
                LutOutputRepresentation::Coefficient,
            )
            .unwrap();
            let output = match prf.output() {
                PrfOutput::Scalar(wire) => Operand::Scalar(wire),
                PrfOutput::Family(family) => {
                    Operand::Member { family, index: IntExpr::constant(members - 1) }
                }
            };
            let program = builder.finish(vec![output]).unwrap();
            let levels = RnsBggDerivedWorkload::from_program(&program)
                .unwrap()
                .level(*program.outputs()[0].source())
                .unwrap();
            let pool = DCRTPolyParams::new(128, levels + 8, 40, 10, None, None);
            let parameters = RnsBggParameters::new(
                (0..levels + 2)
                    .map(|level| {
                        let depth = levels + 6 - level;
                        let modulus =
                            pool.to_crt().0[..depth].iter().map(|p| BigUint::from(*p)).product();
                        pool.select_modulus(&modulus).unwrap()
                    })
                    .collect(),
                bigdecimal::BigDecimal::from(4),
                1,
            )
            .unwrap();
            let mut bindings = child_bindings(&parameters, &program, &prf).unwrap();
            if !contract {
                for binding in bindings.rhs.values_mut() {
                    if let InputBinding::Family(family) = binding {
                        family.max_nonzero_payloads = family.validate_support().unwrap();
                        family.row_length = None;
                    }
                }
            }
            lower_child(&parameters, &program, &bindings, None).unwrap().remove(0)
        };
        let (one, many) = (noise(1, true), noise(3, true));
        let full_support = noise(3, false);
        assert!(
            full_support.bound.error > &many.bound.error << 100,
            "{} vs {}",
            full_support.bound.error.bits(),
            many.bound.error.bits()
        );
        assert_eq!(one.bound.error, many.bound.error);
        assert_eq!(one.bound.payload_l1, many.bound.payload_l1);
        let variance = |value: &RnsBggNoiseValue| {
            let variance = &value.moments.as_ref().unwrap().variance;
            num_traits::ToPrimitive::to_f64(&variance.numerator).unwrap() /
                num_traits::ToPrimitive::to_f64(&variance.denominator).unwrap()
        };
        let (one, many) = (variance(&one), variance(&many));
        assert!((one - many).abs() <= 1e-9 * one, "{one:e} vs {many:e}");
    }

    /// The quadratic through `(c_k, v_k)`, `k < 3`, at `c` (Lagrange form), in floating point
    /// (a parameter study only).
    fn quadratic(c: [f64; 3], v: [f64; 3], at: f64) -> f64 {
        (0..3)
            .map(|i| {
                (0..3).filter(|j| *j != i).fold(v[i], |term, j| term * (at - c[j]) / (c[i] - c[j]))
            })
            .sum::<f64>()
            .max(0.0)
    }

    fn to_f64(value: &BigUint) -> f64 {
        num_traits::ToPrimitive::to_f64(value).unwrap()
    }

    fn rational(value: f64) -> NonnegativeRational {
        let scale = BigUint::one() << 64u32;
        let numerator =
            num_traits::FromPrimitive::from_f64((value * 2f64.powi(64)).ceil()).unwrap();
        NonnegativeRational::new(numerator, scale).unwrap()
    }

    fn rational_f64(value: &NonnegativeRational) -> f64 {
        to_f64(&value.numerator) / to_f64(&value.denominator)
    }

    /// `values[k]` simulated at member scale `scales[k]`, extrapolated to `target` by the
    /// quadratic through the first three; the fourth checks the model.
    fn extrapolate(
        scales: [f64; 4],
        values: [&RnsBggNoiseValue; 4],
        target: f64,
    ) -> RnsBggNoiseValue {
        let at = |pick: &dyn Fn(&RnsBggNoiseValue) -> f64| {
            let c = [scales[0], scales[1], scales[2]];
            let v = [pick(values[0]), pick(values[1]), pick(values[2])];
            let fitted = quadratic(c, v, scales[3]);
            let actual = pick(values[3]);
            assert!(
                (fitted - actual).abs() <= 1e-6 * actual.abs().max(1.0),
                "not quadratic in the member count: fitted {fitted:e}, actual {actual:e}"
            );
            quadratic(c, v, target)
        };
        let mut value = values[1].clone();
        value.bound.error =
            num_traits::FromPrimitive::from_f64(at(&|value| to_f64(&value.bound.error)).ceil())
                .unwrap();
        if let Some(moments) = value.moments.as_mut() {
            let mean = |value: &RnsBggNoiseValue| {
                let moments = value.moments.as_ref().unwrap();
                num_traits::ToPrimitive::to_f64(&moments.mean_numerator).unwrap().abs() /
                    to_f64(&moments.mean_denominator)
            };
            let mean = at(&mean);
            moments.mean_numerator =
                num_traits::FromPrimitive::from_f64((mean * 2f64.powi(64)).ceil()).unwrap();
            moments.mean_denominator = BigUint::one() << 64u32;
            moments.mean_radius =
                rational(at(&|value| rational_f64(&value.moments.as_ref().unwrap().mean_radius)));
            moments.variance =
                rational(at(&|value| rational_f64(&value.moments.as_ref().unwrap().variance)));
        }
        value
    }

    /// Parameter study (CPU only): the smallest source CRT depth `L` at which a refresh of a
    /// fresh level-0 encoding fits, and the largest per-value hiding exponent it reaches, at
    /// the recorded `lwr_prf.rs` parameters (40-bit primes, base `2^10`, one prime per drop,
    /// `sigma = 4`, average case with failure exponent 100, `Q' = 256`, `P = 32`, `nu = 1024`,
    /// `h = 31`, CBD(1) fresh noise).
    ///
    /// Two approximations keep it tractable:
    ///
    /// - A declaration at `n = 2^16` holds `m n + L m n d` key layouts, so the children are
    ///   simulated with a profile of `c in {128, 256, 512, 1024}` coefficients (the same LUT width;
    ///   only the stride differs) and every member-count dependent value is extrapolated to `c = n`
    ///   by the quadratic through three sizes, the fourth checking it. All members share one
    ///   layout.
    /// - The noise model of a multi-member PRF at `nu = 1024` (34 buckets of about 90 cells) is
    ///   quadratic in the cells and does not fit in memory, so the chain depths come from the real
    ///   layout while the noise is simulated with a proxy layout (`nu = 64`, `h = 4`) on the
    ///   proxy's own chain. The children end in LUTs (the sampler's, the smudge digits'), whose
    ///   output noise depends on the LUT level's moduli, not on the PRF before it.
    #[test]
    #[ignore = "parameter study; run with --ignored"]
    fn refresh_parameter_study() {
        use crate::{
            ExpLayout,
            decoder::{
                decoding::{DecoderSmudgeProgram, DecoderTarget},
                simulation::AverageCaseConfig,
            },
            program_gadgets::{
                pbc::{PbcParameters, PbcPublicLayout, PbcRootSeed, generate_key_layout},
                prf::{PrfKeyShape, SparseLwrPrfProfile},
            },
            refresh::FreshNoiseProgram,
        };
        let env = |name: &str, default: usize| {
            std::env::var(name).map_or(default, |value| value.parse().unwrap())
        };
        let n_log2 = env("REFRESH_STUDY_N_LOG2", 16) as u32;
        let n = 1usize << n_log2;
        let average = AverageCaseConfig {
            failure_exponent: 100,
            allow_average_acceptance: true,
            ..AverageCaseConfig::default()
        };
        let sigma = bigdecimal::BigDecimal::from(4);
        let mut rng =
            <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(10187170744214484787);
        // The width limits leave the bucket loads (`3 nu / (h + 3)` on average) a wide margin.
        let mut layout_of = |nu: usize, h: usize, width: usize| {
            let support = rand::seq::index::sample(&mut rng, nu, h).into_vec();
            generate_key_layout(
                &PbcParameters {
                    bucket_width_limit: Some(width),
                    ..PbcParameters::paper_evaluation(nu, h)
                },
                PbcRootSeed(rand::Rng::random(&mut rng)),
                &support,
            )
            .unwrap()
            .public_layout
        };
        let (real, proxy) = (layout_of(1024, 31, 160), layout_of(64, 4, 48));
        let shape = |layout: &PbcPublicLayout, members: usize| {
            PrfKeyShape::new(&vec![layout.clone(); members]).unwrap()
        };
        let profile = |stride_log2: u32, nu: usize| {
            SparseLwrPrfProfile::new(ExpLayout { stride_log2, width_log2: 7 }, 32, nu).unwrap()
        };
        let strides = [0u32, 1, 2, 3];
        let scales = strides.map(|stride| (1u64 << (stride + 7)) as f64);
        let fresh_program = |stride: u32, layout: &PbcPublicLayout, nu: usize| {
            let c = 1usize << (stride + 7);
            FreshNoiseProgram::new(profile(stride, nu), 1, 1, c, shape(layout, c)).unwrap()
        };
        let fresh_level = |program: &FreshNoiseProgram| {
            RnsBggDerivedWorkload::from_program(program.program())
                .unwrap()
                .level(program.output())
                .unwrap()
        };
        let real_fresh_level = fresh_level(&fresh_program(0, &real, 1024));
        let proxy_fresh = strides.map(|stride| fresh_program(stride, &proxy, 64));
        let (source, input) = identity_source();
        let smudge_decoder = |stride: u32, layout: &PbcPublicLayout, nu: usize, digits: usize| {
            let c = 1usize << (stride + 7);
            let program =
                DecoderSmudgeProgram::new(&profile(stride, nu), shape(layout, c * digits), 1)
                    .unwrap();
            DecoderDeclaration::from_smudge(
                &source,
                input,
                profile(stride, nu),
                (1, c, digits),
                program,
                DecoderTarget::Output,
            )
            .unwrap()
        };
        let one = [num_bigint::BigInt::from(1u8)];
        // `REFRESH_STUDY_TOP=1` studies one part of a refresh to the top level
        // (see `crate::refresh::top`): the decoding level of `L` primes keeps
        // the output prime and `L - 1` margin primes, the one encoding carries
        // the mask child's binary polynomial, one prime is decoded over the
        // reference's `w_0` columns, and the source evaluates
        // `REFRESH_STUDY_SOURCE_LEVELS` levels (default: the fresh child's
        // PRF depth) from `Q_0`, so `|Q_0| = L + source levels`.
        let top = env("REFRESH_STUDY_TOP", 0) == 1;
        let source_levels = env("REFRESH_STUDY_SOURCE_LEVELS", real_fresh_level + 1);
        let mask_program = |stride: u32, q: u64| {
            let c = 1usize << (stride + 7);
            FreshNoiseProgram::with_distribution(
                profile(stride, 64),
                crate::program_gadgets::prf_sampler::PrfDistribution::Binary { q_min: q },
                1,
                c,
                shape(&proxy, c),
            )
            .unwrap()
        };
        let min_depth = env("REFRESH_STUDY_MIN_DEPTH", 2);
        let max_depth = env("REFRESH_STUDY_MAX_DEPTH", 8);
        let max_digits = env("REFRESH_STUDY_MAX_DIGITS", 40);
        let real_smudge_levels = (1..=max_digits)
            .map(|digits| {
                RnsBggParameters::decoder_smudge_level(&smudge_decoder(0, &real, 1024, digits))
                    .unwrap()
            })
            .collect::<Vec<_>>();
        eprintln!(
            "study: real fresh level {real_fresh_level}, real smudge levels {real_smudge_levels:?}"
        );
        for depth in min_depth..=max_depth {
            let started = std::time::Instant::now();
            let pool = DCRTPolyParams::new(n as u32, depth + 64, 40, 10, None, None);
            let prefix = |count: usize| {
                let modulus =
                    pool.to_crt().0[..count].iter().map(|prime| BigUint::from(*prime)).product();
                pool.select_modulus(&modulus).unwrap()
            };
            let parameters =
                RnsBggParameters::new(vec![prefix(depth), prefix(depth - 1)], sigma.clone(), 1)
                    .unwrap();
            let level = 0;
            let columns = parameters.level(level).unwrap().modulus_digits();
            let bindings = RnsBggNoiseBindings::from_inputs(
                &source,
                &[RnsBggNoiseValue::sampled_encoding(&parameters, b"refresh-target", &one)],
                &[],
            )
            .unwrap();
            let workload = RnsBggDerivedWorkload::from_program(&source).unwrap();
            let target = crate::api::noise_simulate(
                &source,
                RnsNoiseBuild::new(&parameters, &source, &workload, &bindings).unwrap(),
                |_, outputs| Ok(outputs),
            )
            .unwrap()
            .outputs
            .remove(0);
            let chain_depth = |child_level: usize| {
                parameters
                    .child_chain(level, child_level, &pool, 1)
                    .unwrap()
                    .level(0)
                    .unwrap()
                    .to_crt()
                    .0
                    .len()
            };
            // Fresh carriers on the proxy chain.
            let mut fresh_values = Vec::new();
            let mut fresh_chain = None;
            for program in &proxy_fresh {
                let child_level = fresh_level(program);
                let chain = parameters.child_chain(level, child_level, &pool, 1).unwrap();
                let child =
                    child_bindings(&chain, program.program(), program.sampler().prf()).unwrap();
                chain.audit_evaluation(program.program(), &child, Some(&average)).unwrap();
                let value = lower_child(&chain, program.program(), &child, None).unwrap().remove(0);
                fresh_values.push(value);
                fresh_chain = Some((chain, child_level));
            }
            let (fresh, proxy_fresh_level) = fresh_chain.unwrap();
            let fresh_value = extrapolate(
                scales,
                [&fresh_values[0], &fresh_values[1], &fresh_values[2], &fresh_values[3]],
                n as f64,
            );
            // The mask carriers of a top refresh on the proxy chain.
            let mask = top.then(|| {
                let q = parameters.level(level).unwrap().to_crt().0[0];
                let mut values = Vec::new();
                let mut chain = None;
                for stride in strides {
                    let program = mask_program(stride, q);
                    let child_level = fresh_level(&program);
                    let child_chain = parameters.child_chain(level, child_level, &pool, 1).unwrap();
                    let child =
                        child_bindings(&child_chain, program.program(), program.sampler().prf())
                            .unwrap();
                    child_chain
                        .audit_evaluation(program.program(), &child, Some(&average))
                        .unwrap();
                    values.push(
                        lower_child(&child_chain, program.program(), &child, None)
                            .unwrap()
                            .remove(0),
                    );
                    chain = Some((child_chain, child_level));
                }
                let (chain, child_level) = chain.unwrap();
                let value =
                    extrapolate(scales, [&values[0], &values[1], &values[2], &values[3]], n as f64);
                (chain, child_level, value)
            });
            let top_primes = depth + source_levels;
            let output_columns = if top { (top_primes - 1) * 4 } else { columns };
            let decoded = if top { vec![0] } else { (0..depth).collect::<Vec<_>>() };
            let p_bits = fresh.scale(proxy_fresh_level).unwrap().bits();
            let fresh_rounding =
                fresh.lut_mod_down_bound(proxy_fresh_level, &BigUint::zero()).unwrap().bits();
            eprintln!(
                "study: L={depth} fresh raw error {} bits over p {p_bits} bits; rounding-only \
                 ModDown bound {fresh_rounding} bits; full {} bits",
                fresh_value.bound.error.bits(),
                fresh
                    .lut_mod_down_bound(proxy_fresh_level, &fresh_value.bound.error)
                    .unwrap()
                    .bits()
            );
            let mut best: Option<(usize, i64, RnsBggRefreshNoise)> = None;
            for digits in env("REFRESH_STUDY_MIN_DIGITS", 1)..=max_digits {
                let values = strides.map(|stride| {
                    let decoder = smudge_decoder(stride, &proxy, 64, digits);
                    // The proxy has fewer coefficients than the ring: the chain of
                    // `decoder_smudge_parameters` without its coefficient check.
                    let smudge_level = RnsBggParameters::decoder_smudge_level(&decoder).unwrap();
                    let chain = parameters.child_chain(level, smudge_level, &pool, 1).unwrap();
                    let program = decoder.smudge().program();
                    let child = child_bindings(&chain, program, decoder.smudge().prf()).unwrap();
                    chain.audit_evaluation(program, &child, Some(&average)).unwrap();
                    let value =
                        lower_child(&chain, program, &child, Some(&decoder)).unwrap().remove(0);
                    (value, decoder.smudge_bound())
                });
                let smudge = extrapolate(
                    scales,
                    [&values[0].0, &values[1].0, &values[2].0, &values[3].0],
                    n as f64,
                );
                let smudges = vec![&smudge; decoded.len()];
                let noise = compose_refresh_noise(
                    &parameters,
                    level,
                    &target,
                    mask.as_ref().map(|(chain, level, value)| (chain, *level, value)),
                    &fresh,
                    proxy_fresh_level,
                    (&fresh_value, output_columns),
                    &smudges,
                    &decoded,
                    ([0; 32], values[0].1.clone(), 1),
                    4.578,
                    Some(&average),
                )
                .unwrap();
                if !noise.fits() {
                    break;
                }
                let error = noise.accepted_errors().iter().max().unwrap().clone();
                let bits = (&noise.smudge * 2u8).bits() as i64 - error.bits() as i64 - 1;
                best = Some((digits, bits, noise));
            }
            let events = ((decoded.len() * output_columns * n) as f64).log2().ceil() as i64;
            if top && let Some((digits, bits, noise)) = &best {
                // One chain per prime of `Q_0`; per chain two parts, each with
                // `w_0` smudges of `n d` members, `w_0 n` CBD and `n` mask members.
                let chains = top_primes;
                let members = 2 * (output_columns * n * digits + output_columns * n + n);
                let child_depth =
                    chain_depth(real_fresh_level).max(chain_depth(real_smudge_levels[digits - 1]));
                eprintln!(
                    "TOP n=2^{n_log2} decoding_primes={depth} margin={} |Q_0|={top_primes} \
                     w_0={output_columns} digits={digits} hiding_bits={bits} \
                     average_bits={:?} spacing_bits={:?} chains={chains} \
                     prf_members_per_chain={members} child_top_primes={child_depth} \
                     largest_modulus_primes={}",
                    depth - 1,
                    noise.accepted_errors().iter().map(BigUint::bits).collect::<Vec<_>>(),
                    noise.spacings.iter().map(BigUint::bits).collect::<Vec<_>>(),
                    top_primes.max(child_depth)
                );
                // `REFRESH_STUDY_ESTIMATE=1`: the output gate (binary secret,
                // CBD(1) error, `2 w_0 n` samples over `Q_0`) and the
                // encoding gate on the largest modulus.
                if env("REFRESH_STUDY_ESTIMATE", 0) == 1 {
                    let modulus = |primes: usize| {
                        pool.to_crt().0[..primes]
                            .iter()
                            .map(|prime| BigUint::from(*prime))
                            .product()
                    };
                    let output = mxx_backends::lattice_estimator::binary_secret_cbd_security_bits(
                        n as u32,
                        &modulus(top_primes),
                        1,
                        (2 * output_columns * n) as u64,
                    );
                    let level = pool.select_modulus(&modulus(top_primes.max(child_depth))).unwrap();
                    let encoding = mxx_backends::lattice_estimator::lattice_security_bits(
                        level.ring_dimension(),
                        level.modulus().as_ref(),
                        &sigma,
                    );
                    eprintln!(
                        "TOP-SECURITY n=2^{n_log2} |Q_0|={top_primes} output={output:?} \
                         encoding={encoding:?}"
                    );
                }
            }
            match best {
                Some((digits, bits, noise)) => eprintln!(
                    "STUDY n=2^{n_log2} L={depth} m={columns} decoders={} digits={digits} \
                     hiding_bits={bits} (100 + log2 count = {}) worst_bits={:?} \
                     average_bits={:?} spacing_bits={:?} smudge_bits={} \
                     fresh_chain_depth={} smudge_chain_depth={} elapsed={:?}",
                    depth * columns,
                    100 + events,
                    noise.errors.iter().map(BigUint::bits).collect::<Vec<_>>(),
                    noise.accepted_errors().iter().map(BigUint::bits).collect::<Vec<_>>(),
                    noise.spacings.iter().map(BigUint::bits).collect::<Vec<_>>(),
                    noise.smudge.bits(),
                    chain_depth(real_fresh_level),
                    chain_depth(real_smudge_levels[digits - 1]),
                    started.elapsed()
                ),
                None => eprintln!(
                    "STUDY n=2^{n_log2} L={depth}: no digit count fits; elapsed={:?}",
                    started.elapsed()
                ),
            }
        }
    }
}
