//! Noise of the scheme: the evaluation bound `E_F` from the exponent-lut
//! simulator, and the decryption bound and margin.
//!
//! The simulator evaluates the compiled predicate on the class chain with
//! the gadget output use, so `E_F` bounds the error of the full-gadget
//! carrier `c_F = s (A_F - F(x) G) + e_F` and no direct-decoding action or
//! margin of another consumer is applied. Its material contracts are the
//! setup's: `chi` for the initial state, every companion, and the helpers,
//! and the supplied attribute ciphertexts' bound `B_C` on their fresh error
//! `e_C = e_t`, a `chi` sample.
//!
//! Decryption contracts the family row with the key by blocks. With the key
//! cutoff `B_key`, the anchor blocks contribute `n B_key m_B B_b` each
//! (`W` of them), the helper blocks `n B_key m B_h` each (`W |P|`, for the
//! `|P|` helper automorphisms: `W - 1` without a LUT radix, `|J_B|` with one), and the
//! output block `n B_key m E_F`; with the message error `B_m`,
//!
//! `E_PE <= B_m + n B_key (W m_B B_b + W |P| m B_h + m E_F)`.
//!
//! With compact encodings (the manuscript's Appendix C), the companions are
//! read through their decompression, the anchor `b = round(b_Q / p)` has the
//! error `B_b = ceil(B_chi / p) + ceil((n + 2) / 2)` (Lemma 12 with `j = 0`),
//! and the blocks of `B''` add `W m_c D n B_key B''` for the error
//! `B'' = ceil(B_chi q / p) + ceil((n + 2) / 2)` of `b'' = Dc(b_c)`
//! ([`mxx_exponent_lut::rns_bgg::compact_expansion_bound`]); the block `I_W`
//! contracts zero.
//!
//! The decoder rounds each coefficient of `floor(q/2) m + e` to the nearer of
//! `0` and `floor(q/2)`, i.e. `round(2 x / q) mod 2`; for `q` odd it returns
//! `m` for both symbols whenever `4 |e| + 2 < q` ([`PeNoiseReport::decrypts`]).
//!
//! # Average case
//!
//! [`simulate_average`] keeps every bound above except the extension blocks
//! of the key, whose samples it treats probabilistically. For one coefficient
//! of the phase, the error is
//!
//! `e_m - sum_i sigma_i(e_b) k_i - sum_(i,k) sigma_i(e_(h,k)) v_(i,k) - e_F v_F`.
//!
//! - `e_m` and the anchor blocks enter by their worst-case bounds `B_m` and `W m_B n B_b B_key`:
//!   the preimages `k_i` are bounded by their cutoff only.
//! - The rest is a linear form in the extension coordinates `v`, which key generation samples from
//!   a discrete Gaussian of standard deviation `s_key`, independently of encryption. Its
//!   coefficients are the coefficients of the errors `e_(h,k)` and `e_F` (and of `e''` with compact
//!   encodings), so its energy is at most `W |P| m n B_h^2 + m n E_F^2 (+ W m_c D n B''^2)` on
//!   every run. A linear form in fresh sub-Gaussian coordinates of proxy `s_key^2` with
//!   coefficients of energy at most `E` exceeds `t` in absolute value with probability at most `2
//!   exp(-t^2 / (2 s_key^2 E))`.
//!
//! Decryption fails with probability at most `2^-k` over the `n` coefficients when
//! `B_m + W m_B n B_b B_key + z s_key sqrt(E) < (q - 2) / 4`, with `z^2 = 2 ln 2 (k + ceil(log2 n)
//! + 1)` ([`AverageCaseConfig::z_squared_for_events`]). The argument needs no assumption on the
//! distribution of the evaluation error: `E_F` is its worst-case bound.

use crate::{BranchingProgram, PeClass, PeError, PeParameters};
use mxx_backends::poly::PolyParams;
use mxx_exponent_lut::{
    api,
    decoder::simulation::AverageCaseConfig,
    noise_utils::{NoiseMoments, NonnegativeRational},
    rns_bgg::{
        RnsBggDerivedWorkload, RnsBggDynamicDistributions, RnsBggDynamicNoise, RnsBggNoiseBindings,
        RnsBggNoiseRhs, RnsBggNoiseTrace, RnsBggNoiseValue, RnsBggParameters, RnsNoiseBuild,
        RnsOutputUse,
    },
};
use num_bigint::{BigInt, BigUint};
use std::collections::BTreeMap;

/// The simulated noise of one predicate under one parameter set.
#[derive(Clone, Debug)]
pub struct PeNoiseReport {
    /// `E_F`: the coefficient bound of the evaluated output's error.
    pub evaluation: BigUint,
    /// `E_PE`: the coefficient bound of the decrypted phase's error.
    pub decryption: BigUint,
    /// The ring modulus `q`.
    pub modulus: BigUint,
    /// The exponent-lut simulator's trace of the evaluation.
    pub trace: RnsBggNoiseTrace,
    /// For [`simulate_average`]: the failure exponent `k` of the target
    /// `2^-k`, the tail multiplier `z^2`, and one phase coefficient's error as
    /// a bias (its worst-case terms) and a variance proxy (its linear form in
    /// the extension samples).
    pub average: Option<(u32, NonnegativeRational, NoiseMoments)>,
}

impl PeNoiseReport {
    /// Whether every coefficient of `floor(q/2) m + e` rounds to `m`: in the
    /// worst case `4 E_PE + 2 < q`; in the average case `|bias| + z
    /// sqrt(Var) < (q - 2) / 4` for the bias and variance of
    /// [`simulate_average`].
    pub fn decrypts(&self) -> bool {
        match &self.average {
            None => &self.decryption * 4u8 + 2u8 < self.modulus,
            Some((_, z_squared, moments)) => {
                self.modulus > BigUint::from(2u8) && {
                    let limit = NonnegativeRational::new(&self.modulus - 2u8, 4u8.into())
                        .expect("positive denominator");
                    moments.fits(&limit, z_squared)
                }
            }
        }
    }
}

/// `sigma^2` as an exact rational.
fn squared(sigma: &bigdecimal::BigDecimal) -> NonnegativeRational {
    let (integer, exponent) = sigma.as_bigint_and_exponent();
    let magnitude = integer.magnitude().pow(2u32);
    let power = BigUint::from(10u8).pow(2 * exponent.unsigned_abs() as u32);
    if exponent >= 0 {
        NonnegativeRational::new(magnitude, power)
    } else {
        NonnegativeRational::new(magnitude * power, 1u8.into())
    }
    .expect("positive denominator")
}

/// The material contracts of `program`'s packages over `rns` for `class`, in the source namespace
/// `namespace` (a setup identity): `distributions` for the initial state, the companions, the
/// helpers, and the attribute ciphertexts.
fn package_contracts(
    rns: &RnsBggParameters,
    class: &PeClass,
    program: &crate::CompiledPredicate,
    namespace: [u8; 32],
    distributions: RnsBggDynamicDistributions,
    lut_radix: Option<usize>,
    compact_extension: Option<&[u64]>,
) -> Result<RnsBggDynamicNoise, PeError> {
    let automorphisms = class
        .layout()
        .automorphism_indices(rns.levels()[0].ring_dimension() as usize)
        .map_err(|error| PeError::Parameters(error.to_string()))?;
    let rhs = program
        .program()
        .rhs_inputs()
        .keys()
        .map(|id| Ok((*id, program.binding().rhs(*id)?)))
        .collect::<Result<BTreeMap<_, _>, PeError>>()?;
    let depths = mxx_exponent_lut::rns_bgg::radix_depths(
        &automorphisms,
        rns.levels()[0].ring_dimension() as usize,
        lut_radix,
    );
    Ok(RnsBggDynamicNoise {
        namespace,
        distributions,
        rhs,
        automorphisms,
        depths,
        compact_extension: compact_extension.map(<[u64]>::to_vec),
    })
}

/// The exponent-lut simulator's trace of `program` over `rns` under the given material contracts.
fn evaluation_trace(
    rns: &RnsBggParameters,
    class: &PeClass,
    program: &BranchingProgram,
    namespace: [u8; 32],
    distributions: RnsBggDynamicDistributions,
    lut_radix: Option<usize>,
    compact_extension: Option<&[u64]>,
) -> Result<RnsBggNoiseTrace, PeError> {
    let compiled = program.class().compile()?;
    let initial = RnsBggNoiseValue::sampled_gaussian(
        &distributions.input,
        b"mxx-pe/initial-state",
        &[BigInt::from(1u8)],
    );
    let symbol = RnsBggNoiseRhs { payload_l1: 1u8.into(), payload_l2_squared: 1u8.into() };
    let symbols = vec![symbol; compiled.program().rhs_inputs().len()];
    let bindings = RnsBggNoiseBindings::from_inputs(compiled.program(), &[initial], &symbols)?
        .with_output_use(RnsOutputUse::Gadget)
        .with_dynamic(package_contracts(
            rns,
            class,
            &compiled,
            namespace,
            distributions,
            lut_radix,
            compact_extension,
        )?);
    let workload = RnsBggDerivedWorkload::from_program_with_parameters(rns, compiled.program())?;
    Ok(api::noise_simulate(
        compiled.program(),
        RnsNoiseBuild::new(rns, compiled.program(), &workload, &bindings)?,
        |_, outputs| Ok(outputs),
    )?)
}

fn output_bound(trace: &RnsBggNoiseTrace) -> Result<BigUint, PeError> {
    Ok(trace
        .outputs
        .first()
        .ok_or_else(|| PeError::Parameters("the simulator returned no output".into()))?
        .bound
        .error
        .clone())
}

/// The coefficient bound of the anchor's error as decryption reads it: the cutoff of `chi`, or,
/// with compact encodings, that of `round(b_Q / p)`, `ceil(B_chi / p) + ceil((n + 2) / 2)`.
pub(crate) fn anchor_error(parameters: &PeParameters) -> Result<BigUint, PeError> {
    let cutoff = parameters.error().cutoff();
    if parameters.compact_extension().is_none() {
        return Ok(cutoff.clone());
    }
    let p = parameters.compact_modulus()?;
    let n = BigUint::from(parameters.level().ring_dimension());
    Ok((cutoff + &p - 1u8) / &p + (n + 3u8) / 2u8)
}

/// With compact encodings, the coefficient bound of the error of `b'' = Dc(b_c)`; zero without
/// them.
pub(crate) fn compressed_anchor_error(parameters: &PeParameters) -> Result<BigUint, PeError> {
    Ok(match parameters.compact_extension() {
        Some(extension) => mxx_exponent_lut::rns_bgg::compact_expansion_bound(
            parameters.rns(),
            extension,
            parameters.error().cutoff(),
        )?,
        None => BigUint::ZERO,
    })
}

/// Simulate `program` under `parameters` with the exponent-lut evaluator,
/// and bound its decryption in the worst case.
pub fn simulate(
    parameters: &PeParameters,
    program: &BranchingProgram,
) -> Result<PeNoiseReport, PeError> {
    let trace = evaluation_trace(
        parameters.rns(),
        parameters.class(),
        program,
        parameters.identity(),
        parameters.distributions(),
        parameters.lut_radix(),
        parameters.compact_extension(),
    )?;
    let evaluation = output_bound(&trace)?;
    let w = BigUint::from(parameters.class().lookup_width());
    let p = BigUint::from(parameters.helper_automorphism_count()?);
    let m = BigUint::from(parameters.state_columns());
    let m_b = BigUint::from(parameters.anchor_columns());
    let n = BigUint::from(parameters.level().ring_dimension());
    let error = parameters.error().cutoff();
    let helper = parameters.distributions().helper.cutoff().clone();
    let compressed =
        BigUint::from(parameters.compressed_anchor_width()) * compressed_anchor_error(parameters)?;
    let blocks = &w * &m_b * anchor_error(parameters)? +
        &w * &compressed +
        &w * &p * &m * &helper +
        &m * &evaluation;
    let decryption = error + n * parameters.key_cutoff() * blocks;
    Ok(PeNoiseReport {
        evaluation,
        decryption,
        modulus: parameters.modulus(),
        trace,
        average: None,
    })
}

/// [`simulate`], with the average-case decryption bound of one phase
/// coefficient (see the module documentation) for the failure target
/// `2^-failure_exponent`: the worst-case terms as the bias and the linear form
/// in the extension samples as the variance `s_key^2 E`.
pub fn simulate_average(
    parameters: &PeParameters,
    program: &BranchingProgram,
    failure_exponent: u32,
) -> Result<PeNoiseReport, PeError> {
    let mut report = simulate(parameters, program)?;
    let class = parameters.class();
    let w = BigUint::from(class.lookup_width());
    let p = BigUint::from(parameters.helper_automorphism_count()?);
    let m = BigUint::from(parameters.state_columns());
    let n = BigUint::from(parameters.level().ring_dimension());
    let error = parameters.error().cutoff();
    let helper = parameters.distributions().helper.cutoff().clone();
    let energy = &w * &p * &m * &n * helper.pow(2) +
        &m * &n * report.evaluation.pow(2) +
        &w * BigUint::from(parameters.compressed_anchor_width()) *
            &n *
            compressed_anchor_error(parameters)?.pow(2);
    let bias = error +
        &w * BigUint::from(parameters.anchor_columns()) *
            &n *
            anchor_error(parameters)? *
            parameters.key_cutoff();
    let moments = NoiseMoments {
        mean_numerator: 0.into(),
        mean_denominator: 1u8.into(),
        mean_radius: NonnegativeRational::new(bias, 1u8.into()).expect("positive denominator"),
        variance: squared(parameters.key_width()).scaled(&energy),
    };
    let config = AverageCaseConfig { failure_exponent, ..AverageCaseConfig::default() };
    let z_squared =
        config.z_squared_for_events(&n).map_err(|error| PeError::Parameters(error.to_string()))?;
    report.average = Some((failure_exponent, z_squared, moments));
    Ok(report)
}
