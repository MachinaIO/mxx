//! The simulated noise of every decoder of an evaluation (spec §6.7, §8) and the parameter check
//! it gives.
//!
//! Every step input is a set of rows whose error is CBD(1): the obfuscator's initial input and the
//! anchor hint sample `u - u'` for uniform bits, and every refreshed row carries exactly the
//! refresh's CBD(1) fresh noise. One simulation of a step's Program on inputs of that error
//! therefore bounds the decoders of every step, and the decoders keep the invariant when they
//! decode exactly.
//!
//! Part `(t, j)` of a refreshing step decodes, for every prime `q` of `Q_0`, at the decoding level
//! `L` (modulus `Q = Q_0`, `m` digits, `Q_q = Q / q`),
//!
//! `d = [g | omega] X - c_j Y + f Z + u`, unmasked by `b K_pre`,
//!
//! and its rounding returns `y_j mod q` when the error and the smudge stay below `Q_q / 2`. Every
//! evaluated row is the `Gamma` of a lowered Program output to its form at `L`
//! ([`RnsBggParameters::retarget_noise`]). Term by term:
//!
//! - `[g | omega] X`: `X = G^-1(Q_q A'_j)` has `2m` rows of polynomial digits of at most `B/2`, so
//!   `m n B/2` per block times the error of `g` and of `omega`;
//! - `c_j Y`: `Y` has the `m'` digits of level `L + 1` (`m' B'/2` in `l1`), times the target's
//!   error: the private, state or helper encoding's, the mask's for `ONEPUB`, and for a label the
//!   public product `[P_0 | P_1] G^-1(K_j) + alpha c_j` with the mask (`2 m n B/2` times the
//!   label's error plus `||alpha||_1` times the mask's);
//! - `f Z`: the exact `l1` of `G^-1(Q_q)` times the CBD row's error;
//! - `u`: the smudge row's error;
//! - `e_b K_pre`: `n 2 (w_0 + 2)` products of CBD(1) hint errors and preimage coefficients.
//!
//! The output decoder of the last step decodes `Q_q z + u` for the first prime `q`, with spacing
//! `Q_q`.
//!
//! Under the average-case model the digits of `X` and of `G^-1(K_j)` are uniform (decompositions
//! of hashed matrices), the constant actions scale coherently, the hint term is an independent
//! sample, and the evaluated terms are added with unknown covariance. One tail multiplier covers
//! every decoded coefficient of an evaluation.

use super::{
    Dio2Error, Dio2Parameters, StreamingFunction, anchor_columns, preimage_bound,
    refresh::carrier,
    step::{ChainKeys, ChainProgram, StepPlan, Target},
};
use mxx_backends::poly::PolyParams;
use mxx_exponent_lut::{
    api::noise_simulate,
    decoder::simulation::AverageCaseConfig,
    noise_utils::{NoiseCorrelation, NoiseMoments, NonnegativeRational},
    program::{InputBinding, Operand},
    rns_bgg::{
        RefreshSigmaNoise, RnsBggDerivedWorkload, RnsBggForm, RnsBggNoiseBindings, RnsBggNoiseRhs,
        RnsBggNoiseValue, RnsBggParameters, RnsBggRefreshNoise, RnsNoiseBuild,
    },
};
use num_bigint::{BigInt, BigUint};
use num_traits::{One, Zero};
use rayon::prelude::*;
use sha2::{Digest, Sha256};

/// The average-case model of an evaluation of `function`: failure probability `2^-failure_exponent`
/// for all of its decoded coefficients together.
pub fn average_model(failure_exponent: u32) -> AverageCaseConfig {
    AverageCaseConfig { failure_exponent, allow_average_acceptance: true, ..Default::default() }
}

/// The decoded coefficients of one evaluation of `function`: `2 |targets| |P| w_0 n` per
/// refreshing step, and the output decoder's `n`.
pub fn decoded_coefficients(function: &StreamingFunction, plan: &StepPlan) -> BigUint {
    let n = plan.ring_dimension;
    let refreshing = function.stream_length().saturating_sub(1);
    BigUint::from(refreshing * plan.smudges() * plan.columns * n + n)
}

/// Whether every decoder of every evaluation of `function` on `parameters` decodes exactly and
/// hides its noise by `statistical_bits` (`2 (u + F) < spacing` and `2 u > F 2^bits` for the
/// smudge bound `u = P^d / 2` and the accepted error `F`), under the average-case model
/// `average` when given (else worst case).
pub fn noise_accepted(
    parameters: &Dio2Parameters,
    function: &StreamingFunction,
    statistical_bits: usize,
    average: Option<&AverageCaseConfig>,
) -> Result<bool, Dio2Error> {
    Ok(simulate_noise(parameters, function, average)?
        .iter()
        .all(|noise| noise.fits() && noise.hides(statistical_bits)))
}

/// The simulated noise of every decoder of `function` on `parameters`: every part of every
/// refreshing step for every prime, and the output decoder, on every chunk.
pub fn simulate_noise(
    parameters: &Dio2Parameters,
    function: &StreamingFunction,
    average: Option<&AverageCaseConfig>,
) -> Result<Vec<RnsBggRefreshNoise>, Dio2Error> {
    parameters.validate()?;
    let plan = StepPlan::new(parameters, function)?;
    let z_squared = average
        .map(|config| {
            config
                .z_squared_for_events(&decoded_coefficients(function, &plan))
                .map_err(|_| Dio2Error::Parameters("invalid average-case event count"))
        })
        .transpose()?;
    let chunks = 1usize << function.eval_chunk_bits;
    let jobs = (0..function.stream_length())
        .flat_map(|step| (0..chunks).map(move |chunk| (step, chunk)))
        .collect::<Vec<_>>();
    let chain_parameters = parameters.chain(&parameters.root([0; 32])?)?;
    let noise = jobs
        .into_par_iter()
        .map(|(step, chunk)| {
            let x =
                (0..function.eval_chunk_bits).map(|bit| chunk >> bit & 1 == 1).collect::<Vec<_>>();
            StepNoise {
                parameters,
                chain_parameters: &chain_parameters,
                plan: &plan,
                average,
                z_squared: z_squared.as_ref(),
            }
            .simulate(function, step, &x)
        })
        .collect::<Result<Vec<_>, Dio2Error>>()?;
    Ok(noise.into_iter().flatten().collect())
}

/// The simulation of one step's Program.
struct StepNoise<'a> {
    parameters: &'a Dio2Parameters,
    chain_parameters: &'a RnsBggParameters,
    plan: &'a StepPlan,
    average: Option<&'a AverageCaseConfig>,
    z_squared: Option<&'a NonnegativeRational>,
}

/// An encoding's error at the decoding level.
type Term = RefreshSigmaNoise;

impl StepNoise<'_> {
    /// The noise of every decoder of step `step` on chunk `x`.
    fn simulate(
        &self,
        function: &StreamingFunction,
        step: usize,
        x: &[bool],
    ) -> Result<Vec<RnsBggRefreshNoise>, Dio2Error> {
        let plan = self.plan;
        let last = step + 1 == function.stream_length();
        let shape = |role: &str, members| {
            self.parameters
                .prf
                .declared_shape(0, role, members)
                .map_err(|_| Dio2Error::Parameters("the PRF keys do not fit"))
        };
        let program = if last {
            let keys = ChainKeys {
                sigma: None,
                cbd: None,
                smudge: shape("smudge", plan.output_smudge_members())?,
            };
            ChainProgram::last(self.parameters, function, plan, step, x, &keys)?
        } else {
            let keys = ChainKeys {
                sigma: Some(shape("sigma", plan.ring_dimension)?),
                cbd: Some(shape("cbd", plan.cbd_members())?),
                smudge: shape("smudge", plan.smudge_members())?,
            };
            ChainProgram::refreshing(self.parameters, function, plan, step, x, &keys)?
        };
        let outputs = self.lower(&program)?;
        let workload = RnsBggDerivedWorkload::from_program(&program.program)?;
        let level = self.parameters.decoding_level();
        let level_parameters = self.chain_parameters.level(level)?;
        let modulus: BigUint = level_parameters.modulus().as_ref().clone();
        // `Gamma` of output `output` to the form of integer entries `weights` at `L`.
        let retarget = |output: usize, weights: &[BigUint]| -> Result<Term, Dio2Error> {
            let value = &outputs[output];
            Ok(self.chain_parameters.retarget_noise(
                carrier(&program.program, &workload, output)?,
                (level, weights),
                &RefreshSigmaNoise {
                    error: value.bound.error.clone(),
                    moments: value.moments.clone(),
                },
            )?)
        };
        let half = [(&modulus + 1u8) / 2u8];
        let state = self.chain_parameters.form_weights(level, RnsBggForm::State)?;
        let public = self.chain_parameters.form_weights(level, RnsBggForm::Public)?;
        let outs = &program.outputs;
        let identity = |target: usize, component: usize, prime: usize| {
            let mut digest = Sha256::new();
            digest.update(b"mxx-io/dio2/decoder/v1");
            for value in [
                step,
                x.iter().fold(0, |acc, bit| 2 * acc + usize::from(*bit)),
                target,
                component,
                prime,
            ] {
                digest.update((value as u64).to_le_bytes());
            }
            <[u8; 32]>::from(digest.finalize())
        };
        if last {
            let cofactor = &modulus / self.parameters.primes[0];
            let terms = [retarget(outs.state, &[cofactor.clone()])?, retarget(outs.smudge, &half)?];
            return Ok(vec![self.decoder(identity(0, 0, 0), &terms, cofactor)?]);
        }
        let missing = || Dio2Error::Parameters("a refreshing step has every refresh output");
        let g = retarget(outs.sigma.ok_or_else(missing)?, &public)?;
        let omega = retarget(outs.public.ok_or_else(missing)?, &public)?;
        // `Gamma` of a public row at `L` to the state form at `L`.
        let to_state = |term: &Term| -> Result<Term, Dio2Error> {
            Ok(self.chain_parameters.retarget_noise(
                (level, RnsBggForm::Public),
                (level, &state),
                term,
            )?)
        };
        let mask = [to_state(&g)?, to_state(&omega)?];
        let n = BigUint::from(level_parameters.ring_dimension());
        let block = BigUint::from(level_parameters.modulus_digits()) * &n;
        let digit_variance = self.digit_variance(level)?;
        let next = self.chain_parameters.level(level + 1)?;
        let target_l1 =
            BigUint::from(next.modulus_digits()) * (BigUint::one() << (next.base_bits() - 1));
        let half_base = BigUint::one() << (level_parameters.base_bits() - 1);
        let one = |term: &Term| {
            scaled(term, &(&block * &half_base), |moments| {
                moments.uniform_digit_action(&block, &digit_variance)
            })
        };
        let mut noises = Vec::with_capacity(plan.smudges());
        for (index, target) in plan.targets.iter().enumerate() {
            for component in 0..2 {
                let target_error = match target {
                    Target::Private => retarget(outs.private.ok_or_else(missing)?, &state)?,
                    Target::State => retarget(outs.state, &state)?,
                    Target::Public => mask[component].clone(),
                    Target::Helper(1) if component == 0 => mask[0].clone(),
                    Target::Helper(k) if component == 0 => to_state(&retarget(
                        *outs.automorphed.get(k).ok_or_else(missing)?,
                        &public,
                    )?)?,
                    Target::Helper(_) => mask[1].clone(),
                    Target::Label { row, column } => {
                        let output = outs.labels[row * (plan.label_width + 1) + column];
                        let label = retarget(output, &public)?;
                        let alpha = &outputs[output].bound.payload_l1;
                        let digits = &block * 2u8;
                        sum(
                            &[
                                scaled(&label, &(&digits * &half_base), |moments| {
                                    moments.uniform_digit_action(&digits, &digit_variance)
                                }),
                                scaled(&mask[component], alpha, |moments| moments.scaled(alpha)),
                            ],
                            NoiseCorrelation::Unknown,
                        )
                    }
                    Target::Anchor(_) => zero(),
                };
                let target_term =
                    scaled(&target_error, &target_l1, |moments| moments.scaled(&target_l1));
                let fresh = retarget(outs.cbd.ok_or_else(missing)?, &public)?;
                for (prime, q) in self.parameters.primes.iter().enumerate() {
                    let cofactor = &modulus / *q;
                    let (z_l1, _) = self.chain_parameters.gadget_inverse_norms(level, &cofactor)?;
                    let terms = [
                        one(&g),
                        one(&omega),
                        target_term.clone(),
                        scaled(&fresh, &z_l1, |moments| moments.scaled(&z_l1)),
                        retarget(outs.smudge, &half)?,
                    ];
                    noises.push(self.decoder(
                        identity(index, component, prime),
                        &terms,
                        cofactor,
                    )?);
                }
            }
        }
        Ok(noises)
    }

    /// The decoder whose evaluated terms are `terms`, with the hint term added, at spacing
    /// `spacing`.
    fn decoder(
        &self,
        refresh: [u8; 32],
        terms: &[Term],
        spacing: BigUint,
    ) -> Result<RnsBggRefreshNoise, Dio2Error> {
        let n = BigUint::from(self.plan.ring_dimension);
        let hint_terms = n * BigUint::from(anchor_columns(self.plan.columns));
        let preimage = preimage_bound(self.parameters)?;
        let hint = RefreshSigmaNoise {
            error: &hint_terms * &preimage,
            moments: Some(zero_mean(
                NonnegativeRational::new(&hint_terms * preimage.pow(2u32), BigUint::from(2u8))
                    .expect("positive denominator"),
            )),
        };
        let evaluated = sum(terms, NoiseCorrelation::Unknown);
        let total = sum(&[evaluated, hint], NoiseCorrelation::Independent);
        let profile = self.parameters.profile()?;
        let smudge = BigUint::from(profile.p()).pow(self.parameters.smudge_digits as u32) / 2u8;
        let average_errors = match (self.z_squared, self.average) {
            (Some(z_squared), Some(_)) => Some(vec![
                total
                    .moments
                    .as_ref()
                    .ok_or(Dio2Error::Parameters("average-case moments are missing"))?
                    .tail_bound(z_squared),
            ]),
            _ => None,
        };
        Ok(RnsBggRefreshNoise {
            refresh,
            errors: vec![total.error],
            average_errors,
            spacings: vec![spacing],
            smudge,
            fresh: BigUint::one(),
        })
    }

    /// The variance of a uniform balanced digit of the gadget base at `level`.
    fn digit_variance(&self, level: usize) -> Result<NonnegativeRational, Dio2Error> {
        let base = BigUint::one() << self.chain_parameters.level(level)?.base_bits();
        Ok(NonnegativeRational::new(&base * &base + 2u8, BigUint::from(12u8))
            .expect("positive denominator"))
    }

    /// The simulated value of every output of `program` on step inputs of CBD(1) error: one
    /// sample per root object (the inputs reading one root object share its error), and every
    /// family member its own sample. With the average-case model, the Program's evaluation
    /// budget is audited too.
    fn lower(&self, program: &ChainProgram) -> Result<Vec<RnsBggNoiseValue>, Dio2Error> {
        let one = [BigInt::one()];
        let inputs = program
            .program
            .inputs()
            .iter()
            .map(|(wire, declaration)| {
                let tag = program
                    .root_inputs
                    .get(wire)
                    .ok_or(Dio2Error::Parameters("every Program input reads a root object"))?;
                let value = RnsBggNoiseValue::sampled_cbd(1, tag.as_bytes(), &one);
                Ok(match declaration.shape() {
                    InputBinding::Scalar(()) => value,
                    InputBinding::Family(_) => value.independent_family_members(),
                })
            })
            .collect::<Result<Vec<_>, Dio2Error>>()?;
        let rhs =
            vec![
                RnsBggNoiseRhs { payload_l1: BigUint::one(), payload_l2_squared: BigUint::one() };
                program.program.rhs_inputs().len()
            ];
        let selectors = std::iter::once(&program.smudge)
            .chain(program.sigma.iter().map(|sampler| sampler.prf()))
            .chain(program.cbd.iter().map(|sampler| sampler.prf()))
            .flat_map(|prf| prf.selector_families())
            .collect::<Vec<_>>();
        let bindings = RnsBggNoiseBindings::from_inputs(&program.program, &inputs, &rhs)?
            .with_selector_families(&selectors)?;
        if let Some(config) = self.average {
            self.chain_parameters.audit_evaluation(&program.program, &bindings, Some(config))?;
        }
        let workload = RnsBggDerivedWorkload::from_program(&program.program)?;
        let trace = noise_simulate(
            &program.program,
            RnsNoiseBuild::new(self.chain_parameters, &program.program, &workload, &bindings)?,
            |_, outputs| Ok(outputs),
        )?;
        if trace.outputs.len() != program.program.outputs().len() ||
            program
                .program
                .outputs()
                .iter()
                .any(|output| matches!(output, Operand::Selected { .. }))
        {
            return Err(Dio2Error::Parameters("a step outputs fixed wires"));
        }
        Ok(trace.outputs)
    }
}

/// Zero-mean moments of variance `variance`.
fn zero_mean(variance: NonnegativeRational) -> NoiseMoments {
    NoiseMoments {
        mean_numerator: BigInt::zero(),
        mean_denominator: BigUint::one(),
        mean_radius: NonnegativeRational::zero(),
        variance,
    }
}

/// A zero error.
fn zero() -> Term {
    RefreshSigmaNoise {
        error: BigUint::zero(),
        moments: Some(zero_mean(NonnegativeRational::zero())),
    }
}

/// `term` under an action of worst-case gain `gain` and moment map `moments`.
fn scaled(term: &Term, gain: &BigUint, moments: impl Fn(&NoiseMoments) -> NoiseMoments) -> Term {
    RefreshSigmaNoise { error: &term.error * gain, moments: term.moments.as_ref().map(moments) }
}

/// The sum of `terms` under `correlation`; its moments when every term has them.
fn sum(terms: &[Term], correlation: NoiseCorrelation) -> Term {
    let error = terms.iter().map(|term| &term.error).sum();
    let moments = terms.iter().try_fold(None::<NoiseMoments>, |total, term| {
        let moments = term.moments.as_ref()?;
        Some(Some(match total {
            None => moments.clone(),
            Some(total) => total.add_with_covariance(moments, correlation),
        }))
    });
    RefreshSigmaNoise { error, moments: moments.flatten() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dio2::test_support::toy;

    /// The toy parameters decode exactly in the worst case and under the average-case model.
    #[test]
    fn toy_parameters_decode_exactly() {
        let (parameters, function) = toy();
        let worst = simulate_noise(&parameters, &function, None).unwrap();
        let average = simulate_noise(&parameters, &function, Some(&average_model(40))).unwrap();
        let log = |noises: &[RnsBggRefreshNoise]| {
            noises
                .iter()
                .map(|noise| {
                    (
                        noise.accepted_errors()[0].bits(),
                        noise.spacings[0].bits(),
                        noise.smudge.bits(),
                    )
                })
                .max()
                .unwrap()
        };
        eprintln!("worst (error, spacing, smudge) bits: {:?}", log(&worst));
        eprintln!("average (error, spacing, smudge) bits: {:?}", log(&average));
        assert!(worst.iter().chain(&average).all(RnsBggRefreshNoise::fits));
    }
}
