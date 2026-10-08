//! Samples of a small distribution derived from the sparse-LWR PRF.
//!
//! [`PrfSampler`] evaluates [`SparseLwrPrf`] as a black box, its terminal LUT
//! returning the exponent `e = 2 f(k) - 1` of the half-integer readout in an
//! exponent layout of width `P`, and maps the PRF output with one sampling LUT
//! to a coefficient sample. For a uniform `z = <a, k_o> mod Q` the PRF output
//! is `(-1)^b X^(e)` with the sign bit `b` and the bin `j = (e - 1) / 2` of
//! `[0, P/2)` uniform and independent, and a LUT is linear in its monomial
//! input, so the sample is `(-1)^b g(j)` for the table `g` of the
//! distribution:
//!
//! - [`PrfDistribution::Cbd`]: the centered binomial distribution `CBD_eta`, exactly when `4^eta`
//!   divides `P`. `g` takes the value `0` on `P C(2 eta, eta) / (2 4^eta)` bins and `v > 0` on `P
//!   C(2 eta, eta + v) / 4^eta` bins, so `(-1)^b g(j)` is `v` with probability `C(2 eta, eta + v) /
//!   4^eta`.
//! - [`PrfDistribution::Binary`]: the uniform bit, as an element of `Z_(q_min)` for an odd modulus
//!   `q_min`. The sign-symmetric sample `(-1)^b h` with `h = 2^(-1) mod q_min = (q_min + 1) / 2` is
//!   shifted by the encoding of `h`, so the output `h (1 + (-1)^b)` is `0` or `2 h = 1 mod q_min`.
//!   The shift is derived inside the Program from the encoding of one, the PRF's accumulator start
//!   `mono(0)`: one private multiplication by an RHS bound to one per PRF bucket keeps it at the
//!   PRF's level, and the PRF's readout LUT and the sampling LUT read it at `z = 0` (`e = 1`, sign
//!   `+`), giving `h` at the level of the samples.
use crate::{
    api::PlaintextInputs,
    program::{
        ExpLayout, ExponentLutProgramBuilder, FamilyRange, InputBinding, LutId,
        LutOutputRepresentation, LutTable, Operand, ProgramValidationError, RhsId, WireId,
    },
    program_gadgets::{
        pbc::{PbcGeneratedKeyLayout, PbcPublicLayout},
        prf::{PrfError, PrfOutput, SparseLwrPrf, SparseLwrPrfProfile},
    },
};
use mxx_backends::poly::Poly;
use mxx_ir_core::IntExpr;

/// Errors in a PRF sampler declaration or its bindings.
#[derive(Debug, thiserror::Error, Eq, PartialEq)]
pub enum PrfSamplerError {
    #[error("the distribution is not exactly representable over the PRF output modulus")]
    InvalidDistribution,
    #[error(transparent)]
    Prf(#[from] PrfError),
    #[error(transparent)]
    Program(#[from] ProgramValidationError),
}

/// The distribution of a [`PrfSampler`] sample.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrfDistribution {
    /// The uniform bit in `Z_(q_min)` for an odd `q_min >= 3`.
    Binary { q_min: u64 },
    /// The centered binomial distribution `CBD_eta`, `eta >= 1`.
    Cbd { eta: u32 },
}

impl PrfDistribution {
    /// The table `g` on the `P/2` bins of the PRF output modulus `p`.
    fn bin_values(&self, p: usize) -> Result<Vec<usize>, PrfSamplerError> {
        let invalid = || PrfSamplerError::InvalidDistribution;
        match *self {
            Self::Binary { q_min } => {
                if q_min < 3 || q_min.is_multiple_of(2) {
                    return Err(invalid());
                }
                let half = usize::try_from(q_min.div_ceil(2)).map_err(|_| invalid())?;
                Ok(vec![half; p / 2])
            }
            Self::Cbd { eta } => {
                let eta = usize::try_from(eta).map_err(|_| invalid())?;
                let total = eta
                    .checked_mul(2)
                    .and_then(|bits| u32::try_from(bits).ok())
                    .and_then(|bits| 1usize.checked_shl(bits))
                    .ok_or_else(invalid)?;
                if eta == 0 || !p.is_multiple_of(total) {
                    return Err(invalid());
                }
                // `C(2 eta, eta + v)` for `v = 0..=eta`, over `4^eta` outcomes.
                let mut binomial = vec![1usize; eta + 1];
                for v in (0..eta).rev() {
                    binomial[v] = binomial[v + 1]
                        .checked_mul(eta + v + 1)
                        .map(|product| product / (eta - v))
                        .ok_or_else(invalid)?;
                }
                let scale = p / total;
                let mut values = vec![0; scale * binomial[0] / 2];
                for (value, count) in binomial.iter().enumerate().skip(1) {
                    values.extend(std::iter::repeat_n(value, scale * count));
                }
                debug_assert_eq!(values.len(), p / 2);
                Ok(values)
            }
        }
    }
}

/// A sample of a [`PrfDistribution`] per member of a [`SparseLwrPrf`].
///
/// The output has the PRF's shape: one coefficient scalar for one member,
/// else the coefficient family of the members. The plaintext inputs are those
/// of the PRF; the RHS inputs are the PRF's selectors and, for
/// [`PrfDistribution::Binary`], one RHS per PRF bucket bound to one.
#[derive(Clone, Debug)]
pub struct PrfSampler {
    prf: SparseLwrPrf,
    distribution: PrfDistribution,
    lut: LutId,
    /// The RHS inputs lifting the encoding of one to the PRF's level.
    offset_rhs: Vec<RhsId>,
    output: PrfOutput,
}

impl PrfSampler {
    /// Add the PRF of the members `layouts` (see [`SparseLwrPrf::build`])
    /// and its samples of `distribution` to `builder`.
    pub fn build(
        builder: &mut ExponentLutProgramBuilder,
        profile: SparseLwrPrfProfile,
        layouts: Vec<PbcPublicLayout>,
        distribution: PrfDistribution,
    ) -> Result<Self, PrfSamplerError> {
        profile.validate()?;
        let p = profile.p();
        let bins = distribution.bin_values(p)?;
        // The readout exponent `e < P` lies in an exponent layout of width
        // `P` spanning the ring, so `P` (a power of two dividing `Q`) is at
        // most the ring dimension.
        let width_log2 = p.trailing_zeros();
        let stride_log2 = profile
            .native_ring_log2()
            .checked_sub(width_log2)
            .ok_or(PrfSamplerError::InvalidDistribution)?;
        let readout_layout = ExpLayout { stride_log2, width_log2 };
        let prf = SparseLwrPrf::build(
            builder,
            profile,
            layouts,
            LutOutputRepresentation::Exponent(readout_layout),
        )?;
        // Odd `e = 2 j + 1` reads bin `j`; even exponents are never read.
        let mut values = vec![0; p];
        for (bin, value) in bins.iter().enumerate() {
            values[2 * bin + 1] = *value;
        }
        let output_width = bins.iter().copied().max().unwrap_or(0) + 1;
        let lut = builder.register_lut(LutTable::unary(
            p,
            readout_layout,
            output_width,
            values,
            LutOutputRepresentation::Coefficient,
        )?)?;
        let (offset_rhs, offset) = match distribution {
            PrfDistribution::Cbd { .. } => (Vec::new(), None),
            PrfDistribution::Binary { .. } => {
                let mut state = prf.initial_input();
                let offset_rhs = prf.layouts()[0]
                    .cells
                    .iter()
                    .map(|_| {
                        let rhs = builder.rhs_input()?;
                        state =
                            builder.mul_private(Operand::Scalar(state), Operand::Scalar(rhs))?;
                        Ok(rhs)
                    })
                    .collect::<Result<Vec<_>, ProgramValidationError>>()?;
                let readout = builder.lut(Operand::Scalar(state), prf.lut())?;
                (offset_rhs, Some(builder.lut(Operand::Scalar(readout), lut)?))
            }
        };
        let sample = |builder: &mut ExponentLutProgramBuilder, input| {
            let sample = builder.lut(input, lut)?;
            match offset {
                Some(offset) => builder.add(Operand::Scalar(sample), Operand::Scalar(offset)),
                None => Ok(sample),
            }
        };
        let output = match prf.output() {
            PrfOutput::Scalar(wire) => PrfOutput::Scalar(sample(builder, Operand::Scalar(wire))?),
            PrfOutput::Family(family) => {
                let range = FamilyRange::full(IntExpr::constant(prf.members()))?;
                PrfOutput::Family(builder.parallel(range, |builder, member| {
                    sample(builder, Operand::Member { family, index: member }).map(Operand::Scalar)
                })?)
            }
        };
        Ok(Self { prf, distribution, lut, offset_rhs, output })
    }

    /// The PRF the samples are read from.
    pub fn prf(&self) -> &SparseLwrPrf {
        &self.prf
    }

    pub const fn distribution(&self) -> PrfDistribution {
        self.distribution
    }

    /// The sampling LUT on the PRF's exponent readout.
    pub const fn lut(&self) -> LutId {
        self.lut
    }

    /// The RHS inputs bound to one, one per PRF bucket; empty unless the
    /// distribution is [`PrfDistribution::Binary`].
    pub fn offset_rhs(&self) -> &[RhsId] {
        &self.offset_rhs
    }

    pub const fn output(&self) -> PrfOutput {
        self.output
    }

    /// The RHS plaintexts: the PRF selectors of `keys` (see
    /// [`SparseLwrPrf::selector_plaintexts`]) and one for every offset RHS.
    pub fn rhs_plaintexts<P: Poly>(
        &self,
        parameters: &P::Params,
        keys: &[PbcGeneratedKeyLayout],
    ) -> Result<PlaintextInputs<RhsId, P>, PrfSamplerError> {
        let mut plaintexts = self.prf.selector_plaintexts(parameters, keys)?;
        plaintexts.extend(
            self.offset_rhs
                .iter()
                .map(|rhs| (*rhs, InputBinding::Scalar(P::const_one(parameters)))),
        );
        Ok(plaintexts)
    }

    /// The LHS plaintexts of the PRF on the public vector `a` (see
    /// [`SparseLwrPrf::plaintext_inputs`]).
    pub fn plaintext_inputs<P: Poly>(
        &self,
        parameters: &P::Params,
        a: &[usize],
    ) -> Result<PlaintextInputs<WireId, P>, PrfSamplerError> {
        Ok(self.prf.plaintext_inputs(parameters, a)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        program::ExponentLutProgram,
        program_gadgets::pbc::{PbcParameters, PbcRootSeed, generate_key_layout},
        rns_bgg::{
            RnsBggDerivedWorkload, RnsBggNoiseBindings, RnsBggNoiseFamily, RnsBggNoiseRhs,
            RnsBggNoiseValue, RnsBggParameters, RnsBggPlanRequest, RnsEncryptBuild,
            RnsOnlineEvalBuild, RnsPreprocessBuild, decode_output, verify_output,
        },
    };
    use mxx_backends::{
        artifact::MemoryArtifactStore,
        authority::CpuExecution,
        backend::poly::cpu_backend,
        poly::dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    };
    use mxx_ir_core::ParamEnv;
    use num_bigint::{BigInt, BigUint};

    /// n = 32, Q = 32, P = 16 over an LWR dimension of ten.
    fn profile() -> SparseLwrPrfProfile {
        SparseLwrPrfProfile::new(ExpLayout { stride_log2: 1, width_log2: 4 }, 16, 10).unwrap()
    }

    fn key(support: &[usize], seed_base: u8) -> PbcGeneratedKeyLayout {
        let parameters = PbcParameters::paper_evaluation(10, 4);
        (seed_base..seed_base.saturating_add(32))
            .find_map(|seed| {
                generate_key_layout(&parameters, PbcRootSeed([seed; 32]), support).ok()
            })
            .expect("toy key must find a PBC layout")
    }

    fn keys(members: usize) -> Vec<PbcGeneratedKeyLayout> {
        [(&[0, 3, 5, 9][..], 0), (&[1, 2, 6, 7][..], 100)][..members]
            .iter()
            .map(|(support, seed)| key(support, *seed))
            .collect()
    }

    fn sampler(
        keys: &[PbcGeneratedKeyLayout],
        distribution: PrfDistribution,
    ) -> Result<(ExponentLutProgram, PrfSampler), PrfSamplerError> {
        sampler_on(profile(), keys, distribution)
    }

    fn sampler_on(
        profile: SparseLwrPrfProfile,
        keys: &[PbcGeneratedKeyLayout],
        distribution: PrfDistribution,
    ) -> Result<(ExponentLutProgram, PrfSampler), PrfSamplerError> {
        let mut builder = ExponentLutProgramBuilder::new();
        let sampler = PrfSampler::build(
            &mut builder,
            profile,
            keys.iter().map(|key| key.public_layout.clone()).collect(),
            distribution,
        )?;
        let outputs = match sampler.output() {
            PrfOutput::Scalar(wire) => vec![Operand::Scalar(wire)],
            PrfOutput::Family(family) => (0..keys.len())
                .map(|index| Operand::Member { family, index: IntExpr::constant(index) })
                .collect(),
        };
        Ok((builder.finish(outputs)?, sampler))
    }

    /// The sample at every `z in Z_Q`, through the registered readout and
    /// sampling tables under the LUT contract `±X^(stride e) -> ±T[e]`.
    fn samples(program: &ExponentLutProgram, sampler: &PrfSampler) -> Vec<i64> {
        let inputs = program.encryption_inputs();
        let readout = inputs.luts()[sampler.prf().lut().0 as usize].values().unwrap();
        let table = inputs.luts()[sampler.lut().0 as usize].values().unwrap();
        let profile = sampler.prf().profile();
        let n_prime = profile.native_width();
        (0..profile.q().unwrap())
            .map(|z| {
                let value = table[readout[z % n_prime]] as i64;
                if z < n_prime { value } else { -value }
            })
            .collect()
    }

    fn binomial(n: u64, k: u64) -> u64 {
        (0..k).fold(1, |product, i| product * (n - i) / (i + 1))
    }

    #[test]
    fn cbd_samples_are_exactly_centered_binomial() {
        let q = profile().q().unwrap() as u64;
        for eta in [1_u32, 2] {
            let (program, sampler) = sampler(&keys(1), PrfDistribution::Cbd { eta }).unwrap();
            assert!(sampler.offset_rhs().is_empty());
            let samples = samples(&program, &sampler);
            let total = 1_u64 << (2 * eta);
            for v in -(eta as i64)..=eta as i64 {
                let count = samples.iter().filter(|sample| **sample == v).count() as u64;
                assert_eq!(
                    count * total,
                    q * binomial(2 * eta as u64, (eta as i64 + v) as u64),
                    "eta = {eta}, v = {v}"
                );
            }
            assert_eq!(samples.len() as u64, q);
        }
    }

    #[test]
    fn binary_samples_shifted_by_the_offset_are_uniform_bits_mod_q_min() {
        let q_min = 1_000_003_u64;
        let (program, sampler) = sampler(&keys(1), PrfDistribution::Binary { q_min }).unwrap();
        // The offset reads the PRF readout at `z = 0`.
        let offset = samples(&program, &sampler)[0];
        assert_eq!(offset as u64, q_min.div_ceil(2));
        let bits = samples(&program, &sampler)
            .into_iter()
            .map(|sample| (sample + offset).rem_euclid(q_min as i64))
            .collect::<Vec<_>>();
        assert!(bits.iter().all(|bit| *bit == 0 || *bit == 1));
        assert_eq!(bits.iter().sum::<i64>() * 2, bits.len() as i64);
    }

    /// The offset reaches the samples' level, so the Add is well formed for
    /// the scalar and the member-family shapes.
    #[test]
    fn binary_offset_is_lifted_to_the_sample_level() {
        let parameters = DCRTPolyParams::new(32, 1, 17, 4, None, None);
        for members in [1, 2] {
            let keys = keys(members);
            let (program, sampler) = sampler(&keys, PrfDistribution::Binary { q_min: 7 }).unwrap();
            let workload = RnsBggDerivedWorkload::from_program(&program).unwrap();
            let buckets = keys[0].public_layout.parameters.bucket_count;
            assert_eq!(sampler.offset_rhs().len(), buckets);
            for (level, rhs) in sampler.offset_rhs().iter().enumerate() {
                assert_eq!(workload.rhs_level(*rhs).unwrap(), level);
            }
            let plaintexts = sampler.rhs_plaintexts::<DCRTPoly>(&parameters, &keys).unwrap();
            let one = DCRTPoly::const_one(&parameters);
            for rhs in sampler.offset_rhs() {
                assert!(matches!(&plaintexts[rhs], InputBinding::Scalar(value) if *value == one));
            }
            assert_eq!(plaintexts.len(), program.rhs_inputs().len());
        }
    }

    #[test]
    fn rejects_distributions_not_exact_over_p() {
        let keys = keys(1);
        for distribution in [
            PrfDistribution::Cbd { eta: 0 },
            PrfDistribution::Cbd { eta: 3 },
            PrfDistribution::Binary { q_min: 1 },
            PrfDistribution::Binary { q_min: 8 },
        ] {
            assert_eq!(
                sampler(&keys, distribution).unwrap_err(),
                PrfSamplerError::InvalidDistribution,
                "{distribution:?}"
            );
        }
    }

    /// The noise contracts of the sampler's inputs: fresh encodings of
    /// `sigma` (every member of the public family a separate encryption), a
    /// one-hot or member-indexed selector family per PRF selector, and the
    /// unit offset RHS inputs.
    fn noise_bindings(
        program: &ExponentLutProgram,
        sampler: &PrfSampler,
        sigma: &bigdecimal::BigDecimal,
    ) -> RnsBggNoiseBindings {
        let inputs = program
            .encryption_inputs()
            .inputs()
            .iter()
            .map(|(wire, declaration)| {
                let value = RnsBggNoiseValue::sampled_encoding_sigma(
                    sigma,
                    format!("prf-sampler-input-{wire:?}").as_bytes(),
                    &[BigInt::from(1u8)],
                )
                .unwrap();
                let binding = match declaration.shape() {
                    InputBinding::Scalar(()) => InputBinding::Scalar(value),
                    InputBinding::Family(count) => InputBinding::Family(RnsBggNoiseFamily {
                        element: value.independent_family_members(),
                        range: FamilyRange::full(count.clone()).unwrap(),
                        max_nonzero_payloads: 1u8.into(),
                    }),
                };
                (*wire, binding)
            })
            .collect();
        let unit = RnsBggNoiseRhs {
            payload_l1: BigUint::from(1u8),
            payload_l2_squared: BigUint::from(1u8),
        };
        let rhs = sampler
            .prf()
            .selector_families()
            .into_iter()
            .map(|family| {
                (
                    family.rhs,
                    InputBinding::Family(RnsBggNoiseFamily {
                        element: unit.clone(),
                        range: FamilyRange::full(IntExpr::constant(family.members)).unwrap(),
                        max_nonzero_payloads: family.max_nonzero.into(),
                    }),
                )
            })
            .chain(
                sampler.offset_rhs().iter().map(|rhs| (*rhs, InputBinding::Scalar(unit.clone()))),
            )
            .collect();
        RnsBggNoiseBindings::new(inputs, rhs).unwrap()
    }

    /// The first chain the noise audit accepts for `program` at encoding
    /// sigma `sigma` in a ring of dimension 16 with 30-bit primes and base
    /// `2^15`, over CRT depths from three and one to three primes per drop.
    fn planned_parameters(
        program: &ExponentLutProgram,
        sampler: &PrfSampler,
        sigma: &bigdecimal::BigDecimal,
        bound: u64,
    ) -> RnsBggParameters {
        let bindings = noise_bindings(program, sampler, sigma);
        let bounds = vec![BigUint::from(bound); program.outputs().len()];
        let mut rejected = Vec::new();
        for max_crt_depth in 3..=24 {
            for primes_per_drop in 1..=3 {
                let request = RnsBggPlanRequest {
                    ring_dimension: 16,
                    encoding_error_sigma: sigma.clone(),
                    crt_bits: 30,
                    base_bits: 15,
                    terminal_depth: 1,
                    max_crt_depth,
                    terminal_value_bound: BigUint::from(bound),
                    operations_per_drop: 1,
                    primes_per_drop,
                    rhs_mode: Default::default(),
                };
                match crate::utils::audit_program_at_crt_depth(
                    &request, program, &bindings, &bounds, None, 0,
                ) {
                    Ok((parameters, _)) => {
                        eprintln!(
                            "accepted crt_depth={max_crt_depth} primes_per_drop={primes_per_drop} levels={}",
                            parameters.levels().len()
                        );
                        return parameters;
                    }
                    Err(error) => rejected.push((max_crt_depth, primes_per_drop, error)),
                }
            }
        }
        panic!("no parameter candidate: {rejected:?}");
    }

    /// Preprocesses, encrypts and evaluates the sampler of `members` random
    /// keys on the CPU, then verifies and decrypts every output against the
    /// sample of its key's inner product `z = <a, k_o> mod Q`, read through
    /// the registered tables. Without `sigma` the encodings are noise-free
    /// (sigma 0) in a ring of dimension 16 with three 17-bit primes at every
    /// level; with it, the chain is the first the noise audit accepts (see
    /// [`planned_parameters`]).
    fn check_round_trip(
        distribution: PrfDistribution,
        members: usize,
        sigma: Option<bigdecimal::BigDecimal>,
    ) {
        // n = 16, n' = 8, Q = 16, P = 16 over an LWR dimension of four.
        let profile =
            SparseLwrPrfProfile::new(ExpLayout { stride_log2: 1, width_log2: 3 }, 16, 4).unwrap();
        let keys = crate::decoder::decoding::fixture::keys(members);
        let (program, sampler) = sampler_on(profile, &keys, distribution).unwrap();
        let samples = samples(&program, &sampler);
        // The binary offset is the sample at `z = 0`, added as an encoding.
        let offset = match distribution {
            PrfDistribution::Binary { .. } => samples[0],
            PrfDistribution::Cbd { .. } => 0,
        };
        let bound = samples.iter().map(|sample| sample.unsigned_abs()).max().unwrap() +
            offset.unsigned_abs();
        let parameters = match &sigma {
            Some(sigma) => planned_parameters(&program, &sampler, sigma, bound),
            None => {
                let workload = RnsBggDerivedWorkload::from_program(&program).unwrap();
                let output = match sampler.output() {
                    PrfOutput::Scalar(wire) | PrfOutput::Family(wire) => wire,
                };
                // The coefficient outputs are finalized one level below
                // their own.
                let level = DCRTPolyParams::new(16, 3, 17, 4, None, None);
                RnsBggParameters::new(vec![level; workload.level(output).unwrap() + 2], 0, 1)
                    .unwrap()
            }
        };
        let level = parameters.levels()[0].clone();
        let mut execution = CpuExecution::new(cpu_backend(parameters.levels().iter().cloned()));
        let mut store = MemoryArtifactStore::default();
        let hash_key: [u8; 32] = rand::random();
        let context: [u8; 32] = rand::random();
        let payload = RnsEncryptBuild::new(&parameters, context)
            .sample_payload_secret(&mut execution, &mut store, |_| Ok(()))
            .unwrap();
        let rhs = sampler.rhs_plaintexts::<DCRTPoly>(&level, &keys).unwrap();
        let (preprocess, online_eval) = crate::api::preprocess(
            &program,
            &ParamEnv::default(),
            &rhs,
            Some(&payload),
            &hash_key,
            RnsPreprocessBuild::new(&parameters, context),
            &mut execution,
            &mut store,
            |_, outputs| Ok(outputs),
            |_| Ok(()),
        )
        .unwrap();
        let mask = RnsEncryptBuild::new(&parameters, rand::random())
            .sample_mask_secret(&mut execution, &mut store, |_| Ok(()))
            .unwrap();
        let q = profile.q().unwrap();
        let a = (0..profile.lwr_dimension).map(|i| (7 * i + 3) % q).collect::<Vec<_>>();
        let lhs = sampler.plaintext_inputs::<DCRTPoly>(&level, &a).unwrap();
        let encrypted = crate::api::encrypt(
            &program.encryption_inputs(),
            &ParamEnv::default(),
            &lhs,
            &preprocess,
            &payload,
            Some(&mask),
            &hash_key,
            RnsEncryptBuild::new(&parameters, context),
            &mut execution,
            &mut store,
            |_| Ok(()),
            |_| Ok(()),
        )
        .unwrap();
        let result = crate::api::online_eval(
            &program.without_lut_values(),
            &ParamEnv::default(),
            &online_eval,
            &encrypted,
            &hash_key,
            RnsOnlineEvalBuild::new(&parameters, context),
            &mut execution,
            &mut store,
            |_, outputs| Ok(outputs),
            |_| Ok(()),
        )
        .unwrap();

        let t = RnsBggParameters::coefficient_decoding_modulus(bound).unwrap() as i64;
        let a = a.iter().map(|value| *value as u64).collect::<Vec<_>>();
        let terminal = parameters.levels().last().unwrap();
        let plaintext = |value: i64| {
            let constant =
                DCRTPoly::from_usize_to_constant(terminal, value.unsigned_abs() as usize);
            if value < 0 { -constant } else { constant }
        };
        for (index, key) in keys.iter().enumerate() {
            let z = crate::program_gadgets::pbc::clear_pbc_inner_product(
                &key.public_layout,
                key.private_schedule(),
                &a,
                q as u64,
            )
            .unwrap() as usize;
            let expected = samples[z] + offset;
            let verify = |value: i64, execution: &mut CpuExecution, store: &mut _| {
                verify_output(
                    &program,
                    &result,
                    index,
                    &plaintext(value),
                    &parameters,
                    &payload,
                    Some(&mask),
                    execution,
                    store,
                )
            };
            verify(expected, &mut execution, &mut store)
                .unwrap_or_else(|error| panic!("{distribution:?} output {index}: {error:?}"));
            assert!(verify(expected + 1, &mut execution, &mut store).is_err());
            let decoded = decode_output(
                &program,
                &result,
                index,
                &parameters,
                &payload,
                Some(&mask),
                &[0],
                &mut execution,
                &mut store,
                |_| Ok(()),
            )
            .unwrap();
            assert_eq!(decoded, vec![BigInt::from(expected.rem_euclid(t))], "{distribution:?}");
            if let PrfDistribution::Binary { q_min } = distribution {
                let bit = expected.rem_euclid(q_min as i64);
                assert!(bit == 0 || bit == 1, "z = {z}: {expected} mod {q_min}");
            }
        }
    }

    #[test]
    fn cbd_round_trips_on_the_cpu() {
        for members in [1, 2] {
            check_round_trip(PrfDistribution::Cbd { eta: 2 }, members, None);
        }
    }

    #[test]
    fn binary_round_trips_on_the_cpu() {
        for members in [1, 2] {
            check_round_trip(PrfDistribution::Binary { q_min: 7 }, members, None);
        }
    }

    /// The round trip with encoding noise of sigma one.
    #[test]
    fn noisy_round_trips_on_the_cpu() {
        let sigma = bigdecimal::BigDecimal::from(1);
        for distribution in [PrfDistribution::Cbd { eta: 2 }, PrfDistribution::Binary { q_min: 7 }]
        {
            for members in [1, 2] {
                check_round_trip(distribution, members, Some(sigma.clone()));
            }
        }
    }
}
