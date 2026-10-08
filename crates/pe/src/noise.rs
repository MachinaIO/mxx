//! Noise of the scheme: the evaluation bound `E_F` from the exponent-lut
//! simulator, and the decryption bound and margin.
//!
//! The simulator evaluates the compiled predicate on the class chain with
//! the gadget output use, so `E_F` bounds the error of the full-gadget
//! carrier `c_F = s (A_F - F(x) G) + e_F` and no direct-decoding action or
//! margin of another consumer is applied. Its material contracts are the
//! setup's: `chi_fl` for the initial state and every companion, `chi` for
//! the helpers, and the supplied attribute ciphertexts' bound
//! `B_C = m_B n B_b B_Z` on `e_C = -e_b Z_t`.
//!
//! Decryption contracts the family row with the key by blocks. With the key
//! cutoff `B_key`, the anchor blocks contribute `n B_key m_B B_b` each
//! (`W` of them), the helper blocks `n B_key m B_h` each (`W (W - 1)`), and
//! the output block `n B_key m E_F`; with the message error `B_m`,
//!
//! `E_PE <= B_m + n B_key (W m_B B_b + W (W - 1) m B_h + m E_F)`.
//!
//! The decoder rounds each coefficient of `floor(q/2) m + e` to the nearer of
//! `0` and `floor(q/2)`, i.e. `round(2 x / q) mod 2`; for `q` odd it returns
//! `m` for both symbols whenever `4 |e| + 2 < q` ([`PeNoiseReport::decrypts`]).
use crate::{BranchingProgram, PeError, PeParameters, branching_program::ATTRIBUTE_COORDINATE};
use mxx_backends::poly::PolyParams;
use mxx_exponent_lut::{
    api,
    rns_bgg::{
        RnsBggDerivedWorkload, RnsBggDynamicNoise, RnsBggNoiseBindings, RnsBggNoiseRhs,
        RnsBggNoiseTrace, RnsBggNoiseValue, RnsBggPackageNoise, RnsNoiseBuild, RnsOutputUse,
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
}

impl PeNoiseReport {
    /// Whether every coefficient of `floor(q/2) m + e` rounds to `m`:
    /// `4 E_PE + 2 < q`.
    pub fn decrypts(&self) -> bool {
        &self.decryption * 4u8 + 2u8 < self.modulus
    }
}

/// The material contracts of `program`'s packages under `parameters`, in
/// the source namespace `namespace` (a setup identity).
pub(crate) fn package_contracts(
    parameters: &PeParameters,
    program: &crate::CompiledPredicate,
    namespace: [u8; 32],
) -> Result<RnsBggDynamicNoise, PeError> {
    let automorphisms = parameters
        .class()
        .layout()
        .automorphism_indices(parameters.level().ring_dimension() as usize)
        .map_err(|error| PeError::Parameters(error.to_string()))?;
    let rhs = program
        .program()
        .rhs_inputs()
        .keys()
        .map(|id| {
            Ok((
                *id,
                RnsBggPackageNoise {
                    coordinate: program.binding().rhs(*id)?,
                    ciphertext_error: parameters.ciphertext_error_bound(),
                    ciphertext_sigma: None,
                },
            ))
        })
        .collect::<Result<BTreeMap<_, _>, PeError>>()?;
    debug_assert!(rhs.values().all(|package| package.coordinate == ATTRIBUTE_COORDINATE));
    Ok(RnsBggDynamicNoise {
        namespace,
        distributions: parameters.distributions(),
        rhs,
        automorphisms,
    })
}

/// Simulate `program` under `parameters` with the exponent-lut evaluator,
/// and bound its decryption.
pub fn simulate(
    parameters: &PeParameters,
    program: &BranchingProgram,
) -> Result<PeNoiseReport, PeError> {
    let compiled = program.compile()?;
    let rns = parameters.rns();
    let initial = RnsBggNoiseValue::sampled_gaussian(
        parameters.flooding(),
        b"mxx-pe/initial-state",
        &[BigInt::from(1u8)],
    );
    let symbol = RnsBggNoiseRhs { payload_l1: 1u8.into(), payload_l2_squared: 1u8.into() };
    let bindings = RnsBggNoiseBindings::from_inputs(compiled.program(), &[initial], &[symbol])?
        .with_output_use(RnsOutputUse::Gadget)
        .with_dynamic(package_contracts(parameters, &compiled, parameters.identity())?);
    let workload = RnsBggDerivedWorkload::from_program_with_parameters(rns, compiled.program())?;
    let trace = api::noise_simulate(
        compiled.program(),
        RnsNoiseBuild::new(rns, compiled.program(), &workload, &bindings)?,
        |_, outputs| Ok(outputs),
    )?;
    let evaluation = trace
        .outputs
        .first()
        .ok_or_else(|| PeError::Parameters("the simulator returned no output".into()))?
        .bound
        .error
        .clone();
    let w = BigUint::from(parameters.class().lookup_width());
    let m = BigUint::from(parameters.state_columns());
    let m_b = BigUint::from(parameters.anchor_columns());
    let n = BigUint::from(parameters.level().ring_dimension());
    let error = parameters.error().cutoff();
    let blocks = &w * &m_b * error + &w * (&w - 1u8) * &m * error + &m * &evaluation;
    let decryption = error + n * parameters.key_cutoff() * blocks;
    Ok(PeNoiseReport { evaluation, decryption, modulus: parameters.modulus(), trace })
}
