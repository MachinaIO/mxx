//! Noise of the noise refresh (see [`crate::refresh`]).
//!
//! Decoder `(i, j)` sees `Q_i y_j + M_ij + N_ij` with
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
//! - `e^sm_ij`: decoder `(i, j)`'s simulated smudge sum.
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
    /// `F_i >= ||N_ij||` for every column `j`, by prime.
    pub errors: Vec<BigUint>,
    /// `Q_i = Q / q_i`, by prime.
    pub spacings: Vec<BigUint>,
    /// The smudge magnitude bound `P^d / 2`.
    pub smudge: BigUint,
    /// The fresh noise bound: `||e'||_inf <= eta`.
    pub fresh: BigUint,
}

impl RnsBggRefreshNoise {
    /// Whether every decoder rounds exactly: `P^d / 2 + F_i < Q_i / 2`.
    pub fn fits(&self) -> bool {
        self.errors
            .iter()
            .zip(&self.spacings)
            .all(|(error, spacing)| (&self.smudge + error) * 2u8 < *spacing)
    }

    /// Whether the smudge range `P^d` hides every decoder's noise by
    /// `P^d > F_i 2^bits`, `bits` per smudged value (see
    /// [`crate::decoder::decoding`]).
    pub fn hides(&self, statistical_bits: usize) -> bool {
        self.errors.iter().all(|error| &self.smudge * 2u8 > error << statistical_bits)
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
/// label member a separate encryption, and every RHS input a bit.
fn child_bindings(
    parameters: &RnsBggParameters,
    program: &crate::program::ExponentLutProgram,
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
    )
}

/// The worst-case noise of `declaration`'s refresh of output `target_index`
/// of `source` under `bindings`, with the fresh-noise child on `fresh` and
/// the decoders' smudge children on `smudges` (in decoder order), and the
/// unmask trapdoor sigma `unmask_sigma`.
#[allow(clippy::too_many_arguments)]
pub fn simulate_refresh_noise(
    parameters: &RnsBggParameters,
    fresh: &RnsBggParameters,
    smudges: &[&RnsBggParameters],
    source: &crate::program::ExponentLutProgram,
    bindings: &RnsBggNoiseBindings,
    declaration: &RefreshDeclaration,
    target_index: usize,
    unmask_sigma: f64,
) -> Result<RnsBggRefreshNoise, crate::api::ApiError> {
    use crate::api::ApiError;
    let level = parameters
        .validate_refresh_children(declaration, source, fresh, smudges)
        .map_err(ApiError::scheme)?;
    if source.outputs().get(target_index) !=
        Some(&crate::program::Operand::Scalar(declaration.target_output()))
    {
        return Err(ApiError::scheme(RnsBggError::Decoder(DecoderError::TargetMismatch)));
    }
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
        .bound
        .error
        .clone();
    // Fresh carriers: the raw coefficient outputs moved down once to `Q`.
    let fresh_program = declaration.fresh().program();
    let fresh_level =
        RnsBggParameters::refresh_fresh_level(declaration).map_err(ApiError::scheme)?;
    let fresh_error = lower_child(
        fresh,
        fresh_program,
        &child_bindings(fresh, fresh_program).map_err(ApiError::scheme)?,
        None,
    )
    .map_err(ApiError::scheme)?
    .iter()
    .map(|value| fresh.lut_mod_down_bound(fresh_level, &value.bound.error))
    .try_fold(BigUint::zero(), |largest, error| error.map(|error| largest.max(error)))
    .map_err(ApiError::scheme)?;
    let smudge_errors = smudges
        .iter()
        .zip(declaration.decoders())
        .map(|(smudge, decoder)| {
            let program = decoder.smudge().program();
            let values =
                lower_child(smudge, program, &child_bindings(smudge, program)?, Some(decoder))?;
            Ok(values.into_iter().map(|value| value.bound.error).max().unwrap_or_default())
        })
        .collect::<Result<Vec<_>, RnsBggError>>()
        .map_err(ApiError::scheme)?;
    let level_parameters = parameters.level(level).map_err(ApiError::scheme)?;
    let next = parameters.level(level + 1).map_err(ApiError::scheme)?;
    let n = BigUint::from(level_parameters.ring_dimension());
    let columns = level_parameters.modulus_digits();
    let half_base = |params: &DCRTPolyParams| BigUint::one() << (params.base_bits() - 1);
    // `e_1 X_i` and `e_x Y_i` do not depend on the prime.
    let one_term = BigUint::from(2 * columns) *
        &n *
        half_base(level_parameters) *
        parameters.encoding_error_bound();
    let target_term = BigUint::from(next.modulus_digits()) * half_base(next) * &target;
    let (_, unmask_digits, preimage_bound) =
        parameters.unmask_gadget(level, unmask_sigma).map_err(ApiError::scheme)?;
    let unmask_term = &n *
        BigUint::from(crate::decoder::unmask::UNMASK_ROWS * (unmask_digits + 2)) *
        parameters.encoding_error_bound() *
        &preimage_bound;
    let modulus: BigUint = level_parameters.modulus().as_ref().clone();
    let mut errors = Vec::with_capacity(declaration.primes());
    let mut spacings = Vec::with_capacity(declaration.primes());
    for (prime, q_i) in level_parameters.to_crt().0.into_iter().enumerate() {
        let cofactor = &modulus / q_i;
        let (z_l1, _) =
            parameters.gadget_inverse_norms(level, &cofactor).map_err(ApiError::scheme)?;
        let common = &one_term + &target_term + z_l1 * &fresh_error + &unmask_term;
        let smudge = (0..declaration.columns())
            .map(|column| smudge_errors[prime * declaration.columns() + column].clone())
            .max()
            .unwrap_or_default();
        errors.push(common + smudge);
        spacings.push(cofactor);
    }
    Ok(RnsBggRefreshNoise {
        refresh: *declaration.spec(),
        errors,
        spacings,
        smudge: declaration.decoders()[0].smudge_bound(),
        fresh: BigUint::from(declaration.eta()),
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
                &fresh,
                &smudges,
                &source,
                &bindings,
                &declaration,
                0,
                4.578,
            )
            .unwrap()
        };
        let (small, large) = (report(1), report(14));
        assert!(small.errors[0] < large.errors[0]);
        assert!(large.fits() && large.hides(1), "{large:?}");
        assert!(!small.hides(1));
        assert_eq!(large.fresh, BigUint::one());
    }
}
