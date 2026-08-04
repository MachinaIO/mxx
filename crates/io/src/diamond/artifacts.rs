use super::{DiamondIoConfig, DiamondIoConfigError, DiamondIoFunction};
use thiserror::Error;

/// Canonical public artifact names for one Diamond iO production.
///
/// Names describe protocol roles. Families such as one native ciphertext or
/// one refresh bundle may contain several matrices, but their internal layout
/// is fixed by the validated graph and therefore does not leak into the
/// persisted namespace.
#[derive(Clone, Debug, Default)]
pub struct DiamondIoArtifactNames;

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum DiamondIoArtifactNameError {
    #[error(transparent)]
    Config(#[from] DiamondIoConfigError),
    #[error("the Diamond iO public artifact layout overflowed usize")]
    LayoutOverflow,
}

impl DiamondIoArtifactNames {
    pub const INJECTOR_INITIAL_STATE: &'static str = "diamond-io/injector/initial-state";
    pub const INJECTOR_TRANSITIONS: &'static str = "diamond-io/injector/transitions";
    pub const HASH_KEY: &'static str = "diamond-io/hash-key";
    pub const SCALAR_ONE_PUBLIC_KEY: &'static str = "diamond-io/scalar/one/public-key";
    pub const SCALAR_K_PUBLIC_KEY: &'static str = "diamond-io/scalar/k/public-key";
    pub const ONE_PROJECTION: &'static str = "diamond-io/projection/one";
    pub const K_PROJECTION: &'static str = "diamond-io/projection/k";
    pub const LOOKUP_BASE_PROJECTION: &'static str = "diamond-io/projection/lookup-base";

    pub fn scalar_input_public_key(bit: usize) -> String {
        format!("diamond-io/scalar/input/{bit}/public-key")
    }

    pub fn input_projection(bit: usize) -> String {
        format!("diamond-io/projection/input/{bit}")
    }

    /// One scalar-matrix family obtained from one flattened circuit wire of a
    /// native encrypted seed bit. Core artifacts cannot contain nested
    /// families, so every wire has its own stable artifact name.
    pub fn native_seed_bindings(seed: usize, wire: usize) -> String {
        format!("diamond-io/native-seed-seed-{seed}-{wire}")
    }

    pub fn round_common_public_key(round: usize, wire: usize) -> String {
        format!("diamond-io/prf/round/{round}/wire/{wire}/common-public-key")
    }

    pub fn round_rebase_preimages(round: usize, branch: usize, wire: usize) -> String {
        format!("diamond-io/prf/round/{round}/branch/{branch}/wire/{wire}/rebase-preimages")
    }

    /// `A'` is branch-independent: every selected branch refreshes to the same
    /// next-seed public key for this round and wire.
    pub fn round_refresh_a_prime(round: usize, wire: usize) -> String {
        format!("diamond-io/prf/round/{round}/wire/{wire}/refresh/a-prime")
    }

    pub fn round_refresh_decoder_preimages(round: usize, branch: usize, wire: usize) -> String {
        format!(
            "diamond-io/prf/round/{round}/branch/{branch}/wire/{wire}/refresh/decoder-preimages"
        )
    }

    pub fn final_function_secret_dependent_public_key(output: usize) -> String {
        format!("diamond-io/final/output/{output}/function/secret-dependent-public-key")
    }

    pub fn final_function_public_bottom(output: usize) -> String {
        format!("diamond-io/final/output/{output}/function/public-bottom")
    }

    pub fn final_mask_secret_dependent_public_key(output: usize) -> String {
        format!("diamond-io/final/output/{output}/mask/secret-dependent-public-key")
    }

    pub fn final_mask_public_bottom(output: usize) -> String {
        format!("diamond-io/final/output/{output}/mask/public-bottom")
    }

    pub fn final_decoder_preimages(output: usize) -> String {
        format!("diamond-io/final/output/{output}/decoder-preimages")
    }

    /// Enumerates the fixed Diamond protocol outputs. Public-LUT lowering adds
    /// canonical `lwe_lookup_*` artifacts derived from concrete circuit gate
    /// identities; those dynamic names are validated separately when the
    /// complete producer graph is built.
    pub fn all_public_names(
        config: &DiamondIoConfig,
        function: &DiamondIoFunction,
        ciphertext_wire_count: usize,
    ) -> Result<Vec<String>, DiamondIoArtifactNameError> {
        config.validate(function)?;
        if ciphertext_wire_count == 0 {
            return Err(DiamondIoArtifactNameError::LayoutOverflow);
        }
        let input = config.input_config();
        let input_bits = input.witness_size().map_err(DiamondIoConfigError::from)?;
        let round_wire_count = config
            .seed_bits
            .checked_mul(ciphertext_wire_count)
            .ok_or(DiamondIoArtifactNameError::LayoutOverflow)?;
        let mut names = vec![
            Self::INJECTOR_INITIAL_STATE.to_owned(),
            Self::INJECTOR_TRANSITIONS.to_owned(),
            Self::HASH_KEY.to_owned(),
            Self::SCALAR_ONE_PUBLIC_KEY.to_owned(),
            Self::SCALAR_K_PUBLIC_KEY.to_owned(),
            Self::ONE_PROJECTION.to_owned(),
            Self::K_PROJECTION.to_owned(),
            Self::LOOKUP_BASE_PROJECTION.to_owned(),
        ];

        for bit in 0..input_bits {
            names.push(Self::scalar_input_public_key(bit));
            names.push(Self::input_projection(bit));
        }
        for seed in 0..config.seed_bits {
            for wire in 0..ciphertext_wire_count {
                names.push(Self::native_seed_bindings(seed, wire));
            }
        }
        for round in 0..config.input_count {
            for wire in 0..round_wire_count {
                names.push(Self::round_common_public_key(round, wire));
                names.push(Self::round_refresh_a_prime(round, wire));
                for branch in 0..config.digit_base {
                    names.push(Self::round_rebase_preimages(round, branch, wire));
                    names.push(Self::round_refresh_decoder_preimages(round, branch, wire));
                }
            }
        }
        for output in 0..function.output_bits() {
            names.push(Self::final_function_secret_dependent_public_key(output));
            names.push(Self::final_function_public_bottom(output));
            names.push(Self::final_mask_secret_dependent_public_key(output));
            names.push(Self::final_mask_public_bottom(output));
            names.push(Self::final_decoder_preimages(output));
        }
        Ok(names)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_ir_core::RealExpr;
    use std::collections::BTreeSet;

    fn config() -> DiamondIoConfig {
        DiamondIoConfig {
            modulus: 65_537.into(),
            ring_dimension: 8,
            input_count: 2,
            digit_base: 4,
            batch_bits: 2,
            gadget_base: 4.into(),
            digit_count: 4,
            trapdoor_sigma: RealExpr::from_integer(5),
            error_sigma: RealExpr::from_integer(1),
            bgg_tag: b"diamond-io-artifacts".to_vec(),
            seed_bits: 5,
            prf_mask_output_coeff_bits: 1,
            noise_refresh_v_bits: 1,
            noise_refresh_cbd_n: 2,
            noise_refresh_hash_key: [2; 32],
            goldreich_graph_seed: [3; 32],
            ring_gsw_width: 3,
            ring_gsw_public_key_error_sigma: Some(RealExpr::from_integer(1)),
            refresh_crt_scale_factors: vec![1.into()],
            refresh_crt_plaintext_moduli: vec![2.into()],
            refresh_reconstruction_coefficients: vec![1.into()],
            refresh_decoder_public_columns: 4,
        }
    }

    #[test]
    fn complete_schema_has_the_exact_configured_count_and_no_collisions() {
        let mut config = config();
        let function = DiamondIoFunction::GoldreichPrf { output_bits: 2 };
        config.seed_bits = config.minimum_goldreich_seed_bits(&function).unwrap();
        let ciphertext_wires = 6;
        let names =
            DiamondIoArtifactNames::all_public_names(&config, &function, ciphertext_wires).unwrap();
        let transition_count = (1..=config.input_count)
            .map(|level| {
                config.digit_base * config.input_config().state_count_at_level(level).unwrap()
            })
            .sum::<usize>();
        let input_bits = config.input_bits().unwrap();
        let expected = 7 +
            transition_count +
            2 * input_bits +
            config.seed_bits * ciphertext_wires +
            config.input_count *
                config.seed_bits *
                ciphertext_wires *
                (2 + 2 * config.digit_base) +
            5 * function.output_bits();
        assert_eq!(names.len(), expected);
        assert_eq!(names.iter().collect::<BTreeSet<_>>().len(), expected);
    }

    #[test]
    fn branch_independent_and_branch_specific_refresh_names_are_distinct() {
        assert_ne!(
            DiamondIoArtifactNames::round_refresh_a_prime(1, 11),
            DiamondIoArtifactNames::round_refresh_decoder_preimages(1, 1, 1)
        );
        assert_ne!(
            DiamondIoArtifactNames::round_rebase_preimages(1, 11, 1),
            DiamondIoArtifactNames::round_rebase_preimages(11, 1, 1)
        );
    }
}
