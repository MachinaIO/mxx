/// Canonical artifact names for the Section 5.2 prMIFE-to-iO wrapper.
///
/// Composite private-prFE objects are flattened into typed IR artifacts.  The
/// preprocessing graph, evaluator graph, runtime, and linked noise simulator
/// must all use these names rather than duplicating string layouts.
pub struct Aky24ArtifactNames;

impl Aky24ArtifactNames {
    pub const FINAL_KEY_PREIMAGE: &str = "aky24/final-key/preimage";
    pub const FUNCTION_CIPHERTEXT_C_B: &str = "aky24/function-ciphertext/c-b";
    pub const FUNCTION_CIPHERTEXT_X: &str = "aky24/function-ciphertext/x";
    pub const FUNCTION_CIPHERTEXT_ATTRIBUTE_VECTORS: &str =
        "aky24/function-ciphertext/attribute-vectors";
    pub const OUTPUT: &str = "aky24/output";

    pub fn layer_b_public(layer: usize) -> String {
        format!("aky24/layer/{layer}/b-public")
    }

    pub fn layer_attribute_public(layer: usize) -> String {
        format!("aky24/layer/{layer}/attribute-public")
    }

    pub fn input_ciphertext_preimage(input: usize, bit: bool) -> String {
        format!("aky24/input/{input}/ciphertext/{}/preimage", usize::from(bit))
    }

    pub fn input_ske_nonce(input: usize, bit: bool) -> String {
        format!("aky24/input/{input}/ciphertext/{}/ske-nonce", usize::from(bit))
    }

    pub fn input_ske_masked_payload(input: usize, bit: bool) -> String {
        format!("aky24/input/{input}/ciphertext/{}/ske-masked-payload", usize::from(bit))
    }

    pub fn input_randomness(input: usize, bit: bool) -> String {
        format!("aky24/input/{input}/ciphertext/{}/randomness", usize::from(bit))
    }

    /// Complete public manifest schema for an `input_size + 1` layer AKY24
    /// obfuscation. Layer indices are one-based, matching the paper.
    pub fn all_public_names(input_size: usize) -> BTreeSet<String> {
        let layer_count = input_size.checked_add(1).expect("AKY24 layer count overflowed");
        let mut names = BTreeSet::from([
            Self::FINAL_KEY_PREIMAGE.to_owned(),
            Self::FUNCTION_CIPHERTEXT_C_B.to_owned(),
            Self::FUNCTION_CIPHERTEXT_X.to_owned(),
            Self::FUNCTION_CIPHERTEXT_ATTRIBUTE_VECTORS.to_owned(),
        ]);
        for layer in 1..=layer_count {
            names.insert(Self::layer_b_public(layer));
            names.insert(Self::layer_attribute_public(layer));
        }
        for input in 0..input_size {
            for bit in [false, true] {
                names.insert(Self::input_ciphertext_preimage(input, bit));
                names.insert(Self::input_ske_nonce(input, bit));
                names.insert(Self::input_ske_masked_payload(input, bit));
                names.insert(Self::input_randomness(input, bit));
            }
        }
        names
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_two_ciphertexts_for_one_position_have_distinct_names() {
        assert_ne!(
            Aky24ArtifactNames::input_ciphertext_preimage(3, false),
            Aky24ArtifactNames::input_ciphertext_preimage(3, true)
        );
    }

    #[test]
    fn complete_public_schema_contains_every_layer_and_both_input_choices() {
        let names = Aky24ArtifactNames::all_public_names(3);
        assert_eq!(names.len(), 2 * 4 + 3 * 8 + 4);
        assert!(names.contains(&Aky24ArtifactNames::layer_b_public(4)));
        assert!(names.contains(&Aky24ArtifactNames::input_ciphertext_preimage(2, false)));
        assert!(names.contains(&Aky24ArtifactNames::input_ciphertext_preimage(2, true)));
        assert!(names.contains(&Aky24ArtifactNames::input_ske_nonce(2, true)));
        assert!(names.contains(&Aky24ArtifactNames::input_ske_masked_payload(2, true)));
        assert!(names.contains(&Aky24ArtifactNames::input_randomness(2, true)));
    }
}
use std::collections::BTreeSet;
