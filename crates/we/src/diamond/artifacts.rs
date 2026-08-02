#[derive(Clone, Debug, Default)]
pub struct DiamondArtifactNames;

impl DiamondArtifactNames {
    pub const INITIAL_STATE: &'static str = "diamond_initial_state";
    pub const ONE_PREIMAGE: &'static str = "diamond_one_preimage";
    pub const K_PREIMAGE: &'static str = "diamond_k_preimage";
    pub const DECODER_PREIMAGE: &'static str = "diamond_decoder_preimage";
    pub const R_DECOMPOSED: &'static str = "diamond_r_decomposed";

    pub fn transition(level: usize, digit: usize, state: usize) -> String {
        format!("diamond_transition_{level}_{digit}_{state}")
    }

    pub fn witness_preimage(bit: usize) -> String {
        format!("diamond_witness_preimage_{bit}")
    }

    pub fn public_key(index: usize) -> String {
        format!("diamond_public_key_{index}")
    }
}
