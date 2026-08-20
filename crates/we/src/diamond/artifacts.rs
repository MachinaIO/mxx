#[derive(Clone, Debug, Default)]
pub struct DiamondArtifactNames;

impl DiamondArtifactNames {
    pub const INITIAL_STATE: &'static str = "diamond_initial_state";
    pub const ONE_PREIMAGE: &'static str = "diamond_one_preimage";
    pub const K_PREIMAGE: &'static str = "diamond_k_preimage";
    pub const DECODER_PREIMAGE: &'static str = "diamond_decoder_preimage";
    pub const R_DECOMPOSED: &'static str = "diamond_r_decomposed";
    pub const TRANSITIONS: &'static str = "diamond_transitions";
    pub const WITNESS_PREIMAGES: &'static str = "diamond_witness_preimages";
    pub const PUBLIC_KEYS: &'static str = "diamond_public_keys";
}
