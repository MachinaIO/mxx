pub mod circuit_decrypt;
pub mod circuit_merge;
pub mod circuit_prg;
pub mod naive_vec;
pub mod simulation;

use crate::{
    circuit::evaluable::Evaluable, lookup::PltEvaluator, matrix::PolyMatrix,
    slot_transfer::SlotTransferEvaluator,
};

pub use naive_vec::{
    NoiseRefresherNaiveVec, debug_sample_prg_encoding_wires, debug_sample_prg_plaintext_wires,
    debug_sample_prg_public_key_wires,
};
pub use simulation::{NoiseRefreshErrorSimulation, simulate_noise_refresh_error_growth};

/// Refreshes the noise of one slotwise encoding wire.
///
/// The trait separates the offline public-key preprocessing path from the online encoding path.
/// `K` is the public-key-like evaluable used for preprocessing, `E` is the encoding-like evaluable
/// used online, and `M` is the raw matrix representation used for decoder terms.
pub trait NoiseRefresher<K, E, M>
where
    K: Evaluable,
    E: Evaluable,
    M: PolyMatrix,
{
    /// Builds the public refresh material for `refreshed_input`.
    ///
    /// The returned pair is `(A_prime, refresh_keys)`. `A_prime` is sampled from `refresh_id` in
    /// the same public-key-like shape as `K`; `refresh_keys` contains the preprocessing targets
    /// needed to build decoder preimages for the online path.
    fn preprocess<PE, ST>(
        &self,
        refresh_id: &[u8],
        one: &K,
        refreshed_input: &K,
        enc_seeds: &[K],
        decryption_key: &K,
        plt_evaluator: &PE,
        slot_transfer_evaluator: &ST,
    ) -> (K, Vec<K>)
    where
        PE: PltEvaluator<K>,
        ST: SlotTransferEvaluator<K>;

    /// Computes the online refreshed encoding vector.
    ///
    /// `decoders` is ordered by `slot_idx * crt_depth + crt_idx`. Each decoder is subtracted from
    /// the matching CRT-level online term before rounding and CRT recomposition.
    ///
    /// The output has the same encoding-like shape as `E`.
    fn online_eval<PE, ST>(
        &self,
        refresh_id: &[u8],
        one: &E,
        refreshed_input: &E,
        enc_seeds: &[E],
        decryption_key: &E,
        decoders: &[M],
        plt_evaluator: &PE,
        slot_transfer_evaluator: &ST,
    ) -> E
    where
        PE: PltEvaluator<E>,
        ST: SlotTransferEvaluator<E>;
}
