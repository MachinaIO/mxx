pub mod circuit_decrypt;
pub mod circuit_merge;
pub mod circuit_prg;
pub mod naive_vec;
pub mod simulation;

use crate::{
    circuit::evaluable::Evaluable, lookup::PltEvaluator, matrix::PolyMatrix,
    slot_transfer::SlotTransferEvaluator,
};

pub use naive_vec::NoiseRefresherNaiveVec;
pub use simulation::{NoiseRefreshErrorSimulation, simulate_noise_refresh_error_growth};

/// Refreshes the noise of one slotwise encoding wire.
///
/// The trait separates the offline public-key preprocessing path from the online encoding path.
/// `K` is the public-key-like evaluable used for preprocessing, `E` is the encoding-like evaluable
/// used online, and `M` is the raw matrix representation returned by both paths after the
/// noise-refresh terms have been compressed into gadget columns.
pub trait NoiseRefresher<K, E, M>
where
    K: Evaluable,
    E: Evaluable,
    M: PolyMatrix,
{
    /// Builds the public refresh material for `refreshed_input`.
    ///
    /// The returned pair is `(A_prime, refresh_matrices)`. `A_prime` is the public `1 x m_g`
    /// matrix sampled from `refresh_id`; `refresh_matrices` is ordered by
    /// `slot_idx * crt_depth + crt_idx`, and each matrix has shape `1 x m_g`.
    fn preprocess<PE, ST>(
        &self,
        refresh_id: &[u8],
        one: &K,
        refreshed_input: &K,
        enc_seeds: &[K],
        decryption_key: &K,
        plt_evaluator: &PE,
        slot_transfer_evaluator: &ST,
    ) -> (M, Vec<M>)
    where
        PE: PltEvaluator<K>,
        ST: SlotTransferEvaluator<K>;

    /// Computes the online refreshed encoding vector directly as a matrix.
    ///
    /// `decoders` is ordered by `slot_idx * crt_depth + crt_idx`. Each decoder is subtracted from
    /// the matching CRT-level online term before rounding and CRT recomposition.
    ///
    /// The output matrix has one row per logical slot/reference wire and `m_g` columns. Each row is
    /// reconstructed from the per-CRT `q/q_i`-scaled terms by coefficient-wise rounding followed by
    /// CRT recomposition.
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
    ) -> M
    where
        PE: PltEvaluator<E>,
        ST: SlotTransferEvaluator<E>;
}
