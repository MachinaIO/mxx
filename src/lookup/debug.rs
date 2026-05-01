use crate::{
    bgg::{
        encoding::BggEncoding,
        naive_vec::{NaiveBGGEncodingVec, NaiveBGGPublicKeyVec},
        public_key::BggPublicKey,
    },
    circuit::{evaluable::Evaluable, gate::GateId},
    lookup::{
        PltEvaluator, PublicLut,
        lwe::{derive_a_lt_matrix_for_slot, derive_k_low_for_slot},
    },
    matrix::PolyMatrix,
    poly::Poly,
    sampler::{PolyHashSampler, PolyTrapdoorSampler, trapdoor::DCRTTrapdoor},
    storage::{
        read::read_bytes_from_multi_batch,
        write::{add_lookup_buffer, get_lookup_buffer, get_lookup_buffer_bytes},
    },
};
use rayon::prelude::*;
use std::{
    marker::PhantomData,
    path::{Path, PathBuf},
    sync::Arc,
};

#[cfg(feature = "gpu")]
#[path = "debug_gpu.rs"]
mod gpu;

const DEFAULT_CHECKPOINT_PREFIX: &str = "DEBUG_NAIVE_BGG_PUBLIC_LUT";

pub trait DebugTrapdoorPreimage<M: PolyMatrix>: Send + Sync {
    fn debug_preimage(&self, params: &<M::P as Poly>::Params, target: &M) -> M;
}

impl DebugTrapdoorPreimage<crate::matrix::dcrt_poly::DCRTPolyMatrix> for DCRTTrapdoor {
    fn debug_preimage(
        &self,
        _params: &<<crate::matrix::dcrt_poly::DCRTPolyMatrix as PolyMatrix>::P as Poly>::Params,
        target: &crate::matrix::dcrt_poly::DCRTPolyMatrix,
    ) -> crate::matrix::dcrt_poly::DCRTPolyMatrix {
        let decomposed = target.decompose();
        let r_part = self.r_cpu() * &decomposed;
        let e_part = self.e_cpu() * &decomposed;
        r_part.concat_rows(&[&e_part, &decomposed])
    }
}

#[derive(Debug)]
pub struct DebugNaiveBGGPublicKeyVecPltEvaluator<M, SH, ST>
where
    M: PolyMatrix,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    ST: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    hash_key: [u8; 32],
    b0_trapdoor: ST::Trapdoor,
    b0_matrix: M,
    dir_path: PathBuf,
    checkpoint_prefix: String,
    _sh: PhantomData<SH>,
}

impl<M, SH, ST> DebugNaiveBGGPublicKeyVecPltEvaluator<M, SH, ST>
where
    M: PolyMatrix + Send + Sync + 'static,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    ST: PolyTrapdoorSampler<M = M> + Send + Sync,
    M::P: 'static,
{
    pub fn new(
        hash_key: [u8; 32],
        b0_trapdoor: ST::Trapdoor,
        b0_matrix: M,
        dir_path: PathBuf,
    ) -> Self {
        Self::with_checkpoint_prefix(
            hash_key,
            b0_trapdoor,
            b0_matrix,
            dir_path,
            DEFAULT_CHECKPOINT_PREFIX.to_string(),
        )
    }

    pub fn with_checkpoint_prefix(
        hash_key: [u8; 32],
        b0_trapdoor: ST::Trapdoor,
        b0_matrix: M,
        dir_path: PathBuf,
        checkpoint_prefix: String,
    ) -> Self {
        Self { hash_key, b0_trapdoor, b0_matrix, dir_path, checkpoint_prefix, _sh: PhantomData }
    }

    pub fn checkpoint_prefix(&self) -> &str {
        &self.checkpoint_prefix
    }

    pub fn sample_aux_matrices(&self, _params: &<M::P as Poly>::Params) {
        store_b0_checkpoint::<M>(
            &self.checkpoint_prefix,
            self.b0_matrix.clone(),
            ST::trapdoor_to_bytes(&self.b0_trapdoor),
        );
    }

    pub fn load_b0_matrix_checkpoint(&self, params: &<M::P as Poly>::Params) -> Option<M> {
        load_b0_matrix_checkpoint(params, &self.dir_path, &self.checkpoint_prefix)
    }

    fn assert_b0_row_size(&self, row_size: usize) {
        assert_eq!(
            self.b0_matrix.row_size(),
            row_size,
            "debug lookup b0 matrix row size must match input public-key row size"
        );
    }
}

impl<M, SH, ST> PltEvaluator<NaiveBGGPublicKeyVec<M>>
    for DebugNaiveBGGPublicKeyVecPltEvaluator<M, SH, ST>
where
    M: PolyMatrix + Send + Sync + 'static,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    ST: PolyTrapdoorSampler<M = M> + Send + Sync,
    M::P: 'static,
{
    fn public_lookup(
        &self,
        params: &<NaiveBGGPublicKeyVec<M> as Evaluable>::Params,
        _plt: &PublicLut<<NaiveBGGPublicKeyVec<M> as Evaluable>::P>,
        one: &NaiveBGGPublicKeyVec<M>,
        input: &NaiveBGGPublicKeyVec<M>,
        gate_id: GateId,
        _lut_id: usize,
    ) -> NaiveBGGPublicKeyVec<M> {
        input.assert_compatible(one);
        let num_slots = input.num_slots();
        NaiveBGGPublicKeyVec::new(
            (0..num_slots)
                .into_par_iter()
                .map(|slot_idx| {
                    let input_key = &input.keys[slot_idx];
                    let row_size = input_key.matrix.row_size();
                    self.assert_b0_row_size(row_size);
                    let a_lt = derive_a_lt_matrix_for_slot::<M, SH>(
                        params,
                        row_size,
                        self.hash_key,
                        gate_id,
                        Some(slot_idx),
                    );
                    BggPublicKey::new(a_lt, true)
                })
                .collect(),
        )
    }
}

#[derive(Debug, Clone)]
pub struct DebugNaiveBGGEncodingVecPltEvaluator<M, SH, ST>
where
    M: PolyMatrix,
    SH: PolyHashSampler<[u8; 32], M = M>,
    ST: PolyTrapdoorSampler<M = M>,
{
    hash_key: [u8; 32],
    dir_path: PathBuf,
    checkpoint_prefix: String,
    c_b0_compact_bytes: Arc<[u8]>,
    _sh: PhantomData<SH>,
    _st: PhantomData<ST>,
}

impl<M, SH, ST> DebugNaiveBGGEncodingVecPltEvaluator<M, SH, ST>
where
    M: PolyMatrix,
    SH: PolyHashSampler<[u8; 32], M = M>,
    ST: PolyTrapdoorSampler<M = M>,
{
    pub fn new(hash_key: [u8; 32], dir_path: PathBuf, c_b0: M) -> Self {
        Self::with_checkpoint_prefix(
            hash_key,
            dir_path,
            DEFAULT_CHECKPOINT_PREFIX.to_string(),
            c_b0,
        )
    }

    pub fn with_checkpoint_prefix(
        hash_key: [u8; 32],
        dir_path: PathBuf,
        checkpoint_prefix: String,
        c_b0: M,
    ) -> Self {
        Self {
            hash_key,
            dir_path,
            checkpoint_prefix,
            c_b0_compact_bytes: Arc::<[u8]>::from(c_b0.into_compact_bytes()),
            _sh: PhantomData,
            _st: PhantomData,
        }
    }
}

impl<M, SH, ST> PltEvaluator<NaiveBGGEncodingVec<M>>
    for DebugNaiveBGGEncodingVecPltEvaluator<M, SH, ST>
where
    M: PolyMatrix + Send + Sync + 'static,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    ST: PolyTrapdoorSampler<M = M> + Send + Sync,
    ST::Trapdoor: DebugTrapdoorPreimage<M>,
    M::P: 'static,
    for<'a, 'b> &'a M: std::ops::Mul<&'b M, Output = M>,
{
    fn public_lookup(
        &self,
        params: &<NaiveBGGEncodingVec<M> as Evaluable>::Params,
        plt: &PublicLut<<NaiveBGGEncodingVec<M> as Evaluable>::P>,
        one: &NaiveBGGEncodingVec<M>,
        input: &NaiveBGGEncodingVec<M>,
        gate_id: GateId,
        lut_id: usize,
    ) -> NaiveBGGEncodingVec<M> {
        input.assert_compatible(one);
        let (b0_trapdoor, _b0_matrix) =
            load_b0_checkpoint::<M, ST>(params, &self.dir_path, &self.checkpoint_prefix)
                .expect("debug lookup b0 checkpoint missing; call sample_aux_matrices first");
        let c_b0 = M::from_compact_bytes(params, self.c_b0_compact_bytes.as_ref());
        let num_slots = input.num_slots();
        NaiveBGGEncodingVec::new(
            (0..num_slots)
                .into_par_iter()
                .map(|slot_idx| {
                    public_lookup_for_slot::<M, SH, ST>(
                        params,
                        plt,
                        self.hash_key,
                        &b0_trapdoor,
                        &c_b0,
                        &input.encodings[slot_idx],
                        gate_id,
                        lut_id,
                        slot_idx,
                    )
                })
                .collect(),
        )
    }
}

fn public_lookup_for_slot<M, SH, ST>(
    params: &<BggEncoding<M> as Evaluable>::Params,
    plt: &PublicLut<<BggEncoding<M> as Evaluable>::P>,
    hash_key: [u8; 32],
    b0_trapdoor: &ST::Trapdoor,
    c_b0: &M,
    input: &BggEncoding<M>,
    gate_id: GateId,
    lut_id: usize,
    slot_idx: usize,
) -> BggEncoding<M>
where
    M: PolyMatrix + Send + Sync + 'static,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    ST: PolyTrapdoorSampler<M = M> + Send + Sync,
    ST::Trapdoor: DebugTrapdoorPreimage<M>,
    M::P: 'static,
    for<'a, 'b> &'a M: std::ops::Mul<&'b M, Output = M>,
{
    let input_plaintext =
        input.plaintext.as_ref().expect("debug lookup requires revealed input plaintext");
    let x_u64 = input_plaintext.const_coeff_u64();
    let x_usize = usize::try_from(x_u64).expect("LUT input must fit in usize");
    let (k, y_k) =
        plt.get(params, x_u64).unwrap_or_else(|| panic!("{x_u64:?} is not in public lookup f"));
    let k_usize = usize::try_from(k).expect("LUT row index must fit in usize");
    let row_size = input.pubkey.matrix.row_size();
    let gadget = M::gadget_matrix(params, row_size);
    let x_poly = M::P::from_usize_to_constant(params, x_usize);
    let y_poly = M::P::from_elem_to_constant(params, &y_k);
    let a_out =
        derive_a_lt_matrix_for_slot::<M, SH>(params, row_size, hash_key, gate_id, Some(slot_idx));
    let ext_matrix = input.pubkey.matrix.clone() - &(gadget.clone() * x_poly);
    let target = a_out.clone() - &(gadget * &y_poly);
    let k_low = derive_k_low_for_slot::<M, SH>(
        params,
        row_size,
        hash_key,
        gate_id,
        lut_id,
        k_usize,
        Some(slot_idx),
    );
    let adjusted_target = target - &(ext_matrix * &k_low);
    let preimage = b0_trapdoor.debug_preimage(params, &adjusted_target);
    let mut vector = c_b0 * &preimage;
    vector.add_in_place(&(input.vector.clone() * &k_low));
    BggEncoding::new(vector, BggPublicKey::new(a_out, true), Some(y_poly))
}

fn b0_id_prefix(checkpoint_prefix: &str) -> String {
    format!("{checkpoint_prefix}_b0")
}

fn b0_trapdoor_id_prefix(checkpoint_prefix: &str) -> String {
    format!("{checkpoint_prefix}_b0_trapdoor")
}

fn store_b0_checkpoint<M>(checkpoint_prefix: &str, b0_matrix: M, b0_trapdoor_bytes: Vec<u8>)
where
    M: PolyMatrix + Send,
{
    let _ = add_lookup_buffer(get_lookup_buffer(
        vec![(0, b0_matrix)],
        &b0_id_prefix(checkpoint_prefix),
    ));
    let _ = add_lookup_buffer(get_lookup_buffer_bytes(
        vec![(0, b0_trapdoor_bytes)],
        &b0_trapdoor_id_prefix(checkpoint_prefix),
    ));
}

fn load_b0_matrix_checkpoint<M>(
    params: &<M::P as Poly>::Params,
    dir_path: &Path,
    checkpoint_prefix: &str,
) -> Option<M>
where
    M: PolyMatrix,
{
    let bytes = read_bytes_from_multi_batch(dir_path, &b0_id_prefix(checkpoint_prefix), 0)?;
    Some(M::from_compact_bytes(params, &bytes))
}

fn load_b0_checkpoint<M, ST>(
    params: &<M::P as Poly>::Params,
    dir_path: &Path,
    checkpoint_prefix: &str,
) -> Option<(ST::Trapdoor, M)>
where
    M: PolyMatrix,
    ST: PolyTrapdoorSampler<M = M>,
{
    let b0_matrix = load_b0_matrix_checkpoint::<M>(params, dir_path, checkpoint_prefix)?;
    let trapdoor_bytes =
        read_bytes_from_multi_batch(dir_path, &b0_trapdoor_id_prefix(checkpoint_prefix), 0)?;
    let b0_trapdoor = ST::trapdoor_from_bytes(params, &trapdoor_bytes)?;
    Some((b0_trapdoor, b0_matrix))
}

#[cfg(test)]
mod tests {
    use super::{DebugNaiveBGGEncodingVecPltEvaluator, DebugNaiveBGGPublicKeyVecPltEvaluator};
    use crate::{
        __PAIR, __TestState,
        bgg::{
            naive_vec::{NaiveBGGEncodingVec, NaiveBGGPublicKeyVec},
            sampler::{BGGEncodingSampler, BGGPublicKeySampler},
        },
        circuit::PolyCircuit,
        element::PolyElem,
        lookup::PublicLut,
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{
            Poly, PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
        sampler::{
            PolyTrapdoorSampler, hash::DCRTPolyHashSampler, trapdoor::DCRTPolyTrapdoorSampler,
            uniform::DCRTPolyUniformSampler,
        },
        storage::write::{init_storage_system, storage_test_lock, wait_for_all_writes},
        utils::create_bit_random_poly,
    };
    use keccak_asm::Keccak256;
    use std::{fs, path::Path};

    const SIGMA: f64 = 4.578;

    fn lsb_lut(params: &DCRTPolyParams) -> PublicLut<DCRTPoly> {
        PublicLut::new(
            params,
            16,
            |params: &DCRTPolyParams, input| {
                Some((input & 1, <DCRTPoly as Poly>::Elem::constant(&params.modulus(), input & 1)))
            },
            Some((1, <DCRTPoly as Poly>::Elem::constant(&params.modulus(), 1))),
        )
    }

    fn prepare_clean_storage(dir_path: &str) {
        let dir = Path::new(dir_path);
        if dir.exists() {
            fs::remove_dir_all(dir).unwrap();
        }
        fs::create_dir_all(dir).unwrap();
        init_storage_system(dir.to_path_buf());
    }

    #[tokio::test]
    #[sequential_test::sequential]
    async fn test_debug_naive_bgg_vec_lookup_satisfies_output_vector_relation() {
        let _storage_lock = storage_test_lock().await;
        let params = DCRTPolyParams::default();
        let plt = lsb_lut(&params);

        let mut circuit = PolyCircuit::new();
        let input = circuit.input(1).as_single_wire();
        let plt_id = circuit.register_public_lookup(plt);
        let output = circuit.public_lookup_gate(input, plt_id);
        circuit.output(vec![output]);

        let d = 2;
        let num_slots = 2;
        let hash_key = [0x45u8; 32];
        let pubkey_sampler =
            BGGPublicKeySampler::<_, DCRTPolyHashSampler<Keccak256>>::new(hash_key, d);
        let secrets = vec![create_bit_random_poly(&params); d];
        let plaintexts = vec![
            DCRTPoly::from_usize_to_constant(&params, 5),
            DCRTPoly::from_usize_to_constant(&params, 6),
        ];
        let pubkeys = pubkey_sampler.sample(&params, b"debug-naive-lookup", &[true, true]);
        let encoding_sampler =
            BGGEncodingSampler::<DCRTPolyUniformSampler>::new(&params, &secrets, None);
        let encodings = encoding_sampler.sample(&params, &pubkeys, &plaintexts);
        let secret_vec = DCRTPolyMatrix::from_poly_vec_row(&params, secrets);

        let dir_path = "test_data/test_debug_naive_bgg_vec_lookup";
        prepare_clean_storage(dir_path);

        let trapdoor_sampler = DCRTPolyTrapdoorSampler::new(&params, SIGMA);
        let (b0_trapdoor, b0_matrix) = trapdoor_sampler.trapdoor(&params, d);
        let pubkey_evaluator = DebugNaiveBGGPublicKeyVecPltEvaluator::<
            DCRTPolyMatrix,
            DCRTPolyHashSampler<Keccak256>,
            DCRTPolyTrapdoorSampler,
        >::new(hash_key, b0_trapdoor, b0_matrix, dir_path.into());
        let one_pubkey = NaiveBGGPublicKeyVec::new(vec![pubkeys[0].clone(); num_slots]);
        let input_pubkey = NaiveBGGPublicKeyVec::new(pubkeys[1..].to_vec());
        let result_pubkey = circuit.eval(
            &params,
            one_pubkey,
            vec![input_pubkey],
            Some(&pubkey_evaluator),
            None,
            None,
        );
        pubkey_evaluator.sample_aux_matrices(&params);
        wait_for_all_writes(Path::new(dir_path).to_path_buf()).await.unwrap();

        let b0_matrix = pubkey_evaluator
            .load_b0_matrix_checkpoint(&params)
            .expect("debug b0 checkpoint should exist after sample_aux_matrices");
        let c_b0 = secret_vec.clone() * &b0_matrix;
        let encoding_evaluator = DebugNaiveBGGEncodingVecPltEvaluator::<
            DCRTPolyMatrix,
            DCRTPolyHashSampler<Keccak256>,
            DCRTPolyTrapdoorSampler,
        >::new(hash_key, dir_path.into(), c_b0);

        let one_encoding = NaiveBGGEncodingVec::new(vec![encodings[0].clone(); num_slots]);
        let input_encoding = NaiveBGGEncodingVec::new(encodings[1..].to_vec());
        let result_encoding = circuit.eval(
            &params,
            one_encoding,
            vec![input_encoding],
            Some(&encoding_evaluator),
            None,
            None,
        );

        assert_eq!(result_pubkey.len(), 1);
        assert_eq!(result_encoding.len(), 1);
        let gadget = DCRTPolyMatrix::gadget_matrix(&params, d);
        for slot_idx in 0..num_slots {
            let output_encoding = &result_encoding[0].encodings[slot_idx];
            let output_pubkey = &result_pubkey[0].keys[slot_idx];
            assert_eq!(output_encoding.pubkey, output_pubkey.clone());

            let expected_bit = plaintexts[slot_idx].const_coeff_u64() & 1;
            let expected_plaintext =
                DCRTPoly::from_usize_to_constant(&params, expected_bit as usize);
            assert_eq!(
                output_encoding
                    .plaintext
                    .as_ref()
                    .expect("debug lookup output plaintext should be revealed"),
                &expected_plaintext
            );

            let expected_vector = secret_vec.clone() *
                (output_pubkey.matrix.clone() - &(gadget.clone() * expected_plaintext));
            assert_eq!(output_encoding.vector, expected_vector);
        }
    }
}
