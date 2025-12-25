use crate::{
    bgg::{encoding::BggEncoding, public_key::BggPublicKey},
    circuit::{evaluable::Evaluable, gate::GateId},
    lookup::{PltEvaluator, PublicLut},
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
    sampler::{DistType, PolyHashSampler, PolyTrapdoorSampler},
    storage::{
        read::read_matrix_from_multi_batch,
        write::{BatchLookupBuffer, add_lookup_buffer, get_lookup_buffer},
    },
    utils::timed_read,
};
use rayon::prelude::*;
use std::{marker::PhantomData, path::PathBuf, sync::Arc};
use tracing::{debug, info};

#[derive(Debug)]
pub struct LWEBGGPubKeyPltEvaluator<M, SH, ST>
where
    M: PolyMatrix,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    ST: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    pub hash_key: [u8; 32],
    pub trap_sampler: ST,
    pub pub_matrix: Arc<M>,
    pub trapdoor: Arc<ST::Trapdoor>,
    pub dir_path: PathBuf,
    _sh: PhantomData<SH>,
    _st: PhantomData<ST>,
}

impl<M, SH, ST> PltEvaluator<BggPublicKey<M>> for LWEBGGPubKeyPltEvaluator<M, SH, ST>
where
    M: PolyMatrix + Send + 'static,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    ST: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    fn public_lookup(
        &self,
        params: &<BggPublicKey<M> as Evaluable>::Params,
        plt: &PublicLut<<BggPublicKey<M> as Evaluable>::P>,
        _: BggPublicKey<M>,
        input: BggPublicKey<M>,
        gate_id: GateId,
        _: usize,
    ) -> BggPublicKey<M> {
        let row_size = input.matrix.row_size();
        let a_lt = derive_a_lt_matrix::<M, SH>(params, row_size, self.hash_key, gate_id);
        let buffer = preimage_all::<M, ST, _>(
            plt,
            params,
            &self.trap_sampler,
            &self.pub_matrix,
            &self.trapdoor,
            &input.matrix,
            &a_lt,
            &gate_id,
        );
        add_lookup_buffer(buffer);
        BggPublicKey { matrix: a_lt, reveal_plaintext: true }
    }
}

impl<M, SH, ST> LWEBGGPubKeyPltEvaluator<M, SH, ST>
where
    M: PolyMatrix,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    ST: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    pub fn new(
        hash_key: [u8; 32],
        trap_sampler: ST,
        pub_matrix: std::sync::Arc<M>,
        trapdoor: std::sync::Arc<ST::Trapdoor>,
        dir_path: PathBuf,
    ) -> Self {
        Self {
            hash_key,
            trap_sampler,
            pub_matrix,
            trapdoor,
            dir_path,
            _sh: PhantomData,
            _st: PhantomData,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LWEBGGEncodingPltEvaluator<M, SH>
where
    M: PolyMatrix,
    SH: PolyHashSampler<[u8; 32], M = M>,
{
    pub hash_key: [u8; 32],
    pub dir_path: PathBuf,
    pub c_b: M, // c_b = s*B + e
    _marker: PhantomData<SH>,
}

impl<M, SH> PltEvaluator<BggEncoding<M>> for LWEBGGEncodingPltEvaluator<M, SH>
where
    M: PolyMatrix,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
{
    fn public_lookup(
        &self,
        params: &<BggEncoding<M> as Evaluable>::Params,
        plt: &PublicLut<<BggEncoding<M> as Evaluable>::P>,
        _: BggEncoding<M>,
        input: BggEncoding<M>,
        gate_id: GateId,
        _: usize,
    ) -> BggEncoding<M> {
        let z = &input.plaintext.expect("the BGG encoding should revealed plaintext");
        let z_coeff = z
            .coeffs()
            .first()
            .cloned()
            .expect("z polynomial must contain at least one coefficient");
        debug!("public lookup length is {}", plt.len());
        let (k, y_k_coeff) = plt
            .get(params, &z_coeff)
            .unwrap_or_else(|| panic!("{:?} is not exist in public lookup f", z.to_const_int()));
        debug!("Performing public lookup, k={k}");
        let row_size = input.pubkey.matrix.row_size();
        let a_lt = derive_a_lt_matrix::<M, SH>(params, row_size, self.hash_key, gate_id);
        let pubkey = BggPublicKey::new(a_lt, true);
        let l_k = timed_read(
            &format!("L_{gate_id}_{k}"),
            || {
                read_matrix_from_multi_batch::<M>(
                    params,
                    &self.dir_path,
                    &format!("L_{gate_id}"),
                    k,
                )
                .unwrap_or_else(|| panic!("Matrix with index {} not found in batch", k))
            },
            &mut std::time::Duration::default(),
        );
        let concat = self.c_b.clone().concat_columns(&[&input.vector]);
        let vector = concat * l_k;
        let y_k = <M::P as Poly>::from_elem_to_constant(params, &y_k_coeff);
        BggEncoding::new(vector, pubkey, Some(y_k))
    }
}

impl<M, SH> LWEBGGEncodingPltEvaluator<M, SH>
where
    M: PolyMatrix,
    SH: PolyHashSampler<[u8; 32], M = M>,
{
    pub fn new(hash_key: [u8; 32], dir_path: PathBuf, c_b: M) -> Self {
        Self { hash_key, dir_path, c_b, _marker: PhantomData }
    }
}

fn derive_a_lt_matrix<M, SH>(
    params: &<M::P as Poly>::Params,
    row_size: usize,
    hash_key: [u8; 32],
    id: GateId,
) -> M
where
    M: PolyMatrix,
    SH: PolyHashSampler<[u8; 32], M = M>,
{
    let m = row_size * params.modulus_digits();
    let hash_sampler = SH::new();
    let tag = format!("A_LT_{id}");
    hash_sampler.sample_hash(params, hash_key, tag.into_bytes(), row_size, m, DistType::FinRingDist)
}

fn preimage_all<M, ST, P>(
    plt: &PublicLut<P>,
    params: &<M::P as Poly>::Params,
    trap_sampler: &ST,
    pub_matrix: &M,
    trapdoor: &ST::Trapdoor,
    a_z: &M,
    a_lt: &M,
    id: &GateId,
) -> BatchLookupBuffer
where
    P: Poly,
    M: PolyMatrix<P = P> + Send + 'static,
    ST: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    let row_size = pub_matrix.row_size();
    let gadget = M::gadget_matrix(params, row_size);
    let items: Vec<_> = plt.f.iter().collect();
    info!("start collecting preimages {}", id);
    let preimages = items
        .par_chunks(4)
        .flat_map(|batch| {
            batch
                .iter()
                .map(|(x_k, (k, y_k))| {
                    let x_k = P::from_elem_to_constant(params, x_k);
                    let y_k = P::from_elem_to_constant(params, y_k);
                    let ext_matrix = a_z.clone() - &(gadget.clone() * x_k);
                    let target = a_lt.clone() - &(gadget.clone() * y_k);
                    (
                        *k,
                        trap_sampler.preimage_extend(
                            params,
                            trapdoor,
                            pub_matrix,
                            &ext_matrix,
                            &target,
                        ),
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    info!("finish collecting preimages {}", id);
    get_lookup_buffer(preimages, &format!("L_{id}"))
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        bgg::sampler::{BGGEncodingSampler, BGGPublicKeySampler},
        circuit::PolyCircuit,
        element::finite_ring::FinRingElem,
        lookup::lwe_eval::LWEBGGEncodingPltEvaluator,
        matrix::dcrt_poly::DCRTPolyMatrix,
        poly::dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        sampler::{
            hash::DCRTPolyHashSampler, trapdoor::DCRTPolyTrapdoorSampler,
            uniform::DCRTPolyUniformSampler,
        },
        storage::write::{init_storage_system, wait_for_all_writes},
        utils::create_bit_random_poly,
    };
    use keccak_asm::Keccak256;
    use std::{collections::HashMap, fs, path::Path};
    use tokio;

    fn setup_lsb_constant_binary_plt(t_n: usize, params: &DCRTPolyParams) -> PublicLut<DCRTPoly> {
        let mut f = HashMap::new();
        let modulus = params.modulus();
        for k in 0..t_n {
            let const_elem = FinRingElem::from_u64(k as u64, modulus.clone());
            let lsb_elem = FinRingElem::from_u64((k & 1) as u64, modulus.clone());
            f.insert(const_elem, (k, lsb_elem));
        }
        PublicLut::<DCRTPoly>::new(f)
    }

    const SIGMA: f64 = 4.578;

    #[tokio::test]
    async fn test_lwe_plt_eval() {
        init_storage_system();
        let _ = tracing_subscriber::fmt::try_init();
        let params = DCRTPolyParams::default();
        let plt = setup_lsb_constant_binary_plt(16, &params);
        // Create a simple circuit with the lookup table
        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(1);
        let plt_id = circuit.register_public_lookup(plt.clone());
        let output = circuit.public_lookup_gate(inputs[0], plt_id);
        circuit.output(vec![output]);

        let d = 3;
        let input_size = 1;
        let key: [u8; 32] = rand::random();
        let bgg_pubkey_sampler =
            BGGPublicKeySampler::<_, DCRTPolyHashSampler<Keccak256>>::new(key, d);

        let tag: u64 = rand::random();
        let tag_bytes = tag.to_le_bytes();
        // Create secret and plaintexts
        let secrets = vec![create_bit_random_poly(&params); d];
        let rand_int = (rand::random::<u64>() % 16) as usize;
        let plaintexts = vec![DCRTPoly::from_usize_to_constant(&params, rand_int); input_size];

        // Create random public keys and encodings
        let reveal_plaintexts = vec![true; input_size];
        let bgg_encoding_sampler =
            BGGEncodingSampler::<DCRTPolyUniformSampler>::new(&params, &secrets, None);
        let pubkeys = bgg_pubkey_sampler.sample(&params, &tag_bytes, &reveal_plaintexts);
        let encodings = bgg_encoding_sampler.sample(&params, &pubkeys, &plaintexts);
        let enc_one = encodings[0].clone();
        let enc1 = encodings[1].clone();

        let trapdoor_sampler = DCRTPolyTrapdoorSampler::new(&params, SIGMA);
        let (b_trapdoor, b) = trapdoor_sampler.trapdoor(&params, d);
        let s_vec = DCRTPolyMatrix::from_poly_vec_row(&params, secrets);
        let c_b = s_vec.clone() * &b;

        // Create a public key evaluator
        let dir_path = "test_data/test_lwe_plt_eval";
        let dir = Path::new(&dir_path);
        if !dir.exists() {
            fs::create_dir(dir).unwrap();
        } else {
            // Clean it first to ensure no old files interfere
            fs::remove_dir_all(dir).unwrap();
            fs::create_dir(dir).unwrap();
        }
        let plt_pubkey_evaluator =
            LWEBGGPubKeyPltEvaluator::<DCRTPolyMatrix, DCRTPolyHashSampler<Keccak256>, _>::new(
                key,
                trapdoor_sampler,
                Arc::new(b),
                Arc::new(b_trapdoor),
                dir_path.into(),
            );
        let result_pubkey = circuit.eval(
            &params,
            &enc_one.pubkey,
            std::slice::from_ref(&enc1.pubkey),
            Some(plt_pubkey_evaluator),
        );
        wait_for_all_writes(dir.to_path_buf()).await.unwrap();
        assert_eq!(result_pubkey.len(), 1);
        let result_pubkey = &result_pubkey[0];

        //Create an encoding evaluator
        let plt_encoding_evaluator = LWEBGGEncodingPltEvaluator::<
            DCRTPolyMatrix,
            DCRTPolyHashSampler<Keccak256>,
        >::new(key, dir_path.into(), c_b);

        // Evaluate the circuit
        let result_encoding = circuit.eval(
            &params,
            &enc_one.clone(),
            std::slice::from_ref(&enc1),
            Some(plt_encoding_evaluator),
        );
        assert_eq!(result_encoding.len(), 1);
        let result_encoding = &result_encoding[0];
        assert_eq!(result_encoding.pubkey, result_pubkey.clone());
        let plaintext_const_coeff = plaintexts[0]
            .coeffs()
            .first()
            .cloned()
            .expect("plaintext poly must have at least one coefficient");
        let expected_plaintext_coeff = plt.get(&params, &plaintext_const_coeff).unwrap().1;
        let expected_plaintext =
            DCRTPoly::from_elem_to_constant(&params, &expected_plaintext_coeff);
        assert_eq!(result_encoding.plaintext.clone().unwrap(), expected_plaintext);
        let expected_vector = s_vec.clone() *
            (result_encoding.pubkey.matrix.clone() -
                (DCRTPolyMatrix::gadget_matrix(&params, d) * expected_plaintext));
        assert_eq!(result_encoding.vector, expected_vector);
    }
}
