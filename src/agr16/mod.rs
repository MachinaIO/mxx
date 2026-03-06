pub mod encoding;
pub mod public_key;
pub mod sampler;

#[cfg(test)]
mod tests {
    use crate::{
        agr16::{
            encoding::Agr16Encoding,
            public_key::Agr16PublicKey,
            sampler::{AGR16EncodingSampler, AGR16PublicKeySampler},
        },
        circuit::{PolyCircuit, gate::GateId},
        lookup::{PltEvaluator, PublicLut},
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{
            Poly,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
        sampler::{hash::DCRTPolyHashSampler, uniform::DCRTPolyUniformSampler},
        utils::{block_size, create_random_poly, create_ternary_random_poly},
    };
    use keccak_asm::Keccak256;
    use std::path::{Path, PathBuf};

    const AUXILIARY_DEPTH: usize = 8;

    struct NoopAgr16PkPlt;

    impl PltEvaluator<Agr16PublicKey<DCRTPolyMatrix>> for NoopAgr16PkPlt {
        fn public_lookup(
            &self,
            _params: &<Agr16PublicKey<DCRTPolyMatrix> as crate::circuit::evaluable::Evaluable>::Params,
            _plt: &PublicLut<
                <Agr16PublicKey<DCRTPolyMatrix> as crate::circuit::evaluable::Evaluable>::P,
            >,
            _one: &Agr16PublicKey<DCRTPolyMatrix>,
            _input: &Agr16PublicKey<DCRTPolyMatrix>,
            _gate_id: GateId,
            _lut_id: usize,
        ) -> Agr16PublicKey<DCRTPolyMatrix> {
            panic!("NoopAgr16PkPlt should not be called in these tests");
        }
    }

    struct NoopAgr16EncPlt;

    impl PltEvaluator<Agr16Encoding<DCRTPolyMatrix>> for NoopAgr16EncPlt {
        fn public_lookup(
            &self,
            _params: &<Agr16Encoding<DCRTPolyMatrix> as crate::circuit::evaluable::Evaluable>::Params,
            _plt: &PublicLut<
                <Agr16Encoding<DCRTPolyMatrix> as crate::circuit::evaluable::Evaluable>::P,
            >,
            _one: &Agr16Encoding<DCRTPolyMatrix>,
            _input: &Agr16Encoding<DCRTPolyMatrix>,
            _gate_id: GateId,
            _lut_id: usize,
        ) -> Agr16Encoding<DCRTPolyMatrix> {
            panic!("NoopAgr16EncPlt should not be called in these tests");
        }
    }

    fn sample_fixture_with_aux_depth_and_reveal_flags(
        input_size: usize,
        auxiliary_depth: usize,
        reveal_plaintexts: Vec<bool>,
        params: &DCRTPolyParams,
    ) -> (
        Vec<Agr16PublicKey<DCRTPolyMatrix>>,
        Vec<Agr16Encoding<DCRTPolyMatrix>>,
        Vec<DCRTPoly>,
        DCRTPoly,
    ) {
        assert_eq!(
            reveal_plaintexts.len(),
            input_size,
            "reveal_plaintexts length must match AGR16 input_size"
        );
        let key: [u8; 32] = rand::random();
        let tag: u64 = rand::random();
        let tag_bytes = tag.to_le_bytes();

        let pubkey_sampler =
            AGR16PublicKeySampler::<_, DCRTPolyHashSampler<Keccak256>>::new(key, auxiliary_depth);
        let pubkeys = pubkey_sampler.sample(params, &tag_bytes, &reveal_plaintexts);

        let secret = create_ternary_random_poly(params);
        let secrets = vec![secret.clone()];
        let plaintexts = (0..input_size).map(|_| create_random_poly(params)).collect::<Vec<_>>();
        let encoding_sampler =
            AGR16EncodingSampler::<DCRTPolyUniformSampler>::new(params, &secrets, None);
        let encodings = encoding_sampler.sample(params, &pubkeys, &plaintexts);

        (pubkeys, encodings, plaintexts, encoding_sampler.secret)
    }

    fn sample_fixture_with_aux_depth(
        input_size: usize,
        auxiliary_depth: usize,
        params: &DCRTPolyParams,
    ) -> (
        Vec<Agr16PublicKey<DCRTPolyMatrix>>,
        Vec<Agr16Encoding<DCRTPolyMatrix>>,
        Vec<DCRTPoly>,
        DCRTPoly,
    ) {
        sample_fixture_with_aux_depth_and_reveal_flags(
            input_size,
            auxiliary_depth,
            vec![true; input_size],
            params,
        )
    }

    fn sample_fixture(
        input_size: usize,
        params: &DCRTPolyParams,
    ) -> (
        Vec<Agr16PublicKey<DCRTPolyMatrix>>,
        Vec<Agr16Encoding<DCRTPolyMatrix>>,
        Vec<DCRTPoly>,
        DCRTPoly,
    ) {
        sample_fixture_with_aux_depth(input_size, AUXILIARY_DEPTH, params)
    }

    fn scalar_matrix(params: &DCRTPolyParams, value: DCRTPoly) -> DCRTPolyMatrix {
        DCRTPolyMatrix::from_poly_vec_row(params, vec![value])
    }

    fn create_temp_test_dir(name: &str) -> PathBuf {
        let mut dir = std::env::temp_dir();
        dir.push(format!("mxx_{name}_{}_{}", std::process::id(), rand::random::<u64>()));
        std::fs::create_dir_all(&dir).expect("Failed to create temporary test directory");
        dir
    }

    fn write_matrix_file(dir_path: &Path, id: &str, matrix: &DCRTPolyMatrix) {
        let (nrow, ncol) = matrix.size();
        let default_bsize = block_size();
        let compact_bsize = default_bsize.min(nrow.max(1)).min(ncol.max(1));
        let entries = matrix.block_entries(0..nrow, 0..ncol);
        let entries_bytes: Vec<Vec<Vec<u8>>> = entries
            .iter()
            .map(|row| row.iter().map(|poly| poly.to_compact_bytes()).collect())
            .collect();
        let bytes = bincode::encode_to_vec(&entries_bytes, bincode::config::standard())
            .expect("Failed to encode matrix bytes");
        let mut path = dir_path.to_path_buf();
        path.push(format!("{}_{}_{}.{}_{}.{}.matrix", id, default_bsize, 0, nrow, 0, ncol));
        std::fs::write(&path, &bytes).expect("Failed to write matrix file");
        if compact_bsize != default_bsize {
            let mut compact_path = dir_path.to_path_buf();
            compact_path
                .push(format!("{}_{}_{}.{}_{}.{}.matrix", id, compact_bsize, 0, nrow, 0, ncol));
            std::fs::write(compact_path, bytes).expect("Failed to write compact matrix file");
        }
    }

    fn assert_primary_auxiliary_invariants(
        encoding: &Agr16Encoding<DCRTPolyMatrix>,
        secret: &DCRTPoly,
    ) {
        assert!(
            !encoding.pubkey.c_times_s_pubkeys.is_empty() &&
                !encoding.c_times_s_encodings.is_empty(),
            "AGR16 encoding must keep at least one recursive c_times_s level"
        );
        let expected_c_times_s = (encoding.pubkey.c_times_s_pubkeys[0].clone() * secret) +
            (encoding.vector.clone() * secret);
        assert_eq!(
            encoding.c_times_s_encodings[0], expected_c_times_s,
            "AGR16 c_times_s invariant must hold"
        );
    }

    fn assert_full_auxiliary_invariants(
        params: &DCRTPolyParams,
        encoding: &Agr16Encoding<DCRTPolyMatrix>,
        secret: &DCRTPoly,
    ) {
        let secret_matrix = scalar_matrix(params, secret.clone());
        assert_eq!(
            encoding.pubkey.c_times_s_pubkeys.len(),
            encoding.c_times_s_encodings.len(),
            "AGR16 c_times_s invariant depth mismatch between key and encoding"
        );
        assert_eq!(
            encoding.pubkey.s_power_pubkeys.len(),
            encoding.s_power_encodings.len(),
            "AGR16 s-power advice depth mismatch between key and encoding"
        );

        let mut current_c_level = encoding.vector.clone();
        for level in 0..encoding.c_times_s_encodings.len() {
            let expected = (encoding.pubkey.c_times_s_pubkeys[level].clone() * secret) +
                (current_c_level.clone() * secret);
            assert_eq!(
                encoding.c_times_s_encodings[level], expected,
                "AGR16 c_times_s recursive invariant must hold at level {level}"
            );
            current_c_level = encoding.c_times_s_encodings[level].clone();
        }

        let mut current_s_level = secret_matrix;
        for level in 0..encoding.s_power_encodings.len() {
            let expected = (encoding.pubkey.s_power_pubkeys[level].clone() * secret) +
                (current_s_level.clone() * secret);
            assert_eq!(
                encoding.s_power_encodings[level], expected,
                "AGR16 s-power recursive invariant must hold at level {level}"
            );
            current_s_level = encoding.s_power_encodings[level].clone();
        }
    }

    fn assert_eval_output_matches_equation_5_1(
        params: &DCRTPolyParams,
        secret: &DCRTPoly,
        pk_out: &Agr16PublicKey<DCRTPolyMatrix>,
        enc_out: &Agr16Encoding<DCRTPolyMatrix>,
        expected_plain: DCRTPoly,
        context: &str,
    ) {
        assert_eq!(enc_out.pubkey, *pk_out);
        let expected_ct = (scalar_matrix(params, secret.clone()) * pk_out.matrix.clone()) +
            scalar_matrix(params, expected_plain.clone());
        assert_eq!(enc_out.vector, expected_ct, "{context}");
        assert_primary_auxiliary_invariants(enc_out, secret);
        assert_eq!(enc_out.plaintext, Some(expected_plain));
    }

    #[test]
    fn test_agr16_sampling_satisfies_equation_5_1_without_error() {
        let params = DCRTPolyParams::default();
        let input_size = 3;
        let (pubkeys, encodings, plaintexts, secret) = sample_fixture(input_size, &params);

        let secret_matrix = scalar_matrix(&params, secret.clone());

        // Slot 0 is the constant-1 encoding.
        let all_plaintexts = [&[DCRTPoly::const_one(&params)], plaintexts.as_slice()].concat();
        for idx in 0..encodings.len() {
            let expected = (secret_matrix.clone() * pubkeys[idx].matrix.clone()) +
                scalar_matrix(&params, all_plaintexts[idx].clone());
            assert_eq!(
                encodings[idx].vector, expected,
                "AGR16 base encoding must satisfy Equation 5.1 with zero injected error"
            );
            assert_full_auxiliary_invariants(&params, &encodings[idx], &secret);
        }
    }

    #[test]
    fn test_agr16_circuit_eval_matches_equation_5_1_without_error() {
        let params = DCRTPolyParams::default();
        let (pubkeys, encodings, plaintexts, secret) = sample_fixture(3, &params);

        // f(x1,x2,x3) = (x1 + x2) * x3 + x1
        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(3);
        let add = circuit.add_gate(inputs[0], inputs[1]);
        let mul = circuit.mul_gate(add, inputs[2]);
        let out = circuit.add_gate(mul, inputs[0]);
        circuit.output(vec![out]);

        let pk_outputs = circuit.eval(
            &params,
            pubkeys[0].clone(),
            vec![pubkeys[1].clone(), pubkeys[2].clone(), pubkeys[3].clone()],
            None::<&NoopAgr16PkPlt>,
        );
        let enc_outputs = circuit.eval(
            &params,
            encodings[0].clone(),
            vec![encodings[1].clone(), encodings[2].clone(), encodings[3].clone()],
            None::<&NoopAgr16EncPlt>,
        );

        let pk_out = &pk_outputs[0];
        let enc_out = &enc_outputs[0];
        let expected_plain = (plaintexts[0].clone() + plaintexts[1].clone()) *
            plaintexts[2].clone() +
            plaintexts[0].clone();
        assert_eval_output_matches_equation_5_1(
            &params,
            &secret,
            pk_out,
            enc_out,
            expected_plain,
            "Evaluated AGR16 ciphertext must satisfy Equation 5.1 when error=0",
        );
    }

    #[test]
    fn test_agr16_nested_multiplication_preserves_equation_5_1_without_error() {
        let params = DCRTPolyParams::default();
        let (pubkeys, encodings, plaintexts, secret) = sample_fixture(3, &params);

        // f(x1,x2,x3) = ((x1 * x2) + x3) * x2
        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(3);
        let mul1 = circuit.mul_gate(inputs[0], inputs[1]);
        let add = circuit.add_gate(mul1, inputs[2]);
        let out = circuit.mul_gate(add, inputs[1]);
        circuit.output(vec![out]);

        let pk_outputs = circuit.eval(
            &params,
            pubkeys[0].clone(),
            vec![pubkeys[1].clone(), pubkeys[2].clone(), pubkeys[3].clone()],
            None::<&NoopAgr16PkPlt>,
        );
        let enc_outputs = circuit.eval(
            &params,
            encodings[0].clone(),
            vec![encodings[1].clone(), encodings[2].clone(), encodings[3].clone()],
            None::<&NoopAgr16EncPlt>,
        );

        let pk_out = &pk_outputs[0];
        let enc_out = &enc_outputs[0];
        let expected_plain = ((plaintexts[0].clone() * plaintexts[1].clone()) +
            plaintexts[2].clone()) *
            plaintexts[1].clone();
        assert_eval_output_matches_equation_5_1(
            &params,
            &secret,
            pk_out,
            enc_out,
            expected_plain,
            "Nested AGR16 multiplication output must satisfy Equation 5.1 when error=0",
        );
    }

    #[test]
    fn test_agr16_depth3_multiplication_preserves_equation_5_1_without_error() {
        let params = DCRTPolyParams::default();
        let (pubkeys, encodings, plaintexts, secret) = sample_fixture(4, &params);

        // f(x1,x2,x3,x4) = (((x1 * x2) * x3) * x4)
        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(4);
        let mul1 = circuit.mul_gate(inputs[0], inputs[1]);
        let mul2 = circuit.mul_gate(mul1, inputs[2]);
        let out = circuit.mul_gate(mul2, inputs[3]);
        circuit.output(vec![out]);

        let pk_outputs = circuit.eval(
            &params,
            pubkeys[0].clone(),
            vec![pubkeys[1].clone(), pubkeys[2].clone(), pubkeys[3].clone(), pubkeys[4].clone()],
            None::<&NoopAgr16PkPlt>,
        );
        let enc_outputs = circuit.eval(
            &params,
            encodings[0].clone(),
            vec![
                encodings[1].clone(),
                encodings[2].clone(),
                encodings[3].clone(),
                encodings[4].clone(),
            ],
            None::<&NoopAgr16EncPlt>,
        );

        let expected_plain = ((plaintexts[0].clone() * plaintexts[1].clone()) *
            plaintexts[2].clone()) *
            plaintexts[3].clone();
        assert_eval_output_matches_equation_5_1(
            &params,
            &secret,
            &pk_outputs[0],
            &enc_outputs[0],
            expected_plain,
            "Depth-3 AGR16 multiplication output must satisfy Equation 5.1 when error=0",
        );
    }

    #[test]
    fn test_agr16_depth4_composed_circuit_preserves_equation_5_1_without_error() {
        let params = DCRTPolyParams::default();
        let (pubkeys, encodings, plaintexts, secret) = sample_fixture(8, &params);

        // f(x1..x8) = ((((x1 * x2) + x3) * (x4 * x5)) * (x6 + x7)) * x8
        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(8);
        let mul12 = circuit.mul_gate(inputs[0], inputs[1]);
        let add123 = circuit.add_gate(mul12, inputs[2]);
        let mul45 = circuit.mul_gate(inputs[3], inputs[4]);
        let mul_left = circuit.mul_gate(add123, mul45);
        let add67 = circuit.add_gate(inputs[5], inputs[6]);
        let mul_deep = circuit.mul_gate(mul_left, add67);
        let out = circuit.mul_gate(mul_deep, inputs[7]);
        circuit.output(vec![out]);

        let pk_outputs = circuit.eval(
            &params,
            pubkeys[0].clone(),
            vec![
                pubkeys[1].clone(),
                pubkeys[2].clone(),
                pubkeys[3].clone(),
                pubkeys[4].clone(),
                pubkeys[5].clone(),
                pubkeys[6].clone(),
                pubkeys[7].clone(),
                pubkeys[8].clone(),
            ],
            None::<&NoopAgr16PkPlt>,
        );
        let enc_outputs = circuit.eval(
            &params,
            encodings[0].clone(),
            vec![
                encodings[1].clone(),
                encodings[2].clone(),
                encodings[3].clone(),
                encodings[4].clone(),
                encodings[5].clone(),
                encodings[6].clone(),
                encodings[7].clone(),
                encodings[8].clone(),
            ],
            None::<&NoopAgr16EncPlt>,
        );

        let expected_plain = ((((plaintexts[0].clone() * plaintexts[1].clone()) +
            plaintexts[2].clone()) *
            (plaintexts[3].clone() * plaintexts[4].clone())) *
            (plaintexts[5].clone() + plaintexts[6].clone())) *
            plaintexts[7].clone();
        assert_eval_output_matches_equation_5_1(
            &params,
            &secret,
            &pk_outputs[0],
            &enc_outputs[0],
            expected_plain,
            "Depth-4 AGR16 composed output must satisfy Equation 5.1 when error=0",
        );
    }

    #[test]
    fn test_agr16_complete_binary_tree_depth3_preserves_equation_5_1_without_error() {
        let params = DCRTPolyParams::default();
        let leaf_count = 8;
        let (pubkeys, encodings, plaintexts, secret) = sample_fixture(leaf_count, &params);

        // Complete binary tree multiplication of depth 3:
        // f(x1..x8) = ((x1*x2)*(x3*x4)) * ((x5*x6)*(x7*x8))
        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(leaf_count);

        let level1 = [
            circuit.mul_gate(inputs[0], inputs[1]),
            circuit.mul_gate(inputs[2], inputs[3]),
            circuit.mul_gate(inputs[4], inputs[5]),
            circuit.mul_gate(inputs[6], inputs[7]),
        ];
        let level2 =
            [circuit.mul_gate(level1[0], level1[1]), circuit.mul_gate(level1[2], level1[3])];
        let out = circuit.mul_gate(level2[0], level2[1]);
        circuit.output(vec![out]);

        let pk_outputs = circuit.eval(
            &params,
            pubkeys[0].clone(),
            vec![
                pubkeys[1].clone(),
                pubkeys[2].clone(),
                pubkeys[3].clone(),
                pubkeys[4].clone(),
                pubkeys[5].clone(),
                pubkeys[6].clone(),
                pubkeys[7].clone(),
                pubkeys[8].clone(),
            ],
            None::<&NoopAgr16PkPlt>,
        );
        let enc_outputs = circuit.eval(
            &params,
            encodings[0].clone(),
            vec![
                encodings[1].clone(),
                encodings[2].clone(),
                encodings[3].clone(),
                encodings[4].clone(),
                encodings[5].clone(),
                encodings[6].clone(),
                encodings[7].clone(),
                encodings[8].clone(),
            ],
            None::<&NoopAgr16EncPlt>,
        );

        let expected_plain = plaintexts.iter().cloned().reduce(|acc, next| acc * next).unwrap();
        assert_eval_output_matches_equation_5_1(
            &params,
            &secret,
            &pk_outputs[0],
            &enc_outputs[0],
            expected_plain,
            "Depth-3 complete binary-tree AGR16 multiplication output must satisfy Equation 5.1 when error=0",
        );
    }

    #[test]
    fn test_agr16_complete_binary_tree_depth_env_probe() {
        let params = DCRTPolyParams::default();
        let depth = std::env::var("MXX_AGR16_TREE_DEPTH")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(3);
        let auxiliary_depth = std::env::var("MXX_AGR16_AUX_DEPTH")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(depth + 1);
        assert!(depth > 0, "MXX_AGR16_TREE_DEPTH must be positive");
        assert!(
            auxiliary_depth >= depth + 1,
            "MXX_AGR16_AUX_DEPTH must satisfy auxiliary_depth >= depth + 1"
        );
        let leaf_count = 1usize << depth;
        let (pubkeys, encodings, plaintexts, secret) =
            sample_fixture_with_aux_depth(leaf_count, auxiliary_depth, &params);

        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(leaf_count);
        let mut level = inputs.clone();
        while level.len() > 1 {
            level = level.chunks_exact(2).map(|pair| circuit.mul_gate(pair[0], pair[1])).collect();
        }
        let out = level[0];
        circuit.output(vec![out]);

        let pk_outputs = circuit.eval(
            &params,
            pubkeys[0].clone(),
            pubkeys.iter().skip(1).cloned().collect(),
            None::<&NoopAgr16PkPlt>,
        );
        let enc_outputs = circuit.eval(
            &params,
            encodings[0].clone(),
            encodings.iter().skip(1).cloned().collect(),
            None::<&NoopAgr16EncPlt>,
        );

        let expected_plain = plaintexts.iter().cloned().reduce(|acc, next| acc * next).unwrap();
        assert_eval_output_matches_equation_5_1(
            &params,
            &secret,
            &pk_outputs[0],
            &enc_outputs[0],
            expected_plain,
            "Env-probe complete binary-tree AGR16 multiplication output must satisfy Equation 5.1 when error=0",
        );
    }

    #[test]
    fn test_agr16_mul_eval_works_without_revealed_plaintexts() {
        let params = DCRTPolyParams::default();
        let input_size = 3;
        let (pubkeys, encodings, plaintexts, secret) =
            sample_fixture_with_aux_depth_and_reveal_flags(
                input_size,
                AUXILIARY_DEPTH,
                vec![false; input_size],
                &params,
            );

        assert!(
            pubkeys.iter().skip(1).all(|pk| !pk.reveal_plaintext),
            "AGR16 fixture must hide user-input plaintexts in this test"
        );
        assert!(
            encodings.iter().skip(1).all(|ct| ct.plaintext.is_none()),
            "Sampled AGR16 encodings must hide user-input plaintexts in this test"
        );

        // f(x1,x2,x3) = (x1 * x2) * x3
        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(input_size);
        let mul12 = circuit.mul_gate(inputs[0], inputs[1]);
        let out = circuit.mul_gate(mul12, inputs[2]);
        circuit.output(vec![out]);

        let pk_outputs = circuit.eval(
            &params,
            pubkeys[0].clone(),
            vec![pubkeys[1].clone(), pubkeys[2].clone(), pubkeys[3].clone()],
            None::<&NoopAgr16PkPlt>,
        );
        let enc_outputs = circuit.eval(
            &params,
            encodings[0].clone(),
            vec![encodings[1].clone(), encodings[2].clone(), encodings[3].clone()],
            None::<&NoopAgr16EncPlt>,
        );

        let expected_plain =
            (plaintexts[0].clone() * plaintexts[1].clone()) * plaintexts[2].clone();
        let expected_ct = (scalar_matrix(&params, secret.clone()) * pk_outputs[0].matrix.clone()) +
            scalar_matrix(&params, expected_plain);
        assert_eq!(enc_outputs[0].pubkey, pk_outputs[0]);
        assert_eq!(
            enc_outputs[0].vector, expected_ct,
            "AGR16 hidden-plaintext multiplication output must satisfy Equation 5.1 when error=0"
        );
        assert_primary_auxiliary_invariants(&enc_outputs[0], &secret);
        assert_eq!(
            enc_outputs[0].plaintext, None,
            "AGR16 public evaluation should not require or reveal plaintext for hidden inputs"
        );
    }

    #[test]
    fn test_agr16_pubkey_read_from_files_supports_recursive_depth() {
        let params = DCRTPolyParams::default();
        let dir = create_temp_test_dir("agr16_recursive_read");
        let id = "pk_recursive";
        let nrow = 1;
        let ncol = 1;
        let recursive_depth = 4;

        let matrix = scalar_matrix(&params, create_random_poly(&params));
        write_matrix_file(&dir, &format!("{id}_matrix"), &matrix);

        let c_times_s_pubkeys: Vec<DCRTPolyMatrix> = (0..recursive_depth)
            .map(|level| {
                let level_matrix = scalar_matrix(&params, create_random_poly(&params));
                write_matrix_file(&dir, &format!("{id}_cts_pk_{level}"), &level_matrix);
                level_matrix
            })
            .collect();
        let s_power_pubkeys: Vec<DCRTPolyMatrix> = (0..recursive_depth)
            .map(|level| {
                let level_matrix = scalar_matrix(&params, create_random_poly(&params));
                write_matrix_file(&dir, &format!("{id}_s_power_pk_{level}"), &level_matrix);
                level_matrix
            })
            .collect();

        let loaded = Agr16PublicKey::<DCRTPolyMatrix>::read_from_files(
            &params, nrow, ncol, &dir, id, 4, true,
        );
        let expected = Agr16PublicKey::new(matrix, c_times_s_pubkeys, s_power_pubkeys, true);
        assert_eq!(loaded, expected);

        std::fs::remove_dir_all(&dir).expect("Failed to cleanup temporary test directory");
    }

    #[test]
    fn test_agr16_pubkey_read_from_files_supports_legacy_two_level_names() {
        let params = DCRTPolyParams::default();
        let dir = create_temp_test_dir("agr16_legacy_read");
        let id = "pk_legacy";
        let nrow = 1;
        let ncol = 1;

        let matrix = scalar_matrix(&params, create_random_poly(&params));
        write_matrix_file(&dir, &format!("{id}_matrix"), &matrix);

        let cts_pk = scalar_matrix(&params, create_random_poly(&params));
        let ctss_pk = scalar_matrix(&params, create_random_poly(&params));
        write_matrix_file(&dir, &format!("{id}_cts_pk"), &cts_pk);
        write_matrix_file(&dir, &format!("{id}_ctss_pk"), &ctss_pk);

        let s2_pk = scalar_matrix(&params, create_random_poly(&params));
        let s2s_pk = scalar_matrix(&params, create_random_poly(&params));
        write_matrix_file(&dir, &format!("{id}_s2_pk"), &s2_pk);
        write_matrix_file(&dir, &format!("{id}_s2s_pk"), &s2s_pk);

        let loaded = Agr16PublicKey::<DCRTPolyMatrix>::read_from_files(
            &params, nrow, ncol, &dir, id, 2, false,
        );
        let expected =
            Agr16PublicKey::new(matrix, vec![cts_pk, ctss_pk], vec![s2_pk, s2s_pk], false);
        assert_eq!(loaded, expected);

        std::fs::remove_dir_all(&dir).expect("Failed to cleanup temporary test directory");
    }

    #[test]
    #[should_panic(expected = "AGR16EncodingSampler::new requires at least one secret polynomial")]
    fn test_agr16_sampler_rejects_empty_secret_input() {
        let params = DCRTPolyParams::default();
        let empty_secrets: Vec<DCRTPoly> = Vec::new();
        let _ = AGR16EncodingSampler::<DCRTPolyUniformSampler>::new(&params, &empty_secrets, None);
    }
}
