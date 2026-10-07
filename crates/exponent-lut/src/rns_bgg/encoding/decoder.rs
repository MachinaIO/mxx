//! The encryption and online evaluation side of the smudged trapdoor decoder (see
//! [`crate::decoder`]).
//!
//! Encryption publishes the unmask hint `b = s B + e_b` for the stage mask
//! `s = (s_bar, 1)` at the decoding level `L`. Online evaluation decodes component
//! 1 of the source output under the column selection `G_(L+1)^-1(1)` plus the
//! smudge sum: after the public `-2^-1` its payload is `Delta_L mu + m`.
use super::*;

/// The online stage's decoder: the summed decode row `c`.
impl Decoder for RnsBggDecoder<'_, EncodingLowering<'_>> {
    /// The lowered smudge sum.
    type Smudge = Vec<RnsBggEncoding>;
    /// The `1 x 1` decode row: component 1 of the source under
    /// `G_(L+1)^-1(1)`, plus component 1 of each smudge sum.
    type Output = Mat;
    type Error = RnsBggError;

    fn finish_decoding(
        &mut self,
        declaration: &DecoderDeclaration,
        smudge: Self::Smudge,
    ) -> Result<Self::Output, Self::Error> {
        let parameters = self.lowerer.parameters;
        let wire = self.source.wire();
        if wire != declaration.source_output() || smudge.len() != 1 {
            return Err(RnsBggError::Decoder(DecoderError::TargetMismatch));
        }
        let level = self.lowerer.workload.level(wire)?;
        let ring = parameters.ring(level)?;
        let plaintext = level.checked_add(1).ok_or(RnsBggError::Parameters("level overflow"))?;
        let selection = parameters.gadget_inverse(
            plaintext,
            level,
            parameters.ring(plaintext)?.polynomial([IntExpr::constant(1)]),
        )?;
        let mut row = self.source.row.1.clone().mul_small_rhs(selection);
        for member in smudge {
            row = row + checked_modulus_convert(member.row.1, &ring)?;
        }
        Ok(row)
    }
}

impl RnsEncryptBuild<'_> {
    /// The unmask hint `b = s B + e_b` of `declaration`'s decoder, for the
    /// stage mask `s` at level `level` of `parameters`, the decoding ring,
    /// where `B` is the public matrix preprocessing published with the
    /// preimage.
    pub(super) fn unmask_hint(
        &self,
        declaration: &DecoderDeclaration,
        parameters: &RnsBggParameters,
        level: usize,
    ) -> Result<Mat, ApiError> {
        let production = self
            .preprocess
            .as_ref()
            .ok_or_else(|| ApiError::input("decoder preprocess is not bound"))?;
        // The bound mask is the stage mask: in Dynamic mode, the payload secret.
        let mask = self
            .mask_secret
            .as_ref()
            .ok_or_else(|| ApiError::input("the stage mask secret is not bound"))?;
        let ring = parameters.ring(level).map_err(ApiError::scheme)?;
        let columns = crate::decoder::unmask::UNMASK_ROWS *
            (parameters.level(level).map_err(ApiError::scheme)?.modulus_digits() + 2);
        let public = ring.artifact_input(
            production.clone(),
            decoder_artifact_name(*declaration.commitment(), "unmask-public"),
            (crate::decoder::unmask::UNMASK_ROWS, columns),
            ArtifactAvailability::Transferred,
        );
        let secret = reduce_secret(parameters, &mask.row, level).map_err(ApiError::scheme)?;
        let secret = Mat::concat(mxx_ir_core::node::ConcatAxis::Columns, vec![secret.0, secret.1]);
        let error = encoding_error_row(parameters, &ring, columns).map_err(ApiError::scheme)?.0;
        Ok(crate::decoder::unmask::unmask_hint(secret, public, error))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        api::PlaintextInputs,
        rns_bgg::{RnsBggNoiseBindings, RnsBggNoiseValue, RnsOnlineEvalBuild, RnsPreprocessBuild},
    };
    use mxx_backends::{
        MemoryArtifactStore, authority::CpuExecution, backend::poly::cpu_backend,
        poly::dcrt::poly::DCRTPoly,
    };

    fn read(name: &str, default: usize) -> usize {
        std::env::var(format!("MXX_EXPONENT_LUT_DECODER_TEST_{name}"))
            .map_or(default, |value| value.parse().unwrap())
    }

    /// The identity Program on one private input, and that input.
    fn identity_source() -> (ExponentLutProgram, WireId) {
        let mut builder = ExponentLutProgramBuilder::new();
        let input = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        (builder.finish(vec![Operand::Scalar(input)]).unwrap(), input)
    }

    /// The smudge PRF profile of the round trip: `n' = 8` and `P =
    /// MXX_EXPONENT_LUT_DECODER_TEST_P` (default 16) in the ring of dimension
    /// eight, over an LWR dimension of four.
    fn round_trip_profile() -> crate::program_gadgets::prf::SparseLwrPrfProfile {
        crate::program_gadgets::prf::SparseLwrPrfProfile::new(
            crate::ExpLayout { stride_log2: 0, width_log2: 3 },
            read("P", 16),
            4,
        )
        .unwrap()
    }

    /// The decoder of the identity Program chosen by
    /// [`crate::utils::select_decoder_at_crt_depth`] on a ring of dimension
    /// eight with 30-bit primes and base `2^15`, over increasing candidates
    /// (`primes_per_drop` up to `MAX_PRIMES_PER_DROP`, default 4, and
    /// `max_crt_depth` up to `MAX_PRIMES`, default 40), hiding
    /// `STATISTICAL_BITS` (default 1) bits (all
    /// `MXX_EXPONENT_LUT_DECODER_TEST_*`). Returns the selection and the
    /// smudge keys of its members.
    fn selection(
        mode: RhsMode,
        source: &ExponentLutProgram,
        input: WireId,
    ) -> (crate::utils::DecoderSelection, Vec<crate::program_gadgets::pbc::PbcGeneratedKeyLayout>)
    {
        let generated = std::cell::RefCell::new(BTreeMap::new());
        let layouts = |members: usize| {
            let keys = crate::decoder::decoding::fixture::keys(members);
            let layouts = keys.iter().map(|key| key.public_layout.clone()).collect();
            generated.borrow_mut().insert(members, keys);
            Ok(layouts)
        };
        let decoder = crate::utils::DecoderRequest {
            output: input,
            profile: round_trip_profile(),
            layouts: &layouts,
            statistical_bits: read("STATISTICAL_BITS", 1),
            unmask_sigma: 4.578,
        };
        let one = [BigInt::from(1u8)];
        let bindings = RnsBggNoiseBindings::from_inputs(
            source,
            &[RnsBggNoiseValue::sampled_encoding_sigma(&1.into(), b"decoder-source", &one)
                .unwrap()],
            &[],
        )
        .unwrap();
        let mut rejected = Vec::new();
        for primes_per_drop in 1..=read("MAX_PRIMES_PER_DROP", 4) {
            for max_crt_depth in 2..=read("MAX_PRIMES", 40) {
                let request = crate::rns_bgg::RnsBggPlanRequest {
                    ring_dimension: 8,
                    encoding_error_sigma: 1.into(),
                    crt_bits: 30,
                    base_bits: 15,
                    terminal_depth: 1,
                    max_crt_depth,
                    terminal_value_bound: BigUint::from(1u8),
                    operations_per_drop: 1,
                    primes_per_drop,
                    rhs_mode: mode,
                };
                match crate::utils::select_decoder_at_crt_depth(
                    &request,
                    source,
                    &bindings,
                    &[BigUint::from(1u8)],
                    None,
                    &decoder,
                ) {
                    Ok(selection) => {
                        let members = selection.declaration.smudge().members();
                        let keys = generated.borrow_mut().remove(&members).unwrap();
                        return (selection, keys);
                    }
                    Err(error) => rejected.push((primes_per_drop, max_crt_depth, error)),
                }
            }
        }
        panic!("no decoder candidate: {rejected:?}");
    }

    /// The selector picks the smallest digit count whose smudge hides the
    /// simulated noise and fits the decoding margin, on a smudge chain that
    /// descends to the decoding level.
    #[test]
    fn decoder_selection_hides_and_fits_the_simulated_noise() {
        let (source, input) = identity_source();
        let (selection, keys) = selection(RhsMode::Static, &source, input);
        let noise = &selection.noise;
        let bits = read("STATISTICAL_BITS", 1);
        assert!(noise.fits() && noise.hides(bits), "{noise:?}");
        let digits = selection.declaration.digits();
        assert_eq!(keys.len(), 8 * digits);
        let smudge = selection.smudge_parameters.levels();
        assert!(smudge.windows(2).all(|pair| pair[0].to_crt().0.len() > pair[1].to_crt().0.len()));
    }

    /// Every artifact import a parallel loop of `graph` captures, with its
    /// loop input mode.
    fn loop_imports(graph: &mxx_ir_core::Graph) -> Vec<(String, mxx_ir_core::node::LoopInputMode)> {
        let mut imports = Vec::new();
        for scope in graph.scopes().values() {
            for node in scope.nodes() {
                let mxx_ir_core::node::NodeKind::ParallelLoop(control) = node.kind() else {
                    continue;
                };
                for (argument, mode) in node.arguments().iter().zip(&control.input_modes) {
                    if let mxx_ir_core::node::NodeKind::Input { name, artifact: Some(_), .. } =
                        argument.node().kind()
                    {
                        imports.push((name.clone(), *mode));
                    }
                }
            }
        }
        imports
    }

    /// Preprocessing, encryption and online evaluation of the decoder
    /// of the identity Program on a private input `X^b`, in Static and
    /// Dynamic mode. The simulated noise must fit the decoding margin, and
    /// `round((p/q) x) mod p` of the decoded `x = (q/p) X^b + m + E` must be
    /// the plaintext `X^b` modulo the plaintext level.
    #[test]
    #[ignore = "the CPU round trip of both modes takes about five hours"]
    fn decoder_round_trip_returns_the_source_plaintext() {
        for mode in [RhsMode::Static, RhsMode::Dynamic] {
            let dynamic = mode == RhsMode::Dynamic;
            let (source, input) = identity_source();
            let (selection, keys) = selection(mode, &source, input);
            let crate::utils::DecoderSelection {
                parameters,
                smudge_parameters,
                declaration,
                noise,
                ..
            } = selection;
            assert!(
                noise.fits() && noise.hides(read("STATISTICAL_BITS", 1)),
                "{mode:?}: error 2^{}, smudge 2^{}, spacing 2^{}",
                noise.error.bits(),
                noise.smudge.bits(),
                noise.spacing.bits()
            );
            let smudge_program = declaration.smudge().program();
            let level = parameters.levels()[0].clone();
            let smudge_level = smudge_parameters.levels()[0].clone();
            let selectors = declaration
                .smudge()
                .prf()
                .selector_plaintexts::<DCRTPoly>(&smudge_level, &keys)
                .unwrap();
            let a = (0..4).map(|_| rand::random::<u32>() as usize % 8).collect::<Vec<_>>();
            let smudge_lhs =
                declaration.smudge().prf().plaintext_inputs::<DCRTPoly>(&smudge_level, &a).unwrap();

            let mut execution = CpuExecution::new(cpu_backend(
                parameters.levels().iter().chain(smudge_parameters.levels()).cloned(),
            ));
            let mut store = MemoryArtifactStore::default();
            let hash_key: [u8; 32] = rand::random();
            let context: [u8; 32] = rand::random();
            let env = ParamEnv::default();
            let sample_payload = |execution: &mut CpuExecution, store: &mut MemoryArtifactStore| {
                RnsEncryptBuild::new(&parameters, context)
                    .sample_payload_secret(execution, store, |_| Ok(()))
                    .unwrap()
            };
            let static_payload = (!dynamic).then(|| sample_payload(&mut execution, &mut store));

            let no_rhs = PlaintextInputs::<RhsId, DCRTPoly>::new();
            let mut smudge_preprocess = RnsPreprocessBuild::new(&smudge_parameters, context);
            smudge_preprocess
                .bind_rhs_inputs(
                    smudge_program,
                    &env,
                    if dynamic { &no_rhs } else { &selectors },
                    static_payload
                        .as_ref()
                        .map(|payload| payload.at_level(&smudge_parameters, 0).unwrap())
                        .as_ref(),
                    &hash_key,
                )
                .unwrap();
            smudge_preprocess.bind_decoder_smudge(&declaration).unwrap();
            let (preprocess, online_eval) = crate::api::preprocess(
                &source,
                &env,
                &no_rhs,
                static_payload.as_ref(),
                &hash_key,
                RnsPreprocessBuild::new(&parameters, context),
                &mut execution,
                &mut store,
                |build, outputs| {
                    build.apply_decoder(
                        0,
                        &declaration,
                        outputs[0].clone(),
                        smudge_preprocess,
                        &"4.578".parse().unwrap(),
                    )?;
                    Ok(outputs)
                },
                |_| Ok(()),
            )
            .unwrap();

            let payload =
                static_payload.unwrap_or_else(|| sample_payload(&mut execution, &mut store));
            let mask = (!dynamic).then(|| {
                RnsEncryptBuild::new(&parameters, rand::random())
                    .sample_mask_secret(&mut execution, &mut store, |_| Ok(()))
                    .unwrap()
            });
            let batch: [u8; 32] = rand::random();
            let mut smudge_encrypt = RnsEncryptBuild::new(&smudge_parameters, context);
            smudge_encrypt
                .bind_lhs_inputs(
                    &smudge_program.encryption_inputs(),
                    &env,
                    &smudge_lhs,
                    &payload.at_level(&smudge_parameters, 0).unwrap(),
                    mask.as_ref()
                        .map(|mask| mask.at_level(&smudge_parameters, 0).unwrap())
                        .as_ref(),
                    &hash_key,
                )
                .unwrap();
            smudge_encrypt.bind_decoder_smudge(&declaration).unwrap();
            smudge_encrypt.bind_preprocess_artifacts(&preprocess).unwrap();
            if dynamic {
                smudge_encrypt.bind_rhs_inputs(&env, &selectors, batch).unwrap();
            }
            let bit = rand::random::<u32>() as usize % 8;
            let lhs = BTreeMap::from([(
                input,
                InputBinding::Scalar(DCRTPoly::const_rotate_poly(&level, bit)),
            )]);
            let encrypted = crate::api::encrypt(
                &source.encryption_inputs(),
                &env,
                &lhs,
                &preprocess,
                &payload,
                mask.as_ref(),
                &hash_key,
                RnsEncryptBuild::new(&parameters, context),
                &mut execution,
                &mut store,
                |build| {
                    if dynamic {
                        build.bind_rhs_inputs(&env, &no_rhs, batch)?;
                    }
                    build.bind_decoder(&declaration, smudge_encrypt)
                },
                |validated| {
                    // The Dynamic companion loop reads the prelude's family
                    // ciphertexts through its index family, ahead of use.
                    let ciphertexts = loop_imports(&validated.source)
                        .into_iter()
                        .filter(|(name, _)| name.contains("/family/ciphertext."))
                        .collect::<Vec<_>>();
                    assert_eq!(!ciphertexts.is_empty(), dynamic, "{ciphertexts:?}");
                    assert!(
                        ciphertexts.iter().all(|(_, mode)| matches!(
                            mode,
                            mxx_ir_core::node::LoopInputMode::Gather { .. }
                        )),
                        "{ciphertexts:?}"
                    );
                    Ok(())
                },
            )
            .unwrap();

            let mut smudge_online = RnsOnlineEvalBuild::new(&smudge_parameters, context);
            smudge_online
                .bind_decoder_smudge(&declaration, &online_eval, &encrypted, &hash_key)
                .unwrap();
            let result = crate::api::online_eval(
                &source,
                &env,
                &online_eval,
                &encrypted,
                &hash_key,
                RnsOnlineEvalBuild::new(&parameters, context),
                &mut execution,
                &mut store,
                |build, outputs| {
                    build.apply_decoder(0, &declaration, smudge_online)?;
                    Ok(outputs)
                },
                |validated| {
                    // Every smudge selector member is read at the bare root
                    // binder, so the wave importer reads it ahead. A Dynamic
                    // RHS's digit-companion family is one root import shared
                    // by every lane.
                    let selectors = loop_imports(&validated.source)
                        .into_iter()
                        .filter(|(name, _)| {
                            name.contains("/smudge/rhs/") && !name.contains("/digit-encoding.")
                        })
                        .collect::<Vec<_>>();
                    assert!(!selectors.is_empty());
                    assert!(
                        selectors
                            .iter()
                            .all(|(_, mode)| matches!(mode, mxx_ir_core::node::LoopInputMode::Zip)),
                        "{selectors:?}"
                    );
                    Ok(())
                },
            )
            .unwrap();
            let plaintext = <CpuExecution as StageExecution<MemoryArtifactStore>>::download_matrix(
                &execution,
                &result.outputs["output-0"],
            )
            .unwrap();
            // Final check: `round((p/q) x) mod p` recovers the payload.
            let expected = DCRTPoly::const_rotate_poly(&parameters.levels()[1], bit);
            assert_eq!(plaintext.entry(0, 0), expected, "{mode:?}");
        }
    }
}
