//! The encryption and online evaluation side of the masked trapdoor-decoder
//! refresh (see [`crate::noise_refresh`]).
//!
//! Encryption publishes the decoder row `b = s B + e_b` for the stage mask
//! `s = (s_bar, 1)` at the source level. Online evaluation decodes component
//! 1 of the source output under the column selection `G_(L+1)^-1(1)` plus the
//! two mask family sums: its payload is `Delta_L mu + m`.
use super::*;
use crate::noise_refresh::refresh::MaskFamily;

/// The online stage's refresh: the summed decode row `c`.
impl NoiseRefresh for RnsBggRefresh<'_, EncodingLowering<'_>> {
    /// The two lowered mask family sums.
    type Mask = Vec<RnsBggEncoding>;
    /// The `1 x 1` decode row: component 1 of the source under
    /// `G_(L+1)^-1(1)`, plus component 1 of each mask sum.
    type Output = Mat;
    type Error = RnsBggError;

    fn finish_noise_refresh(
        &mut self,
        declaration: &RefreshDeclaration,
        mask: Self::Mask,
    ) -> Result<Self::Output, Self::Error> {
        let parameters = self.lowerer.parameters;
        let wire = self.source.wire();
        if wire != declaration.source_output() || mask.len() != MaskFamily::ALL.len() {
            return Err(RnsBggError::Refresh(RefreshError::TargetMismatch));
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
        for member in mask {
            row = row + checked_modulus_convert(member.row.1, &ring)?;
        }
        Ok(row)
    }
}

impl RnsEncryptBuild<'_> {
    /// The decoder row `b = s B + e_b` of `declaration`'s refresh, for the
    /// stage mask `s` at the source level, where `B` is the preprocess
    /// republication of the trusted setup's public matrix.
    pub(super) fn refresh_decoder_row(
        &self,
        declaration: &RefreshDeclaration,
        level_parameters: &RnsBggParameters,
    ) -> Result<Mat, ApiError> {
        let production = self
            .preprocess
            .as_ref()
            .ok_or_else(|| ApiError::input("refresh preprocess is not bound"))?;
        // The bound mask is the stage mask: in Dynamic mode, the payload secret.
        let mask = self
            .mask_secret
            .as_ref()
            .ok_or_else(|| ApiError::input("refresh mask secret is not bound"))?;
        // The mask child's chain is the source level repeated: its level 0 is
        // the decoding ring.
        let ring = level_parameters.ring(0).map_err(ApiError::scheme)?;
        let columns = crate::noise_refresh::decoder::DECODER_ROWS *
            (level_parameters.level(0).map_err(ApiError::scheme)?.modulus_digits() + 2);
        let public = ring.artifact_input(
            production.clone(),
            refresh_artifact_name(*declaration.commitment(), "decoder-public"),
            (crate::noise_refresh::decoder::DECODER_ROWS, columns),
            ArtifactAvailability::Transferred,
        );
        let secret = reduce_secret(level_parameters, &mask.row, 0).map_err(ApiError::scheme)?;
        let secret = Mat::concat(mxx_ir_core::node::ConcatAxis::Columns, vec![secret.0, secret.1]);
        let error =
            encoding_error_row(level_parameters, &ring, columns).map_err(ApiError::scheme)?.0;
        Ok(crate::noise_refresh::decoder::decoder_row(secret, public, error))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        api::PlaintextInputs,
        rns_bgg::{
            RnsBggNoiseBindings, RnsBggNoiseRhs, RnsBggNoiseValue, RnsNoiseBuild,
            RnsOnlineEvalBuild, RnsPreprocessBuild, sample_refresh_decoder,
        },
    };
    use mxx_backends::{
        MemoryArtifactStore, authority::CpuExecution, backend::poly::cpu_backend,
        poly::dcrt::poly::DCRTPoly,
    };

    /// A source level of `MXX_EXPONENT_LUT_REFRESH_TEST_SOURCE_PRIMES` (default
    /// five) 30-bit primes over a one-prime plaintext level, ring dimension
    /// eight. The mask chain repeats the source modulus, so its noise is never
    /// divided by a ModDown: the source drop `Delta_0 = 2^120` must exceed
    /// the worst-case bound `2^101` of a two-bucket mask; four primes do not.
    fn parameters(mode: RhsMode) -> RnsBggParameters {
        let read = |name: &str, default: usize| {
            std::env::var(format!("MXX_EXPONENT_LUT_REFRESH_TEST_{name}"))
                .map_or(default, |value| value.parse().unwrap())
        };
        let source = DCRTPolyParams::new(8, read("SOURCE_PRIMES", 5), 30, 15, None, None);
        let plaintext = source.select_modulus(&BigUint::from(source.to_crt().0[0])).unwrap();
        RnsBggParameters::new(vec![source, plaintext], 1, 1).unwrap().with_rhs_mode(mode)
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

    /// The simulated refresh noise of `declaration` on the identity source.
    fn refresh_noise(
        parameters: &RnsBggParameters,
        mask_parameters: &RnsBggParameters,
        source: &ExponentLutProgram,
        declaration: &RefreshDeclaration,
    ) -> crate::rns_bgg::RnsBggRefreshNoise {
        let source_workload = RnsBggDerivedWorkload::from_program(source).unwrap();
        let one = [BigInt::from(1u8)];
        let bindings = RnsBggNoiseBindings::from_inputs(
            source,
            &[RnsBggNoiseValue::sampled_encoding(parameters, b"refresh-source", &one)],
            &[],
        )
        .unwrap();
        let mask_program = declaration.mask().program();
        let mask_workload = RnsBggDerivedWorkload::from_program(mask_program).unwrap();
        let mask_bindings = RnsBggNoiseBindings::from_inputs(
            mask_program,
            &[
                RnsBggNoiseValue::sampled_encoding(mask_parameters, b"mask-start", &one),
                RnsBggNoiseValue::sampled_encoding(mask_parameters, b"mask-a", &one)
                    .independent_family_members(),
            ],
            &vec![
                RnsBggNoiseRhs {
                    payload_l1: BigUint::from(1u8),
                    payload_l2_squared: BigUint::from(1u8)
                };
                mask_program.rhs_inputs().len()
            ],
        )
        .unwrap();
        let mut mask =
            RnsNoiseBuild::new(mask_parameters, mask_program, &mask_workload, &mask_bindings)
                .unwrap();
        mask.bind_refresh_mask(declaration).unwrap();
        let trace = crate::api::noise_simulate(
            source,
            RnsNoiseBuild::new(parameters, source, &source_workload, &bindings).unwrap(),
            |build, outputs| {
                build
                    .apply_refresh(0, declaration, outputs[0].clone(), mask, 4.578)
                    .map_err(ApiError::scheme)?;
                Ok(outputs)
            },
        )
        .unwrap();
        trace.refresh.unwrap()
    }

    /// Setup, preprocessing, encryption and online evaluation of the refresh
    /// of the identity Program on a private input `X^b`, in Static and
    /// Dynamic mode. The simulated noise must fit the decoding margin, and
    /// `round((p/q) x) mod p` of the decoded `x = (q/p) X^b + m + E` must be
    /// the plaintext `X^b` modulo the plaintext level.
    #[test]
    #[ignore = "the CPU round trip takes about three minutes in Static and fifty minutes in Dynamic mode"]
    fn refresh_round_trip_returns_the_source_plaintext() {
        for mode in [RhsMode::Static, RhsMode::Dynamic] {
            let parameters = parameters(mode);
            let dynamic = mode == RhsMode::Dynamic;
            let mut builder = ExponentLutProgramBuilder::new();
            let input = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
            let source = builder.finish(vec![Operand::Scalar(input)]).unwrap();
            let coefficients = parameters.level(0).unwrap().ring_dimension() as usize;
            let (declaration, keys) = crate::noise_refresh::refresh::fixture::declaration(
                &source,
                input,
                coefficients,
                1,
            );
            let profile = crate::noise_refresh::refresh::fixture::profile();
            let mask_parameters =
                parameters.refresh_mask_parameters(&declaration, &source).unwrap();
            let noise = refresh_noise(&parameters, &mask_parameters, &source, &declaration);
            assert!(
                noise.fits(),
                "{mode:?}: error 2^{} and mask {} do not fit spacing 2^{}",
                noise.error.bits(),
                noise.mask,
                noise.spacing.bits()
            );
            let mask_program = declaration.mask().program();
            let level = parameters.levels()[0].clone();
            let selectors = declaration
                .mask()
                .selector_plaintexts::<DCRTPoly>(&level, [&keys[0], &keys[1]])
                .unwrap();
            let a = (0..4).map(|_| rand::random::<u32>() as usize % 8).collect::<Vec<_>>();
            let mask_lhs =
                declaration.mask().plaintext_inputs::<DCRTPoly>(&profile, &level, &a).unwrap();

            let mut execution = CpuExecution::new(cpu_backend(parameters.levels().iter().cloned()));
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
            let setup = sample_refresh_decoder(
                &parameters,
                &declaration,
                &source,
                context,
                &"4.578".parse().unwrap(),
                &mut execution,
                &mut store,
                |_| Ok(()),
            )
            .unwrap();

            let no_rhs = PlaintextInputs::<RhsId, DCRTPoly>::new();
            let mut mask_preprocess = RnsPreprocessBuild::new(&mask_parameters, context);
            mask_preprocess
                .bind_rhs_inputs(
                    mask_program,
                    &env,
                    if dynamic { &no_rhs } else { &selectors },
                    static_payload
                        .as_ref()
                        .map(|payload| payload.at_level(&mask_parameters, 0).unwrap())
                        .as_ref(),
                    &hash_key,
                )
                .unwrap();
            mask_preprocess.bind_refresh_mask(&declaration).unwrap();
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
                    build.apply_refresh(
                        0,
                        &declaration,
                        outputs[0].clone(),
                        mask_preprocess,
                        &setup,
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
            let mut mask_encrypt = RnsEncryptBuild::new(&mask_parameters, context);
            mask_encrypt
                .bind_lhs_inputs(
                    &mask_program.encryption_inputs(),
                    &env,
                    &mask_lhs,
                    &payload.at_level(&mask_parameters, 0).unwrap(),
                    mask.as_ref().map(|mask| mask.at_level(&mask_parameters, 0).unwrap()).as_ref(),
                    &hash_key,
                )
                .unwrap();
            mask_encrypt.bind_refresh_mask(&declaration).unwrap();
            mask_encrypt.bind_preprocess_artifacts(&preprocess).unwrap();
            if dynamic {
                mask_encrypt.bind_rhs_inputs(&env, &selectors, batch).unwrap();
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
                    build.bind_refresh(&declaration, mask_encrypt).map_err(ApiError::scheme)
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

            let mut mask_online = RnsOnlineEvalBuild::new(&mask_parameters, context);
            mask_online
                .bind_refresh_mask(&declaration, &online_eval, &encrypted, &hash_key)
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
                    build.apply_refresh(0, &declaration, mask_online)?;
                    Ok(outputs)
                },
                |validated| {
                    // Every mask selector member is read at the bare root
                    // binder, so the wave importer reads it ahead. A Dynamic
                    // RHS's digit-companion family is one root import shared
                    // by every lane.
                    let selectors = loop_imports(&validated.source)
                        .into_iter()
                        .filter(|(name, _)| {
                            name.contains("/mask/rhs/") && !name.contains("/digit-encoding.")
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
