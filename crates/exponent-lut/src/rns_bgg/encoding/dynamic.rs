//! Dynamic RHS digit companions: their encryption and their import at
//! online evaluation.
use super::*;

impl EncodingLowering<'_> {
    /// Import the digit-companion family of every Dynamic RHS of the Program
    /// (see [`Self::companion_family`]); Static RHS have none.
    pub(super) fn import_rhs_companions(&mut self) -> Result<(), RnsBggError> {
        if self.parameters.rhs_mode() == RhsMode::Static {
            return Ok(());
        }
        for (rhs, shape) in self.program.rhs_inputs() {
            let members = match shape {
                InputBinding::Scalar(()) => IntExpr::constant(1),
                InputBinding::Family(count) => count.clone(),
            };
            let family = self.companion_family(
                self.workload.rhs_level(*rhs)?,
                format!("rhs/{rhs:?}/digit-encoding"),
                members,
            )?;
            self.rhs_companions.insert(*rhs, family);
        }
        Ok(())
    }

    /// The encryption's family of digit-companion encodings of `members`
    /// ciphertexts consumed at `level`, stored under `role`: member `t`,
    /// output column `j`, and digit row `a` at `(t m_(i+1) + j) m_i + a`.
    pub(super) fn companion_family(
        &self,
        level: usize,
        role: String,
        members: IntExpr,
    ) -> Result<Family<RnsBggRow>, RnsBggError> {
        let ring = self.parameters.ring(level)?;
        let columns = self.parameters.level(level)?.modulus_digits();
        let count =
            (members * IntExpr::constant(companion_count(self.parameters, level)?)).canonicalize();
        let families = [0, 1].map(|port| {
            ring.family_artifact_input(
                self.encryption.clone(),
                self.encryption_artifact_name(format!("{role}.{port}")),
                count.clone(),
                (1, columns),
                ArtifactAvailability::Transferred,
            )
        });
        let element: RnsBggRow =
            (families[0].at(Int::constant(0)), families[1].at(Int::constant(0)));
        let fields =
            families.iter().map(|family| family.value_handle().clone()).collect::<Vec<_>>();
        family_of_fields(&element, count, &fields).map_err(graph)
    }

    /// The pre-rounding Dynamic key rows `T` and companion row `E` (see
    /// [`super::companion_columns`]) of state key `key` at `level`, with the
    /// companions read from `companions`. An RHS product passes the RHS
    /// `member`. A LUT passes its automorphism indices `sigmas`: branch `s`
    /// is member `s`, its key is `sigma_s(A)`, and its stored `z^0` encodes
    /// the inverse-automorphed digit under `B^0`, so `sigma_s(z^0)` encodes
    /// the digit under `sigma_s(B^0)` and the transformed secret.
    pub(super) fn companion_rows(
        &self,
        level: usize,
        key: &RnsBggMatrix,
        material: String,
        member: Option<IntExpr>,
        sigmas: Option<&[usize]>,
        companions: &Family<RnsBggRow>,
    ) -> Result<Vec<Mat>, RnsBggError> {
        let count = companion_count(self.parameters, level)?;
        let digit_rows = 2 * self.parameters.component_columns(level)?;
        let automorphism = |branch: &IntExpr, entry: Mat| match sigmas {
            Some(sigmas) => entry.ring_automorphism(IntExpr::Select {
                selector: Box::new(branch.clone()),
                branches: sigmas.iter().map(|sigma| IntExpr::constant(*sigma)).collect(),
            }),
            None => entry,
        };
        let member_of = |branch: &IntExpr| match sigmas {
            Some(_) => Some(branch.clone()),
            None => member.clone(),
        };
        let rows = [&key.0, &key.1].map(super::join_row);
        let key_entry =
            |branch: IntExpr, r: usize, a| Ok(automorphism(&branch, super::row_entry(&rows[r], a)));
        let companion_key = |branch: IntExpr, a, j| {
            let tag = super::companion_tag(
                self.parameters,
                &self.context,
                self.program.id(),
                self.refresh_prf_namespace,
                material.clone(),
                member_of(&branch),
                level,
            );
            let key =
                super::companion_key(self.parameters, level, self.hash_key.clone(), &tag, a, j)?;
            Ok(map_matrix(key, |entry| automorphism(&branch, entry)))
        };
        let companion = |branch: IntExpr, a: IntExpr, j: IntExpr| {
            let index = member_of(&branch).unwrap_or(IntExpr::constant(0)) *
                IntExpr::constant(count) +
                j * IntExpr::constant(digit_rows) +
                a;
            Ok(map_row(companions.at(Int::evaluate(index.canonicalize())), |entry| {
                automorphism(&branch, entry)
            }))
        };
        super::companion_columns(
            self.parameters,
            level,
            sigmas.map_or(1, <[usize]>::len),
            &key_entry,
            &companion_key,
            Some(&companion),
        )
    }
}

impl<'a> RnsEncryptBuild<'a> {
    /// Bind the RHS plaintexts of a Dynamic encryption and its batch
    /// identifier, after the preprocessing artifacts (or, for a PRF child,
    /// its refresh binding) are bound, e.g. in the `api::encrypt` compose
    /// callback.
    ///
    /// `batch_id` is a fresh public identifier of this encryption request,
    /// retained only to resume the same request. It selects the sampling
    /// session, so every encryption batch under one setup draws its own
    /// ciphertexts, digit companions, and helpers; it never enters a
    /// preprocessing key. An empty map is valid when no RHS is declared.
    pub fn bind_rhs_inputs<P: Poly>(
        &mut self,
        parameters: &ParamEnv,
        rhs: &PlaintextInputs<RhsId, P>,
        batch_id: [u8; 32],
    ) -> Result<(), ApiError> {
        if self.parameters.rhs_mode() != RhsMode::Dynamic {
            return Err(ApiError::input("Static RHS plaintexts are bound by preprocessing"));
        }
        if self.dynamic_rhs.is_some() {
            return Err(ApiError::input("Dynamic RHS inputs are already bound"));
        }
        let (schema, _) = self.dynamic_schema.as_ref().ok_or_else(|| {
            ApiError::input("bind the preprocessing artifacts before the Dynamic RHS inputs")
        })?;
        if rhs.len() != schema.len() || rhs.keys().any(|id| !schema.contains_key(id)) {
            return Err(ApiError::input("RHS bindings do not exactly match Program inputs"));
        }
        let ring = self.parameters.ring(0).map_err(ApiError::scheme)?;
        let prefix = super::plaintext_input_prefix();
        let mut plaintexts = BTreeMap::new();
        for (id, (shape, _)) in schema {
            let value = &rhs[id];
            match (shape, value) {
                (InputBinding::Scalar(()), InputBinding::Scalar(_)) => {}
                (InputBinding::Family(count), InputBinding::Family(values)) => {
                    if resolved_input_cardinality(shape, parameters).map_err(ApiError::scheme)? !=
                        Some(values.len())
                    {
                        return Err(ApiError::input(format!(
                            "RHS family count {count:?} does not match Program"
                        )));
                    }
                }
                _ => return Err(ApiError::input("RHS plaintext shape does not match Program")),
            }
            let plaintext = super::plaintext_inputs(
                &ring,
                &format!("{prefix}/rhs/{id:?}"),
                value,
                &mut self.runtime,
            )
            .map_err(ApiError::input)?;
            plaintexts.insert(*id, plaintext);
        }
        self.dynamic_rhs = Some((batch_id, plaintexts));
        Ok(())
    }

    /// Dynamic encryption material under the single secret `s`: every RHS
    /// ciphertext and LUT switching ciphertext of the bound schema, with top
    /// rows sampled in this encryption, and the digit-companion encodings of
    /// every digit of each stored ciphertext (spec Eq. 043-045, 083-084).
    /// `program` and `namespace` name the companion keys, as preprocessing
    /// does; `name` maps a role to its artifact name.
    pub(super) fn dynamic_material(
        &self,
        mut graph: DslContext,
        hash_key: &Bytes,
        program: crate::program::ExponentLutProgramId,
        namespace: Option<([u8; 32], usize)>,
        name: impl Fn(String) -> String,
    ) -> Result<DslContext, ApiError> {
        let (rhs_schema, luts) = self
            .dynamic_schema
            .as_ref()
            .ok_or_else(|| ApiError::input("Dynamic encryption schema is not bound"))?;
        let (_, plaintexts) = self.dynamic_rhs.as_ref().ok_or_else(|| {
            ApiError::input("Dynamic encryption requires bind_rhs_inputs with a batch identifier")
        })?;
        let payload = &self
            .payload_secret
            .as_ref()
            .ok_or_else(|| ApiError::input("secret is not bound"))?
            .row;
        let inputs = self
            .inputs
            .as_ref()
            .ok_or_else(|| ApiError::input("encryption inputs are not bound"))?;
        let tag = |material: String, member: Option<IntExpr>, level: usize| {
            super::companion_tag(
                self.parameters,
                &self.context,
                program,
                namespace,
                material,
                member,
                level,
            )
        };
        let top = |level: usize| -> Result<RnsBggRow, RnsBggError> {
            let next = level.checked_add(1).ok_or(RnsBggError::Parameters("level overflow"))?;
            Ok(uniform_row(&self.parameters.ring(next)?, self.parameters.component_columns(next)?))
        };
        let plaintext_ring = self.parameters.ring(0).map_err(ApiError::scheme)?;
        for (rhs, (_, level)) in rhs_schema {
            let level = *level;
            let material = format!("rhs/{rhs:?}");
            match plaintexts.get(rhs) {
                Some(InputBinding::Scalar(plaintext)) => {
                    let ciphertext = super::public_key::rhs_ciphertext(
                        self.parameters,
                        level,
                        payload,
                        plaintext.clone(),
                        || top(level),
                    )
                    .map_err(ApiError::scheme)?;
                    let companions = self
                        .companion_encodings(
                            level,
                            1,
                            &|_| ciphertext.clone(),
                            &|_| tag(material.clone(), None, level),
                            None,
                            hash_key,
                        )
                        .map_err(ApiError::scheme)?;
                    graph = graph
                        .transferred_output(name(format!("{material}/ciphertext")), ciphertext)
                        .map_err(ApiError::scheme)?;
                    graph = graph
                        .transferred_output(name(format!("{material}/digit-encoding")), companions)
                        .map_err(ApiError::scheme)?;
                }
                Some(InputBinding::Family(family)) => {
                    let members = family
                        .count()
                        .evaluate(&ParamEnv::default())
                        .ok()
                        .and_then(|count| count.to_usize())
                        .ok_or_else(|| ApiError::input("RHS family count is not concrete"))?;
                    let ciphertexts = mxx_dsl::parallel(IntExpr::constant(members), |member| {
                        let plaintext = family.member(&plaintext_ring, member);
                        super::public_key::rhs_ciphertext(
                            self.parameters,
                            level,
                            payload,
                            plaintext,
                            || top(level),
                        )
                        .map_err(|_| DslError::Schema)
                    })
                    .map_err(ApiError::scheme)?;
                    let companions = self
                        .companion_encodings(
                            level,
                            members,
                            &|member| ciphertexts.at(member),
                            &|member| tag(material.clone(), Some(member), level),
                            None,
                            hash_key,
                        )
                        .map_err(ApiError::scheme)?;
                    graph = graph
                        .transferred_output(
                            name(format!("{material}/family/ciphertext")),
                            ciphertexts,
                        )
                        .map_err(ApiError::scheme)?;
                    graph = graph
                        .transferred_output(name(format!("{material}/digit-encoding")), companions)
                        .map_err(ApiError::scheme)?;
                }
                None => return Err(ApiError::input("missing Dynamic RHS plaintext")),
            }
        }
        for (level, lut) in luts {
            let level = *level;
            let table = inputs
                .luts()
                .get(lut.0 as usize)
                .ok_or_else(|| ApiError::scheme(RnsBggError::Lut))?;
            let layout = table.input_layout();
            let dimension =
                self.parameters.level(level).map_err(ApiError::scheme)?.ring_dimension() as usize;
            let sigmas = layout
                .automorphism_indices(dimension)
                .map_err(|_| ApiError::scheme(RnsBggError::Lut))?;
            let inverses = sigmas
                .iter()
                .map(|sigma| automorphism_inverse(*sigma, dimension))
                .collect::<Result<Vec<_>, _>>()
                .map_err(ApiError::scheme)?;
            let select = |indices: &[usize], branch: IntExpr| IntExpr::Select {
                selector: Box::new(branch),
                branches: indices.iter().map(|index| IntExpr::constant(*index)).collect(),
            };
            let ciphertexts = mxx_dsl::parallel(IntExpr::constant(sigmas.len()), |branch| {
                let sigma = select(&sigmas, branch.expression()?);
                super::public_key::lut_switching_ciphertext(
                    self.parameters,
                    level,
                    table,
                    payload,
                    sigma,
                    || top(level),
                )
                .map_err(|_| DslError::Schema)
            })
            .map_err(ApiError::scheme)?;
            let material = format!("lut/{lut:?}");
            let companions = self
                .companion_encodings(
                    level,
                    sigmas.len(),
                    &|branch| ciphertexts.at(branch),
                    &|branch| tag(material.clone(), Some(branch), level),
                    Some(&|branch| select(&inverses, branch)),
                    hash_key,
                )
                .map_err(ApiError::scheme)?;
            graph = graph
                .transferred_output(name(lut_switching_role(level, *lut, layout, "")), ciphertexts)
                .map_err(ApiError::scheme)?;
            graph = graph
                .transferred_output(
                    name(lut_switching_role(level, *lut, layout, "/digit-encoding")),
                    companions,
                )
                .map_err(ApiError::scheme)?;
        }
        Ok(graph)
    }

    /// The digit-companion encodings, under the single secret, of `members`
    /// ciphertexts consumed at `level`, member `t`'s stored ciphertext being
    /// `ciphertext(t)` and its companion-key tag `tag(t)`. Element
    /// `(t m_(i+1) + j) m_i + a` (see [`Self::companion_family`]) is
    /// `z_{a,j} = s B_{a,j} - d_{a,j} s G_i + eps` under
    /// [`super::companion_key`] for digit `d_{a,j} = Dec_{q_(i+1)}(C)_{a,j}`
    /// of the stored ciphertext, rebased to `q_i`; a LUT branch encodes
    /// `sigma^-1(d)`, with the inverse automorphism index `inverse_sigma(t)`
    /// (spec Eq. 045, 084). Each lane decomposes only column `j` of `C`.
    fn companion_encodings(
        &self,
        level: usize,
        members: usize,
        ciphertext: &dyn Fn(Int) -> RnsBggMatrix,
        tag: &dyn Fn(IntExpr) -> mxx_dsl::HashTag,
        inverse_sigma: Option<&dyn Fn(IntExpr) -> IntExpr>,
        hash_key: &Bytes,
    ) -> Result<Family<RnsBggRow>, RnsBggError> {
        let ring = self.parameters.ring(level)?;
        let next = level.checked_add(1).ok_or(RnsBggError::Parameters("level overflow"))?;
        let next_parameters = self.parameters.level(next)?;
        let base = 1u64
            .checked_shl(next_parameters.base_bits())
            .ok_or(RnsBggError::Parameters("decomposition base overflow"))?;
        let digits = next_parameters.modulus_digits();
        let digit_rows = 2 * self.parameters.component_columns(level)?;
        let output_columns = 2 * self.parameters.component_columns(next)?;
        let count = companion_count(self.parameters, level)?;
        let gadget = self.parameters.public_gadget(level)?;
        let width = self.parameters.level(level)?.modulus_digits();
        let mask = &self.payload_secret.as_ref().ok_or(RnsBggError::Binding("secret"))?.row;
        mxx_dsl::parallel(IntExpr::constant(members * count), |index| {
            let index = Box::new(index.expression()?);
            let quotient = |divisor: usize| {
                IntExpr::FloorDiv(index.clone(), Box::new(IntExpr::constant(divisor)))
                    .canonicalize()
            };
            let member = quotient(count);
            let column = quotient(digit_rows);
            let j = IntExpr::Rem(Box::new(column), Box::new(IntExpr::constant(output_columns)))
                .canonicalize();
            let a = IntExpr::Rem(index, Box::new(IntExpr::constant(digit_rows))).canonicalize();
            let ciphertext = ciphertext(Int::evaluate(member.clone()));
            let [top, bottom] = [&ciphertext.0, &ciphertext.1].map(super::join_row);
            let column_digits = Mat::concat(
                mxx_ir_core::node::ConcatAxis::Rows,
                vec![super::row_entry(&top, j.clone()), super::row_entry(&bottom, j.clone())],
            )
            .decompose(base, digits)
            .centered_rebase(&ring);
            let mut digit =
                super::unit_row(&ring, digit_rows, a.clone()).mul_small_rhs(column_digits);
            if let Some(inverse_sigma) = inverse_sigma {
                digit = digit.ring_automorphism(inverse_sigma(member.clone()));
            }
            let key =
                super::companion_key(self.parameters, level, hash_key.clone(), &tag(member), a, j)
                    .map_err(|_| DslError::Schema)?;
            public_encoding_row(
                self.parameters,
                key,
                mask,
                digit.tensor(gadget.clone()),
                width,
                level,
            )
            .map_err(|_| DslError::Schema)
        })
        .map_err(graph)
    }
}

/// A row of two uniform blocks of `columns` columns over `ring`: the top row
/// of a ciphertext sampled in the encryption that publishes it.
pub(super) fn uniform_row(ring: &Ring, columns: usize) -> RnsBggRow {
    (ring.uniform_residue((1, columns)), ring.uniform_residue((1, columns)))
}

/// The automorphism index `sigma^-1 mod 2n` inverting `X -> X^sigma`.
fn automorphism_inverse(sigma: usize, ring_dimension: usize) -> Result<usize, RnsBggError> {
    let order = BigUint::from(2 * ring_dimension);
    mxx_backends::utils::mod_inverse_biguints(&(BigUint::from(sigma) % &order), &order)
        .and_then(|inverse| inverse.to_usize())
        .ok_or(RnsBggError::Lut)
}

#[cfg(test)]
mod test {
    use super::*;
    use mxx_backends::{
        MemoryArtifactStore, backend::poly::cpu_backend, poly::dcrt::poly::DCRTPoly,
    };

    /// A three-level Dynamic chain with a multiple-prime drop at every
    /// level. Ring dimension, base bits, and sigma are overridable by
    /// `MXX_EXPONENT_LUT_DYNAMIC_TEST_{RING_DIMENSION,BASE_BITS,SIGMA}`.
    fn dynamic_test_parameters() -> RnsBggParameters {
        let read = |name: &str, default: &str| {
            std::env::var(format!("MXX_EXPONENT_LUT_DYNAMIC_TEST_{name}"))
                .unwrap_or_else(|_| default.to_owned())
        };
        let ring_dimension = read("RING_DIMENSION", "16").parse().unwrap();
        let base_bits = read("BASE_BITS", "6").parse().unwrap();
        let sigma = read("SIGMA", "1").parse::<bigdecimal::BigDecimal>().unwrap();
        let high = DCRTPolyParams::new(ring_dimension, 3, 17, base_bits, None, None);
        let primes = high.to_crt().0;
        let middle =
            high.select_modulus(&(BigUint::from(primes[0]) * BigUint::from(primes[1]))).unwrap();
        let low = high.select_modulus(&BigUint::from(primes[0])).unwrap();
        RnsBggParameters::new(vec![high, middle, low], sigma, 1)
            .unwrap()
            .with_rhs_mode(RhsMode::Dynamic)
    }

    /// One plaintext assignment of a Dynamic batch: the LHS and RHS
    /// plaintexts, the expected terminal output, and a wrong output.
    type DynamicBatch =
        (PlaintextInputs<WireId, DCRTPoly>, PlaintextInputs<RhsId, DCRTPoly>, DCRTPoly, DCRTPoly);

    /// Preprocess `program` once with no secret and no RHS, then run two
    /// encryption batches, each under a freshly sampled single secret with
    /// RHS values chosen only then, verify each online evaluation output, and check that
    /// every online evaluation output key equals the preprocessed one.
    fn check_dynamic_batches(
        parameters: &RnsBggParameters,
        program: &ExponentLutProgram,
        batch: impl Fn() -> DynamicBatch,
    ) {
        let mut execution = mxx_backends::authority::CpuExecution::new(cpu_backend(
            parameters.levels().iter().cloned(),
        ));
        let mut store = MemoryArtifactStore::default();
        let hash_key: [u8; 32] = rand::random();
        let context: [u8; 32] = rand::random();
        let (preprocess, online_eval) = crate::api::preprocess(
            program,
            &ParamEnv::default(),
            &PlaintextInputs::<RhsId, DCRTPoly>::new(),
            None,
            &hash_key,
            super::RnsPreprocessBuild::new(parameters, context),
            &mut execution,
            &mut store,
            |_, outputs| Ok(outputs),
            // Preprocessing reads only the hash key and samples nothing: no
            // secret, ciphertext, or secret-dependent helper exists yet.
            |validated| {
                for (_, scope) in validated.source.scopes() {
                    for node in scope.nodes() {
                        assert!(!matches!(
                            node.kind(),
                            mxx_ir_core::node::NodeKind::UniformResidueSample { .. } |
                                mxx_ir_core::node::NodeKind::UniformIntervalSample { .. } |
                                mxx_ir_core::node::NodeKind::GaussianSample { .. }
                        ));
                    }
                }
                for node in validated.source.root_scope().nodes() {
                    if let mxx_ir_core::node::NodeKind::Input { name, .. } = node.kind() {
                        assert_eq!(name, super::HASH_KEY_INPUT);
                    }
                }
                Ok(())
            },
        )
        .unwrap();
        let Operand::Scalar(output) = program.outputs()[0] else { panic!("scalar output") };
        let level = RnsBggDerivedWorkload::from_program(program).unwrap().level(output).unwrap();
        let ring = parameters.ring(level).unwrap();
        let columns = parameters.component_columns(level).unwrap();
        let mut graph = DslContext::new("dynamic-preprocessed-key");
        for port in 0..4 {
            graph = graph
                .output(
                    format!("key-{port}"),
                    ring.artifact_input(
                        online_eval.production().clone(),
                        format!("public-key/{output:?}.{port}"),
                        (1, columns),
                        ArtifactAvailability::Transferred,
                    ),
                )
                .unwrap();
        }
        let preprocessed = execute_graph(
            StageGraph::new(
                graph.build().unwrap(),
                BTreeMap::new(),
                BTreeMap::from([(
                    online_eval.production().clone(),
                    online_eval.manifest().clone(),
                )]),
            ),
            &ParamEnv::default(),
            &mut execution,
            &mut store,
            |_| Ok(()),
        )
        .unwrap();
        let download = |execution: &mxx_backends::authority::CpuExecution,
                        outputs: &StageOutputs,
                        name: String| {
            <mxx_backends::authority::CpuExecution as crate::api::StageExecution<
                MemoryArtifactStore,
            >>::download_matrix(execution, &outputs.outputs[&name])
            .unwrap()
        };
        let preprocessed_key = (0..4)
            .map(|port| download(&execution, &preprocessed, format!("key-{port}")))
            .collect::<Vec<_>>();
        for _ in 0..2 {
            let (lhs, rhs, expected, wrong) = batch();
            let secret = RnsEncryptBuild::new(parameters, context)
                .sample_payload_secret(&mut execution, &mut store, |_| Ok(()))
                .unwrap();
            let encrypted = crate::api::encrypt(
                &program.encryption_inputs(),
                &ParamEnv::default(),
                &lhs,
                &preprocess,
                &secret,
                None,
                &hash_key,
                RnsEncryptBuild::new(parameters, context),
                &mut execution,
                &mut store,
                |build| build.bind_rhs_inputs(&ParamEnv::default(), &rhs, rand::random()),
                |_| Ok(()),
            )
            .unwrap();
            let result = crate::api::online_eval(
                program,
                &ParamEnv::default(),
                &online_eval,
                &encrypted,
                &hash_key,
                RnsOnlineEvalBuild::new(parameters, context),
                &mut execution,
                &mut store,
                |_, outputs| Ok(outputs),
                |_| Ok(()),
            )
            .unwrap();
            verify_output(
                program,
                &result,
                0,
                &expected,
                parameters,
                &secret,
                None,
                &mut execution,
                &mut store,
            )
            .unwrap();
            assert!(
                verify_output(
                    program,
                    &result,
                    0,
                    &wrong,
                    parameters,
                    &secret,
                    None,
                    &mut execution,
                    &mut store,
                )
                .is_err(),
                "a wrong Dynamic output must be rejected"
            );
            for (port, key) in preprocessed_key.iter().enumerate() {
                assert_eq!(
                    &download(&execution, &result, format!("output-0.{port}")),
                    key,
                    "the online evaluation key must be the key fixed by preprocessing"
                );
            }
        }
    }

    fn monomial(parameters: &RnsBggParameters, exponent: usize) -> DCRTPoly {
        DCRTPoly::const_rotate_poly(parameters.levels().last().unwrap(), exponent)
    }

    /// Two successive private products by a scalar RHS and a family RHS
    /// member, both chosen after preprocessing.
    #[test]
    fn dynamic_private_products_verify_late_rhs_under_one_setup() {
        let parameters = dynamic_test_parameters();
        let mut builder = ExponentLutProgramBuilder::new();
        let input = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let first = builder.rhs_input().unwrap();
        let second = builder.rhs_family(IntExpr::constant(2)).unwrap();
        let product = builder.mul_private(Operand::Scalar(input), Operand::Scalar(first)).unwrap();
        let output = builder
            .mul_private(
                Operand::Scalar(product),
                Operand::Member { family: second, index: IntExpr::constant(1) },
            )
            .unwrap();
        let program = builder.finish(vec![Operand::Scalar(output)]).unwrap();
        let dimension = parameters.level(0).unwrap().ring_dimension() as usize;
        check_dynamic_batches(&parameters, &program, || {
            // Products past X^n are negated monomials.
            let exponents = [0; 4].map(|_| rand::random::<u32>() as usize % dimension);
            let [x, y, z0, z1] = exponents.map(|exponent| monomial(&parameters, exponent));
            let expected = x.clone() * y.clone() * z1.clone();
            let wrong = expected.clone() * monomial(&parameters, 1);
            (
                BTreeMap::from([(input, InputBinding::Scalar(x))]),
                BTreeMap::from([
                    (first, InputBinding::Scalar(y)),
                    (second, InputBinding::Family(vec![z0, z1])),
                ]),
                expected,
                wrong,
            )
        });
    }

    /// A Dynamic product inside a Program family body over member 1 only, by
    /// the RHS member the body index selects, then by a scalar RHS.
    #[test]
    fn dynamic_family_body_products_verify_indexed_rhs_members() {
        let parameters = dynamic_test_parameters();
        let mut builder = ExponentLutProgramBuilder::new();
        let inputs =
            builder.input_family(IntExpr::constant(2), AttributeKnowledge::PrivatePayload).unwrap();
        let rhs = builder.rhs_family(IntExpr::constant(2)).unwrap();
        let scale = builder.rhs_input().unwrap();
        let range = crate::program::FamilyRange::bounded(
            IntExpr::constant(1),
            IntExpr::constant(1),
            IntExpr::constant(2),
        )
        .unwrap();
        let products = builder
            .parallel(range, |builder, index| {
                builder
                    .mul_private(
                        Operand::Member { family: inputs, index: index.clone() },
                        Operand::Member { family: rhs, index },
                    )
                    .map(Operand::Scalar)
            })
            .unwrap();
        let sum = builder.sum(products).unwrap();
        let output = builder.mul_private(Operand::Scalar(sum), Operand::Scalar(scale)).unwrap();
        let program = builder.finish(vec![Operand::Scalar(output)]).unwrap();
        let dimension = parameters.level(0).unwrap().ring_dimension() as usize;
        check_dynamic_batches(&parameters, &program, || {
            let exponents = [0; 5].map(|_| rand::random::<u32>() as usize % dimension);
            let [x0, x1, y0, y1, z] = exponents.map(|exponent| monomial(&parameters, exponent));
            let expected = x1.clone() * y1.clone() * z.clone();
            let wrong = expected.clone() * monomial(&parameters, 1);
            (
                BTreeMap::from([(inputs, InputBinding::Family(vec![x0, x1]))]),
                BTreeMap::from([
                    (rhs, InputBinding::Family(vec![y0, y1])),
                    (scale, InputBinding::Scalar(z)),
                ]),
                expected,
                wrong,
            )
        });
    }

    /// A Dynamic LUT, whose switching ciphertexts and digit companions
    /// encryption generates, followed by a product by a late RHS.
    #[test]
    fn dynamic_lut_switching_verifies_through_a_late_rhs_product() {
        let parameters = dynamic_test_parameters();
        let layout = ExpLayout { stride_log2: 0, width_log2: 1 };
        let mut builder = ExponentLutProgramBuilder::new();
        let input = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let rhs = builder.rhs_input().unwrap();
        let table = builder
            .register_lut(
                LutTable::unary(
                    2,
                    layout,
                    2,
                    vec![1, 0],
                    crate::program::LutOutputRepresentation::Exponent(layout),
                )
                .unwrap(),
            )
            .unwrap();
        let negated = builder.lut(Operand::Scalar(input), table).unwrap();
        let output = builder.mul_private(Operand::Scalar(negated), Operand::Scalar(rhs)).unwrap();
        let program = builder.finish(vec![Operand::Scalar(output)]).unwrap();
        let dimension = parameters.level(0).unwrap().ring_dimension() as usize;
        check_dynamic_batches(&parameters, &program, || {
            let bit = rand::random::<u32>() as usize % 2;
            let exponent = rand::random::<u32>() as usize % dimension;
            let expected = monomial(&parameters, 1 - bit) * monomial(&parameters, exponent);
            let wrong = monomial(&parameters, bit) * monomial(&parameters, exponent);
            (
                BTreeMap::from([(input, InputBinding::Scalar(monomial(&parameters, bit)))]),
                BTreeMap::from([(rhs, InputBinding::Scalar(monomial(&parameters, exponent)))]),
                expected,
                wrong,
            )
        });
    }

    /// Dynamic stages take no preprocessing secret or RHS, no separate mask,
    /// and require their RHS binding at encryption; Static stages reject the
    /// Dynamic RHS binding.
    #[test]
    fn dynamic_stages_reject_preprocessing_secrets_and_separate_masks() {
        let parameters = dynamic_test_parameters();
        let mut builder = ExponentLutProgramBuilder::new();
        let input = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let rhs = builder.rhs_input().unwrap();
        let output = builder.mul_private(Operand::Scalar(input), Operand::Scalar(rhs)).unwrap();
        let program = builder.finish(vec![Operand::Scalar(output)]).unwrap();
        let mut execution = mxx_backends::authority::CpuExecution::new(cpu_backend(
            parameters.levels().iter().cloned(),
        ));
        let mut store = MemoryArtifactStore::default();
        let secret = RnsEncryptBuild::new(&parameters, [1; 32])
            .sample_payload_secret(&mut execution, &mut store, |_| Ok(()))
            .unwrap();
        assert!(
            RnsEncryptBuild::new(&parameters, [1; 32])
                .sample_mask_secret(&mut execution, &mut store, |_| Ok(()))
                .is_err()
        );
        let mut preprocess = super::RnsPreprocessBuild::new(&parameters, [1; 32]);
        assert!(
            <super::RnsPreprocessBuild<'_> as crate::api::PreprocessBuild<DCRTPoly>>::bind_inputs(
                &mut preprocess,
                &program,
                &ParamEnv::default(),
                &BTreeMap::new(),
                Some(&secret),
                &[0; 32],
            )
            .is_err(),
            "Dynamic preprocessing takes no secret"
        );
        let rhs_values = BTreeMap::from([(rhs, InputBinding::Scalar(monomial(&parameters, 1)))]);
        assert!(
            <super::RnsPreprocessBuild<'_> as crate::api::PreprocessBuild<DCRTPoly>>::bind_inputs(
                &mut super::RnsPreprocessBuild::new(&parameters, [1; 32]),
                &program,
                &ParamEnv::default(),
                &rhs_values,
                None,
                &[0; 32],
            )
            .is_err(),
            "Dynamic preprocessing takes no RHS plaintexts"
        );
        let lhs = BTreeMap::from([(input, InputBinding::Scalar(monomial(&parameters, 0)))]);
        let mask = RnsBggMaskSecret { row: secret.row.clone(), binding: secret.binding.clone() };
        assert!(
            <RnsEncryptBuild<'_> as EncryptBuild<DCRTPoly>>::bind_inputs(
                &mut RnsEncryptBuild::new(&parameters, [1; 32]),
                &program.encryption_inputs(),
                &ParamEnv::default(),
                &lhs,
                &secret,
                Some(&mask),
                &[0; 32],
            )
            .is_err(),
            "Dynamic encryption takes no separate mask"
        );
        let (preprocess, _) = crate::api::preprocess(
            &program,
            &ParamEnv::default(),
            &PlaintextInputs::<RhsId, DCRTPoly>::new(),
            None,
            &[0; 32],
            super::RnsPreprocessBuild::new(&parameters, [1; 32]),
            &mut execution,
            &mut store,
            |_, outputs| Ok(outputs),
            |_| Ok(()),
        )
        .unwrap();
        let encrypt = |compose: &dyn Fn(&mut RnsEncryptBuild<'_>) -> Result<(), ApiError>,
                       execution: &mut mxx_backends::authority::CpuExecution,
                       store: &mut MemoryArtifactStore| {
            crate::api::encrypt(
                &program.encryption_inputs(),
                &ParamEnv::default(),
                &lhs,
                &preprocess,
                &secret,
                None,
                &[0; 32],
                RnsEncryptBuild::new(&parameters, [1; 32]),
                execution,
                store,
                compose,
                |_| Ok(()),
            )
            .map(|_| ())
        };
        assert!(
            encrypt(&|_| Ok(()), &mut execution, &mut store).is_err(),
            "Dynamic encryption requires its RHS binding"
        );
        assert!(
            encrypt(
                &|build| build.bind_rhs_inputs(
                    &ParamEnv::default(),
                    &PlaintextInputs::<RhsId, DCRTPoly>::new(),
                    [2; 32]
                ),
                &mut execution,
                &mut store,
            )
            .is_err(),
            "a missing RHS plaintext is rejected"
        );
        let static_parameters = RnsBggParameters::new(parameters.levels().to_vec(), 1, 1).unwrap();
        assert!(
            RnsEncryptBuild::new(&static_parameters, [1; 32])
                .bind_rhs_inputs(&ParamEnv::default(), &rhs_values, [2; 32])
                .is_err(),
            "Static encryption rejects late RHS plaintexts"
        );
    }
}
