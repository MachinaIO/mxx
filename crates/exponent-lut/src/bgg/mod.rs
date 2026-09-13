//! BGG+ implementation of Exponent-LUT encoding contracts.
pub mod encoding;
pub mod noise;
pub mod public_key;
use crate::{
    noise_refresh::ExponentLUTRefreshScheme,
    program::{ProgramFamilyBindings, ProgramValue},
    scheme::ExponentLUTEncodingScheme,
};
use encoding::ExponentArtifactImportError;
use mxx_bgg::{BggEncodingWire, BggPublicKeyWire, EncodingCompileError};
use mxx_dsl::{Bytes, HashTag, Mat, Preimage};
use mxx_ir_core::{
    IntExpr, ParamEnv,
    artifact::{ArtifactConfidentiality, ArtifactType, Manifest, ProductionId},
    types::ConcreteMatrixType,
};
use num_bigint::{BigInt, BigUint, ToBigInt};
use num_traits::{FromPrimitive, Signed, ToPrimitive, Zero};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

#[cfg(test)]
use crate::scheme::PreparedProgram;
#[derive(Debug, Error, Eq, PartialEq)]
/// Errors raised while validating or lowering an Exponent-LUT operation.
pub enum BggError {
    #[error(transparent)]
    Pbc(#[from] crate::noise_refresh::pbc::PbcError),
    #[error(transparent)]
    /// A generic BGG encoding operation rejected its inputs.
    Bgg(#[from] EncodingCompileError),
    #[error(transparent)]
    /// An RHS package failed material or artifact validation.
    Rhs(#[from] ExponentRhsPackageError),
    #[error(transparent)]
    /// An imported artifact manifest is not compatible with the requested value.
    Artifact(#[from] ExponentArtifactImportError),
    #[error("artifact transport or binding failed: {0}")]
    ArtifactTransport(String),
    #[error(transparent)]
    /// A shared Exponent-LUT program failed validation or lowering.
    Program(#[from] crate::program::ProgramValidationError),
    #[error("LUT dimensions, output exponents, or RHS width are invalid")]
    /// A table, exponent, or matrix shape is inconsistent with the operation.
    InvalidLut,
    #[error("sparse-LWR block layout or package count is invalid")]
    /// A sparse-LWR block has no valid public/package pairing.
    InvalidSparseLwrBlock,
    #[error("program operation {gate} failed: {detail}")]
    /// A backend rejected one identified primitive operation while lowering.
    ProgramOperation { gate: String, detail: String },
    #[error("graph construction failed: {0}")]
    Graph(String),
    #[error("online PRF stage `{stage}` failed: {detail}")]
    /// Structural online batch construction failed before a graph was emitted.
    OnlinePrfStage { stage: &'static str, detail: String },
}

impl From<BggError> for crate::noise_refresh::setup::RefreshSetupError {
    fn from(error: BggError) -> Self {
        Self::Backend(Box::new(error))
    }
}

impl From<encoding::ExponentLutSamplingError> for crate::noise_refresh::setup::RefreshSetupError {
    fn from(error: encoding::ExponentLutSamplingError) -> Self {
        Self::Backend(Box::new(error))
    }
}

impl From<mxx_bgg::EncodingCompileError> for crate::noise_refresh::setup::RefreshSetupError {
    fn from(error: mxx_bgg::EncodingCompileError) -> Self {
        Self::Backend(Box::new(error))
    }
}

impl From<mxx_bgg::BggSampleError> for crate::noise_refresh::setup::RefreshSetupError {
    fn from(error: mxx_bgg::BggSampleError) -> Self {
        Self::Backend(Box::new(error))
    }
}

impl From<mxx_dsl::DslError> for BggError {
    fn from(error: mxx_dsl::DslError) -> Self {
        Self::Graph(error.to_string())
    }
}

impl From<crate::program::ProgramLowerError<BggError>> for BggError {
    fn from(error: crate::program::ProgramLowerError<BggError>) -> Self {
        match error {
            crate::program::ProgramLowerError::Validation(error) => Self::Program(error),
            crate::program::ProgramLowerError::Graph(error) => Self::Graph(error.to_string()),
            crate::program::ProgramLowerError::Operation { gate, source } => {
                Self::ProgramOperation { gate: format!("{gate:?}"), detail: source.to_string() }
            }
        }
    }
}

impl From<crate::ExponentLutError> for BggError {
    fn from(error: crate::ExponentLutError) -> Self {
        match error {
            crate::ExponentLutError::Program(error) => Self::Program(error),
            crate::ExponentLutError::InvalidLut => Self::InvalidLut,
            crate::ExponentLutError::ProgramOperation { gate, detail } => {
                Self::ProgramOperation { gate, detail }
            }
        }
    }
}
/// BGG+ specialization; only this implementation knows gadget/preimage types.
pub struct BggScheme {
    public: crate::bgg::public_key::ExponentLutPublicKeyCompiler,
    encoding: crate::bgg::encoding::ExponentLutEncodingCompiler,
}

impl BggScheme {
    pub fn new(
        public_key: mxx_bgg::BggPublicKeyCompiler,
        layout: crate::layout::ExpLayout,
    ) -> Self {
        Self {
            encoding: crate::bgg::encoding::ExponentLutEncodingCompiler::with_layout(
                public_key.clone(),
                layout,
            ),
            public: crate::bgg::public_key::ExponentLutPublicKeyCompiler::with_layout(
                public_key, layout,
            ),
        }
    }
}

// Fixed-product preparation uses the same external product as MulPrivate.
// Selection and step traversal are owned by the common PRF implementation.
impl crate::scheme::ExponentLUTFixedRhsScheme for BggScheme {
    type FixedRhs = encoding::SetupRhsFamily;
    type PreparedFixedProduct = Family<Preimage>;
    type FixedProductBinding = (IntExpr, IntExpr, IntExpr, IntExpr, crate::layout::ExpLayout);

    fn fixed_product_binding(&self) -> Self::FixedProductBinding {
        let matrix = self.public.public_key.ring.matrix_type((1, 1));
        (
            matrix.modulus,
            matrix.ring_dimension,
            self.public.public_key.base.clone(),
            self.public.public_key.digit_count.clone(),
            self.public.exponent_layout,
        )
    }

    fn combine_fixed_rhs(
        &self,
        rhs: &Self::FixedRhs,
        weights: &mxx_dsl::Family<Mat>,
        active: &crate::program::FamilyRange,
        _: [u8; 32],
    ) -> Result<Self::FixedRhs, BggError> {
        let gsw = crate::utils::weighted_family_combinations(rhs.gsw(), weights, active)
            .map_err(|_| BggError::InvalidSparseLwrBlock)?;
        match rhs.layout_id() {
            Some(layout) => encoding::SetupRhsFamily::new_for_layout(gsw, layout),
            None => encoding::SetupRhsFamily::new(gsw),
        }
    }

    fn prepare_fixed_product(
        &self,
        input: &crate::program::ProgramValue<Self::PublicKey, Family<Self::PublicKey>>,
        rhs: &Self::FixedRhs,
        _: [u8; 32],
    ) -> Result<
        (
            crate::program::ProgramValue<Self::PublicKey, Family<Self::PublicKey>>,
            Self::PreparedFixedProduct,
        ),
        BggError,
    > {
        let prepared = mxx_dsl::parallel(rhs.gsw().count().clone(), |index| {
            let decomposition = rhs.gsw().at(&index).decompose(
                self.public.public_key.base.clone(),
                self.public.public_key.digit_count.clone(),
            );
            let key = input.at(index);
            Ok((
                BggPublicKeyWire {
                    matrix: decomposition.clone().mul_small_rhs(key.matrix),
                    reveal_plaintext: false,
                },
                decomposition,
            ))
        })?;
        let public = prepared.field(|value| value.0)?;
        let decompositions = prepared.field(|value| value.1)?;
        Ok((crate::program::ProgramValue::Family(public), decompositions))
    }

    fn evaluate_fixed_product(
        &self,
        input: &crate::program::ProgramValue<Self::Encoding, Family<Self::Encoding>>,
        prepared: &Self::PreparedFixedProduct,
    ) -> Result<crate::program::ProgramValue<Self::Encoding, Family<Self::Encoding>>, BggError>
    {
        let values = mxx_dsl::parallel(prepared.count().clone(), |index| {
            let input = input.at(&index);
            Ok(BggEncodingWire {
                vector: prepared.at(index).mul_small_rhs(input.vector),
                plaintext: None,
            })
        })?;
        Ok(crate::program::ProgramValue::Family(values))
    }
}

impl ExponentLUTEncodingScheme for BggScheme {
    fn encoding_dimensions(
        &self,
        key: &Self::PublicKey,
    ) -> (mxx_ir_core::IntExpr, mxx_ir_core::IntExpr) {
        let shape = key.matrix.matrix_type();
        (shape.columns.clone(), shape.ring_dimension.clone())
    }
    fn reduce_public(
        &self,
        values: Family<Self::PublicKey>,
        _: crate::program::ProgramGateInstance,
        _: &mut Self::Cache,
    ) -> Result<Self::PublicKey, Self::Error> {
        Ok(BggPublicKeyWire {
            matrix: crate::utils::balanced_sum_family(values.field(|value| value.matrix)?)?,
            reveal_plaintext: false,
        })
    }
    fn reduce_encoding(
        &self,
        values: Family<Self::Encoding>,
        _: crate::program::ProgramGateInstance,
        _: &Self::Cache,
    ) -> Result<Self::Encoding, Self::Error> {
        let plaintext = if mxx_dsl::GraphValue::schema(&values).element.plaintext.is_some() {
            Some(crate::utils::balanced_sum_family(
                values.field(|value| value.plaintext.expect("known plaintext schema"))?,
            )?)
        } else {
            None
        };
        Ok(BggEncodingWire {
            vector: crate::utils::balanced_sum_family(values.field(|value| value.vector)?)?,
            plaintext,
        })
    }
    type PublicKey = BggPublicKeyWire;
    type Encoding = BggEncodingWire;
    type PublicRhs = ExponentRhsPackage;
    type EncodingRhs = ();
    type PublicHelpers = crate::bgg::public_key::LutPublicHelperSet;
    type EncodingHelpers = crate::bgg::encoding::LutHelperSet;
    type Cache = crate::bgg::public_key::ProgramCacheBindings;
    type NoiseState = noise::NoiseProgramWire;
    type NoiseModel<'a> = noise::BggNoiseModel<'a>;
    type NoiseEvaluator<'a> = noise::BggNoiseEvaluator<'a>;
    fn noise_evaluator<'a>(
        program: &'a crate::program::ExponentLutProgram,
        model: Self::NoiseModel<'a>,
        ranges: &'a ProgramFamilyBindings,
    ) -> Self::NoiseEvaluator<'a>
    where
        Self: 'a,
    {
        noise::BggNoiseEvaluator {
            ranges,
            program,
            helper_cutoff: model.helper_cutoff,
            helper_variance: model.helper_variance,
            actions: model.actions,
            operation_residuals: model.operation_residuals,
            records: Default::default(),
        }
    }
    fn reduce_noise<'a>(
        evaluator: &mut Self::NoiseEvaluator<'a>,
        value: &Self::NoiseState,
        range: &crate::program::FamilyRange,
        gate: crate::program::ProgramGateInstance,
    ) -> Result<Self::NoiseState, noise::ProgramNoiseError>
    where
        Self: 'a,
    {
        evaluator.reduce_add(value, range, gate)
    }
    type Error = crate::bgg::BggError;
    type PublicEvaluator<'a> = crate::bgg::public_key::PublicProgramLowering<'a>;
    type EncodingEvaluator<'a> = crate::bgg::encoding::EncodingProgramLowering<'a>;

    fn public_evaluator<'a>(
        &'a self,
        cache: &'a mut Self::Cache,
        shared_cache: &'a Self::Cache,
        _ranges: &'a ProgramFamilyBindings,
    ) -> Self::PublicEvaluator<'a> {
        crate::bgg::public_key::PublicProgramLowering {
            compiler: &self.public,
            cache: Some(cache),
            shared_cache: Some(shared_cache),
        }
    }
    fn encoding_evaluator<'a>(
        &'a self,
        cache: &'a Self::Cache,
        shared_cache: &'a Self::Cache,
        _ranges: &'a ProgramFamilyBindings,
    ) -> Self::EncodingEvaluator<'a> {
        crate::bgg::encoding::EncodingProgramLowering {
            compiler: &self.encoding,
            cache: Some(cache),
            shared_cache: Some(shared_cache),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        bgg::{
            encoding::{ExponentLutEncodingSampler, LutHelperSet},
            public_key::{ExponentLutPublicKeySampler, LutPublicHelper, LutPublicHelperSet},
        },
        layout::ExpLayout,
        program::{
            AttributeKnowledge, ExponentLutProgramBuilder, LutTable, ProgramBindings,
            ProgramWireType,
        },
    };
    use mxx_bgg::{BggPublicKeyCompiler, BggSamplerLayout};
    use mxx_dsl::DslContext;
    use mxx_ir_core::ParamEnv;
    use mxx_primitives::{
        matrix::dcrt_poly::DCRTPolyMatrix,
        poly::{PolyParams, dcrt::params::DCRTPolyParams},
    };
    use mxx_runtime::{
        RuntimeValue, artifact::MemoryArtifactStore, backend::poly::cpu_backend, execute,
        transcript::SamplingMode,
    };
    use num_bigint::BigInt;
    use std::collections::BTreeMap;

    #[test]
    fn test_bgg_shared_family_rhs_reuses_one_decomposition() {
        use crate::program::{
            FamilyProgram, FamilyRange, FamilyRhsSource, FamilyWireSource, Gate, ProgramFamilyId,
            RhsId, WireId,
        };
        use mxx_dsl::{Family, GraphValue, GraphValueSchema};
        use mxx_runtime::ArtifactStore;
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .ok()
            .map(|value| value.parse().unwrap())
            .unwrap_or(16);
        let parameters = DCRTPolyParams::new(n, 1, 17, 2, None, None);
        let layout = BggSamplerLayout {
            modulus: BigInt::from(parameters.modulus().as_ref().clone()).into(),
            ring_dimension: (n as usize).into(),
            secret_dimension: 2,
            digit_count: parameters.modulus_digits(),
            gadget_base: 4.into(),
        };
        let exponent = ExpLayout { stride_log2: 0, width_log2: 2 };
        let sampler = ExponentLutEncodingSampler {
            layout: layout.clone(),
            exponent_layout: exponent,
            gaussian_sigma: None,
            gaussian_max_coefficient_bound: None,
        };
        let ring = layout.ring();
        let secret = sampler.sample_secret().unwrap();
        let payload_secret = sampler.sample_secret().unwrap();
        let hash_key = ring.bytes_input("hash-key", 32);
        let keys =
            ExponentLutPublicKeySampler { layout: layout.clone(), exponent_layout: exponent }
                .sample_input_keys(hash_key.clone(), b"shared-family-inputs".as_slice(), 3)
                .unwrap();
        let plaintexts =
            (0..3).map(|i| exponent.mono(&ring, &BigInt::from(i % 2)).unwrap()).collect::<Vec<_>>();
        let encodings = sampler
            .sample_encodings_for_public_matrices(
                secret.clone(),
                Some(payload_secret.clone()),
                &keys,
                &plaintexts,
            )
            .unwrap();
        let nu = exponent.mono(&ring, &1.into()).unwrap();
        let rhs = sampler
            .sample_cross_secret_rhs(
                payload_secret.clone(),
                payload_secret.clone(),
                nu.clone(),
                hash_key.clone(),
                b"shared-family-rhs".as_slice(),
            )
            .unwrap();
        let mut builder = ExponentLutProgramBuilder::new();
        builder.set_layout(exponent, n as usize).unwrap();
        let members = builder.wire_family(2).unwrap();
        let rhs_family = builder.rhs_family(2).unwrap();
        let rhs_id = builder.rhs_input(rhs_family, 2, 2).unwrap();
        let input = WireId::from_index(0);
        let product = WireId::from_index(1);
        let local_rhs = RhsId::from_index(0);
        let output = builder
            .family(
                3,
                FamilyProgram::new(
                    BTreeMap::from([(input, FamilyWireSource::Member(members))]),
                    BTreeMap::from([(local_rhs, FamilyRhsSource::Broadcast(rhs_id))]),
                    vec![Gate::MulPrivate { lhs: input, rhs: local_rhs, output: product }],
                    product,
                )
                .unwrap(),
            )
            .unwrap();
        builder.output(output).unwrap();
        let program = builder.build().unwrap();
        let mut ranges = ProgramFamilyBindings::new();
        ranges.bind(ProgramFamilyId::from_index(0), FamilyRange::bounded(1, 2, 3).unwrap());
        let public_members = Family::pack(keys.clone()).unwrap();
        let public_rhs = BTreeMap::from([(rhs_id, rhs)]);
        let at = crate::program::ProgramGateInstance::FamilyGate {
            step: 0,
            gate: 0,
            family: ProgramFamilyId::from_index(0),
        };
        let mut shared_cache = public_key::ProgramCacheBindings::new();
        shared_cache.prepare_private_rhs(
            at,
            &public_rhs[&rhs_id],
            &BggPublicKeyCompiler {
                ring: ring.clone(),
                base: 4.into(),
                digit_count: layout.digit_count.into(),
            },
        );
        let prepared = PreparedProgram::new(
            BggScheme::new(
                BggPublicKeyCompiler {
                    ring: ring.clone(),
                    base: 4.into(),
                    digit_count: layout.digit_count.into(),
                },
                exponent,
            ),
            program,
            &ProgramBindings::new(
                &BTreeMap::new(),
                &public_rhs,
                &BTreeMap::new(),
                &BTreeMap::from([(members, (public_members, FamilyRange::full(3).unwrap()))]),
                &BTreeMap::new(),
                None,
            ),
            ranges,
            Some(shared_cache),
        )
        .unwrap();
        // A common family body can capture one outer prepared RHS and return
        // its local scalar cache explicitly through the structural loop.
        {
            use crate::program::{ExponentProgramLowering, ProgramGateInstance};
            let at = ProgramGateInstance::FamilyGate {
                step: 0,
                gate: 0,
                family: ProgramFamilyId::from_index(0),
            };
            let rhs = &public_rhs[&rhs_id];
            let mut seed = public_key::ProgramCacheBindings::new();
            let shared = seed.prepare_private_rhs(at, rhs, &prepared.scheme.public.public_key);
            let members =
                Family::pack(keys.iter().map(|key| key.matrix.clone()).collect()).unwrap();
            let body = mxx_dsl::parallel(2, |index| {
                let mut local = public_key::ProgramCacheBindings::new();
                let mut evaluator =
                    prepared.scheme.public_evaluator(&mut local, &seed, &prepared.ranges);
                let output = evaluator
                    .mul_private(
                        &BggPublicKeyWire {
                            matrix: members.at(index.add(mxx_dsl::Int::constant(1))),
                            reveal_plaintext: false,
                        },
                        rhs,
                        None,
                        at,
                    )
                    .map_err(|_| mxx_dsl::DslError::Schema)?;
                assert!(
                    local.gate_entry(at).is_none(),
                    "shared cache is not copied into body output"
                );
                assert!(
                    output
                        .matrix
                        .value_handle()
                        .node()
                        .arguments()
                        .iter()
                        .any(|argument| { argument.node() == shared.value_handle().node() }),
                    "the scalar product uses the captured decomposition"
                );
                Ok((output, local))
            })
            .unwrap();
            let graph = DslContext::new("bgg-shared-rhs-scalar-body")
                .public_output("members", body)
                .unwrap()
                .build()
                .unwrap();
            let decomposition_count = |scope: &mxx_ir_core::GraphScope| {
                scope
                    .nodes()
                    .iter()
                    .filter(|node| {
                        matches!(node.kind(), mxx_ir_core::node::NodeKind::GadgetDecompose { .. })
                    })
                    .count()
            };
            assert_eq!(decomposition_count(graph.graph.root_scope()), 1);
            assert_eq!(graph.graph.scopes().values().map(decomposition_count).sum::<usize>(), 1);
            graph.validate(&ParamEnv::default()).unwrap();
        }
        assert!(prepared.shared_cache.gate_entry(at).is_some());
        assert!(
            prepared.family_caches[&ProgramFamilyId::from_index(0)]
                .at(mxx_dsl::Int::constant(0))
                .gate_entry(at)
                .is_none()
        );
        let member_encodings = Family::pack(encodings.clone()).unwrap();
        let outputs = prepared
            .evaluate(&ProgramBindings::new(
                &BTreeMap::new(),
                &BTreeMap::from([(rhs_id, ())]),
                &BTreeMap::new(),
                &BTreeMap::from([(members, (member_encodings, FamilyRange::full(3).unwrap()))]),
                &BTreeMap::new(),
                None,
            ))
            .unwrap();
        let actual = &outputs[&output];
        let key = &prepared.public_values()[&output];
        let expected = (plaintexts[1].clone() + plaintexts[2].clone()) * nu;
        let residual = actual.vector.clone() - secret.clone() * key.matrix.clone() +
            expected.clone() * payload_secret.clone() * ring.gadget(2, 4, layout.digit_count);
        let vectors =
            Family::pack(encodings.iter().map(|value| value.vector.clone()).collect()).unwrap();
        let vector_schema = vectors.schema();
        let values = Family::pack(plaintexts.clone()).unwrap();
        let plaintext_schema = values.schema();
        let binding = (
            layout.modulus.clone(),
            layout.ring_dimension.clone(),
            layout.secret_dimension,
            layout.gadget_base.clone(),
            layout.digit_count,
            exponent,
        );
        let (graph, descriptor) = prepared
            .export(binding.clone(), DslContext::new("bgg-shared-family-rhs"), "program")
            .unwrap();
        let identity = descriptor.identity().unwrap();
        let descriptor = serde_json::from_slice(&serde_json::to_vec(&descriptor).unwrap()).unwrap();
        let graph = graph
            .public_output("plaintexts", values)
            .unwrap()
            .private_output("encodings", vectors)
            .unwrap()
            .private_output("mask", secret.clone())
            .unwrap()
            .private_output("payload", payload_secret.clone())
            .unwrap()
            .output("residual", residual)
            .unwrap()
            .build()
            .unwrap();
        let decompositions = graph
            .graph
            .scopes()
            .values()
            .flat_map(|scope| scope.nodes())
            .filter(|node| {
                matches!(node.kind(), mxx_ir_core::node::NodeKind::GadgetDecompose { .. })
            })
            .count();
        assert_eq!(decompositions, 1);
        let graph = graph.validate(&ParamEnv::default()).unwrap();
        std::fs::create_dir_all("test_data/bgg_shared_family_rhs_artifacts").unwrap();
        let directory = tempfile::tempdir_in("test_data/bgg_shared_family_rhs_artifacts").unwrap();
        let mut store = mxx_runtime::FileArtifactStore::new(directory.path()).unwrap();
        let results = execute(
            &graph,
            &mut cpu_backend([parameters.clone()]),
            BTreeMap::from([(
                "hash-key".into(),
                RuntimeValue::Bytes(rand::random::<[u8; 32]>().to_vec()),
            )]),
            &mut store,
            SamplingMode::Fresh,
        )
        .unwrap();
        let RuntimeValue::Matrix(matrix) = &results.outputs["residual"] else { panic!("matrix") };
        assert_eq!(
            matrix.as_ref(),
            &DCRTPolyMatrix::zero(&parameters, 1, layout.public_key_columns())
        );
        let production = results.production_id.unwrap();
        drop(store);
        let mut store = mxx_runtime::FileArtifactStore::open(directory.path()).unwrap();
        let manifests =
            BTreeMap::from([(production.clone(), store.load_manifest(&production).unwrap())]);
        let prepared = PreparedProgram::import(
            binding,
            prepared.scheme,
            prepared.program,
            prepared.ranges,
            &descriptor,
            production.clone(),
            identity,
        )
        .unwrap();
        let decomposition = prepared.shared_cache.gate_entry(at).unwrap();
        let vectors = vector_schema
            .artifact_input(production.clone(), "encodings", ArtifactConfidentiality::Private)
            .unwrap();
        let plaintexts = plaintext_schema
            .artifact_input(production.clone(), "plaintexts", ArtifactConfidentiality::Public)
            .unwrap();
        let members = BTreeMap::from([(
            members,
            (
                vectors
                    .zip_map(plaintexts, |vector, plaintext| {
                        Ok(BggEncodingWire { vector, plaintext: Some(plaintext) })
                    })
                    .unwrap(),
                FamilyRange::full(3).unwrap(),
            ),
        )]);
        let mask = ring.artifact_input(
            production.clone(),
            "mask",
            (1, 2),
            ArtifactConfidentiality::Private,
        );
        let payload = ring.artifact_input(
            production.clone(),
            "payload",
            (1, 2),
            ArtifactConfidentiality::Private,
        );
        let key = prepared.public_values()[&output].matrix.clone();
        let mut consumer = DslContext::new("bgg-shared-family-rhs-reopened");
        for output_index in 0..2 {
            let cached = prepared.shared_cache.gate_entry(at).unwrap();
            assert_eq!(cached.value_handle().node(), decomposition.value_handle().node());
            let outputs = prepared
                .evaluate(&ProgramBindings::new(
                    &BTreeMap::new(),
                    &BTreeMap::from([(rhs_id, ())]),
                    &BTreeMap::new(),
                    &members,
                    &BTreeMap::new(),
                    None,
                ))
                .unwrap();
            let actual = &outputs[&output];
            let residual = actual.vector.clone() - mask.clone() * key.clone() +
                expected.clone() * payload.clone() * ring.gadget(2, 4, layout.digit_count);
            consumer = consumer.output(format!("residual-{output_index}"), residual).unwrap();
        }
        let consumer = consumer
            .build()
            .unwrap()
            .validate_with_manifests(&ParamEnv::default(), &manifests)
            .unwrap();
        assert!(!consumer.source.scopes().values().flat_map(|scope| scope.nodes()).any(
            |node| matches!(node.kind(), mxx_ir_core::node::NodeKind::GadgetDecompose { .. })
        ));
        let results = execute(
            &consumer,
            &mut cpu_backend([parameters.clone()]),
            BTreeMap::new(),
            &mut store,
            SamplingMode::Fresh,
        )
        .unwrap();
        for output in results.outputs.values() {
            let RuntimeValue::Matrix(matrix) = output else { panic!("matrix") };
            assert_eq!(
                matrix.as_ref(),
                &DCRTPolyMatrix::zero(&parameters, 1, layout.public_key_columns())
            );
        }
    }

    #[test]
    fn test_bgg_common_prf_composes_fixed_rhs_before_decomposition() {
        use crate::{
            noise_refresh::prf::{
                PreparedSparseLwrPrf, PrfSelectionInputs, SparseLwrPrfProfile, SparseLwrPrfProgram,
            },
            program::FamilyRange,
        };
        use mxx_dsl::{Family, GraphValueSchema};
        use mxx_runtime::ArtifactStore;
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .ok()
            .map(|v| v.parse().unwrap())
            .unwrap_or(16);
        let parameters = DCRTPolyParams::new(n, 1, 17, 2, None, None);
        let layout = BggSamplerLayout {
            modulus: BigInt::from(parameters.modulus().as_ref().clone()).into(),
            ring_dimension: (n as usize).into(),
            secret_dimension: 2,
            digit_count: parameters.modulus_digits(),
            gadget_base: 4.into(),
        };
        let exponent = ExpLayout { stride_log2: 0, width_log2: n.trailing_zeros() };
        let sampler = ExponentLutEncodingSampler {
            layout: layout.clone(),
            exponent_layout: exponent,
            gaussian_sigma: None,
            gaussian_max_coefficient_bound: None,
        };
        let ring = layout.ring();
        let mask = sampler.sample_secret().unwrap();
        let payload = sampler.sample_secret().unwrap();
        let seed = ring.bytes_input("hash-key", 32);
        let key = ExponentLutPublicKeySampler { layout: layout.clone(), exponent_layout: exponent }
            .sample_input_keys(seed.clone(), b"initial".as_slice(), 1)
            .unwrap()
            .remove(0);
        let scheme = BggScheme::new(
            BggPublicKeyCompiler {
                ring: ring.clone(),
                base: 4.into(),
                digit_count: layout.digit_count.into(),
            },
            exponent,
        );
        let program =
            SparseLwrPrfProgram::new(SparseLwrPrfProfile::new(exponent, 1).unwrap(), 4, 2).unwrap();
        let mut expected = [0usize; 3];
        let mut blocks = Vec::new();
        for step in 0..2 {
            let selected = 1 + usize::from(rand::random::<bool>());
            let selectors = (0..4)
                .map(|cell| {
                    sampler
                        .sample_cross_secret_rhs(
                            payload.clone(),
                            payload.clone(),
                            ring.polynomial([usize::from(
                                cell == 0 || cell == 3 || cell == selected,
                            )
                            .into()]),
                            seed.clone(),
                            format!("selector-{step}-{cell}").into_bytes(),
                        )
                        .unwrap()
                        .gsw_ciphertext()
                        .clone()
                })
                .collect();
            let mut labels = Vec::new();
            for (label, total) in expected.iter_mut().enumerate() {
                for cell in 0..4 {
                    let value = (label + 1) * (step + 1) + cell * (n as usize - 1);
                    labels.push(exponent.mono(&ring, &BigInt::from(value)).unwrap());
                    if cell == selected {
                        *total += value;
                    }
                }
            }
            blocks.push(
                PrfSelectionInputs::new(
                    encoding::SetupRhsFamily::new(Family::pack(selectors).unwrap()).unwrap(),
                    &Family::pack(labels).unwrap(),
                    FamilyRange::bounded(1, 2, 4).unwrap(),
                    rand::random(),
                )
                .unwrap(),
            );
        }
        let (public, prepared) = PreparedSparseLwrPrf::new(
            &scheme,
            program.clone(),
            crate::program::ProgramValue::Scalar(key.clone()),
            &blocks,
        )
        .unwrap();
        let initial = sampler
            .sample_encodings_for_public_matrices(
                mask.clone(),
                Some(payload.clone()),
                &[key],
                &[ring.polynomial([1.into()])],
            )
            .unwrap()
            .remove(0);
        let producer = DslContext::new("bgg-common-fixed-prf-cache")
            .private_output("initial", initial.vector.clone())
            .unwrap()
            .private_output("mask", mask.clone())
            .unwrap()
            .private_output("payload", payload.clone())
            .unwrap();
        let public_schema = mxx_dsl::GraphValue::schema(&public);
        let producer = producer.public_output("public-key", public).unwrap();
        let (producer, descriptor) = prepared.export(&scheme, producer, "fixed").unwrap();
        let identity = descriptor.identity().unwrap();
        let descriptor = serde_json::from_slice(&serde_json::to_vec(&descriptor).unwrap()).unwrap();
        let producer = producer.build().unwrap().validate(&ParamEnv::default()).unwrap();
        let decomposition_count = producer
            .source
            .scopes()
            .values()
            .flat_map(|scope| scope.nodes())
            .filter(|node| {
                matches!(node.kind(), mxx_ir_core::node::NodeKind::GadgetDecompose { .. })
            })
            .count();
        assert_eq!(decomposition_count, 2, "one structural decomposition body per composed step");
        std::fs::create_dir_all("test_data/bgg_common_fixed_prf_artifacts").unwrap();
        let directory = tempfile::tempdir_in("test_data/bgg_common_fixed_prf_artifacts").unwrap();
        let mut store = mxx_runtime::FileArtifactStore::new(directory.path()).unwrap();
        let produced = execute(
            &producer,
            &mut cpu_backend([parameters.clone()]),
            BTreeMap::from([(
                "hash-key".into(),
                RuntimeValue::Bytes(rand::random::<[u8; 32]>().to_vec()),
            )]),
            &mut store,
            SamplingMode::Fresh,
        )
        .unwrap();
        let production = produced.production_id.unwrap();
        drop(store);
        let mut store = mxx_runtime::FileArtifactStore::open(directory.path()).unwrap();
        let manifests =
            BTreeMap::from([(production.clone(), store.load_manifest(&production).unwrap())]);
        let prepared = PreparedSparseLwrPrf::import(
            &scheme,
            program,
            &descriptor,
            production.clone(),
            identity,
        )
        .unwrap();
        let initial = BggEncodingWire {
            vector: ring.artifact_input(
                production.clone(),
                "initial",
                (1, layout.public_key_columns()),
                ArtifactConfidentiality::Private,
            ),
            plaintext: None,
        };
        let mask = ring.artifact_input(
            production.clone(),
            "mask",
            (1, layout.secret_dimension),
            ArtifactConfidentiality::Private,
        );
        let payload = ring.artifact_input(
            production.clone(),
            "payload",
            (1, layout.secret_dimension),
            ArtifactConfidentiality::Private,
        );
        let output =
            prepared.evaluate(&scheme, crate::program::ProgramValue::Scalar(initial)).unwrap();
        let crate::program::ProgramValue::Family(public) = public_schema
            .artifact_input(production.clone(), "public-key", ArtifactConfidentiality::Public)
            .unwrap()
        else {
            panic!("batched public output")
        };
        let crate::program::ProgramValue::Family(output) = output else {
            panic!("batched encoding output")
        };
        assert_eq!(public.count(), output.count());
        assert_eq!(output.count().evaluate(&ParamEnv::default()).unwrap(), 3.into());
        assert!(mxx_dsl::GraphValue::schema(&output).element.plaintext.is_none());
        let gadget = ring.gadget(2, 4, layout.digit_count);
        let mut graph = DslContext::new("bgg-common-fixed-prf");
        for (label, value) in expected.into_iter().enumerate() {
            let residual = output.at(label).vector - mask.clone() * public.at(label).matrix +
                exponent.mono(&ring, &BigInt::from(value)).unwrap() *
                    payload.clone() *
                    gadget.clone();
            graph = graph.output(format!("residual-{label}"), residual).unwrap();
        }
        let graph = graph
            .build()
            .unwrap()
            .validate_with_manifests(&ParamEnv::default(), &manifests)
            .unwrap();
        assert!(
            !graph.source.scopes().values().flat_map(|scope| scope.nodes()).any(|node| {
                matches!(node.kind(), mxx_ir_core::node::NodeKind::GadgetDecompose { .. })
            }),
            "reopened online evaluation consumes cached decompositions"
        );
        assert!(!graph.source.scopes().values().flat_map(|scope| scope.nodes()).any(|node| {
            matches!(node.kind(), mxx_ir_core::node::NodeKind::BlockModSwitch { .. })
        }));
        let result = execute(
            &graph,
            &mut cpu_backend([parameters.clone()]),
            BTreeMap::new(),
            &mut store,
            SamplingMode::Fresh,
        )
        .unwrap();
        let zero = DCRTPolyMatrix::zero(&parameters, 1, layout.public_key_columns());
        for output in result.outputs.values() {
            let RuntimeValue::Matrix(matrix) = output else { panic!("matrix residual") };
            assert_eq!(matrix.as_ref(), &zero);
        }
    }

    #[test]
    fn test_bgg_scheme_binary_lut_all_inputs() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .ok()
            .map(|v| v.parse().unwrap())
            .unwrap_or(16);
        let parameters = DCRTPolyParams::new(n, 1, 17, 2, None, None);
        let layout = BggSamplerLayout {
            modulus: BigInt::from(parameters.modulus().as_ref().clone()).into(),
            ring_dimension: (n as usize).into(),
            secret_dimension: 2,
            digit_count: parameters.modulus_digits(),
            gadget_base: 4.into(),
        };
        let exponent = ExpLayout { stride_log2: 0, width_log2: 3 };
        let sampler = ExponentLutEncodingSampler {
            layout: layout.clone(),
            exponent_layout: exponent,
            gaussian_sigma: None,
            gaussian_max_coefficient_bound: None,
        };
        let public_sampler =
            ExponentLutPublicKeySampler { layout: layout.clone(), exponent_layout: exponent };
        let ring = layout.ring();
        let secret = sampler.sample_secret().unwrap();
        let payload_secret = sampler.sample_secret().unwrap();
        let hash_key = ring.bytes_input("hash-key", 32);
        let key = public_sampler
            .sample_input_keys(hash_key.clone(), b"input".as_slice(), 1)
            .unwrap()
            .remove(0);
        let bank = sampler
            .sample_mask_bank(secret.clone(), hash_key.clone(), 8, b"bank".as_slice())
            .unwrap();
        let public_bank =
            public_sampler.sample_mask_bank(hash_key.clone(), 8, b"bank".as_slice()).unwrap();
        let table = LutTable::binary(2, 4, 8, vec![0, 3, 1, 6, 2, 7, 5, 4]).unwrap();
        let mut builder = ExponentLutProgramBuilder::new();
        builder.set_layout(exponent, n as usize).unwrap();
        let input = builder
            .input(2, ProgramWireType { stride_log2: 0, knowledge: AttributeKnowledge::Private })
            .unwrap();
        let family = builder.rhs_family(4).unwrap();
        let rhs_id = builder.rhs_input(family, 2, 4).unwrap();
        let lut = builder.register_lut(table.clone()).unwrap();
        let output = builder.lut(builder.input_wire(input).unwrap(), lut, Some(rhs_id)).unwrap();
        builder.output(output).unwrap();
        let program = builder.build().unwrap();
        assert_eq!(program.gates().len(), 2);
        let crate::program::Gate::Lut { lut_id: lut, .. } = program.gates()[1] else {
            panic!("unary LUT")
        };
        let table = program.lut(lut).unwrap();
        let mut graph = DslContext::new("bgg-binary-scheme");
        let helpers = sampler
            .sample_helpers_for_lut(
                secret.clone(),
                Some(payload_secret.clone()),
                hash_key.clone(),
                &table,
                &bank,
                b"shared-lut-helpers".as_slice(),
            )
            .unwrap();
        let public_helpers = helpers
            .iter()
            .map(|helper| {
                LutPublicHelper::with_mask_bank(
                    helper.sigma(),
                    helper.switch().public_projection(),
                    public_bank.clone(),
                )
                .unwrap()
            })
            .collect();
        let gate = crate::program::ProgramGateInstance::Gate { step: 1 };
        let public_helpers = BTreeMap::from([(
            gate,
            crate::program::ProgramValue::Scalar(
                LutPublicHelperSet::new(&table, public_helpers).unwrap(),
            ),
        )]);
        let encoding_helpers = BTreeMap::from([(
            gate,
            crate::program::ProgramValue::Scalar(LutHelperSet::new(&table, helpers).unwrap()),
        )]);
        for y in 0..4 {
            let rhs_plain = exponent.mono(&ring, &BigInt::from(2 * y)).unwrap();
            let rhs = sampler
                .sample_cross_secret_rhs(
                    payload_secret.clone(),
                    payload_secret.clone(),
                    rhs_plain,
                    hash_key.clone(),
                    format!("rhs-{y}").into_bytes(),
                )
                .unwrap();
            let public_inputs = BTreeMap::from([(input, key.clone())]);
            let public_rhs = BTreeMap::from([(rhs_id, rhs)]);
            let prepared = PreparedProgram::new(
                BggScheme::new(
                    BggPublicKeyCompiler {
                        ring: ring.clone(),
                        base: 4.into(),
                        digit_count: layout.digit_count.into(),
                    },
                    exponent,
                ),
                program.clone(),
                &ProgramBindings::new(
                    &public_inputs,
                    &public_rhs,
                    &BTreeMap::new(),
                    &BTreeMap::new(),
                    &BTreeMap::new(),
                    Some(&public_helpers),
                ),
                ProgramFamilyBindings::new(),
                None,
            )
            .unwrap();
            assert!(
                prepared
                    .cache()
                    .gate_entry(crate::program::ProgramGateInstance::Gate { step: 0 })
                    .is_some()
            );
            assert!(
                prepared
                    .cache()
                    .lut_entry(crate::program::ProgramGateInstance::Gate { step: 1 })
                    .is_some()
            );
            let output_key = &prepared.public_values()[&output];
            for x in 0..2 {
                let mu = exponent.mono(&ring, &BigInt::from(x)).unwrap();
                let mut encoding = sampler
                    .sample_encodings_for_public_matrices(
                        secret.clone(),
                        Some(payload_secret.clone()),
                        &[key.clone()],
                        &[mu.clone()],
                    )
                    .unwrap()
                    .remove(0);
                // Known left plaintext must not survive as incorrect unary
                // metadata when the second operand remains private.
                encoding.plaintext = Some(mu);
                let inputs = BTreeMap::from([(input, encoding)]);
                let rhs = BTreeMap::from([(rhs_id, ())]);
                let values = prepared
                    .evaluate(&ProgramBindings::new(
                        &inputs,
                        &rhs,
                        &BTreeMap::new(),
                        &BTreeMap::new(),
                        &BTreeMap::new(),
                        Some(&encoding_helpers),
                    ))
                    .unwrap();
                let actual = &values[&output];
                assert!(actual.plaintext.is_none());
                let mu = exponent.mono(&ring, &BigInt::from(table.values()[x + 2 * y])).unwrap();
                let residual = actual.vector.clone() - secret.clone() * output_key.matrix.clone() +
                    mu * payload_secret.clone() * ring.gadget(2, 4, layout.digit_count);
                graph = graph.output(format!("residual-{x}-{y}"), residual).unwrap();
            }
        }
        let graph = graph.build().unwrap().validate(&ParamEnv::default()).unwrap();
        let results = execute(
            &graph,
            &mut cpu_backend([parameters.clone()]),
            BTreeMap::from([(
                "hash-key".into(),
                RuntimeValue::Bytes(rand::random::<[u8; 32]>().to_vec()),
            )]),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .unwrap();
        let zero = DCRTPolyMatrix::zero(&parameters, 1, layout.public_key_columns());
        for value in results.outputs.values() {
            let RuntimeValue::Matrix(matrix) = value else { panic!("residual matrix") };
            assert_eq!(matrix.as_ref(), &zero);
        }
    }
}

/// Artifact names for one setup-fixed RHS ciphertext.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExponentRhsPackageArtifactNames {
    /// The fixed GSW ciphertext `C`.
    pub gsw_ciphertext: String,
}

/// Setup-fixed RHS package consumed by the encoding evaluator.
///
/// `C` is deliberately the only payload. Its value is fixed when the package
/// is prepared, so the public-key projection is allowed to use the same
/// matrix and computes `A * G^{-1}(C)`.
#[derive(Clone, PartialEq)]
pub struct ExponentRhsPackage<M = Mat> {
    gsw_ciphertext: M,
}

impl mxx_dsl::GraphValue for ExponentRhsPackage {
    type Schema = ExponentRhsPackage<mxx_dsl::MatType>;
    fn flatten(&self) -> Vec<mxx_ir_core::ValueHandle> {
        mxx_dsl::GraphValue::flatten(&self.gsw_ciphertext)
    }
    fn schema(&self) -> Self::Schema {
        ExponentRhsPackage { gsw_ciphertext: mxx_dsl::GraphValue::schema(&self.gsw_ciphertext) }
    }
    fn from_values(
        schema: &Self::Schema,
        values: &[mxx_ir_core::ValueHandle],
    ) -> Result<Self, mxx_dsl::DslError> {
        Ok(Self {
            gsw_ciphertext: <Mat as mxx_dsl::GraphValue>::from_values(
                &schema.gsw_ciphertext,
                values,
            )?,
        })
    }
}
impl mxx_dsl::GraphValueSchema for ExponentRhsPackage<mxx_dsl::MatType> {
    type Value = ExponentRhsPackage;
    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        ExponentRhsPackage {
            gsw_ciphertext: mxx_dsl::GraphValueSchema::placeholders_from(
                &self.gsw_ciphertext,
                next,
            ),
        }
    }
    fn wire_types(&self) -> Vec<mxx_ir_core::types::WireType> {
        mxx_dsl::GraphValueSchema::wire_types(&self.gsw_ciphertext)
    }
}

/// Metadata stored by artifact producers so imports can reject a ciphertext
/// prepared for a different source/target BGG layout. These fields are
/// import-only provenance and are not runtime RHS data.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(crate) struct ManifestRhsMetadata {
    pub(crate) source: ManifestSecretMetadata,
    pub(crate) target: ManifestSecretMetadata,
}

impl ManifestRhsMetadata {
    pub(crate) fn from_layouts(
        source: &mxx_bgg::BggSamplerLayout,
        source_identity: [u8; 32],
        target: &mxx_bgg::BggSamplerLayout,
        target_identity: [u8; 32],
    ) -> Self {
        Self {
            source: ManifestSecretMetadata::from_layout(source, source_identity),
            target: ManifestSecretMetadata::from_layout(target, target_identity),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(crate) struct ManifestSecretMetadata {
    pub(crate) modulus: mxx_ir_core::IntExpr,
    pub(crate) ring_dimension: mxx_ir_core::IntExpr,
    pub(crate) secret_dimension: usize,
    pub(crate) digit_count: usize,
    pub(crate) gadget_base: mxx_ir_core::IntExpr,
    pub(crate) identity: [u8; 32],
}

impl ManifestSecretMetadata {
    pub(crate) fn from_layout(layout: &mxx_bgg::BggSamplerLayout, identity: [u8; 32]) -> Self {
        Self {
            modulus: layout.modulus.clone(),
            ring_dimension: layout.ring_dimension.clone(),
            secret_dimension: layout.secret_dimension,
            digit_count: layout.digit_count,
            gadget_base: layout.gadget_base.clone(),
            identity,
        }
    }
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum ExponentRhsPackageError {
    #[error("RHS ciphertext has an invalid shape")]
    InvalidShape,
}

impl ExponentRhsPackage {
    /// Constructs a package from the one fixed GSW ciphertext.
    pub(crate) fn new(gsw_ciphertext: Mat) -> Result<Self, ExponentRhsPackageError> {
        let matrix_type = gsw_ciphertext.matrix_type();
        let rows =
            matrix_type.rows.evaluate(&ParamEnv::default()).ok().and_then(|value| value.to_usize());
        let columns = matrix_type
            .columns
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize());
        if rows != Some(2) || columns.is_none() || columns == Some(0) {
            return Err(ExponentRhsPackageError::InvalidShape);
        }
        Ok(Self { gsw_ciphertext })
    }

    /// Imports the fixed ciphertext after checking its manifest identity and
    /// concrete matrix type. The ciphertext `C` is an ordinary matrix: its
    /// coefficients live in the full ciphertext modulus and therefore it is
    /// not a bounded compact `SmallMatrix` artifact. The bounded object is
    /// the gadget decomposition used by `mul_small_rhs`, which is derived
    /// from this imported ciphertext at the operation boundary.
    pub fn artifact_input(
        production_id: ProductionId,
        manifest: &Manifest,
        names: ExponentRhsPackageArtifactNames,
        expected_source: &mxx_bgg::BggSamplerLayout,
        expected_target: &mxx_bgg::BggSamplerLayout,
    ) -> Result<Self, ExponentArtifactImportError> {
        if manifest.ir_version != mxx_ir_core::encoding::IR_VERSION {
            return Err(ExponentArtifactImportError::InvalidMetadata);
        }
        if manifest.production_id != production_id {
            return Err(ExponentArtifactImportError::ProductionMismatch);
        }
        let artifact = manifest
            .artifacts
            .get(&names.gsw_ciphertext)
            .ok_or(ExponentArtifactImportError::MissingArtifact)?;
        if artifact.confidentiality != ArtifactConfidentiality::Public ||
            artifact.family_count.is_some()
        {
            return Err(ExponentArtifactImportError::ConfidentialityMismatch);
        }
        mxx_ir_core::artifact::validate_manifest(manifest)
            .map_err(|_| ExponentArtifactImportError::InvalidMetadata)?;
        let modulus = expected_source
            .modulus
            .evaluate(&ParamEnv::default())
            .map_err(|_| ExponentArtifactImportError::MatrixTypeMismatch)?;
        let ring_dimension = expected_source
            .ring_dimension
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(ExponentArtifactImportError::MatrixTypeMismatch)?;
        let target_columns = expected_target
            .secret_dimension
            .checked_mul(expected_target.digit_count)
            .filter(|columns| *columns > 0)
            .ok_or(ExponentArtifactImportError::MatrixTypeMismatch)?;
        let expected = ArtifactType::Matrix(ConcreteMatrixType {
            modulus,
            ring_dimension,
            rows: expected_source.secret_dimension,
            columns: target_columns,
        });
        if artifact.artifact_type != expected {
            return Err(ExponentArtifactImportError::MatrixTypeMismatch);
        }
        let ciphertext = expected_source.ring().artifact_input(
            production_id,
            names.gsw_ciphertext,
            (expected_source.secret_dimension, target_columns),
            ArtifactConfidentiality::Public,
        );
        Self::new(ciphertext).map_err(|_| ExponentArtifactImportError::MatrixTypeMismatch)
    }

    pub(crate) fn gsw_ciphertext(&self) -> &Mat {
        &self.gsw_ciphertext
    }

    /// Returns the public setup-fixed descriptor. The same fixed ciphertext
    /// is intentionally retained because it is an input to public Fuse.
    pub fn public_projection(&self) -> ExponentRhsPackage {
        self.clone()
    }
}

#[cfg(test)]
mod rhs_tests {
    use super::*;
    use mxx_bgg::{BggPublicKeyCompiler, BggSamplerLayout};
    use mxx_dsl::DslContext;
    use mxx_ir_core::{
        ParamEnv,
        artifact::{ArtifactType, ManifestArtifact, SpecHash},
        node::NodeKind,
    };
    use std::collections::BTreeMap;

    const ARTIFACT_NAME: &str = "fixed-c";

    fn sampler(gadget_base: usize) -> BggSamplerLayout {
        BggSamplerLayout {
            modulus: 97.into(),
            ring_dimension: 4.into(),
            secret_dimension: 2,
            digit_count: 1,
            gadget_base: gadget_base.into(),
        }
    }

    fn manifest_fixture()
    -> (ProductionId, Manifest, ExponentRhsPackageArtifactNames, BggSamplerLayout, BggSamplerLayout)
    {
        let production_id = ProductionId { spec_hash: SpecHash([1; 32]), execution_nonce: [2; 32] };
        let source = sampler(4);
        let target = sampler(8);
        let artifact = ManifestArtifact {
            artifact_type: ArtifactType::Matrix(ConcreteMatrixType {
                modulus: 97.into(),
                ring_dimension: 4,
                rows: source.secret_dimension,
                columns: target.public_key_columns(),
            }),
            family_count: None,
            confidentiality: ArtifactConfidentiality::Public,
            content_hash: Some([5; 32]),
            layout: None,
        };
        let names = ExponentRhsPackageArtifactNames { gsw_ciphertext: ARTIFACT_NAME.to_owned() };
        let manifest = Manifest {
            ir_version: mxx_ir_core::encoding::IR_VERSION,
            production_id: production_id.clone(),
            artifacts: BTreeMap::from([(ARTIFACT_NAME.to_owned(), artifact)]),
        };
        (production_id, manifest, names, source, target)
    }

    fn import_fixture() -> (ExponentRhsPackage, ProductionId, Manifest) {
        let (production_id, manifest, names, source, target) = manifest_fixture();
        let package = ExponentRhsPackage::artifact_input(
            production_id.clone(),
            &manifest,
            names,
            &source,
            &target,
        )
        .unwrap();
        (package, production_id, manifest)
    }

    #[test]
    fn imports_public_fixed_c_and_preserves_public_artifact_binding() {
        let (package, production_id, _) = import_fixture();
        let NodeKind::Input { artifact: Some(artifact), .. } =
            package.gsw_ciphertext().value_handle().node().kind()
        else {
            panic!("imported fixed C must remain an artifact input")
        };
        assert_eq!(artifact.production_id, production_id);
        assert_eq!(artifact.artifact_name, ARTIFACT_NAME);
        assert_eq!(artifact.confidentiality, ArtifactConfidentiality::Public);
    }

    #[test]
    fn rejects_wrong_source_matrix_schema() {
        let (production_id, manifest, names, _, target) = manifest_fixture();
        let wrong_layout = sampler(16);
        let mut wrong_layout = wrong_layout;
        wrong_layout.ring_dimension = 8.into();
        let wrong_layout = ExponentRhsPackage::artifact_input(
            production_id,
            &manifest,
            names,
            &wrong_layout,
            &target,
        );
        assert!(matches!(wrong_layout, Err(ExponentArtifactImportError::MatrixTypeMismatch)));
    }

    #[test]
    fn rejects_wrong_target_matrix_schema() {
        let (production_id, manifest, names, source, _) = manifest_fixture();
        let wrong_layout = sampler(16);
        let mut wrong_layout = wrong_layout;
        wrong_layout.secret_dimension = 3;
        let wrong_layout = ExponentRhsPackage::artifact_input(
            production_id,
            &manifest,
            names,
            &source,
            &wrong_layout,
        );
        assert!(matches!(wrong_layout, Err(ExponentArtifactImportError::MatrixTypeMismatch)));
    }

    #[test]
    fn rejects_wrong_confidentiality_and_matrix_type() {
        let (production_id, mut manifest, names, source, target) = manifest_fixture();
        manifest.artifacts.get_mut(ARTIFACT_NAME).unwrap().confidentiality =
            ArtifactConfidentiality::Private;
        assert!(matches!(
            ExponentRhsPackage::artifact_input(
                production_id.clone(),
                &manifest,
                names.clone(),
                &source,
                &target,
            ),
            Err(ExponentArtifactImportError::ConfidentialityMismatch)
        ));

        let artifact = manifest.artifacts.get_mut(ARTIFACT_NAME).unwrap();
        artifact.confidentiality = ArtifactConfidentiality::Public;
        artifact.artifact_type = ArtifactType::Matrix(ConcreteMatrixType {
            modulus: 97.into(),
            ring_dimension: 4,
            rows: 2,
            columns: 1,
        });
        assert!(matches!(
            ExponentRhsPackage::artifact_input(production_id, &manifest, names, &source, &target,),
            Err(ExponentArtifactImportError::MatrixTypeMismatch)
        ));
    }

    #[test]
    fn imported_public_fixed_c_feeds_public_compiler_graph() {
        let (package, production_id, manifest) = import_fixture();
        let source = sampler(4);
        let ring = source.ring();
        let compiler = mxx_exponent_lut_public_compiler(&source);
        let fused = compiler.fuse_public(&ring.input("public-a", (2, 2)), &package).unwrap();
        let graph = DslContext::new("rhs-public-import")
            .public_output("fused", fused)
            .unwrap()
            .build()
            .unwrap();
        graph
            .validate_with_manifests(
                &ParamEnv::default(),
                &BTreeMap::from([(production_id, manifest)]),
            )
            .unwrap();
    }

    fn mxx_exponent_lut_public_compiler(
        layout: &BggSamplerLayout,
    ) -> crate::bgg::public_key::ExponentLutPublicKeyCompiler {
        crate::bgg::public_key::ExponentLutPublicKeyCompiler::new(BggPublicKeyCompiler {
            ring: layout.ring(),
            base: layout.gadget_base.clone(),
            digit_count: layout.digit_count.into(),
        })
    }
}

pub struct BggRefreshRecomposition {
    pub plaintext_moduli: Vec<IntExpr>,
    pub coefficients: Vec<IntExpr>,
    pub full_modulus: IntExpr,
}

impl ExponentLUTRefreshScheme for BggScheme {
    type RefreshRow = Mat;
    type RefreshDecoder = Mat;
    type RefreshLevel = IntExpr;
    type RefreshRecomposition = BggRefreshRecomposition;
    fn add_refresh_rows(&self, lhs: Mat, rhs: Mat) -> Mat {
        lhs + rhs
    }
    fn switch_refresh_row(&self, row: Mat, level: &IntExpr) -> Mat {
        row.modulus_switch(level.clone())
    }
    fn apply_refresh_decoder(&self, switched: Mat, decoder: Mat) -> Mat {
        switched - decoder
    }
    fn extract_refresh_residue(&self, row: Mat, _level: &IntExpr) -> Mat {
        row
    }
    fn recompose_refresh_rows(
        &self,
        rows: Vec<Mat>,
        preparation: &BggRefreshRecomposition,
    ) -> BggEncodingWire {
        BggEncodingWire {
            vector: Mat::crt_recompose(
                rows,
                preparation.plaintext_moduli.clone(),
                preparation.coefficients.clone(),
                preparation.full_modulus.clone(),
            ),
            plaintext: None,
        }
    }
    type RefreshPublicRow = Mat;
    type RefreshTargetScale = Mat;
    fn add_refresh_public(&self, lhs: Mat, rhs: Mat) -> Mat {
        lhs + rhs
    }
    fn reduce_refresh_public(&self, value: Mat, level: &IntExpr) -> Mat {
        value.reduce_modulus(level.clone())
    }
    fn switch_refresh_public(&self, value: Mat, level: &IntExpr) -> Mat {
        value.modulus_switch(level.clone())
    }
    fn scale_refresh_public(&self, value: Mat, scale: Mat) -> Mat {
        scale * value
    }
    fn subtract_refresh_public(&self, lhs: Mat, rhs: Mat) -> Mat {
        lhs - rhs
    }
}

/// Public setup parameters for a shared low-modulus BGG decoder anchor.
pub struct BggRefreshAnchorParameters {
    pub components: usize,
    pub gadget_base: IntExpr,
    pub digits: usize,
    pub sigma: mxx_ir_core::RealExpr,
    pub preimage_bound: IntExpr,
}

impl crate::noise_refresh::ExponentLUTRefreshWitnessScheme for BggScheme {
    type RefreshScaling = (Mat, mxx_dsl::Preimage);
    type RefreshDecoderBase = Mat;
    type RefreshDecoderWitness = mxx_dsl::Preimage;

    fn scale_refresh_input(&self, state: &BggEncodingWire, scaling: &Self::RefreshScaling) -> Mat {
        self.encoding.bgg.large_scalar_mul(state, &scaling.0, scaling.1.clone()).vector
    }
    fn prepare_refresh_decoder(&self, base: Mat, witness: mxx_dsl::Preimage) -> Mat {
        witness.mul_small_rhs(base)
    }

    type RefreshPublicScaling = mxx_dsl::Preimage;

    fn route_refresh_row(&self, row: Mat, route: mxx_dsl::Preimage) -> Mat {
        route.mul_small_rhs(row)
    }

    fn scale_refresh_public_input(&self, state: Mat, scaling: mxx_dsl::Preimage) -> Mat {
        scaling.mul_small_rhs(state)
    }
    type RefreshAnchor = mxx_dsl::Trapdoor;
    type RefreshAnchorParameters = BggRefreshAnchorParameters;

    fn sample_refresh_anchor(
        &self,
        level: &IntExpr,
        parameters: &BggRefreshAnchorParameters,
    ) -> mxx_dsl::Trapdoor {
        let dimension = self.public.public_key.ring.matrix_type((1, 1)).ring_dimension;
        mxx_dsl::Ring::new(level.clone(), dimension).sample_trapdoor(
            parameters.components,
            parameters.sigma.clone(),
            parameters.gadget_base.clone(),
            parameters.digits,
            parameters.preimage_bound.clone(),
        )
    }
    fn refresh_anchor_public(&self, anchor: &mxx_dsl::Trapdoor) -> Mat {
        anchor.public_matrix()
    }
    fn sample_refresh_witness(&self, anchor: &mxx_dsl::Trapdoor, target: Mat) -> mxx_dsl::Preimage {
        let rows = anchor.public_matrix().matrix_type().columns.clone();
        let columns = target.matrix_type().columns.clone();
        anchor.sample_preimage(target, (rows, columns))
    }
    type RefreshSecret = Mat;
    type RefreshBaseParameters = (mxx_ir_core::RealExpr, IntExpr);

    fn sample_refresh_decoder_base(
        secret: &Mat,
        public: &Mat,
        level: &IntExpr,
        parameters: &Self::RefreshBaseParameters,
    ) -> Mat {
        let ring = mxx_dsl::Ring::new(level.clone(), public.matrix_type().ring_dimension.clone());
        let reduced = secret.clone().reduce_modulus(level.clone());
        let error = ring.gaussian(
            (1, public.matrix_type().columns.clone()),
            parameters.0.clone(),
            parameters.1.clone(),
        );
        reduced * public.clone() + error
    }
}

impl From<crate::noise_refresh::prf::PrfError> for BggError {
    fn from(error: crate::noise_refresh::prf::PrfError) -> Self {
        match error {
            crate::noise_refresh::prf::PrfError::InvalidSparseLwrBlock => {
                Self::InvalidSparseLwrBlock
            }
            crate::noise_refresh::prf::PrfError::Program(error) => Self::Program(error),
            crate::noise_refresh::prf::PrfError::Artifact(detail) => {
                Self::ArtifactTransport(detail)
            }
        }
    }
}

use crate::noise_refresh::pbc::{
    PbcError, PbcLayoutId, PbcPublicLayout, PbcSelectorArtifacts,
    artifacts::PbcSelectorFamilyMetadata,
};
use encoding::EncodingRhsFamily;
use mxx_bgg::BggSamplerLayout;
use mxx_dsl::{Family, Ring};
use public_key::PublicRhsFamily;
/// Selector artifact families supplied by the trusted artifact importer. A
/// family has one element per non-padding cell; dynamic indexing occurs only
/// inside the structural loop body.
#[allow(dead_code)]
pub(crate) struct PbcSelectorFamilyInputs {
    layout_id: crate::noise_refresh::pbc::PbcLayoutId,
    key_instance_id: [u8; 32],
    package_count: usize,
    gsw: Family<Mat>,
}

#[allow(dead_code)]
impl PbcSelectorFamilyInputs {
    /// Construct selector family handles from the validated artifact family.
    /// Names are derived from the public layout/key namespace and package
    /// schema, never from hidden selector values or a private schedule.  The
    /// imported family is validated against the exact public artifact
    /// manifest before being exposed to lowering.
    /// Binds artifact-backed private/public family handles for structural
    /// lowering. The sampler supplies only matrix dimensions; layout and
    /// artifact identities are checked before any family is exposed.
    pub(crate) fn from_artifacts(
        ring: &Ring,
        layout: &PbcPublicLayout,
        artifacts: &PbcSelectorArtifacts<ManifestRhsMetadata>,
        source: &BggSamplerLayout,
        source_identity: [u8; 32],
        target: &BggSamplerLayout,
        target_identity: [u8; 32],
    ) -> Result<Self, BggError> {
        layout.validate().map_err(|_| BggError::InvalidSparseLwrBlock)?;
        if artifacts.layout_id() != layout.layout_id {
            return Err(BggError::InvalidSparseLwrBlock);
        }
        let package_count = artifacts.package_count();
        if package_count == 0 {
            return Err(BggError::InvalidSparseLwrBlock);
        }
        artifacts
            .validate_family_binding(&pbc_selector_metadata(
                layout,
                artifacts.key_instance_id(),
                source,
                source_identity,
                target,
                target_identity,
            )?)
            .map_err(|_| BggError::InvalidSparseLwrBlock)?;
        let key_instance_id = artifacts.key_instance_id();
        let gsw_shape = (source.secret_dimension, target.public_key_columns());
        let gsw_name =
            crate::noise_refresh::pbc::artifacts::selector_family_artifact_name(artifacts, "gsw");

        let manifest = artifacts.validated_manifest().ok_or(BggError::InvalidSparseLwrBlock)?;
        crate::noise_refresh::pbc::artifacts::require_family_descriptor(
            manifest,
            &gsw_name,
            ArtifactType::Matrix(crate::noise_refresh::pbc::artifacts::concrete_matrix_type(
                ring, gsw_shape,
            )?),
            package_count,
            ArtifactConfidentiality::Public,
            true,
        )
        .map_err(|_| BggError::InvalidSparseLwrBlock)?;
        let production_id = artifacts.production_id().ok_or(BggError::InvalidSparseLwrBlock)?;
        let gsw = ring.family_artifact_input(
            production_id.clone(),
            gsw_name,
            package_count,
            gsw_shape,
            ArtifactConfidentiality::Public,
        );
        Ok(Self { layout_id: layout.layout_id, key_instance_id, package_count, gsw })
    }

    /// Test-only constructor for graphs whose runtime values are supplied
    /// directly. Production compilation must use `from_artifacts`, so an
    /// unvalidated family can never cross the artifact-backed boundary.
    #[cfg(test)]
    pub(crate) fn from_unvalidated_for_tests(
        ring: &Ring,
        layout: &PbcPublicLayout,
        artifacts: &PbcSelectorArtifacts<ManifestRhsMetadata>,
        sampler: &BggSamplerLayout,
    ) -> Result<Self, BggError> {
        layout.validate().map_err(|_| BggError::InvalidSparseLwrBlock)?;
        if artifacts.layout_id() != layout.layout_id {
            return Err(BggError::InvalidSparseLwrBlock);
        }
        let package_count = artifacts.package_count();
        if package_count == 0 {
            return Err(BggError::InvalidSparseLwrBlock);
        }
        let key_instance_id = artifacts.key_instance_id();
        let gsw_shape = (sampler.secret_dimension, sampler.public_key_columns());
        let gsw = ring.input_family(
            crate::noise_refresh::pbc::artifacts::selector_family_artifact_name(artifacts, "gsw"),
            package_count,
            gsw_shape,
        );
        Ok(Self { layout_id: layout.layout_id, key_instance_id, package_count, gsw })
    }

    pub(crate) fn gsw(&self) -> &Family<Mat> {
        &self.gsw
    }

    /// Converts the fixed-C family to the generic Exponent-LUT selector binding.
    pub(crate) fn encoding_family(&self) -> Result<EncodingRhsFamily, BggError> {
        EncodingRhsFamily::new_for_layout(self.gsw.clone(), self.layout_id)
    }

    /// Converts public projections to the public-key native selector binding.
    pub(crate) fn public_family(&self) -> Result<PublicRhsFamily, BggError> {
        PublicRhsFamily::new_for_layout(self.gsw.clone(), self.layout_id)
    }
}

#[cfg(test)]
mod pbc_selector_tests {
    use super::*;
    use crate::noise_refresh::pbc::{PbcActiveCellIndex, routing_tests::layout_with_padding};
    #[test]
    fn selector_views_preserve_the_imported_layout_identity() {
        let layout = layout_with_padding();
        let ring = Ring::new(257, 8);
        let package_count = PbcActiveCellIndex::build(&layout).unwrap().len();
        let inputs = PbcSelectorFamilyInputs {
            layout_id: layout.layout_id,
            key_instance_id: [0x51; 32],
            package_count,
            gsw: ring.input_family("selector-layout-identity", package_count, (2, 4)),
        };

        assert_eq!(
            inputs.encoding_family().expect("encoding selector view").layout_id(),
            Some(layout.layout_id)
        );
        assert_eq!(
            inputs.public_family().expect("public selector view").layout_id(),
            Some(layout.layout_id)
        );
    }
}

use crate::noise_refresh::refresh::{
    RefreshError, RefreshSlot, RefreshSlotPreparation, RefreshSlotValues, evaluate_refresh,
    group_refresh_levels, prepare_refresh_slot,
};
use encoding::ExponentLutEncodingCompiler;
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
/// Protocol declaration used to validate an RNS refresh graph.
pub struct RefreshDeclaration {
    /// Stable declaration identity.
    pub identity: [u8; 32],
    /// Identity of the setup that produced this declaration.
    pub setup_identity: [u8; 32],
    /// Owning protocol/component name.
    pub owner: String,
    /// Named refresh plan.
    pub plan: String,
    /// Declared value type.
    pub value_type: String,
    /// Matrix row count.
    pub rows: usize,
    /// Matrix column count.
    pub columns: usize,
    /// CRT slots in canonical order.
    pub slots: Vec<RefreshSlot>,
    /// CRT recomposition coefficients in slot order.
    pub reconstruction_coefficients: Vec<BigInt>,
}

#[derive(Clone)]
/// Output of refresh, retaining the declaration beside the resulting wire.
pub struct RefreshResult {
    encoding: BggEncodingWire,
    declaration: RefreshDeclaration,
}

impl RefreshResult {
    /// Borrows the refreshed BGG encoding wire.
    pub fn encoding(&self) -> &BggEncodingWire {
        &self.encoding
    }
    /// Borrows the declaration checked for this result.
    pub fn declaration(&self) -> &RefreshDeclaration {
        &self.declaration
    }
    /// Extracts the refreshed encoding and discards the declaration.
    pub fn into_encoding(self) -> BggEncodingWire {
        self.encoding
    }
}

impl RefreshDeclaration {
    fn validate_parameters(&self) -> Result<(), RefreshError> {
        if self.owner != "mxx-exponent-lut" ||
            self.setup_identity == [0; 32] ||
            self.plan != "section-7-refresh" ||
            self.value_type != "BggEncodingVector" ||
            self.rows == 0 ||
            self.columns == 0 ||
            self.slots.is_empty() ||
            self.reconstruction_coefficients.len() != self.slots.len()
        {
            return Err(RefreshError::InvalidLayout);
        }
        for (expected, slot) in self.slots.iter().enumerate() {
            if slot.slot != expected ||
                slot.q <= 0.into() ||
                slot.q_t <= 0.into() ||
                slot.p <= 1.into() ||
                &slot.p % 2 != 1.into() ||
                &slot.q % &slot.p != 0.into() ||
                &slot.p % &slot.q_t != 0.into()
            {
                return Err(RefreshError::InvalidLayout);
            }
        }
        Ok(())
    }
}

/// Typed per-slot package for the RNS refresh equation.
///
/// The package keeps the old state, scaled state, mask, fresh-error
/// source, and decoder preimage together with their graph identities. Mask and
/// fresh-error material remain separate fields so callers cannot accidentally
/// decode before combining them. The online `b`, preprocessing-fixed `k`, and
/// base identity are linkage evidence, not additional artifact outputs.
#[derive(Clone)]
struct RefreshScalarPackage {
    slot: usize,
    values: RefreshSlotValues<BggScheme>,
    decoder_base_handle: mxx_ir_core::ValueHandle,
    target_public_matrix: Mat,
    b: Mat,
    k: Preimage,
}

impl RefreshScalarPackage {
    /// Validates and assembles one CRT-slot refresh package.
    ///
    /// The caller constructs the decoder from the imported witness and shared
    /// online base. This private constructor checks dimensions and retains the
    /// decoder-base handle used to bind shared slots. The sole caller,
    /// `bind_imported_wires`, constructs every slot from the manifest's state.
    #[allow(clippy::too_many_arguments)]
    fn new(
        slot: usize,
        state: &BggEncodingWire,
        values: RefreshSlotValues<BggScheme>,
        decoder_base_handle: mxx_ir_core::ValueHandle,
        target_public_matrix: Mat,
        b: Mat,
        k: Preimage,
    ) -> Result<Self, RefreshError> {
        let low = target_public_matrix.matrix_type();
        let high = state.vector.matrix_type();
        if !same_matrix_type(
            b.matrix_type(),
            &mxx_ir_core::types::MatrixType {
                modulus: low.modulus.clone(),
                ring_dimension: low.ring_dimension.clone(),
                ..b.matrix_type().clone()
            },
        ) || !same_matrix_type(
            k.matrix_type(),
            &mxx_ir_core::types::MatrixType {
                rows: b.matrix_type().columns.clone(),
                columns: high.columns.clone(),
                ..low.clone()
            },
        ) || !same_matrix_type(
            low,
            &mxx_ir_core::types::MatrixType {
                rows: b.matrix_type().rows.clone(),
                columns: high.columns.clone(),
                ring_dimension: high.ring_dimension.clone(),
                ..low.clone()
            },
        ) || !same_matrix_type(
            values.decoder.matrix_type(),
            &mxx_ir_core::types::MatrixType { modulus: low.modulus.clone(), ..high.clone() },
        ) {
            return Err(RefreshError::InvalidLayout);
        }
        Ok(Self { slot, values, decoder_base_handle, target_public_matrix, b, k })
    }
}

/// Opaque, validated setup for one refresh invocation. It owns the exact input
/// state and refreshed public matrix, so evaluation cannot substitute either
/// after setup binding.
#[derive(Clone)]
pub struct RefreshSetupManifest {
    identity: [u8; 32],
    state: BggEncodingWire,
    a_prime: Mat,
    packages: Vec<RefreshScalarPackage>,
}

impl RefreshSetupManifest {
    fn bind(
        state: BggEncodingWire,
        a_prime: Mat,
        packages: Vec<RefreshScalarPackage>,
    ) -> Result<Self, RefreshError> {
        if packages.is_empty() {
            return Err(RefreshError::InvalidLayout);
        }
        let mut anchors = std::collections::BTreeMap::new();
        for (slot, package) in packages.iter().enumerate() {
            if package.slot != slot ||
                !same_matrix_type(package.values.mask.matrix_type(), state.vector.matrix_type()) ||
                !same_matrix_type(
                    package.values.fresh_error.matrix_type(),
                    state.vector.matrix_type(),
                )
            {
                return Err(if package.slot != slot {
                    RefreshError::SlotOrderMismatch
                } else {
                    RefreshError::InvalidLayout
                });
            }
            let modulus = package.b.matrix_type().modulus.clone();
            let binding = (package.b.value_handle().clone(), package.decoder_base_handle.clone());
            if anchors.get(&modulus).is_some_and(|previous| previous != &binding) {
                return Err(RefreshError::AnchorMismatch);
            }
            anchors.insert(modulus, binding);
        }
        let mut hash = Sha256::new();
        hash.update(b"mxx.exponent-lut.refresh-setup");
        hash.update(format!("{:?}", state.vector.value_handle()).as_bytes());
        hash.update(format!("{:?}", a_prime.value_handle()).as_bytes());
        for package in &packages {
            hash.update(package.slot.to_le_bytes());
            hash.update(format!("{:?}", package.target_public_matrix.value_handle()).as_bytes());
            hash.update(format!("{:?}", package.b.value_handle()).as_bytes());
            hash.update(format!("{:?}", package.decoder_base_handle).as_bytes());
            hash.update(format!("{:?}", package.k.value_handle()).as_bytes());
        }
        Ok(Self { identity: hash.finalize().into(), state, a_prime, packages })
    }

    /// Returns the identity of this validated setup.
    pub fn identity(&self) -> &[u8; 32] {
        &self.identity
    }

    /// Returns the exact state encoding sealed into the setup.
    pub fn state(&self) -> &BggEncodingWire {
        &self.state
    }

    /// Returns the public matrix targeted by refresh.
    pub fn refreshed_public_matrix(&self) -> &Mat {
        &self.a_prime
    }
}

/// Configuration for exact q/q_t gadget scaling and CRT recomposition.
#[derive(Clone)]
pub struct RefreshCompiler {
    /// Full ciphertext modulus `q`.
    pub full_modulus: IntExpr,
    /// CRT plaintext moduli `q_t`, in slot order.
    pub crt_plaintext_moduli: Vec<IntExpr>,
    /// Coefficients used to recombine CRT slot outputs.
    pub reconstruction_coefficients: Vec<IntExpr>,
}

impl RefreshCompiler {
    /// Binds setup material imported from the canonical Phase-2 manifest.
    ///
    /// The online shared decoder base and imported per-slot preimages are
    /// deliberately plain wires at this boundary. This helper is the only
    /// crate-internal adapter that turns them into typed per-slot packages:
    /// each target is recomputed from the imported state/mask/fresh material
    /// and the decoder is rebuilt as the shared `base * K_t`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn bind_imported_wires(
        &self,
        compiler: &ExponentLutEncodingCompiler,
        state: BggEncodingWire,
        state_public: BggPublicKeyWire,
        a_prime: Mat,
        public_b: Vec<Mat>,
        refresh_targets: Vec<Mat>,
        scaled_freshes: Vec<Mat>,
        masks: Vec<Mat>,
        state_scaling: Vec<Preimage>,
        decoder_bases: Vec<Mat>,
        preimages: Vec<Preimage>,
    ) -> Result<RefreshSetupManifest, RefreshError> {
        self.validate_layout()?;
        if masks.len() != self.crt_plaintext_moduli.len() ||
            scaled_freshes.len() != masks.len() ||
            state_scaling.len() != masks.len() ||
            preimages.len() != masks.len() ||
            public_b.len() != masks.len() ||
            decoder_bases.len() != masks.len() ||
            refresh_targets.len() != masks.len() ||
            !same_matrix_type(a_prime.matrix_type(), state_public.matrix.matrix_type())
        {
            return Err(RefreshError::InvalidLayout);
        }
        let scheme = crate::bgg::BggScheme::new(
            mxx_bgg::BggPublicKeyCompiler {
                ring: compiler.ring.clone(),
                base: compiler.gadget_base.clone(),
                digit_count: compiler.digit_count.clone(),
            },
            compiler.exponent_layout,
        );
        // Slots sharing P use the same B_P and online anchor handles. Distinct
        // rings remain separate, including when their matrix shapes coincide.
        let mut packages = Vec::with_capacity(masks.len());
        for (slot, (((mask, scaled_fresh), scale_decomposition), k)) in masks
            .into_iter()
            .zip(scaled_freshes.into_iter())
            .zip(state_scaling.into_iter())
            .zip(preimages.into_iter())
            .enumerate()
        {
            let base = decoder_bases[slot].clone();
            let decoder_base_handle = base.value_handle().clone();
            // `scale = mu_t = q/q_t`; multiplying by its gadget decomposition
            // implements `u_t=c G^{-1}(mu_t G)` and `A_t=A G^{-1}(mu_t G)`.
            let scale = mxx_dsl::Ring::new(
                state.vector.matrix_type().modulus.clone(),
                state.vector.matrix_type().ring_dimension.clone(),
            )
            .polynomial([self.scale_expression(slot)?]);
            let target = refresh_targets.get(slot).cloned().ok_or(RefreshError::InvalidLayout)?;
            let p = target
                .matrix_type()
                .modulus
                .evaluate(&Default::default())
                .map_err(|_| RefreshError::InvalidLayout)?;
            let q = self
                .full_modulus
                .evaluate(&Default::default())
                .map_err(|_| RefreshError::InvalidLayout)?;
            let qt = self.crt_plaintext_moduli[slot]
                .evaluate(&Default::default())
                .map_err(|_| RefreshError::InvalidLayout)?;
            if p <= 1.into() || &p % 2 != 1.into() || &q % &p != 0.into() || &p % &qt != 0.into() {
                return Err(RefreshError::InvalidLayout);
            }
            // K_t targets MS(A_sum,t) - Delta_t reduce(A') in R_P.
            let prepared = prepare_refresh_slot(
                &scheme,
                &state,
                mask,
                scaled_fresh,
                RefreshSlotPreparation {
                    level: target.matrix_type().modulus.clone(),
                    scaling: (scale, scale_decomposition),
                    decoder_base: base,
                    witness: k.clone(),
                },
            );
            packages.push(RefreshScalarPackage::new(
                slot,
                &state,
                prepared,
                decoder_base_handle,
                target,
                public_b[slot].clone(),
                k,
            )?);
        }
        RefreshSetupManifest::bind(state, a_prime, packages)
    }

    /// Build the canonical event from the actual refresh output and equation
    /// operands. The event is attached to the output graph value below.
    fn refresh_declaration(
        &self,
        setup: &RefreshSetupManifest,
        output: &Mat,
    ) -> Result<RefreshDeclaration, RefreshError> {
        let state = &setup.state;
        let rows = state
            .vector
            .matrix_type()
            .rows
            .evaluate(&Default::default())
            .map_err(|_| RefreshError::InvalidLayout)?
            .to_usize()
            .ok_or(RefreshError::InvalidLayout)?;
        let columns = state
            .vector
            .matrix_type()
            .columns
            .evaluate(&Default::default())
            .map_err(|_| RefreshError::InvalidLayout)?
            .to_usize()
            .ok_or(RefreshError::InvalidLayout)?;
        let q = self
            .full_modulus
            .evaluate(&Default::default())
            .map_err(|_| RefreshError::InvalidLayout)?;
        let slots = self
            .crt_plaintext_moduli
            .iter()
            .enumerate()
            .map(|(slot, q_t)| {
                Ok(RefreshSlot {
                    slot,
                    q: q.clone(),
                    p: setup.packages[slot]
                        .target_public_matrix
                        .matrix_type()
                        .modulus
                        .evaluate(&Default::default())
                        .map_err(|_| RefreshError::InvalidLayout)?,
                    q_t: q_t
                        .evaluate(&Default::default())
                        .map_err(|_| RefreshError::InvalidLayout)?,
                })
            })
            .collect::<Result<Vec<_>, RefreshError>>()?;
        let declaration = RefreshDeclaration {
            identity: matrix_identity(output),
            setup_identity: setup.identity,
            owner: "mxx-exponent-lut".into(),
            plan: "section-7-refresh".into(),
            value_type: "BggEncodingVector".into(),
            rows,
            columns,
            slots,
            reconstruction_coefficients: self
                .reconstruction_coefficients
                .iter()
                .map(|coefficient| {
                    coefficient
                        .evaluate(&Default::default())
                        .map_err(|_| RefreshError::InvalidLayout)
                })
                .collect::<Result<Vec<_>, _>>()?,
        };
        if output.matrix_type().rows.evaluate(&Default::default()).ok().and_then(|v| v.to_usize()) !=
            Some(rows) ||
            output
                .matrix_type()
                .columns
                .evaluate(&Default::default())
                .ok()
                .and_then(|v| v.to_usize()) !=
                Some(columns) ||
            setup.packages.is_empty()
        {
            return Err(RefreshError::InvalidLayout);
        }
        declaration.validate_parameters()?;
        Ok(declaration)
    }
    /// Validates CRT divisibility and coefficient count.
    pub fn validate_layout(&self) -> Result<(), RefreshError> {
        // CRT-backed refresh requires at least two plaintext-modulus slots;
        // a single slot is not a supported CRT layout.
        if self.crt_plaintext_moduli.len() < 2 ||
            self.crt_plaintext_moduli.len() != self.reconstruction_coefficients.len()
        {
            return Err(RefreshError::InvalidLayout);
        }
        let q = self
            .full_modulus
            .evaluate(&Default::default())
            .map_err(|_| RefreshError::InvalidLayout)?;
        if q <= BigInt::from(0) {
            return Err(RefreshError::InvalidLayout);
        }
        let q_t = self
            .crt_plaintext_moduli
            .iter()
            .map(|qt| qt.evaluate(&Default::default()).map_err(|_| RefreshError::InvalidLayout))
            .collect::<Result<Vec<_>, _>>()?;
        if q_t.iter().any(|value| *value <= BigInt::from(1) || &q % value != BigInt::from(0)) {
            return Err(RefreshError::InvalidLayout);
        }
        for (index, left) in q_t.iter().enumerate() {
            for right in q_t.iter().skip(index + 1) {
                if bigint_gcd(left, right) != BigInt::from(1) {
                    return Err(RefreshError::InvalidLayout);
                }
            }
        }
        let product = q_t.iter().fold(BigInt::from(1), |product, value| product * value);
        if product != q {
            return Err(RefreshError::InvalidLayout);
        }
        Ok(())
    }

    /// Returns the exact `q / q_t` scale expression for one CRT slot.
    pub fn scale_expression(&self, slot: usize) -> Result<IntExpr, RefreshError> {
        self.crt_plaintext_moduli
            .get(slot)
            .cloned()
            .map(|q_t| IntExpr::Div(Box::new(self.full_modulus.clone()), Box::new(q_t)))
            .ok_or(RefreshError::InvalidLayout)
    }

    /// Refreshes the state sealed into a validated manifest. Slot order and
    /// shared decoder-base identity are established by binding; the manifest
    /// exposes no mutation of these values. All masks/errors are added before
    /// switching and subtracting the combined decoder.
    pub fn refresh(
        &self,
        compiler: &ExponentLutEncodingCompiler,
        setup: &RefreshSetupManifest,
    ) -> Result<RefreshResult, RefreshError> {
        self.validate_layout()?;
        let packages = &setup.packages;
        if packages.len() != self.crt_plaintext_moduli.len() {
            return Err(RefreshError::InvalidLayout);
        }
        let scheme = crate::bgg::BggScheme::new(
            mxx_bgg::BggPublicKeyCompiler {
                ring: compiler.ring.clone(),
                base: compiler.gadget_base.clone(),
                digit_count: compiler.digit_count.clone(),
            },
            compiler.exponent_layout,
        );
        let evaluated = evaluate_refresh(
            &scheme,
            packages.iter().map(|package| &package.values),
            &crate::bgg::BggRefreshRecomposition {
                plaintext_moduli: self.crt_plaintext_moduli.clone(),
                coefficients: self.reconstruction_coefficients.clone(),
                full_modulus: self.full_modulus.clone(),
            },
        )
        .map_err(|_| RefreshError::InvalidLayout)?;
        let vector = evaluated.vector;
        let declaration = self.refresh_declaration(setup, &vector)?;
        Ok(RefreshResult { encoding: BggEncodingWire { vector, plaintext: None }, declaration })
    }
}

fn bigint_gcd(left: &BigInt, right: &BigInt) -> BigInt {
    let mut a = left.clone().abs();
    let mut b = right.clone().abs();
    while !b.is_zero() {
        let remainder = a % &b;
        a = b;
        b = remainder;
    }
    a
}

fn matrix_identity(matrix: &Mat) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(format!("{:?}", matrix.value_handle()).as_bytes());
    h.finalize().into()
}

fn same_matrix_type(
    left: &mxx_ir_core::types::MatrixType,
    right: &mxx_ir_core::types::MatrixType,
) -> bool {
    let environment = mxx_ir_core::ParamEnv::default();
    [
        (&left.modulus, &right.modulus),
        (&left.ring_dimension, &right.ring_dimension),
        (&left.rows, &right.rows),
        (&left.columns, &right.columns),
    ]
    .into_iter()
    .all(|(left, right)| matches!((left.evaluate(&environment), right.evaluate(&environment)), (Ok(left), Ok(right)) if left == right))
}

#[cfg(test)]
mod refresh_compiler_tests {
    use super::*;
    #[test]
    fn refresh_layout_rejects_duplicate_or_non_coprime_crt_moduli() {
        let duplicate = RefreshCompiler {
            full_modulus: 12.into(),
            crt_plaintext_moduli: vec![2.into(), 2.into(), 3.into()],
            reconstruction_coefficients: vec![1.into(), 1.into(), 1.into()],
        };
        assert!(matches!(duplicate.validate_layout(), Err(RefreshError::InvalidLayout)));

        let non_coprime = RefreshCompiler {
            full_modulus: 24.into(),
            crt_plaintext_moduli: vec![4.into(), 6.into()],
            reconstruction_coefficients: vec![1.into(), 1.into()],
        };
        assert!(matches!(non_coprime.validate_layout(), Err(RefreshError::InvalidLayout)));

        let length_mismatch = RefreshCompiler {
            full_modulus: 6.into(),
            crt_plaintext_moduli: vec![2.into(), 3.into()],
            reconstruction_coefficients: vec![1.into()],
        };
        assert!(matches!(length_mismatch.validate_layout(), Err(RefreshError::InvalidLayout)));
    }

    #[test]
    fn refresh_layout_rejects_crt_product_mismatch_and_unit_modulus() {
        let mismatch = RefreshCompiler {
            full_modulus: 24.into(),
            crt_plaintext_moduli: vec![3.into(), 5.into()],
            reconstruction_coefficients: vec![1.into(), 1.into()],
        };
        assert!(matches!(mismatch.validate_layout(), Err(RefreshError::InvalidLayout)));

        let unit = RefreshCompiler {
            full_modulus: 6.into(),
            crt_plaintext_moduli: vec![1.into(), 6.into()],
            reconstruction_coefficients: vec![1.into(), 1.into()],
        };
        assert!(matches!(unit.validate_layout(), Err(RefreshError::InvalidLayout)));
    }

    #[test]
    fn refresh_layout_rejects_single_crt_slot() {
        let layout = RefreshCompiler {
            full_modulus: 97.into(),
            crt_plaintext_moduli: vec![97.into()],
            reconstruction_coefficients: vec![1.into()],
        };
        assert!(matches!(layout.validate_layout(), Err(RefreshError::InvalidLayout)));
    }

    #[test]
    fn grouped_low_p_refresh_preserves_slot_order_and_anchor_binding() {
        let ring = Ring::new(105, 4);
        let compiler = ExponentLutEncodingCompiler::with_layout(
            mxx_bgg::BggPublicKeyCompiler {
                ring: ring.clone(),
                base: 4.into(),
                digit_count: 2.into(),
            },
            crate::layout::ExpLayout { stride_log2: 1, width_log2: 1 },
        );
        for (moduli, plaintext) in [([15, 15, 21], [3, 5, 7]), ([21, 15, 15], [7, 3, 5])] {
            let refresh = RefreshCompiler {
                full_modulus: 105.into(),
                crt_plaintext_moduli: plaintext.into_iter().map(IntExpr::constant).collect(),
                reconstruction_coefficients: vec![1.into(); 3],
            };
            let state = BggEncodingWire { vector: ring.input("state", (1, 4)), plaintext: None };
            let state_public =
                BggPublicKeyWire { matrix: ring.input("A", (2, 4)), reveal_plaintext: false };
            let a_prime = ring.input("A-prime", (2, 4));
            let groups = [15, 21]
                .map(|p| {
                    let low = Ring::new(p, 4);
                    (p, (low.input(format!("B-{p}"), (2, 6)), low.input(format!("b-{p}"), (1, 6))))
                })
                .into_iter()
                .collect::<std::collections::BTreeMap<_, _>>();
            let public_b = moduli.map(|p| groups[&p].0.clone()).to_vec();
            let bases = moduli.map(|p| groups[&p].1.clone()).to_vec();
            let targets = moduli
                .iter()
                .enumerate()
                .map(|(slot, p)| Ring::new(*p, 4).input(format!("target-{slot}"), (2, 4)))
                .collect::<Vec<_>>();
            let preimages = moduli
                .iter()
                .enumerate()
                .map(|(slot, p)| Ring::new(*p, 4).preimage_input(format!("K-{slot}"), (6, 4), 1))
                .collect::<Vec<_>>();
            let masks =
                (0..3).map(|slot| ring.input(format!("mask-{slot}"), (1, 4))).collect::<Vec<_>>();
            let fresh =
                (0..3).map(|slot| ring.input(format!("fresh-{slot}"), (1, 4))).collect::<Vec<_>>();
            let scaling = (0..3)
                .map(|slot| ring.preimage_input(format!("scale-{slot}"), (4, 4), 1))
                .collect::<Vec<_>>();
            let bind = |bases| {
                refresh.bind_imported_wires(
                    &compiler,
                    state.clone(),
                    state_public.clone(),
                    a_prime.clone(),
                    public_b.clone(),
                    targets.clone(),
                    fresh.clone(),
                    masks.clone(),
                    scaling.clone(),
                    bases,
                    preimages.clone(),
                )
            };
            let setup = bind(bases.clone()).unwrap();
            let result = refresh.refresh(&compiler, &setup).unwrap();
            assert_eq!(result.encoding().vector.matrix_type(), state.vector.matrix_type());
            assert_eq!(
                result.declaration().slots.iter().map(|slot| slot.p.clone()).collect::<Vec<_>>(),
                moduli.map(BigInt::from)
            );
            let graph = mxx_dsl::DslContext::new("grouped-refresh")
                .output("result", result.into_encoding().vector)
                .unwrap()
                .build()
                .unwrap();
            graph.validate(&Default::default()).unwrap();
            let switches = graph
                .graph
                .scopes()
                .values()
                .flat_map(|scope| scope.nodes())
                .filter(|node| {
                    matches!(node.kind(), mxx_ir_core::node::NodeKind::ModulusSwitch { .. })
                })
                .count();
            assert_eq!(switches, 2, "one symbolic complete-sum switch body per distinct P");
            let shared_slot = moduli.iter().position(|p| *p == 15).unwrap();
            let mut tampered = bases;
            tampered[shared_slot] = Ring::new(15, 4).input("substituted-anchor", (1, 6));
            assert!(matches!(bind(tampered), Err(RefreshError::AnchorMismatch)));
        }
    }
}

impl encoding::ExponentLutEncodingSampler {
    /// Samples all selector RHS packages in one reusable `ParallelLoop`.
    ///
    /// The selector bit is the only selector-dependent input to the structural
    /// loop.  Each resulting GSW ciphertext is fixed during setup and is emitted
    /// as one public family element.  Its concrete value also depends on the
    /// sampled source/target RHS inputs.  The public-key evaluator consumes this
    /// concrete C family directly; its issuer need not know selector plaintext,
    /// private schedule, or sampler encoding.  No digit-level companion family is
    /// created.
    pub fn sample_pbc_selectors(
        &self,
        bits: Family<Mat>,
        source: Mat,
        target: Mat,
        hash_key: Bytes,
        layout: &PbcPublicLayout,
        key_instance_id: [u8; 32],
    ) -> Result<Family<Mat>, PbcError> {
        let sampler = self;
        layout.validate()?;
        let active = crate::noise_refresh::pbc::PbcActiveCellIndex::build(layout)?;
        let expected = active.len();
        if expected == 0 || *bits.count() != mxx_ir_core::IntExpr::constant(expected) {
            return Err(PbcError::InvalidSchedule(
                "private selector family count does not match canonical active order".into(),
            ));
        }
        if bits.element_type() != &sampler.layout.ring().matrix_type((1, 1)) {
            return Err(PbcError::InvalidSchedule(
                "private selector family must contain constant-polynomial scalar matrices".into(),
            ));
        }
        if sampler.layout.secret_dimension < 2 ||
            sampler.layout.public_key_columns() == 0 ||
            !same_secret_shape(&source, &sampler.layout) ||
            !same_secret_shape(&target, &sampler.layout) ||
            sampler.gaussian_sigma.is_some() != sampler.gaussian_max_coefficient_bound.is_some()
        {
            return Err(PbcError::InvalidSchedule("selector sampler has an empty shape".into()));
        }
        let sampler = sampler.clone();
        let layout_id = layout.layout_id;
        let gsw = {
            let source_family = bits;
            mxx_dsl::parallel(source_family.count().clone(), |family_index| {
                let bit = source_family.at(&family_index);
                let index = family_index;
                Ok({
                    let mut tag = selector_rhs_tag(layout_id, key_instance_id);
                    tag.push(index);
                    let package = sampler
                        .sample_cross_secret_rhs(
                            source.clone(),
                            target.clone(),
                            bit,
                            hash_key.clone(),
                            tag,
                        )
                        .unwrap_or_else(|error: encoding::ExponentLutSamplingError| {
                            panic!(
                                "validated selector sampler failed while building graph: {error}"
                            )
                        });
                    package.gsw_ciphertext().clone()
                })
            })
        }
        .map_err(|_| PbcError::InvalidSchedule("selector RHS loop construction failed".into()))?;
        Ok(gsw)
    }
}

fn same_secret_shape(value: &Mat, layout: &mxx_bgg::BggSamplerLayout) -> bool {
    let expected = layout.ring().matrix_type((1, layout.secret_dimension));
    let actual = value.matrix_type();
    actual.modulus.canonicalize() == expected.modulus.canonicalize() &&
        actual.ring_dimension.canonicalize() == expected.ring_dimension.canonicalize() &&
        actual.rows.canonicalize() == expected.rows.canonicalize() &&
        actual.columns.canonicalize() == expected.columns.canonicalize()
}

fn selector_rhs_tag(layout_id: PbcLayoutId, key_instance_id: [u8; 32]) -> HashTag {
    let mut prefix = Vec::with_capacity(32 + 32 + 40);
    prefix.extend_from_slice(b"mxx-exponent-lut/pbc/selector-rhs/v1");
    prefix.extend_from_slice(&(layout_id.0.len() as u64).to_le_bytes());
    prefix.extend_from_slice(&layout_id.0);
    prefix.extend_from_slice(&(key_instance_id.len() as u64).to_le_bytes());
    prefix.extend_from_slice(&key_instance_id);
    HashTag::from(prefix)
}

#[cfg(test)]
mod pbc_sampling_tests {
    use super::pbc_selector_metadata;
    use crate::noise_refresh::pbc::*;
    fn seed(byte: u8) -> PbcRootSeed {
        PbcRootSeed([byte; 32])
    }
    #[test]
    fn large_selector_bits_remain_one_structural_family_input() {
        use crate::bgg::encoding::ExponentLutEncodingSampler;
        use mxx_bgg::BggSamplerLayout;
        use mxx_dsl::DslContext;
        use mxx_ir_core::node::NodeKind;

        // This is the accepted benchmark topology: 16,384 * 3 real cells plus
        // one dummy in each of 20 buckets gives exactly 49,172 active bits.
        let parameters = PbcParameters::custom(16_384, 17, 3, 20, 128, None);
        let support = (0..17).collect::<Vec<_>>();
        let generated = generate_key_layout(&parameters, seed(17), &support).unwrap();
        let sampler_layout = BggSamplerLayout {
            modulus: 257.into(),
            ring_dimension: 4.into(),
            secret_dimension: 2,
            digit_count: 1,
            gadget_base: 2.into(),
        };
        let trusted =
            PbcTrustedSelectorBits::from_schedule(&generated, &sampler_layout.ring(), [8; 32])
                .unwrap();
        assert_eq!(trusted.runtime_bits().len(), 49_172);
        assert_eq!(
            trusted.family().count(),
            &mxx_ir_core::IntExpr::constant(trusted.runtime_bits().len())
        );

        let sampler = ExponentLutEncodingSampler {
            layout: sampler_layout,
            exponent_layout: crate::layout::ExpLayout { stride_log2: 0, width_log2: 0 },
            gaussian_sigma: None,
            gaussian_max_coefficient_bound: None,
        };
        let families = sampler
            .sample_pbc_selectors(
                trusted.family().clone(),
                sampler.sample_secret().unwrap(),
                sampler.sample_secret().unwrap(),
                sampler.layout.ring().bytes_input("large-selector-hash-key", 32),
                &generated.public_layout,
                [8; 32],
            )
            .unwrap();
        let built = DslContext::new("large-structural-selector-input")
            .public_output("gsw", families)
            .unwrap()
            .build()
            .unwrap();
        let nodes =
            built.graph.scopes().values().flat_map(|scope| scope.nodes()).collect::<Vec<_>>();
        assert!(nodes.len() < 64, "selector graph must not grow with active count");
        let bit_name = trusted.input_name().to_owned();
        assert_eq!(
            nodes
                .iter()
                .filter(|node| {
                    matches!(node.kind(), NodeKind::Input { name, .. } if name == &bit_name)
                })
                .count(),
            1
        );
        assert!(!nodes.iter().any(|node| {
            matches!(node.kind(), NodeKind::FamilyGetStatic { .. } | NodeKind::FamilyPack { .. })
        }));
    }

    #[test]
    fn selector_sampling_uses_one_structural_loop_and_runtime_bits() {
        use crate::bgg::encoding::ExponentLutEncodingSampler;
        use mxx_bgg::BggSamplerLayout;
        use mxx_dsl::DslContext;
        use mxx_ir_core::node::NodeKind;

        let parameters = PbcParameters::custom(5, 1, 2, 3, 1, None);
        let first = generate_key_layout(&parameters, seed(16), &[0]).unwrap();
        let second = generate_key_layout(&parameters, seed(16), &[1]).unwrap();
        assert_eq!(first.public_layout, second.public_layout);
        let sampler_layout = BggSamplerLayout {
            modulus: 257.into(),
            ring_dimension: 4.into(),
            secret_dimension: 2,
            digit_count: 1,
            gadget_base: 2.into(),
        };
        let ring = sampler_layout.ring();
        let sampler = ExponentLutEncodingSampler {
            layout: sampler_layout.clone(),
            exponent_layout: crate::layout::ExpLayout { stride_log2: 0, width_log2: 0 },
            gaussian_sigma: None,
            gaussian_max_coefficient_bound: None,
        };
        let bits = PbcTrustedSelectorBits::from_schedule(&first, &ring, [7; 32]).unwrap();
        let source = sampler.sample_secret().unwrap();
        let target = sampler.sample_secret().unwrap();
        let hash_key = ring.bytes_input("selector-hash-key", 32);
        let families = sampler
            .sample_pbc_selectors(
                bits.family().clone(),
                source,
                target,
                hash_key,
                &first.public_layout,
                [7; 32],
            )
            .unwrap();
        let names =
            PbcSelectorArtifactNames::canonicalize_schema(&first.public_layout, [7; 32]).unwrap();
        let artifacts = pbc_selector_metadata(
            &first.public_layout,
            [7; 32],
            &sampler_layout,
            [7; 32],
            &sampler_layout,
            [7; 32],
        )
        .and_then(|metadata| {
            PbcSelectorArtifacts::from_structural(&first.public_layout, [7; 32], names, metadata)
        })
        .unwrap();
        let context = artifacts
            .add_structural_family_outputs(
                DslContext::new("trusted-selector-structural-loop"),
                &first.public_layout,
                families,
                "gsw",
            )
            .unwrap();
        let built = context.build().unwrap();
        let loops = built
            .graph
            .root_scope()
            .nodes()
            .iter()
            .filter(|node| matches!(node.kind(), NodeKind::ParallelLoop(_)))
            .count();
        assert_eq!(loops, 1);
        let bit_name = bits.input_name().to_owned();
        let bit_inputs = built.graph.root_scope().nodes().iter().filter(|node| {
            matches!(node.kind(), NodeKind::Input { name, artifact: None, .. } if name == &bit_name)
        });
        assert_eq!(bit_inputs.count(), 1);

        // A different private schedule still produces the same structural graph
        // schema.  Concrete fixed-C family contents are runtime artifacts and may
        // differ with the sampled RHS inputs; this assertion concerns only the
        // graph shape and public namespace, not matrix-value invariance.
        let second_bits = PbcTrustedSelectorBits::from_schedule(&second, &ring, [7; 32]).unwrap();
        let second_families = sampler
            .sample_pbc_selectors(
                second_bits.family().clone(),
                sampler.sample_secret().unwrap(),
                sampler.sample_secret().unwrap(),
                ring.bytes_input("selector-hash-key", 32),
                &second.public_layout,
                [7; 32],
            )
            .unwrap();
        let second_names =
            PbcSelectorArtifactNames::canonicalize_schema(&second.public_layout, [7; 32]).unwrap();
        let second_artifacts = pbc_selector_metadata(
            &second.public_layout,
            [7; 32],
            &sampler_layout,
            [7; 32],
            &sampler_layout,
            [7; 32],
        )
        .and_then(|metadata| {
            PbcSelectorArtifacts::from_structural(
                &second.public_layout,
                [7; 32],
                second_names,
                metadata,
            )
        })
        .unwrap();
        let second_context = second_artifacts
            .add_structural_family_outputs(
                DslContext::new("trusted-selector-structural-loop"),
                &second.public_layout,
                second_families,
                "gsw",
            )
            .unwrap();
        let second_built = second_context.build().unwrap();
        let first_hash =
            mxx_ir_core::encoding::spec_hash(&built.graph, &Default::default()).unwrap();
        let second_hash =
            mxx_ir_core::encoding::spec_hash(&second_built.graph, &Default::default()).unwrap();
        assert_eq!(first_hash, second_hash);
    }
}

pub(crate) fn pbc_selector_metadata(
    layout: &PbcPublicLayout,
    key_instance_id: [u8; 32],
    source: &mxx_bgg::BggSamplerLayout,
    source_identity: [u8; 32],
    target: &mxx_bgg::BggSamplerLayout,
    target_identity: [u8; 32],
) -> Result<PbcSelectorFamilyMetadata<ManifestRhsMetadata>, PbcError> {
    if source.modulus != target.modulus || source.ring_dimension != target.ring_dimension {
        return Err(PbcError::ArtifactIdentityMismatch);
    }
    let columns =
        target.secret_dimension.checked_mul(target.digit_count).ok_or(PbcError::SizeOverflow)?;
    let modulus = source
        .modulus
        .evaluate(&ParamEnv::default())
        .map_err(|_| PbcError::ArtifactIdentityMismatch)?;
    let ring_dimension = source
        .ring_dimension
        .evaluate(&ParamEnv::default())
        .ok()
        .and_then(|value| value.to_usize())
        .ok_or(PbcError::ArtifactIdentityMismatch)?;
    let element_type =
        ConcreteMatrixType { modulus, ring_dimension, rows: source.secret_dimension, columns };
    PbcSelectorFamilyMetadata::new(
        layout,
        key_instance_id,
        ManifestRhsMetadata::from_layouts(source, source_identity, target, target_identity),
        element_type,
    )
}

#[cfg(test)]
mod pbc_metadata_tests {
    use super::*;
    use crate::noise_refresh::pbc::*;
    fn seed(byte: u8) -> PbcRootSeed {
        PbcRootSeed([byte; 32])
    }
    #[test]
    fn selector_artifact_import_rejects_production_mismatch() {
        use mxx_ir_core::artifact::{Manifest, ProductionId, SpecHash};

        let parameters = PbcParameters::custom(7, 2, 2, 3, 1, None);
        let layout =
            PbcPublicLayout::build(&parameters, derive_attempt_seed(seed(12), 0), 0).unwrap();
        let names = PbcSelectorArtifactNames { selector_packages: Vec::new() };
        let expected = ProductionId { spec_hash: SpecHash([1; 32]), execution_nonce: [2; 32] };
        let wrong_manifest = Manifest {
            ir_version: 1,
            production_id: ProductionId { spec_hash: SpecHash([3; 32]), execution_nonce: [4; 32] },
            artifacts: Default::default(),
        };
        let sampler_layout = BggSamplerLayout {
            modulus: 257.into(),
            ring_dimension: 4.into(),
            secret_dimension: 2,
            digit_count: 1,
            gadget_base: 2.into(),
        };
        assert!(
            pbc_selector_metadata(
                &layout,
                [4; 32],
                &sampler_layout,
                [4; 32],
                &sampler_layout,
                [4; 32]
            )
            .and_then(|metadata| PbcSelectorArtifacts::import(
                expected,
                &layout,
                [4; 32],
                &wrong_manifest,
                names,
                metadata,
                "gsw"
            ))
            .is_err()
        );
    }
    #[test]
    fn fixed_rhs_selector_schema_exposes_only_gsw_component() {
        let parameters = PbcParameters::custom(7, 2, 2, 3, 1, None);
        let layout =
            PbcPublicLayout::build(&parameters, derive_attempt_seed(seed(14), 0), 0).unwrap();
        let names = PbcSelectorArtifactNames::canonicalize_schema(&layout, [8; 32]).unwrap();
        assert_eq!(
            names.selector_packages.len(),
            parameters.universe_size * parameters.hash_count + parameters.bucket_count
        );
        assert!(
            names.selector_packages.iter().all(|package| !package.ciphertext.contains("selected"))
        );
        assert!(
            names.selector_packages.iter().all(|package| !package.ciphertext.contains("support"))
        );
        let sampler_layout = BggSamplerLayout {
            modulus: 257.into(),
            ring_dimension: 4.into(),
            secret_dimension: 2,
            digit_count: 1,
            gadget_base: 2.into(),
        };
        let artifacts = pbc_selector_metadata(
            &layout,
            [8; 32],
            &sampler_layout,
            [8; 32],
            &sampler_layout,
            [8; 32],
        )
        .and_then(|metadata| {
            PbcSelectorArtifacts::from_structural(&layout, [8; 32], names.clone(), metadata)
        })
        .unwrap();
        let family_name = selector_family_artifact_name(&artifacts, "gsw");
        assert!(!family_name.is_empty());
    }
    fn selector_family_manifest_fixture() -> (
        PbcSelectorArtifacts<ManifestRhsMetadata>,
        PbcPublicLayout,
        PbcSelectorArtifactNames,
        BggSamplerLayout,
        ProductionId,
        Manifest,
        [u8; 32],
        [u8; 32],
        [u8; 32],
    ) {
        use mxx_ir_core::{
            artifact::{ArtifactConfidentiality, ArtifactType, ManifestArtifact, SpecHash},
            types::ConcreteMatrixType,
        };

        use std::collections::BTreeMap;

        let parameters = PbcParameters::custom(5, 1, 2, 2, 1, None);
        let layout =
            PbcPublicLayout::build(&parameters, derive_attempt_seed(seed(17), 0), 0).unwrap();
        let sampler = BggSamplerLayout {
            modulus: 97.into(),
            ring_dimension: 4.into(),
            secret_dimension: 2,
            digit_count: 1,
            gadget_base: 4.into(),
        };
        let key_instance_id = [4; 32];
        let source_identity = [5; 32];
        let target_identity = [6; 32];
        let names =
            PbcSelectorArtifactNames::canonicalize_schema(&layout, key_instance_id).unwrap();
        let artifacts = pbc_selector_metadata(
            &layout,
            key_instance_id,
            &sampler,
            source_identity,
            &sampler,
            target_identity,
        )
        .and_then(|metadata| {
            PbcSelectorArtifacts::from_structural(&layout, key_instance_id, names.clone(), metadata)
        })
        .unwrap();
        let family_name = selector_family_artifact_name(&artifacts, "gsw");
        let active_count = names.selector_packages.len();
        let mut manifest = Manifest {
            ir_version: mxx_ir_core::encoding::IR_VERSION,
            production_id: ProductionId { spec_hash: SpecHash([7; 32]), execution_nonce: [8; 32] },
            artifacts: BTreeMap::from([(
                family_name,
                ManifestArtifact {
                    artifact_type: ArtifactType::Matrix(ConcreteMatrixType {
                        modulus: 97.into(),
                        ring_dimension: 4,
                        rows: sampler.secret_dimension,
                        columns: sampler.public_key_columns(),
                    }),
                    family_count: Some(active_count),
                    confidentiality: ArtifactConfidentiality::Public,
                    content_hash: Some([9; 32]),
                    layout: None,
                },
            )]),
        };
        artifacts.finalize_export_manifest(&mut manifest, "gsw").unwrap();
        (
            artifacts,
            layout,
            names,
            sampler,
            manifest.production_id.clone(),
            manifest,
            key_instance_id,
            source_identity,
            target_identity,
        )
    }
    #[test]
    fn selector_family_finalization_preserves_runtime_descriptor() {
        let (artifacts, _, _, _, _, mut manifest, _, _, _) = selector_family_manifest_fixture();
        let before = manifest.clone();
        artifacts.finalize_export_manifest(&mut manifest, "gsw").unwrap();
        assert_eq!(manifest, before);
        assert!(manifest.artifacts.values().all(|descriptor| descriptor.layout.is_none()));
    }
    #[test]
    fn selector_family_import_round_trip_validates_rhs_layouts_and_identities() {
        let (_, layout, names, sampler, production_id, manifest, key_id, source_id, target_id) =
            selector_family_manifest_fixture();
        let imported =
            pbc_selector_metadata(&layout, key_id, &sampler, source_id, &sampler, target_id)
                .and_then(|metadata| {
                    PbcSelectorArtifacts::import(
                        production_id,
                        &layout,
                        key_id,
                        &manifest,
                        names,
                        metadata,
                        "gsw",
                    )
                })
                .unwrap();
        assert_eq!(imported.layout_id(), layout.layout_id);
        assert_eq!(imported.key_instance_id(), key_id);
        assert_eq!(imported.package_count(), layout.parameters.universe_size * 2 + 2);
    }
    #[test]
    fn selector_family_import_rejects_wrong_rhs_bindings_and_family_schema() {
        use mxx_ir_core::{artifact::ArtifactType, types::ConcreteMatrixType};

        let (_, layout, names, sampler, production_id, manifest, key_id, source_id, target_id) =
            selector_family_manifest_fixture();
        let import = |manifest: &Manifest,
                      names: PbcSelectorArtifactNames,
                      layout: &PbcPublicLayout,
                      key_id: [u8; 32],
                      source: &BggSamplerLayout,
                      source_id: [u8; 32],
                      target: &BggSamplerLayout,
                      target_id: [u8; 32]| {
            pbc_selector_metadata(layout, key_id, source, source_id, target, target_id).and_then(
                |metadata| {
                    PbcSelectorArtifacts::import(
                        production_id.clone(),
                        layout,
                        key_id,
                        manifest,
                        names,
                        metadata,
                        "gsw",
                    )
                },
            )
        };

        let mut wrong_source = sampler.clone();
        wrong_source.gadget_base = 8.into();
        assert!(
            import(
                &manifest,
                names.clone(),
                &layout,
                key_id,
                &wrong_source,
                source_id,
                &sampler,
                target_id
            )
            .is_err()
        );
        assert!(
            import(
                &manifest,
                names.clone(),
                &layout,
                key_id,
                &sampler,
                [10; 32],
                &sampler,
                target_id
            )
            .is_err()
        );
        let mut wrong_target = sampler.clone();
        wrong_target.digit_count = 2;
        assert!(
            import(
                &manifest,
                names.clone(),
                &layout,
                key_id,
                &sampler,
                source_id,
                &wrong_target,
                target_id
            )
            .is_err()
        );
        assert!(
            import(
                &manifest,
                names.clone(),
                &layout,
                [11; 32],
                &sampler,
                source_id,
                &sampler,
                target_id
            )
            .is_err()
        );

        let mut wrong_count = manifest.clone();
        let family_name = selector_family_artifact_name(
            &pbc_selector_metadata(&layout, key_id, &sampler, source_id, &sampler, target_id)
                .and_then(|metadata| {
                    PbcSelectorArtifacts::from_structural(&layout, key_id, names.clone(), metadata)
                })
                .unwrap(),
            "gsw",
        );
        wrong_count.artifacts.get_mut(&family_name).unwrap().family_count = Some(1);
        assert!(
            import(
                &wrong_count,
                names.clone(),
                &layout,
                key_id,
                &sampler,
                source_id,
                &sampler,
                target_id
            )
            .is_err()
        );

        let mut wrong_type = manifest.clone();
        wrong_type.artifacts.get_mut(&family_name).unwrap().artifact_type =
            ArtifactType::Matrix(ConcreteMatrixType {
                modulus: 97.into(),
                ring_dimension: 4,
                rows: 2,
                columns: 1,
            });
        assert!(
            import(
                &wrong_type,
                names.clone(),
                &layout,
                key_id,
                &sampler,
                source_id,
                &sampler,
                target_id
            )
            .is_err()
        );

        let mut wrong_metadata = manifest.clone();
        let descriptor = wrong_metadata.artifacts.get_mut(&family_name).unwrap();
        // The runtime owns the generic descriptor layout and persists it as
        // `None`.  A PBC-specific layout payload is therefore malformed rather
        // than a second source of metadata that import would have to reconcile.
        descriptor.layout = Some("unexpected-pbc-metadata".to_owned());
        assert!(
            import(
                &wrong_metadata,
                names.clone(),
                &layout,
                key_id,
                &sampler,
                source_id,
                &sampler,
                target_id
            )
            .is_err()
        );

        let mut wrong_order_names = names;
        wrong_order_names.selector_packages.swap(0, 1);
        assert!(
            import(
                &manifest,
                wrong_order_names,
                &layout,
                key_id,
                &sampler,
                source_id,
                &sampler,
                target_id
            )
            .is_err()
        );
    }
}

#[cfg(test)]
mod setup_sampling_tests {
    use super::*;
    use crate::noise_refresh::setup::RefreshSetupError;
    use encoding::{ExponentLutEncodingSampler, refresh_tests::parameters};
    use mxx_dsl::DslContext;
    use mxx_ir_core::node::NodeKind;
    #[test]
    fn fixed_rhs_preparation_does_not_unroll_active_members() {
        use crate::{
            noise_refresh::prf::{
                PreparedSparseLwrPrf, PrfSelectionInputs, SparseLwrPrfProfile, SparseLwrPrfProgram,
            },
            program::{FamilyRange, ProgramValue},
        };
        let ring = mxx_dsl::Ring::new(257, 8);
        let scheme = BggScheme::new(
            BggPublicKeyCompiler { ring: ring.clone(), base: 4.into(), digit_count: 4.into() },
            ExpLayout { stride_log2: 1, width_log2: 2 },
        );
        let profile =
            SparseLwrPrfProfile::new(ExpLayout { stride_log2: 1, width_log2: 2 }, 2).unwrap();
        let program = SparseLwrPrfProgram::new(profile, 49_172, 1).unwrap();
        let selectors =
            encoding::SetupRhsFamily::new(ring.input_family("selectors", 49_172, (2, 8))).unwrap();
        let labels = ring.input_family("labels", 2 * 49_172, (1, 1));
        let block = PrfSelectionInputs::new(
            selectors,
            &labels,
            FamilyRange::full(49_172).unwrap(),
            rand::random(),
        )
        .unwrap();
        let (public, prepared) = PreparedSparseLwrPrf::new(
            &scheme,
            program,
            ProgramValue::Scalar(BggPublicKeyWire {
                matrix: ring.input("initial-public", (2, 8)),
                reveal_plaintext: false,
            }),
            &[block],
        )
        .unwrap();
        let ProgramValue::Family(public) = public else { panic!("output family") };
        assert_eq!(public.count().evaluate(&ParamEnv::default()).unwrap(), 2.into());
        assert_eq!(prepared.steps().len(), 1);
        let context =
            DslContext::new("structural-fixed-rhs").public_output("public", public).unwrap();
        let (context, _) = prepared.export(&scheme, context, "fixed-product").unwrap();
        let graph = context.build().unwrap();
        graph.validate(&ParamEnv::default()).unwrap();
        let nodes =
            graph.graph.scopes().values().flat_map(|scope| scope.nodes()).collect::<Vec<_>>();
        // The reduction may emit logarithmically many addition scopes. The
        // expensive operations must remain one structural weight multiplication,
        // one decomposition, and one public external product regardless of size.
        assert_eq!(
            nodes
                .iter()
                .filter(|node| matches!(
                    node.kind(),
                    NodeKind::MatrixBinary(mxx_ir_core::node::MatrixBinaryOp::Multiply)
                ))
                .count(),
            1
        );
        assert_eq!(
            nodes.iter().filter(|node| matches!(node.kind(), NodeKind::MatrixMulSmallRhs)).count(),
            1
        );
        assert_eq!(
            nodes
                .iter()
                .filter(|node| matches!(node.kind(), NodeKind::GadgetDecompose { .. }))
                .count(),
            1
        );
    }
    #[test]
    fn initial_encoding_keeps_online_and_payload_secrets_distinct() -> Result<(), RefreshSetupError>
    {
        let setup = parameters(4, 2);
        let sampler = ExponentLutEncodingSampler {
            layout: setup.layout.clone(),
            exponent_layout: crate::layout::ExpLayout { stride_log2: 1, width_log2: 1 },
            gaussian_sigma: Some(setup.encoding_error_sigma.clone()),
            gaussian_max_coefficient_bound: Some(setup.encoding_error_bound.clone()),
        };
        let ring = setup.layout.ring();
        let s = sampler.sample_secret().expect("online secret s");
        let t = sampler.sample_secret().expect("setup secret t");
        let key = mxx_bgg::BggPublicKeySampler { layout: sampler.layout.clone() }
            .sample(
                ring.bytes_input("online-s-test-key", 32),
                b"online-s-test".as_slice(),
                &[false],
            )
            .into_iter()
            .nth(1)
            .expect("public key");
        let plaintext = ring.identity(1);
        let initial = sampler
            .sample_encodings_for_public_matrices(
                s.clone(),
                Some(t.clone()),
                std::slice::from_ref(&key),
                std::slice::from_ref(&plaintext),
            )?
            .into_iter()
            .next()
            .ok_or(RefreshSetupError::InvalidManifest)?
            .vector;
        let context = DslContext::new("online-s-shared-test")
            .private_output("initial", initial)
            .expect("initial output")
            .build()?;
        let uniform_nodes = context
            .graph
            .scopes()
            .values()
            .flat_map(|scope| scope.nodes())
            .filter(|node| matches!(node.kind(), NodeKind::UniformIntervalSample { .. }))
            .count();
        assert_eq!(
            uniform_nodes, 2,
            "initial encoding must retain distinct mask s and payload t samples"
        );
        assert_ne!(s.value_handle(), t.value_handle());
        Ok(())
    }
}

#[cfg(test)]
mod public_refresh_route_tests {
    use super::*;
    use crate::noise_refresh::{
        refresh::{
            RefreshPrfFamilyMaterial, RefreshPrfLabelIndex, aggregate_refresh_prf,
            prepare_refresh_public_routes,
        },
        simulation::RefreshTranscriptTopology,
    };
    use mxx_dsl::{DslContext, Family, Ring};
    use mxx_primitives::poly::{PolyParams, dcrt::params::DCRTPolyParams};
    use mxx_runtime::{
        RuntimeValue, artifact::MemoryArtifactStore, backend::poly::cpu_backend, execute,
        transcript::SamplingMode,
    };
    use std::collections::BTreeMap;

    #[test]
    fn public_refresh_routes_reuse_prepared_witnesses_and_shared_fresh_outputs() {
        let parameters = DCRTPolyParams::new(4, 1, 20, 4, None, None);
        let ring = Ring::new(BigInt::from(parameters.modulus().as_ref().clone()), 4);
        let digits = parameters.modulus_digits();
        let columns = 2 * digits;
        let compiler = mxx_bgg::BggPublicKeyCompiler {
            ring: ring.clone(),
            base: IntExpr::constant(BigInt::from(1) << parameters.base_bits()),
            digit_count: digits.into(),
        };
        let scheme = BggScheme::new(
            compiler.clone(),
            crate::layout::ExpLayout { stride_log2: 0, width_log2: 1 },
        );
        let labels = RefreshPrfLabelIndex::new(
            rand::random(),
            &RefreshTranscriptTopology {
                crt_slot_count: 2,
                component_columns: 1,
                coefficient_count: 2,
                mask_digit_counts: vec![1, 2],
                fresh_digit_count: 2,
            },
        )
        .unwrap();
        let (fresh_start, fresh_count) = labels.group_range(None);
        let public =
            Family::pack((0..labels.len()).map(|_| ring.uniform_residue((2, columns))).collect())
                .unwrap();
        let masks =
            Family::pack((0..fresh_start).map(|_| ring.uniform_residue((2, columns))).collect())
                .unwrap();
        let fresh = Family::pack(
            (0..fresh_count * 2).map(|_| ring.uniform_residue((2, columns))).collect(),
        )
        .unwrap();
        let decompose = |targets: &Family<Mat>| {
            mxx_dsl::parallel(targets.count().clone(), |index| {
                Ok(targets.at(index).decompose(compiler.base.clone(), compiler.digit_count.clone()))
            })
            .unwrap()
        };
        let mask_routes = decompose(&masks);
        let fresh_routes = decompose(&fresh);
        let (actual_masks, actual_fresh) = prepare_refresh_public_routes(
            &scheme,
            &labels,
            public.clone(),
            mask_routes.clone(),
            fresh_routes.clone(),
        )
        .unwrap();
        let online_rows =
            Family::pack((0..labels.len()).map(|_| ring.uniform_residue((1, columns))).collect())
                .unwrap();
        let mask_offsets = (0..2).map(|_| ring.uniform_residue((1, columns))).collect::<Vec<_>>();
        let fresh_offsets = (0..2).map(|_| ring.uniform_residue((1, columns))).collect::<Vec<_>>();
        let online = RefreshPrfFamilyMaterial::new(online_rows.clone(), labels.clone()).unwrap();
        let (online_masks, online_fresh) = aggregate_refresh_prf(
            &scheme,
            &online,
            mask_routes,
            fresh_routes,
            mask_offsets.clone(),
            fresh_offsets.clone(),
        )
        .unwrap();
        let actual_context = || {
            let mut context = DslContext::new("cached-public-refresh-routes");
            for slot in 0..2 {
                context = context
                    .output(format!("mask-{slot}"), actual_masks[slot].clone())
                    .unwrap()
                    .output(format!("fresh-{slot}"), actual_fresh[slot].clone())
                    .unwrap()
                    .output(format!("online-mask-{slot}"), online_masks[slot].clone())
                    .unwrap()
                    .output(format!("online-fresh-{slot}"), online_fresh[slot].clone())
                    .unwrap();
            }
            context
        };
        let graph = actual_context().build().unwrap();
        assert_eq!(
            graph
                .graph
                .scopes()
                .values()
                .flat_map(|scope| scope.nodes())
                .filter(|node| {
                    matches!(node.kind(), mxx_ir_core::node::NodeKind::GadgetDecompose { .. })
                })
                .count(),
            2,
            "one prepared decomposition body per route family"
        );
        let mut context = actual_context();
        // Compare with the existing uncached BGG primitive, using the same
        // raw public keys and targets. Fresh input indices repeat across slots;
        // fresh route indices do not. Unequal mask ranges retain slot order.
        for slot in 0..2 {
            let (mask_start, mask_count) = labels.group_range(Some(slot));
            for (purpose, input_start, route_start, count, targets) in [
                ("mask", mask_start, mask_start, mask_count, &masks),
                ("fresh", fresh_start, slot * fresh_count, fresh_count, &fresh),
            ] {
                let expected = (0..count)
                    .map(|index| {
                        compiler
                            .matrix_mul(
                                &BggPublicKeyWire {
                                    matrix: public.at(input_start + index),
                                    reveal_plaintext: false,
                                },
                                &targets.at(route_start + index),
                            )
                            .matrix
                    })
                    .reduce(|left, right| left + right)
                    .unwrap();
                context = context.output(format!("expected-{purpose}-{slot}"), expected).unwrap();
                let expected_online = (0..count)
                    .map(|index| {
                        mxx_bgg::BggEncodingCompiler
                            .matrix_mul(
                                &BggEncodingWire {
                                    vector: online_rows.at(input_start + index),
                                    plaintext: None,
                                },
                                targets
                                    .at(route_start + index)
                                    .decompose(compiler.base.clone(), compiler.digit_count.clone()),
                            )
                            .vector
                    })
                    .reduce(|left, right| left + right)
                    .unwrap();
                let offset =
                    if purpose == "mask" { &mask_offsets[slot] } else { &fresh_offsets[slot] };
                context = context
                    .output(
                        format!("expected-online-{purpose}-{slot}"),
                        expected_online + offset.clone(),
                    )
                    .unwrap();
            }
        }
        let graph = context.build().unwrap().validate(&ParamEnv::default()).unwrap();
        let result = execute(
            &graph,
            &mut cpu_backend([parameters]),
            BTreeMap::new(),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .unwrap();
        for slot in 0..2 {
            for purpose in ["mask", "fresh", "online-mask", "online-fresh"] {
                let RuntimeValue::Matrix(actual) = &result.outputs[&format!("{purpose}-{slot}")]
                else {
                    panic!("routed public matrix")
                };
                let RuntimeValue::Matrix(expected) =
                    &result.outputs[&format!("expected-{purpose}-{slot}")]
                else {
                    panic!("reference public matrix")
                };
                assert_eq!(actual.as_ref(), expected.as_ref());
            }
        }
    }
}

/// Builds the noiseless refresh verification residual and threshold decoder.
///
/// The expected plaintext is explicit and is never read from
/// `BggEncodingWire::plaintext`.  All arithmetic is ordinary matrix
/// multiplication, matching the separate-secret BGG+ relation
/// `c' = sA' - X^w*(tG) + e'`: the mask secret `s` is used for the public-key
/// term and the payload secret `t` is used for the gadget term. The supplied
/// `expected_plaintext` is the scalar/monomial `X^w` multiplier in that
/// residual, so the noiseless check is `c' - sA' + X^w(tG) = 0`.
///
/// Threshold decoding is scalar-only.  Since a BGG+ residual has one row and
/// one column per public-key column, this function slices each concrete
/// residual column to a 1x1 matrix before decoding it.  The returned Boolean
/// values are flattened in column-major, then coefficient order.  A `true`
/// value means that the corresponding residual decoded as nonzero.
pub fn build_refresh_verification(
    encoding: &BggEncodingWire,
    public_key: &Mat,
    mask_secret: &Mat,
    payload_secret: &Mat,
    expected_plaintext: &Mat,
    gadget_base: impl Into<mxx_ir_core::IntExpr>,
    gadget_digit_count: usize,
    plaintext_modulus: impl Into<mxx_ir_core::IntExpr>,
    decode_length: usize,
) -> Result<
    crate::noise_refresh::setup::RefreshVerification<Mat>,
    crate::noise_refresh::setup::RefreshSetupError,
> {
    use crate::noise_refresh::setup::{RefreshSetupError, RefreshVerification};
    let scalar = expected_plaintext.matrix_type();
    if scalar.rows != 1.into() || scalar.columns != 1.into() {
        return Err(RefreshSetupError::InvalidParameters("expected plaintext must be 1x1"));
    }
    let gadget = public_key.matrix_type();
    let expected_vector = mxx_ir_core::types::MatrixType {
        modulus: gadget.modulus.clone(),
        ring_dimension: gadget.ring_dimension.clone(),
        rows: 1.into(),
        columns: gadget.columns.clone(),
    };
    let secret_shape_matches = |secret: &Mat| {
        secret.matrix_type().rows == 1.into() &&
            secret.matrix_type().columns.evaluate(&ParamEnv::default()).ok() ==
                gadget.rows.evaluate(&ParamEnv::default()).ok() &&
            secret.matrix_type().modulus.evaluate(&ParamEnv::default()).ok() ==
                gadget.modulus.evaluate(&ParamEnv::default()).ok() &&
            secret.matrix_type().ring_dimension.evaluate(&ParamEnv::default()).ok() ==
                gadget.ring_dimension.evaluate(&ParamEnv::default()).ok()
    };
    if !crate::utils::same_matrix_type(encoding.vector.matrix_type(), &expected_vector) ||
        !secret_shape_matches(mask_secret) ||
        !secret_shape_matches(payload_secret)
    {
        return Err(RefreshSetupError::InvalidParameters(
            "mask and payload secrets must both be 1 x gadget-row matrices compatible with the encoding",
        ));
    }
    let public_key_columns = public_key
        .matrix_type()
        .columns
        .evaluate(&ParamEnv::default())
        .ok()
        .and_then(|value| value.to_usize())
        .ok_or(RefreshSetupError::InvalidParameters(
            "public-key column count must be concrete and fit usize",
        ))?;
    if public_key_columns == 0 || decode_length == 0 {
        return Err(RefreshSetupError::InvalidParameters(
            "verification requires positive public-key columns and decode length",
        ));
    }
    let ring = mxx_dsl::Ring::new(
        public_key.matrix_type().modulus.clone(),
        public_key.matrix_type().ring_dimension.clone(),
    );
    let g = ring.gadget(mask_secret.matrix_type().columns.clone(), gadget_base, gadget_digit_count);
    let anchor = mask_secret.clone() * public_key.clone() -
        expected_plaintext.clone() * (payload_secret.clone() * g);
    let residual = encoding.vector.clone() - anchor.clone();
    let mut decoder_residuals = Vec::with_capacity(public_key_columns);
    for column in 0..public_key_columns {
        let scalar_residual = residual.clone().slice(
            None,
            Some(mxx_ir_core::node::IndexRange { start: column.into(), end: (column + 1).into() }),
        );
        decoder_residuals.push(scalar_residual.clone());
    }
    RefreshVerification::new(
        encoding.vector.clone(),
        anchor,
        residual,
        decoder_residuals,
        plaintext_modulus,
        decode_length,
    )
}

#[cfg(test)]
mod refresh_verification_tests {
    use super::*;
    use mxx_dsl::{DslContext, Ring};

    #[test]
    fn refresh_verification_keeps_column_decoder_operands() {
        let ring = Ring::new(97, 4);
        let vector = ring.input("encoding", (1, 4));
        let public = ring.input("public", (2, 4));
        let s = ring.input("s", (1, 2));
        let t = ring.input("t", (1, 2));
        let expected = ring.polynomial([1.into()]);
        let verification = build_refresh_verification(
            &BggEncodingWire { vector, plaintext: None },
            &public,
            &s,
            &t,
            &expected,
            4,
            2,
            2,
            2,
        )
        .unwrap();
        assert_eq!(verification.decoder_residuals().len(), 4);
        assert_eq!(verification.decoded().len(), 8);
        verification.validate_decoder_residuals(verification.decoder_residuals()).unwrap();
        let graph = verification
            .add_outputs(DslContext::new("bgg-refresh-verification"), "residual", "decoded")
            .unwrap()
            .build()
            .unwrap();
        graph.validate(&ParamEnv::default()).unwrap();
        let nodes =
            graph.graph.scopes().values().flat_map(|scope| scope.nodes()).collect::<Vec<_>>();
        assert_eq!(
            nodes
                .iter()
                .filter(|node| matches!(
                    node.kind(),
                    mxx_ir_core::node::NodeKind::ThresholdDecode { .. }
                ))
                .count(),
            4
        );
        assert!(graph.graph.outputs().contains_key("refresh-verification-vector"));
        assert!(graph.graph.outputs().contains_key("refresh-verification-anchor"));
        for column in 0..4 {
            let name = format!("refresh-decoder-residual-{column}");
            assert!(graph.graph.outputs().contains_key(&name));
        }
    }
}

#[cfg(test)]
mod route_preimage_tests {
    use super::*;
    use crate::noise_refresh::refresh::RefreshError;
    use mxx_dsl::Ring;

    #[test]
    fn route_preimages_use_the_two_row_public_key_shape() {
        let ring = Ring::new(97, 8);
        let public_key = ring.input("route-public-key", (2, 4));
        let routes = route_preimage_family(
            public_key.matrix_type(),
            4,
            4,
            2,
            &[2],
            None,
            IntExpr::constant(2),
            IntExpr::constant(3),
        )
        .expect("valid route family");
        let graph = mxx_dsl::DslContext::new("route-preimages-two-row-target")
            .public_output("routes", routes)
            .expect("route output")
            .build()
            .expect("route graph");
        graph
            .validate(&mxx_ir_core::ParamEnv::default())
            .expect("u_2 must be valid for the two-row public-key target");

        let one_row_public_key = ring.input("invalid-route-public-key", (1, 4));
        assert!(matches!(
            route_preimage_family(
                one_row_public_key.matrix_type(),
                4,
                4,
                2,
                &[2],
                None,
                IntExpr::constant(2),
                IntExpr::constant(3),
            ),
            Err(RefreshError::InvalidLayout)
        ));
    }
}

/// Preprocess BGG digit routes `-scale * base^digit * X^coefficient * u_2 * u_j^T`.
/// One scale family is authoritative: callers supplying corrected digit weights
/// use base one. Only the source schema is needed, never a public-key value or
/// the online mask secret. Persisted preimages serve public and online routing.
pub(crate) fn route_preimage_family(
    public_key_type: &mxx_ir_core::types::MatrixType,
    base_p: usize,
    component_count: usize,
    coefficient_count: usize,
    digit_counts: &[usize],
    scales: Option<mxx_dsl::Family<Mat>>,
    decomposition_base: IntExpr,
    decomposition_digit_count: IntExpr,
) -> Result<mxx_dsl::Family<Preimage>, crate::noise_refresh::refresh::RefreshError> {
    use crate::noise_refresh::refresh::{RefreshError, route_coordinates};
    let total_digits = digit_counts.iter().try_fold(0usize, |sum, digits| {
        sum.checked_add(*digits).ok_or(RefreshError::InvalidLayout)
    })?;
    let count = component_count
        .checked_mul(coefficient_count)
        .and_then(|coordinates| coordinates.checked_mul(total_digits))
        .filter(|count| *count > 0)
        .ok_or(RefreshError::InvalidLayout)?;
    if public_key_type.rows.evaluate(&ParamEnv::default()).ok().and_then(|value| value.to_usize()) !=
        Some(2)
    {
        return Err(RefreshError::InvalidLayout);
    }
    let ring =
        mxx_dsl::Ring::new(public_key_type.modulus.clone(), public_key_type.ring_dimension.clone());
    let targets = mxx_dsl::parallel(count, |index| {
        let (_, digit, coefficient, component) = route_coordinates(
            index.expression()?,
            component_count,
            coefficient_count,
            digit_counts,
        )
        .map_err(|_| mxx_dsl::DslError::Schema)?;
        let scalar = ring.constant(
            (1, 1),
            mxx_ir_core::node::ConstantMatrix::PowerOfBase {
                base: IntExpr::constant(base_p),
                exponent: digit,
            },
        ) * ring.constant(
            (1, 1),
            mxx_ir_core::node::ConstantMatrix::Rotation { exponent: coefficient },
        );
        Ok(-scalar *
            ring.constant(
                (public_key_type.rows.clone(), 1),
                mxx_ir_core::node::ConstantMatrix::UnitColumn { index: 1.into() },
            ) *
            ring.constant(
                (1, public_key_type.columns.clone()),
                mxx_ir_core::node::ConstantMatrix::UnitRow { index: component },
            ))
    })
    .map_err(|_| RefreshError::InvalidLayout)?;
    let targets = if let Some(scales) = scales {
        if targets.count() != scales.count() {
            return Err(RefreshError::InvalidLayout);
        }
        mxx_dsl::parallel(count, |index| Ok(scales.at(&index) * targets.at(&index)))
            .map_err(|_| RefreshError::InvalidLayout)?
    } else {
        targets
    };
    mxx_dsl::parallel(count, |index| {
        Ok(targets
            .at(index)
            .decompose(decomposition_base.clone(), decomposition_digit_count.clone()))
    })
    .map_err(|_| RefreshError::InvalidLayout)
}

/// Returns the public vector added after centered native PRF routing.
///
/// Value readout returns `g = d - 1/2` for each base-`P` digit.  The routed
/// digit therefore needs the public offset
/// `1/2 * sum_c P^c * (1 + X + ... + X^(n-1))` in every public-key column.
/// This is a public ring expression and contributes no operational noise.
pub(crate) fn public_half_offset_vector(
    ring: &mxx_dsl::Ring,
    base_p: usize,
    digit_count: usize,
    columns: usize,
    normalize_mask: bool,
) -> Result<Mat, crate::noise_refresh::refresh::RefreshError> {
    use crate::noise_refresh::refresh::RefreshError;
    if base_p < 2 || digit_count == 0 || columns == 0 {
        return Err(RefreshError::InvalidLayout);
    }
    let ring_type = ring.matrix_type((1, 1));
    let modulus =
        ring_type.modulus.evaluate(&Default::default()).map_err(|_| RefreshError::InvalidLayout)?;
    if modulus <= BigInt::from(1) || (&modulus & BigInt::from(1)) == BigInt::from(0) {
        return Err(RefreshError::InvalidLayout);
    }
    let ring_dimension = ring_type
        .ring_dimension
        .evaluate(&Default::default())
        .map_err(|_| RefreshError::InvalidLayout)?
        .to_usize()
        .ok_or(RefreshError::InvalidLayout)?;
    if ring_dimension == 0 {
        return Err(RefreshError::InvalidLayout);
    }
    let inverse_two = (&modulus + BigInt::from(1)) / BigInt::from(2);
    let mut powers = BigInt::from(1);
    let mut digit_sum = BigInt::from(0);
    let base = BigInt::from(base_p);
    for _ in 0..digit_count {
        digit_sum += &powers;
        powers *= &base;
    }
    // Native digits d=g+1/2 span [1-p/2, p/2]. Masks shift this
    // consecutive interval to exactly [-floor(p^digits/2), ..., ceil(p^digits/2)-1].
    let scalar = if normalize_mask {
        let minimum_digit = BigInt::from(1) - BigInt::from(base_p / 2);
        digit_sum * (inverse_two - minimum_digit) - powers / BigInt::from(2)
    } else {
        digit_sum * inverse_two
    };
    let all_ones = ring.polynomial((0..ring_dimension).map(|_| IntExpr::constant(1)));
    let offset = ring.polynomial([IntExpr::constant(scalar)]) * all_ones;
    Ok(Mat::concat(mxx_dsl::ConcatAxis::Columns, (0..columns).map(|_| offset.clone()).collect()))
}

#[cfg(test)]
mod public_offset_tests {
    use super::*;
    use mxx_dsl::Ring;

    #[test]
    fn public_half_offset_supports_native_p16_and_p32() {
        let ring = Ring::new(97, 8);
        for (base_p, digit_count) in [(16, 2), (32, 1)] {
            let offset = public_half_offset_vector(&ring, base_p, digit_count, 2, false)
                .expect("valid public half offset");
            assert_eq!(
                offset.matrix_type().columns.evaluate(&mxx_ir_core::ParamEnv::default()),
                Ok(BigInt::from(2))
            );
            let graph = mxx_dsl::DslContext::new("public-half-offset")
                .public_output("offset", offset)
                .expect("offset output")
                .build()
                .expect("offset graph");
            graph.validate(&mxx_ir_core::ParamEnv::default()).expect("valid offset graph");
        }
    }
}

/// Explicit cache artifacts consumed by one value-readout batch.
///
/// Every decomposition is supplied by preprocessing (or imported through
/// [`crate::bgg::ReadoutCacheFamily::member`]). The online path
/// only clones these typed graph handles while constructing each label branch;
/// it never calls decomposition or public-key compilation as a fallback.
#[derive(Clone)]
#[cfg(test)]
pub struct NativePrfReadoutArtifacts {
    target_decomposition: Preimage,
    target_schema: ReadoutPreimageSchema,
    helpers: Vec<NativePrfReadoutHelper>,
    normalization_decomposition: Preimage,
    normalization_schema: ReadoutPreimageSchema,
}

/// Cache-only description of one readout branch.  The s-dependent mask
/// encoding is deliberately absent and is supplied by the online caller.
#[derive(Clone)]
#[cfg(test)]
pub(crate) struct NativePrfReadoutHelper {
    layout: ExpLayout,
    sigma: usize,
    gadget_base: BigInt,
    digit_count: usize,
    max_coefficient_bound: BigInt,
    switch_decomposition: Preimage,
    public_decomposition: Preimage,
}

pub(crate) fn helper_sigma_for_flat_index(index: IntExpr, branches: Vec<IntExpr>) -> IntExpr {
    // Flattening is label-major: every label visits all helper sigmas.
    // The helper coordinate is a remainder, not an exact integer quotient.
    IntExpr::Select {
        selector: Box::new(IntExpr::Rem(
            Box::new(index),
            Box::new(IntExpr::constant(branches.len())),
        )),
        branches,
    }
}

#[cfg(test)]
mod readout_index_tests {
    use super::helper_sigma_for_flat_index;
    use mxx_ir_core::{IntExpr, ParamEnv};

    #[test]
    fn test_readout_helper_sigma_visits_every_flat_label_coordinate() {
        for sigmas in [vec![3], vec![3, 5, 7], vec![3, 5, 7, 9, 11, 13, 15]] {
            for label in 0..4 {
                for (helper, sigma) in sigmas.iter().enumerate() {
                    let flat = label * sigmas.len() + helper;
                    let selected = helper_sigma_for_flat_index(
                        IntExpr::constant(flat),
                        sigmas.iter().copied().map(IntExpr::constant).collect(),
                    );
                    assert_eq!(selected.evaluate(&ParamEnv::default()).unwrap(), (*sigma).into());
                }
            }
        }
    }
}

/// Errors raised while validating or constructing a value readout.
#[derive(Debug, Eq, PartialEq, Error)]
pub enum BggReadoutError {
    /// The normalization scalar has no inverse in the target ring.
    #[error("2*n' is not invertible in the target ring modulus")]
    NonInvertibleNormalization,
    /// A bounded preimage requirement must use a nontrivial gadget base.
    #[error("normalization gadget base must be greater than one")]
    InvalidGadgetBase,
    /// A bounded preimage requirement must have at least one digit.
    #[error("normalization gadget digit count must be positive")]
    InvalidDigitCount,
    /// A normalization preimage bound must be nonnegative.
    #[error("normalization preimage bound must be nonnegative")]
    InvalidPreimageBound,
    /// Normalization requires the canonical gadget declaration for its base and digits.
    #[error(
        "normalization preimage gadget must be a canonical gadget with the requested base and digits"
    )]
    InvalidNormalizationGadget,
    /// A cache preimage does not match the exact source shape, bound, or
    /// decomposition schema declared by the readout operation.
    #[error("readout preimage does not match its declared source shape, bound, or gadget schema")]
    InvalidPreimageSchema,
    #[error(transparent)]
    Declaration(#[from] ValueReadoutError),
    #[error(transparent)]
    Cache(#[from] ReadoutCacheImportError),
    #[error(transparent)]
    Graph(#[from] crate::bgg::BggError),
}

impl From<LayoutError> for BggReadoutError {
    fn from(error: LayoutError) -> Self {
        Self::Declaration(error.into())
    }
}

/// Exact non-matrix schema attached to one cache-required readout preimage.
/// The matrix type is carried by [`Preimage`] itself; these fields bind the
/// gadget decomposition parameters and coefficient bound that produced it.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ReadoutPreimageSchema {
    pub gadget_base: BigInt,
    pub digit_count: usize,
    pub max_coefficient_bound: BigInt,
}

/// A non-identity value-readout branch with all deterministic right actions
/// already decomposed.  The two preimages are relation-bearing cache inputs;
/// no branch construction is allowed to call `Mat::decompose`.
#[derive(Clone)]
pub struct ValueReadoutHelper {
    layout: ExpLayout,
    sigma: usize,
    gadget_base: BigInt,
    digit_count: usize,
    max_coefficient_bound: BigInt,
    switch_decomposition: Preimage,
    public_decomposition: Preimage,
    mask: BggEncodingWire,
}

/// Public projection of one non-identity value-readout branch. It carries the
/// decomposition of the switched public target and the ordinary public mask;
/// the private switched ciphertext decomposition is not part of this path.
#[derive(Clone)]
pub struct ValueReadoutPublicHelper {
    layout: ExpLayout,
    sigma: usize,
    gadget_base: BigInt,
    digit_count: usize,
    max_coefficient_bound: BigInt,
    public_decomposition: Preimage,
    mask: Mat,
}

/// Descriptor for the concrete matrix target used by value-readout normalization.
///
/// The actual target is `(numerator / denominator) * G`, where `G` is supplied
/// by the BGG setup.  Keeping the gadget out of this descriptor prevents this
/// planning layer from pretending to have sampled or executed a preimage.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NormalizationTargetDescriptor {
    /// Exact concrete matrix type of the gadget target.
    pub matrix: ConcreteMatrixType,
    /// Layout from which `n'` and the native trace denominator are derived.
    pub layout: ExpLayout,
    /// Numerator of the normalization scalar; currently one by definition.
    pub numerator: BigInt,
    /// Denominator `2*n'` in the full ring modulus.
    pub denominator: usize,
}

/// Explicit bounded-preimage requirement for normalization.
///
/// This is metadata for a preprocessing artifact request.  It is not a
/// `Preimage`, and construction does not run the sampler or evaluator.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NormalizationPreimageRequirement {
    pub target: NormalizationTargetDescriptor,
    /// This is a relation-bearing bounded artifact, not a sampled witness.
    pub artifact_type: ArtifactType,
    /// The target is `(2*n')^-1 G` for this native layout.
    pub role: NormalizationPreimageRole,
    pub max_coefficient_bound: BigInt,
    pub gadget_base: BigInt,
    pub digit_count: usize,
    /// Stable association to the exact readout declaration.
    pub readout_identity: ValueReadoutIdentity,
}

/// Semantic role of the typed normalization preimage artifact.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum NormalizationPreimageRole {
    InverseNativeTraceGadget,
}

fn validate_canonical_gadget(
    gadget: &Mat,
    base: &BigInt,
    digit_count: usize,
) -> Result<(), BggReadoutError> {
    let mxx_ir_core::node::NodeKind::ConstantMatrix {
        matrix_type,
        value: mxx_ir_core::node::ConstantMatrix::Gadget { base: declared_base, small: false },
    } = gadget.value_handle().node().kind()
    else {
        return Err(BggReadoutError::InvalidNormalizationGadget.into());
    };
    let mxx_ir_core::IntExpr::Const(declared_base) = declared_base else {
        return Err(BggReadoutError::InvalidNormalizationGadget.into());
    };
    if declared_base != base {
        return Err(BggReadoutError::InvalidNormalizationGadget.into());
    }
    let concrete = concrete_matrix_type(matrix_type)?;
    if concrete.rows == 0 || concrete.columns != concrete.rows.saturating_mul(digit_count) {
        return Err(BggReadoutError::InvalidNormalizationGadget.into());
    }
    Ok(())
}

fn concrete_matrix_type(
    matrix_type: &mxx_ir_core::types::MatrixType,
) -> Result<ConcreteMatrixType, BggReadoutError> {
    let env = ParamEnv::default();
    let evaluate = |expr: &mxx_ir_core::IntExpr| {
        expr.evaluate(&env).map_err(|_| ValueReadoutError::NonConcreteRingParameter)
    };
    Ok(ConcreteMatrixType {
        modulus: evaluate(&matrix_type.modulus)?,
        ring_dimension: evaluate(&matrix_type.ring_dimension)?
            .to_usize()
            .ok_or(ValueReadoutError::NonConcreteRingParameter)?,
        rows: evaluate(&matrix_type.rows)?
            .to_usize()
            .ok_or(ValueReadoutError::NonConcreteRingParameter)?,
        columns: evaluate(&matrix_type.columns)?
            .to_usize()
            .ok_or(ValueReadoutError::NonConcreteRingParameter)?,
    })
}

fn reduce_matrix_family_segments(
    family: Family<Mat>,
    segment_count: usize,
    segment_size: usize,
) -> Result<Family<Mat>, BggReadoutError> {
    if segment_count == 0 || segment_size == 0 {
        return Err(ValueReadoutError::InvalidTraceSet.into());
    }
    crate::utils::reduce_family_segments(family, segment_count, segment_size, |left, right| {
        left + right
    })
    .map_err(|error| {
        BggReadoutError::Graph(crate::bgg::BggError::OnlinePrfStage {
            stage: "reduce readout helper segments",
            detail: error.to_string(),
        })
    })
}

fn validate_preimage_ring(preimage: &Preimage, ring: &Ring) -> Result<(), BggReadoutError> {
    let expected = ring.matrix_type((1, 1));
    let actual = preimage.matrix_type();
    if actual.modulus != expected.modulus || actual.ring_dimension != expected.ring_dimension {
        return Err(ValueReadoutError::RingTypeMismatch.into());
    }
    let bound = preimage
        .max_coefficient_bound()
        .evaluate(&ParamEnv::default())
        .map_err(|_| ValueReadoutError::NonConcreteRingParameter)?;
    if bound < BigInt::zero() {
        return Err(BggReadoutError::InvalidPreimageBound.into());
    }
    Ok(())
}

fn validate_matrix_ring(matrix: &Mat, ring: &Ring) -> Result<(), BggReadoutError> {
    let expected = ring.matrix_type((1, 1));
    let actual = matrix.matrix_type();
    if actual.modulus != expected.modulus || actual.ring_dimension != expected.ring_dimension {
        return Err(ValueReadoutError::RingTypeMismatch.into());
    }
    Ok(())
}

fn validate_family_preimage_shape(
    actual: &mxx_ir_core::types::MatrixType,
    source: &Mat,
    digit_count: usize,
) -> Result<(), BggReadoutError> {
    let actual = concrete_matrix_type(actual)?;
    let source = concrete_matrix_type(source.matrix_type())?;
    if actual.modulus != source.modulus ||
        actual.ring_dimension != source.ring_dimension ||
        actual.rows != source.rows.saturating_mul(digit_count) ||
        actual.columns != source.columns
    {
        return Err(BggReadoutError::InvalidPreimageSchema.into());
    }
    Ok(())
}

fn validate_encoding_action_preimage_family_type(
    actual: &mxx_ir_core::types::MatrixType,
    carrier: &mxx_ir_core::types::MatrixType,
    digit_count: usize,
) -> Result<(), BggReadoutError> {
    let actual = concrete_matrix_type(actual)?;
    let carrier = concrete_matrix_type(carrier)?;
    if actual.modulus != carrier.modulus ||
        actual.ring_dimension != carrier.ring_dimension ||
        actual.rows != carrier.columns ||
        actual.columns != carrier.columns ||
        digit_count == 0 ||
        carrier.columns % digit_count != 0
    {
        return Err(BggReadoutError::InvalidPreimageSchema.into());
    }
    Ok(())
}

fn validate_family_output_shape(
    actual: &mxx_ir_core::types::MatrixType,
    source: &Mat,
    ring: &Ring,
) -> Result<(), BggReadoutError> {
    validate_family_output_type(actual, source.matrix_type(), ring)
}

fn validate_family_output_type(
    actual: &mxx_ir_core::types::MatrixType,
    source: &mxx_ir_core::types::MatrixType,
    ring: &Ring,
) -> Result<(), BggReadoutError> {
    let expected_ring = ring.matrix_type((1, 1));
    if actual.modulus != expected_ring.modulus ||
        actual.ring_dimension != expected_ring.ring_dimension
    {
        return Err(ValueReadoutError::RingTypeMismatch.into());
    }
    let actual = concrete_matrix_type(actual)?;
    let source = concrete_matrix_type(source)?;
    if actual.rows != source.rows || actual.columns != source.columns {
        return Err(BggReadoutError::InvalidPreimageSchema.into());
    }
    Ok(())
}

fn validate_output_shape(output: &Mat, source: &Mat, ring: &Ring) -> Result<(), BggReadoutError> {
    validate_matrix_ring(output, ring)?;
    let expected = concrete_matrix_type(source.matrix_type())?;
    let actual = concrete_matrix_type(output.matrix_type())?;
    if actual.rows != expected.rows || actual.columns != expected.columns {
        return Err(BggReadoutError::InvalidPreimageSchema.into());
    }
    Ok(())
}

fn validate_preimage_schema(
    preimage: &Preimage,
    source: &Mat,
    gadget_base: &BigInt,
    digit_count: usize,
    expected_bound: Option<&BigInt>,
) -> Result<(), BggReadoutError> {
    let actual = concrete_matrix_type(preimage.matrix_type())?;
    let source = concrete_matrix_type(source.matrix_type())?;
    if actual.modulus != source.modulus ||
        actual.ring_dimension != source.ring_dimension ||
        actual.rows != source.rows.saturating_mul(digit_count) ||
        actual.columns != source.columns
    {
        return Err(BggReadoutError::InvalidPreimageSchema.into());
    }
    let bound = preimage
        .max_coefficient_bound()
        .evaluate(&ParamEnv::default())
        .map_err(|_| ValueReadoutError::NonConcreteRingParameter)?;
    if bound < BigInt::zero() || expected_bound.is_some_and(|expected| expected != &bound) {
        return Err(BggReadoutError::InvalidPreimageSchema.into());
    }
    if gadget_base <= &BigInt::one() || digit_count == 0 {
        return Err(BggReadoutError::InvalidPreimageSchema.into());
    }
    Ok(())
}

/// Validates a cached right action consumed by an online encoding carrier.
///
/// A carrier has shape `1 x m_b`, while the cached value is `D(A)` for an
/// underlying public matrix `A` with `m_b` columns.  Consequently the
/// executable `mul_small_rhs` consumes an `m_b x m_b` preimage.  Treating the
/// carrier itself as the decomposition source would incorrectly require only
/// `digit_count x m_b` rows and accept a matrix that cannot be multiplied.
fn validate_encoding_action_preimage_schema(
    preimage: &Preimage,
    carrier: &Mat,
    gadget_base: &BigInt,
    digit_count: usize,
    expected_bound: Option<&BigInt>,
) -> Result<(), BggReadoutError> {
    validate_encoding_action_preimage_schema_type(
        preimage,
        carrier.matrix_type(),
        gadget_base,
        digit_count,
        expected_bound,
    )
}

fn validate_encoding_action_preimage_schema_type(
    preimage: &Preimage,
    carrier: &mxx_ir_core::types::MatrixType,
    gadget_base: &BigInt,
    digit_count: usize,
    expected_bound: Option<&BigInt>,
) -> Result<(), BggReadoutError> {
    let actual = concrete_matrix_type(preimage.matrix_type())?;
    let carrier = concrete_matrix_type(carrier)?;
    if actual.modulus != carrier.modulus ||
        actual.ring_dimension != carrier.ring_dimension ||
        actual.rows != carrier.columns ||
        actual.columns != carrier.columns ||
        digit_count == 0 ||
        carrier.columns % digit_count != 0
    {
        return Err(BggReadoutError::InvalidPreimageSchema.into());
    }
    let bound = preimage
        .max_coefficient_bound()
        .evaluate(&ParamEnv::default())
        .map_err(|_| ValueReadoutError::NonConcreteRingParameter)?;
    if bound < BigInt::zero() || expected_bound.is_some_and(|expected| expected != &bound) {
        return Err(BggReadoutError::InvalidPreimageSchema.into());
    }
    if gadget_base <= &BigInt::one() {
        return Err(BggReadoutError::InvalidPreimageSchema.into());
    }
    Ok(())
}

fn public_bgg_schema(compiler: &BggPublicKeyCompiler) -> Result<(BigInt, usize), BggReadoutError> {
    let base = compiler
        .base
        .evaluate(&ParamEnv::default())
        .map_err(|_| ValueReadoutError::NonConcreteRingParameter)?;
    let digits = compiler
        .digit_count
        .evaluate(&ParamEnv::default())
        .map_err(|_| ValueReadoutError::NonConcreteRingParameter)?
        .to_usize()
        .ok_or(ValueReadoutError::NonConcreteRingParameter)?;
    Ok((base, digits))
}

fn normalization_denominator(
    layout: &ExpLayout,
    ring_dimension: usize,
) -> Result<usize, BggReadoutError> {
    native_width(layout, ring_dimension)?
        .checked_mul(2)
        .ok_or(ValueReadoutError::NormalizationDenominatorOverflow.into())
}

fn modular_inverse(value: &BigInt, modulus: &BigInt) -> Option<BigInt> {
    if modulus <= &BigInt::one() {
        return None;
    }
    let mut old_r = value % modulus;
    if old_r.sign() == Sign::Minus {
        old_r += modulus;
    }
    let mut r = modulus.clone();
    let mut old_s = BigInt::one();
    let mut s = BigInt::zero();
    while !r.is_zero() {
        let quotient = &old_r / &r;
        let next_r = old_r - &quotient * &r;
        old_r = r;
        r = next_r;
        let next_s = old_s - &quotient * &s;
        old_s = s;
        s = next_s;
    }
    (old_r == BigInt::one()).then(|| {
        let mut result = old_s % modulus;
        if result.sign() == Sign::Minus {
            result += modulus;
        }
        result
    })
}

/// The operand role that caused a deterministic decomposition to be cached.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub enum DecompositionOperandRole {
    LutHelperRhs,
    ReadoutAutomorphism,
    ReadoutNormalization,
}

/// Complete application-owned association for one cache artifact.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct DecompositionCacheKey {
    pub layout: ExpLayout,
    /// Identity of the value-readout declaration whose family is cached.
    /// This is separate from the program identity because the same graph
    /// location can host different readout tables over its lifetime.
    pub readout_identity: ValueReadoutIdentity,
    pub setup_identity: [u8; 32],
    pub program_identity: [u8; 32],
    pub gate_or_readout_location: String,
    pub family_coordinate: Vec<usize>,
    pub operand_role: DecompositionOperandRole,
}

/// A manifest-ready cache artifact declaration.  The payload must be a typed
/// relation-bearing preimage; ordinary public matrices are intentionally not
/// representable by this type.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DecompositionCacheArtifact {
    key: DecompositionCacheKey,
    artifact_name: String,
    /// Versioned schema name is part of the artifact identity, not merely
    /// manifest documentation.  Changing the schema can therefore never
    /// alias an older cache member.
    artifact_schema: String,
    artifact_type: ArtifactType,
    family_shape: Option<Vec<usize>>,
}

/// Errors raised while importing a native readout cache family.
#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum ReadoutCacheImportError {
    #[error("cache artifact belongs to a different production")]
    ProductionMismatch,
    #[error("cache artifact is absent from the producer manifest")]
    MissingArtifact,
    #[error("cache artifact manifest declaration is incompatible")]
    DescriptorMismatch,
    #[error("cache artifact is not a family")]
    MissingFamilyShape,
    #[error("cache family member index is out of range")]
    MemberOutOfRange,
    #[error("cache family shape is not representable")]
    InvalidFamilyShape,
    #[error("cache artifact matrix type is not representable")]
    InvalidMatrixType,
}

/// A producer-side declaration and online importer for one readout
/// decomposition artifact. Family-shaped declarations load one requested
/// member; scalar declarations use index zero and remain equally lazy.
///
/// The producer owns only the typed manifest declaration.  The preimage
/// payloads are emitted by the preprocessing graph under this name.  Online,
/// [`Self::member`] constructs a typed artifact input; the runtime's
/// `FileArtifactStore` consequently loads only that requested member, never
/// all family members into memory.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReadoutCacheFamily {
    production_id: ProductionId,
    declaration: DecompositionCacheArtifact,
    confidentiality: ArtifactConfidentiality,
}

fn canonical_artifact_name(
    key: &DecompositionCacheKey,
    artifact_schema: &str,
    artifact_type: &ArtifactType,
    family_shape: &Option<Vec<usize>>,
) -> String {
    let declaration = serde_json::json!({
        "key": key,
        "artifact_schema": artifact_schema,
        "artifact_type": artifact_type,
        "family_shape": family_shape,
    });
    let digest: [u8; 32] = Sha256::digest(
        serde_json::to_vec(&declaration).expect("cache artifact declaration is serializable"),
    )
    .into();
    let hex = digest.iter().map(|byte| format!("{byte:02x}")).collect::<String>();
    format!("exponent-lut/decomposition/{hex}")
}

/// Production readout cache.  The non-identity helper witnesses stay as two
/// aligned structural families, so setup and online lowering never enumerate
/// or statically select helper members on the host.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BggPreparedReadout<F = Family<Preimage>, P = Preimage> {
    target_decompositions: F,
    target_schema: ReadoutPreimageSchema,
    helper_sigmas: Vec<usize>,
    helper_switch_decompositions: F,
    helper_public_decompositions: F,
    normalization_decomposition: P,
    normalization_schema: ReadoutPreimageSchema,
    output_count: usize,
}

impl ExponentLUTReadoutScheme for BggScheme {
    type ReadoutArtifacts = BggPreparedReadout;
    // Canonical helper sigmas, public switch ciphertext family, normalization
    // gadget. Fresh online mask vectors are intentionally absent.
    type ReadoutPreparation = (std::sync::Arc<Vec<usize>>, Family<Mat>, Mat);
    type ReadoutPublicMasks = Family<Mat>;
    type ReadoutEncodingMasks = Family<Mat>;
    type ReadoutPublicOutput = Family<Mat>;
    type ReadoutEncodingOutput = Family<Mat>;
    type ReadoutError = BggReadoutError;

    fn prepare_readout(
        &self,
        readout: &ValueReadout,
        inputs: &crate::scheme::PublicProgramValue<Self>,
        preparation: &Self::ReadoutPreparation,
    ) -> Result<Self::ReadoutArtifacts, Self::ReadoutError> {
        let compiler = &self.public;
        let (sigmas, switch_ciphertexts, normalization_gadget) = preparation;
        let inputs = match inputs {
            ProgramValue::Scalar(key) => Family::pack(vec![key.matrix.clone()]),
            ProgramValue::Family(keys) => keys.field(|key| key.matrix),
        }
        .map_err(BggError::from)?;
        let normalization_gadget = normalization_gadget.clone();
        let ring = compiler.public_key.ring.clone();
        readout.validate(&ring)?;
        let ring_dimension = ring
            .matrix_type((1, 1))
            .ring_dimension
            .evaluate(&mxx_ir_core::ParamEnv::default())
            .map_err(|_| ValueReadoutError::NonConcreteRingParameter)?
            .to_usize()
            .ok_or(ValueReadoutError::NonConcreteRingParameter)?;
        let plan = readout.trace_plan(ring_dimension)?;
        let output_count = inputs
            .count()
            .evaluate(&mxx_ir_core::ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .filter(|count| *count != 0)
            .ok_or(ValueReadoutError::InvalidTraceSet)?;
        if sigmas.as_slice() != plan.non_identity_helpers() {
            return Err(ValueReadoutError::InvalidTraceSet.into());
        }
        let gadget_base = compiler
            .public_key
            .base
            .evaluate(&mxx_ir_core::ParamEnv::default())
            .map_err(|_| ValueReadoutError::NonConcreteRingParameter)?;
        let digit_count = compiler
            .public_key
            .digit_count
            .evaluate(&mxx_ir_core::ParamEnv::default())
            .map_err(|_| ValueReadoutError::NonConcreteRingParameter)?
            .to_usize()
            .ok_or(ValueReadoutError::NonConcreteRingParameter)?;
        let actual_bound = &gadget_base / BigInt::from(2u8);
        let schema = ReadoutPreimageSchema::new(gadget_base.clone(), digit_count, actual_bound)?;
        let target = readout.target(&ring)?;
        let target_for_outputs = target.clone();
        let target_compiler = compiler.public_key.clone();
        let target_decompositions = {
            let source_family = inputs.clone();
            mxx_dsl::parallel(source_family.count().clone(), |family_index| {
                let input = source_family.at(&family_index);
                let _ = family_index;
                Ok({
                    target_compiler.large_scalar_decomposition(
                        &BggPublicKeyWire { matrix: input, reveal_plaintext: false },
                        &target_for_outputs,
                    )
                })
            })
        }
        .map_err(|_| BggReadoutError::Graph(BggError::InvalidLut))?;
        let transformed = inputs
            .zip_map(target_decompositions.clone(), |input, target| Ok(target.mul_small_rhs(input)))
            .map_err(|_| BggReadoutError::Graph(BggError::InvalidLut))?;
        let switch_decompositions = {
            let source_family = switch_ciphertexts.clone();
            mxx_dsl::parallel(source_family.count().clone(), |family_index| {
                let switch = source_family.at(&family_index);
                let _ = family_index;
                Ok(switch.decompose(compiler.public_key.base.clone(), digit_count))
            })
        }
        .map_err(|_| BggReadoutError::Graph(BggError::InvalidLut))?;
        let helper_count = sigmas.len();
        let flat_count =
            output_count.checked_mul(helper_count).ok_or(ValueReadoutError::InvalidTraceSet)?;
        let label_indices = mxx_dsl::parallel(flat_count, |index| {
            Ok(index.clone().div(mxx_dsl::Int::constant(helper_count)))
        })
        .map_err(|_| BggReadoutError::Graph(BggError::InvalidLut))?;
        let flat_transformed = transformed
            .gather(label_indices)
            .map_err(|_| BggReadoutError::Graph(BggError::InvalidLut))?;
        let flat_switches = {
            let source_family = switch_decompositions.clone();
            let gather_indices = mxx_dsl::parallel(flat_count, |index| {
                Ok(index.clone().rem(mxx_dsl::Int::constant(helper_count)))
            })
            .map_err(|_| BggReadoutError::Graph(BggError::InvalidLut))?;
            mxx_dsl::parallel(gather_indices.count().clone(), |family_index| {
                Ok(source_family.at(gather_indices.at(&family_index)))
            })
        }
        .map_err(|_| BggReadoutError::Graph(BggError::InvalidLut))?;
        let sigma_branches = sigmas.iter().copied().map(IntExpr::constant).collect::<Vec<_>>();
        let public_decompositions = {
            let source_families = (flat_transformed, flat_switches);
            if source_families.0.count() != source_families.1.count() {
                Err(mxx_dsl::DslError::Schema)
            } else {
                mxx_dsl::parallel(source_families.0.count().clone(), |family_index| {
                    let (transformed, switch) =
                        (source_families.0.at(&family_index), source_families.1.at(&family_index));
                    let index = family_index;
                    Ok({
                        let sigma = crate::bgg::helper_sigma_for_flat_index(
                            index.expression()?,
                            sigma_branches.clone(),
                        );
                        switch
                            .mul_small_rhs(transformed.ring_automorphism(sigma))
                            .decompose(compiler.public_key.base.clone(), digit_count)
                    })
                })
            }
        }
        .map_err(|_| BggReadoutError::Graph(BggError::InvalidLut))?;
        let normalization_target = readout.normalization_target(&ring, normalization_gadget)?;
        let normalization_decomposition =
            normalization_target.decompose(compiler.public_key.base.clone(), digit_count);
        BggPreparedReadout::new(
            target_decompositions,
            schema.clone(),
            sigmas.as_ref().clone(),
            switch_decompositions,
            public_decompositions,
            normalization_decomposition,
            schema,
            output_count,
        )
    }

    fn project_readout(
        &self,
        readout: &ValueReadout,
        inputs: &crate::scheme::PublicProgramValue<Self>,
        artifacts: &Self::ReadoutArtifacts,
        masks: &Self::ReadoutPublicMasks,
    ) -> Result<Self::ReadoutPublicOutput, Self::ReadoutError> {
        let inputs = match inputs {
            ProgramValue::Scalar(key) => Family::pack(vec![key.matrix.clone()]),
            ProgramValue::Family(keys) => keys.field(|key| key.matrix),
        }
        .map_err(BggError::from)?;
        readout.project_public_batch_with_preprocessed_family_artifacts(
            &self.public.public_key,
            inputs.clone(),
            artifacts.output_count,
            artifacts.target_decompositions.clone(),
            artifacts.target_schema.clone(),
            &artifacts.helper_sigmas,
            artifacts.helper_public_decompositions.clone(),
            masks.clone(),
            artifacts.normalization_decomposition.clone(),
            artifacts.normalization_schema.clone(),
        )
    }

    fn evaluate_readout(
        &self,
        readout: &ValueReadout,
        inputs: &crate::scheme::EncodingProgramValue<Self>,
        artifacts: &Self::ReadoutArtifacts,
        masks: &Self::ReadoutEncodingMasks,
    ) -> Result<Self::ReadoutEncodingOutput, Self::ReadoutError> {
        let inputs = match inputs {
            ProgramValue::Scalar(encoding) => Family::pack(vec![encoding.vector.clone()]),
            ProgramValue::Family(encodings) => encodings.field(|encoding| encoding.vector),
        }
        .map_err(BggError::from)?;
        let vectors = with_benchmark_role(BenchmarkRole::EncodingReadout, || {
            readout.execute_batch_with_preprocessed_family_artifacts(
                &self.encoding.bgg,
                inputs.clone(),
                artifacts.output_count,
                artifacts.target_decompositions.clone(),
                artifacts.target_schema.clone(),
                &artifacts.helper_sigmas,
                artifacts.helper_switch_decompositions.clone(),
                artifacts.helper_public_decompositions.clone(),
                masks.clone(),
                artifacts.normalization_decomposition.clone(),
                artifacts.normalization_schema.clone(),
            )
        })?;
        Ok(vectors)
    }
}

#[cfg(test)]
use crate::noise_refresh::prf::SparseLwrPrfProfile;
#[cfg(test)]
use crate::readout::canonical_table_declaration;
use crate::{
    layout::{ExpLayout, LayoutError},
    noise_refresh::prf::SparseLwrPrfProgram,
    readout::{
        ExponentLUTReadoutScheme, ValueReadout, ValueReadoutError, ValueReadoutIdentity,
        concrete_ring_dimension, native_width,
    },
};
#[cfg(test)]
use encoding::LutHelperSet;
use mxx_bgg::BggPublicKeyCompiler;
use mxx_dsl::DslContext;
use mxx_ir_core::{BenchmarkRole, artifact::ManifestArtifact, with_benchmark_role};
use mxx_runtime::ArtifactStore;
use num_bigint::Sign;
use num_traits::One;

use crate::noise_refresh::{
    refresh::RefreshPrfContract, setup::RefreshSetupError, simulation::RefreshTranscriptTopology,
};
use bigdecimal::BigDecimal;
use mxx_primitives::sampler::bounds::TrapdoorPreimageCutoffPolicy;

#[derive(Clone)]
/// Public dimensions and identities used to build one refresh setup.
///
/// The fields describe both the BGG sampler and the CRT refresh.
/// `lut_width` is the Exponent-LUT coefficient-sieve width; it is independent of
/// any PBC bucket width. The constructor sets the authoritative decoder bound
/// policy, and validation is repeated at every producer/import boundary.
pub struct BggRefreshParameters {
    /// Application-level refresh instance identity.
    pub refresh_id: [u8; 32],
    /// Sparse-LWR modulus used by the PRF layer.
    pub base_p: usize,
    /// Number of secret components represented by each BGG matrix.
    pub component_count: usize,
    /// Number of refresh coefficients processed per CRT slot.
    pub coefficient_count: usize,
    /// Number of base-`p` digits represented by each mask PRF label. This is
    /// independent of `layout.digit_count`, which is the gadget digit count.
    pub mask_base_p_digit_counts: Vec<usize>,
    /// Number of base-`p` digits represented by each fresh-error PRF label.
    pub fresh_error_base_p_digit_count: usize,
    /// Statistical security parameter used for the joint mask transcript.
    pub mask_statistical_security_bits: usize,
    /// Explicit Exponent-LUT sieve width `W`; this is not the PBC bucket width.
    pub lut_width: usize,
    /// BGG sampler dimensions shared by setup artifacts.
    pub layout: BggSamplerLayout,
    /// CRT modulus and reconstruction data for the refresh equations.
    pub refresh: RefreshCompiler,
    /// Exact selected CRT-subset destination modulus for each refresh slot.
    pub switch_moduli: Vec<BigUint>,
    /// Gaussian width used when sampling the decoder trapdoor.
    pub decoder_sigma: mxx_ir_core::RealExpr,
    /// Gaussian width used by every setup helper and the shared anchor error.
    pub encoding_error_sigma: mxx_ir_core::RealExpr,
    /// Common hard coefficient cutoff `B_chi` for setup helper errors.
    pub encoding_error_bound: mxx_ir_core::IntExpr,
    /// Policy for the decoder preimage rejection cutoff.  The policy is
    /// resolved to one concrete integer before the producer graph is built.
    pub decoder_preimage_bound: TrapdoorPreimageCutoffPolicy,
    /// Human-readable setup name used by callers when selecting a production.
    pub name: String,
}

/// Public security-analysis inputs, not a security estimate or joint-view proof.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BggRefreshAnchorSecurityInputs {
    pub modulus: BigUint,
    pub complementary_modulus: BigUint,
    pub slots: Vec<usize>,
    pub ring_dimension: usize,
    pub random_secret_polynomials: usize,
    pub secret_distribution: String,
    pub fixed_augmented_coordinate: i32,
    pub error_sigma: mxx_ir_core::RealExpr,
    pub error_coefficient_cutoff: mxx_ir_core::IntExpr,
    pub anchor_columns: usize,
    pub execution_budget: usize,
    pub ring_samples: usize,
    pub estimated_security_bits: Option<u64>,
    pub status: String,
}

fn resolve_encoding_error_bound(
    sigma: &mxx_ir_core::RealExpr,
) -> Result<mxx_ir_core::IntExpr, RefreshSetupError> {
    let sigma = sigma
        .evaluate_f64(&ParamEnv::default())
        .map_err(|_| RefreshSetupError::InvalidParameters("encoding sigma must be concrete"))?;
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(RefreshSetupError::InvalidParameters(
            "encoding sigma must be positive and finite",
        ));
    }
    let sigma = BigDecimal::from_f64(sigma)
        .ok_or(RefreshSetupError::InvalidParameters("encoding sigma is not finite"))?;
    mxx_primitives::sampler::bounds::hard_cutoff_from_sigma_bound(&sigma)
        .to_bigint()
        .map(Into::into)
        .ok_or(RefreshSetupError::InvalidParameters("encoding cutoff overflow"))
}

fn encoding_error_bound_or_zero(sigma: &mxx_ir_core::RealExpr) -> mxx_ir_core::IntExpr {
    resolve_encoding_error_bound(sigma).unwrap_or_else(|_| 0.into())
}

fn is_power_of_two_expr(value: &mxx_ir_core::IntExpr) -> bool {
    value
        .evaluate(&ParamEnv::default())
        .ok()
        .and_then(|value| value.to_biguint())
        .is_some_and(|value| value > 1u8.into() && (BigUint::one() << (value.bits() - 1)) == value)
}

impl std::fmt::Debug for BggRefreshParameters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BggRefreshParameters")
            .field("refresh_id", &self.refresh_id)
            .field("base_p", &self.base_p)
            .field("component_count", &self.component_count)
            .field("coefficient_count", &self.coefficient_count)
            .field("mask_base_p_digit_count", &self.mask_base_p_digit_counts)
            .field("fresh_error_base_p_digit_count", &self.fresh_error_base_p_digit_count)
            .field("mask_statistical_security_bits", &self.mask_statistical_security_bits)
            .field("lut_width", &self.lut_width)
            .field("layout", &self.layout)
            .field("decoder_sigma", &self.decoder_sigma)
            .field("encoding_error_sigma", &self.encoding_error_sigma)
            .field("encoding_error_bound", &self.encoding_error_bound)
            .field("decoder_preimage_bound", &self.decoder_preimage_bound)
            .field("name", &self.name)
            .finish()
    }
}
impl BggRefreshParameters {
    /// Validates the contract against the refresh setup dimensions.
    pub(crate) fn validate_prf(&self, contract: &RefreshPrfContract) -> Result<(), RefreshError> {
        let ring_dimension = self
            .layout
            .ring_dimension
            .evaluate(&Default::default())
            .map_err(|_| RefreshError::InvalidLayout)?
            .to_usize()
            .ok_or(RefreshError::InvalidLayout)?;
        // The contract owns a validated native profile. Only compatibility
        // with this setup remains to be checked at this boundary.
        if contract.p() != self.base_p || contract.ring_dimension() != ring_dimension {
            return Err(RefreshError::PrfOutputMismatch);
        }
        Ok(())
    }

    /// Creates refresh setup parameters with the official preimage cutoff
    /// policy.  Call [`Self::with_decoder_preimage_bound`] only for an
    /// explicitly reviewed alternative cutoff.
    pub fn new(
        refresh_id: [u8; 32],
        base_p: usize,
        component_count: usize,
        coefficient_count: usize,
        mask_base_p_digit_count: usize,
        fresh_error_base_p_digit_count: usize,
        mask_statistical_security_bits: usize,
        lut_width: usize,
        layout: BggSamplerLayout,
        refresh: RefreshCompiler,
        decoder_sigma: mxx_ir_core::RealExpr,
        encoding_error_sigma: mxx_ir_core::RealExpr,
        name: impl Into<String>,
    ) -> Self {
        let encoding_error_bound = encoding_error_bound_or_zero(&encoding_error_sigma);
        let switch_moduli = vec![
            refresh
                .full_modulus
                .evaluate(&ParamEnv::default())
                .ok()
                .and_then(|value| value.to_biguint())
                .unwrap_or_default();
            refresh.crt_plaintext_moduli.len()
        ];
        Self {
            refresh_id,
            base_p,
            component_count,
            coefficient_count,
            mask_base_p_digit_counts: vec![mask_base_p_digit_count; switch_moduli.len()],
            fresh_error_base_p_digit_count,
            mask_statistical_security_bits,
            lut_width,
            layout,
            refresh,
            switch_moduli,
            decoder_sigma,
            encoding_error_sigma,
            encoding_error_bound,
            decoder_preimage_bound: TrapdoorPreimageCutoffPolicy::Default,
            name: name.into(),
        }
    }

    /// Replaces the default trapdoor-sampler cutoff policy with an explicit policy.
    pub fn with_decoder_preimage_bound(mut self, bound: TrapdoorPreimageCutoffPolicy) -> Self {
        self.decoder_preimage_bound = bound;
        self
    }

    /// Builds the native sparse-LWR profile for a selected power-of-two `Q`.
    ///
    /// Refresh setup owns `p` and `W`; the caller supplies the concrete
    /// plaintext modulus because a refresh may contain several CRT moduli.
    /// Keeping this check explicit prevents a PBC bucket width from being
    /// accidentally reused as the Exponent-LUT domain width.
    pub fn sparse_lwr_profile(
        &self,
        q_l: usize,
    ) -> Result<crate::noise_refresh::prf::SparseLwrPrfProfile, RefreshSetupError> {
        let ring_dimension = self
            .layout
            .ring_dimension
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(RefreshSetupError::InvalidParameters("ring dimension must be concrete"))?;
        if q_l < 2 || !q_l.is_power_of_two() {
            return Err(RefreshSetupError::InvalidParameters("native PRF Q must be a power of two"));
        }
        let width_log2 = q_l
            .trailing_zeros()
            .checked_sub(1)
            .ok_or(RefreshSetupError::InvalidParameters("native PRF Q is too small"))?;
        let stride_log2 = ring_dimension
            .trailing_zeros()
            .checked_sub(width_log2)
            .ok_or(RefreshSetupError::InvalidParameters("native PRF layout exceeds ring"))?;
        let layout = crate::layout::ExpLayout { stride_log2, width_log2 };
        crate::noise_refresh::prf::SparseLwrPrfProfile::new(layout, self.base_p / 2)
            .map_err(|_| RefreshSetupError::InvalidParameters("invalid sparse-LWR profile"))
    }

    /// Number of PRF components covered by every mask and fresh-error group.
    /// This is the BGG public-key column count (`2 * ell_beta`), not the
    /// secret dimension stored in `component_count`.
    pub fn prf_component_count(&self) -> usize {
        self.layout.public_key_columns()
    }

    /// One transcript shape shared by PRF label generation and noise accounting.
    pub fn transcript_topology(&self) -> RefreshTranscriptTopology {
        RefreshTranscriptTopology {
            crt_slot_count: self.refresh.crt_plaintext_moduli.len(),
            component_columns: self.prf_component_count(),
            coefficient_count: self.coefficient_count,
            mask_digit_counts: self.mask_base_p_digit_counts.clone(),
            fresh_digit_count: self.fresh_error_base_p_digit_count,
        }
    }

    pub(crate) fn validate_dimensions(&self) -> Result<(), RefreshSetupError> {
        let n = self
            .layout
            .ring_dimension
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|x| x.to_usize())
            .ok_or(RefreshSetupError::InvalidParameters("ring dimension must be concrete"))?;
        if self.base_p < 2 ||
            self.component_count != self.layout.secret_dimension ||
            self.coefficient_count == 0 ||
            self.coefficient_count != n ||
            self.lut_width == 0 ||
            !self.lut_width.is_power_of_two() ||
            self.lut_width > n ||
            n % self.lut_width != 0 ||
            self.mask_base_p_digit_counts.len() != self.switch_moduli.len() ||
            self.mask_base_p_digit_counts.iter().any(|count| *count == 0) ||
            self.fresh_error_base_p_digit_count == 0 ||
            self.mask_statistical_security_bits == 0 ||
            self.layout.digit_count == 0 ||
            self.layout.secret_dimension != 2 ||
            !is_power_of_two_expr(&self.layout.gadget_base) ||
            self.refresh.full_modulus.canonicalize() != self.layout.modulus.canonicalize()
        {
            return Err(RefreshSetupError::InvalidParameters("inconsistent dimensions"));
        }
        let expected_prf_component_count = self
            .layout
            .secret_dimension
            .checked_mul(self.layout.digit_count)
            .ok_or(RefreshSetupError::InvalidParameters("public-key column count overflow"))?;
        if self.prf_component_count() != expected_prf_component_count {
            return Err(RefreshSetupError::InvalidParameters("invalid public-key column count"));
        }
        self.refresh.validate_layout()?;
        let expected_bound = resolve_encoding_error_bound(&self.encoding_error_sigma)?;
        if self.encoding_error_bound != expected_bound {
            return Err(RefreshSetupError::InvalidParameters(
                "encoding error cutoff does not match encoding sigma",
            ));
        }
        Ok(())
    }

    /// Resolves the preimage cutoff using the authoritative primitives formula.
    /// All dimensions and sigma must be concrete and valid; symbolic,
    /// non-positive, non-finite, and overflowing inputs fail closed.
    pub fn resolve_decoder_preimage_bound(
        &self,
    ) -> Result<mxx_ir_core::IntExpr, RefreshSetupError> {
        self.validate_dimensions()?;
        let ring_dimension = self
            .layout
            .ring_dimension
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_u32())
            .ok_or(RefreshSetupError::InvalidParameters(
                "decoder cutoff requires a concrete ring dimension that fits in u32",
            ))?;
        let gadget_base = self
            .layout
            .gadget_base
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_u32())
            .ok_or(RefreshSetupError::InvalidParameters(
                "decoder cutoff requires a concrete gadget base that fits in u32",
            ))?;
        let decoder_sigma =
            self.decoder_sigma.evaluate_f64(&ParamEnv::default()).map_err(|_| {
                RefreshSetupError::InvalidParameters(
                    "decoder cutoff requires a concrete finite sigma",
                )
            })?;
        let selected = (0..self.switch_moduli.len())
            .map(|slot| {
                self.decoder_preimage_bound
                    .resolve(
                        ring_dimension,
                        self.layout.secret_dimension,
                        self.decoder_digit_count(slot)?,
                        gadget_base,
                        decoder_sigma,
                    )
                    .map_err(|_| {
                        RefreshSetupError::InvalidParameters(
                            "invalid decoder trapdoor preimage cutoff policy",
                        )
                    })
            })
            .collect::<Result<Vec<_>, RefreshSetupError>>()?
            .into_iter()
            .max()
            .ok_or(RefreshSetupError::InvalidParameters("empty decoder moduli"))?;
        Ok(IntExpr::constant(BigInt::from_biguint(num_bigint::Sign::Plus, selected)))
    }

    /// Gadget digits for the exact retained CRT basis of one low-modulus anchor.
    /// Tower order is inherited from the source; arbitrary subsets are supported.
    pub fn decoder_digit_count(&self, slot: usize) -> Result<usize, RefreshSetupError> {
        let selected = self
            .switch_moduli
            .get(slot)
            .filter(|modulus| **modulus > BigUint::one())
            .ok_or(RefreshSetupError::InvalidParameters("missing switching modulus"))?;
        let towers = &self.refresh.crt_plaintext_moduli;
        let base = self
            .layout
            .gadget_base
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_u64())
            .filter(|base| *base >= 2 && base.is_power_of_two())
            .ok_or(RefreshSetupError::InvalidParameters("invalid decoder gadget base"))?;
        let base_bits = base.trailing_zeros() as usize;
        let mut product = BigUint::one();
        let mut count = 0usize;
        let mut maximum_bits = 0usize;
        for tower in towers {
            let modulus = tower
                .evaluate(&ParamEnv::default())
                .ok()
                .and_then(|value| value.to_biguint())
                .filter(|value| *value > BigUint::one())
                .ok_or(RefreshSetupError::InvalidParameters("invalid decoder CRT tower"))?;
            if selected % &modulus == BigUint::zero() {
                maximum_bits = maximum_bits.max(modulus.bits() as usize);
                product *= modulus;
                count = count
                    .checked_add(1)
                    .ok_or(RefreshSetupError::InvalidParameters("decoder digit count overflow"))?;
            }
        }
        if &product != selected {
            return Err(RefreshSetupError::InvalidParameters(
                "switching modulus is not a CRT subset",
            ));
        }
        count
            .checked_mul(maximum_bits.div_ceil(base_bits))
            .ok_or(RefreshSetupError::InvalidParameters("decoder digit count overflow"))
    }

    /// Describe every new marginal and complementary view without borrowing
    /// security bits from the original high-q BGG distribution. The declared
    /// budget counts fresh anchor samples across executions of this setup.
    pub fn anchor_security_inputs(
        &self,
        execution_budget: usize,
    ) -> Result<Vec<BggRefreshAnchorSecurityInputs>, RefreshSetupError> {
        self.validate_dimensions()?;
        if execution_budget == 0 {
            return Err(RefreshSetupError::InvalidParameters("zero security execution budget"));
        }
        let q = self
            .refresh
            .full_modulus
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_biguint())
            .ok_or(RefreshSetupError::InvalidParameters("invalid security modulus"))?;
        group_refresh_levels(self.switch_moduli.iter().cloned()).into_iter().map(|(modulus, slots)| {
            let digits = self.decoder_digit_count(slots[0])?;
            let columns = digits.checked_add(2)
                .and_then(|value| value.checked_mul(self.layout.secret_dimension))
                .ok_or(RefreshSetupError::InvalidParameters("anchor sample count overflow"))?;
            let samples = columns.checked_mul(execution_budget)
                .ok_or(RefreshSetupError::InvalidParameters("anchor sample budget overflow"))?;
            Ok(BggRefreshAnchorSecurityInputs {
                complementary_modulus: &q / &modulus,
                modulus,
                slots,
                ring_dimension: self.coefficient_count,
                random_secret_polynomials: self.layout.secret_dimension - 1,
                secret_distribution: "one coefficientwise uniform ternary polynomial, shared across all moduli".into(),
                fixed_augmented_coordinate: 1,
                error_sigma: self.encoding_error_sigma.clone(),
                error_coefficient_cutoff: self.encoding_error_bound.clone(),
                anchor_columns: columns,
                execution_budget,
                ring_samples: samples,
                estimated_security_bits: None,
                status: "not estimated: low-P marginal with fixed coordinate, complementary J-view with correlated operational errors, and their joint view with high-q encodings/targets/preimages require analysis; noise acceptance is not security acceptance".into(),
            })
        }).collect()
    }

    /// Deterministic public anchor groups, ordered by their exact modulus.
    pub(crate) fn switch_ring(&self, slot: usize) -> Result<mxx_dsl::Ring, RefreshSetupError> {
        let modulus = self.switch_moduli.get(slot).ok_or(RefreshSetupError::InvalidManifest)?;
        Ok(mxx_dsl::Ring::new(
            IntExpr::constant(BigInt::from(modulus.clone())),
            self.layout.ring_dimension.clone(),
        ))
    }

    pub(crate) fn mask_lifts(&self) -> Result<Vec<Mat>, RefreshSetupError> {
        let q = self
            .refresh
            .full_modulus
            .evaluate(&ParamEnv::default())
            .map_err(|_| RefreshSetupError::InvalidManifest)?;
        Ok(self
            .switch_moduli
            .iter()
            .map(|modulus| {
                self.layout
                    .ring()
                    .polynomial([IntExpr::constant(&q / BigInt::from(modulus.clone()))])
            })
            .collect())
    }

    pub(crate) fn validate(&self) -> Result<(), RefreshSetupError> {
        self.validate_dimensions()?;
        self.resolve_decoder_preimage_bound().map(|_| ())
    }
}

/// Computes the canonical public identity for refresh preprocessing setup.
///
/// The digest includes dimensions, CRT data, PRF identity, decoder policy,
/// and public artifact schema; private vectors, support, schedules, and
/// plaintext values are deliberately excluded.
pub(crate) fn identity_digest(
    p: &BggRefreshParameters,
    contract: &RefreshPrfContract,
    pbc_layout_id: crate::noise_refresh::pbc::PbcLayoutId,
    decoder_preimage_bound: &mxx_ir_core::IntExpr,
) -> [u8; 32] {
    #[derive(Serialize)]
    struct Layout {
        modulus: mxx_ir_core::IntExpr,
        ring_dimension: mxx_ir_core::IntExpr,
        secret_dimension: usize,
        gadget_digit_count: usize,
        gadget_base: mxx_ir_core::IntExpr,
    }
    #[derive(Serialize)]
    struct Refresh {
        full_modulus: mxx_ir_core::IntExpr,
        crt_plaintext_moduli: Vec<mxx_ir_core::IntExpr>,
        reconstruction_coefficients: Vec<mxx_ir_core::IntExpr>,
    }
    #[derive(Serialize)]
    struct Payload<'a> {
        schema: &'static str,
        refresh_id: [u8; 32],
        program_id: crate::program::ExponentLutProgramId,
        prf_q: usize,
        prf_p: usize,
        prf_native_width: usize,
        prf_ring_dimension: usize,
        prf_output_wire: crate::program::WireId,
        pbc_layout_id: crate::noise_refresh::pbc::PbcLayoutId,
        name: &'a str,
        base_p: usize,
        component_count: usize,
        prf_component_count: usize,
        coefficient_count: usize,
        mask_base_p_digit_counts: &'a [usize],
        switch_moduli: &'a [BigUint],
        fresh_error_base_p_digit_count: usize,
        mask_statistical_security_bits: usize,
        gadget_digit_count: usize,
        lut_width: usize,
        layout: Layout,
        refresh: Refresh,
        decoder_sigma: &'a mxx_ir_core::RealExpr,
        decoder_preimage_bound: &'a mxx_ir_core::IntExpr,
        encoding_error_sigma: &'a mxx_ir_core::RealExpr,
        encoding_error_bound: &'a mxx_ir_core::IntExpr,
    }
    let payload = Payload {
        schema: "mxx-exponent-lut/refresh-setup/v7",
        refresh_id: p.refresh_id,
        program_id: contract.program_id(),
        prf_q: contract.q(),
        prf_p: contract.p(),
        prf_native_width: contract.native_width(),
        prf_ring_dimension: contract.ring_dimension(),
        prf_output_wire: contract.output_wire(),
        pbc_layout_id,
        name: &p.name,
        base_p: p.base_p,
        component_count: p.component_count,
        prf_component_count: p.prf_component_count(),
        coefficient_count: p.coefficient_count,
        mask_base_p_digit_counts: &p.mask_base_p_digit_counts,
        switch_moduli: &p.switch_moduli,
        fresh_error_base_p_digit_count: p.fresh_error_base_p_digit_count,
        mask_statistical_security_bits: p.mask_statistical_security_bits,
        gadget_digit_count: p.layout.digit_count,
        lut_width: p.lut_width,
        layout: Layout {
            modulus: p.layout.modulus.clone().canonicalize(),
            ring_dimension: p.layout.ring_dimension.clone().canonicalize(),
            secret_dimension: p.layout.secret_dimension,
            gadget_digit_count: p.layout.digit_count,
            gadget_base: p.layout.gadget_base.clone().canonicalize(),
        },
        refresh: Refresh {
            full_modulus: p.refresh.full_modulus.clone().canonicalize(),
            crt_plaintext_moduli: p
                .refresh
                .crt_plaintext_moduli
                .iter()
                .cloned()
                .map(|value| value.canonicalize())
                .collect(),
            reconstruction_coefficients: p
                .refresh
                .reconstruction_coefficients
                .iter()
                .cloned()
                .map(|value| value.canonicalize())
                .collect(),
        },
        decoder_sigma: &p.decoder_sigma,
        decoder_preimage_bound,
        encoding_error_sigma: &p.encoding_error_sigma,
        encoding_error_bound: &p.encoding_error_bound,
    };
    mxx_ir_core::encoding::hash_canonical(&payload).expect("canonical refresh setup identity")
}

/// Extends the refresh identity with the exact value-form readout.  The
/// legacy geometry identity remains available to noise planning, while the
/// artifact declaration must also bind the readout table/trace semantics.
pub(crate) fn setup_identity_digest(
    p: &BggRefreshParameters,
    contract: &RefreshPrfContract,
    pbc_layout_id: crate::noise_refresh::pbc::PbcLayoutId,
    decoder_preimage_bound: &mxx_ir_core::IntExpr,
    readout_identity: crate::readout::ValueReadoutIdentity,
) -> [u8; 32] {
    #[derive(Serialize)]
    struct Payload {
        base: [u8; 32],
        readout_identity: crate::readout::ValueReadoutIdentity,
    }
    mxx_ir_core::encoding::hash_canonical(&Payload {
        base: identity_digest(p, contract, pbc_layout_id, decoder_preimage_bound),
        readout_identity,
    })
    .expect("canonical refresh setup/readout identity")
}
