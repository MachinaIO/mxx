//! Small integer-LWE reference protocol used to test the centered-residual endpoint, integer hash
//! families, and integer matrix-vector products in the shared correctness machinery.

use crate::{DslContext, Family, GraphValue, IdealSpec, Int, Ring, artifact_bindings};
use mxx_ir_core::{
    artifact::{ArtifactAvailability, ProductionId, SpecHash},
    protocol::{
        ClosedProtocolBundle, ComparatorEndpointBinding, ComparatorSpec, EndpointBinding,
        EndpointBindings, EndpointSemanticBinding, EndpointSpecId, InputContract,
        InputContractEntry, InputValueContract, OperationalDecoderKind, OperationalDecoderTarget,
        OutputRef, ProtocolDecl, ProtocolInputBinding, ProtocolInputDestination, ProtocolInputId,
        ProtocolPreconditionSpec, ProtocolStage, StageId, StageInputName, Workflow,
    },
};

const MODULUS: i64 = 64;

/// The placeholder production of the encryption stage's ciphertext artifact.
pub fn lwe_production() -> ProductionId {
    ProductionId { spec_hash: SpecHash([0; 32]), execution_nonce: [0; 32] }
}

/// Encryption transfers the ciphertext `(a, b = <a, s> + 32 m - 16 mod 64)` under a hash-derived
/// mask `a` as a record of an integer family and an integer; decryption decodes the phase `b -
/// <a, s> mod 64` as `phase < 32`.
pub fn lwe_protocol(decoder_uses_phase: bool) -> Result<ProtocolDecl, Box<dyn std::error::Error>> {
    let encryption = DslContext::new("toy-lwe-encrypt");
    let key = Ring::from_crt_moduli(vec![257.into()], 1).bytes_input("hash_key", 32);
    let secret = encryption.int_family_input("secret", 2);
    let message: Int = encryption.input("message", crate::IntType)?;
    let mask = encryption.hash_int_family(key, b"toy-lwe".to_vec(), 2, MODULUS);
    let dot = mask.matrix_vector_product(&secret).at(0);
    let body = dot.add(message.mul(32)).add(MODULUS - 16).rem(MODULUS);
    let ciphertext = (mask, body);
    let schema = ciphertext.schema();
    let encryption = encryption.transferred_output("ciphertext", ciphertext)?.build()?;

    let decryption = DslContext::new("toy-lwe-decrypt");
    let secret = decryption.int_family_input("secret", 2);
    let (mask, body): (Family<Int>, Int) = decryption.artifact_input(
        lwe_production(),
        "ciphertext",
        schema.clone(),
        ArtifactAvailability::Transferred,
    )?;
    let phase = body.sub(mask.matrix_vector_product(&secret).at(0)).rem(MODULUS);
    let decoded = if decoder_uses_phase { phase.clone() } else { secret.at(0) }
        .less(Int::constant(MODULUS / 2))
        .to_int();
    let decryption = decryption.output("decoded", decoded)?.output("phase", phase)?.build()?;

    let ideal = DslContext::new("toy-lwe-ideal");
    let ideal_message: Int = ideal.input("message", crate::IntType)?;
    let ideal = IdealSpec::new(ideal.output("result", ideal_message)?.build()?.graph)?;

    let encrypt = StageId("encrypt".to_owned());
    let decrypt = StageId("decrypt".to_owned());
    let stage_id = decrypt.clone();
    let endpoint = EndpointSpecId::CenteredResidual;
    let bit = || InputValueContract::IntegerRange { lower: 0.into(), upper: 1.into() };
    let inputs = [
        ("hash_key", InputValueContract::Bytes { length: 32.into() }),
        ("secret", InputValueContract::Family { count: 2.into(), element: Box::new(bit()) }),
        ("message", bit()),
    ];
    let bindings = inputs
        .iter()
        .map(|(name, _)| {
            let stage = |stage: &StageId| ProtocolInputDestination::WorkflowStage {
                stage: stage.clone(),
                input: StageInputName((*name).to_owned()),
            };
            let destinations = match *name {
                "hash_key" => vec![stage(&encrypt)],
                "secret" => vec![stage(&encrypt), stage(&decrypt)],
                _ => vec![
                    stage(&encrypt),
                    ProtocolInputDestination::Ideal { input: "message".to_owned() },
                ],
            };
            ProtocolInputBinding { input: ProtocolInputId::from(*name), destinations }
        })
        .collect();
    Ok(ProtocolDecl::new(ProtocolDecl {
        params: Vec::new(),
        bundle: ClosedProtocolBundle {
            workflow: Workflow {
                stages: vec![
                    ProtocolStage {
                        id: encrypt.clone(),
                        graph: encryption.graph,
                        bindings: Vec::new(),
                    },
                    ProtocolStage {
                        id: decrypt.clone(),
                        graph: decryption.graph,
                        bindings: artifact_bindings(&schema, "ciphertext", &encrypt, "ciphertext")?,
                    },
                ],
                entrypoint: decrypt,
            },
            ideal,
            requirements: Vec::new(),
            comparator: ComparatorSpec::Equality {
                endpoints: vec![ComparatorEndpointBinding {
                    endpoint,
                    actual_input: "decoded".to_owned(),
                    ideal_input: "result".to_owned(),
                    result_output: "failure".to_owned(),
                    failure_value: true,
                }],
            },
            endpoints: EndpointBindings {
                entries: vec![EndpointBinding {
                    spec: endpoint,
                    semantics: EndpointSemanticBinding::Exact,
                    workflow_output: OutputRef {
                        stage: stage_id.clone(),
                        output: "decoded".to_owned(),
                    },
                    ideal_output: "result".to_owned(),
                }],
            },
            operational_decoder_targets: vec![OperationalDecoderTarget {
                target_id: "toy-lwe-phase".to_owned(),
                residual: OutputRef { stage: stage_id, output: "phase".to_owned() },
                endpoint,
                kind: OperationalDecoderKind::CenteredResidual,
            }],
            endpoint_specs: vec![endpoint],
            input_contract: InputContract {
                inputs: inputs
                    .into_iter()
                    .map(|(name, value)| InputContractEntry {
                        id: ProtocolInputId::from(name),
                        name: name.to_owned(),
                        value,
                    })
                    .collect(),
            },
            input_bindings: bindings,
            precondition_spec: ProtocolPreconditionSpec::default(),
        },
    })?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_ir_core::{
        ParamEnv,
        artifact::export_validated_manifest,
        lean::{
            claim::{ClaimBackend, ClaimSemantics},
            protocol::export_claim,
        },
        protocol::BundleValidationError,
        validate,
    };
    use std::{collections::BTreeMap, fs, path::Path};

    #[test]
    fn test_centered_residual_lwe_protocol_export() {
        let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../test_data/lean_ir_fixtures/lwe_protocol");
        // The stage layout is part of the fixture; drop modules of an earlier layout.
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir_all(&directory).unwrap();
        let declaration = lwe_protocol(true).unwrap();
        let producer = validate(&declaration.stages()[0].graph, &ParamEnv::default()).unwrap();
        let manifests = BTreeMap::from([(
            lwe_production(),
            export_validated_manifest(lwe_production(), &producer).unwrap(),
        )]);
        fs::write(directory.join("LweFixture.lean"), LWE_SEMANTICS).unwrap();
        fs::write(directory.join("LweProof.lean"), LWE_PROOF).unwrap();
        export_claim(
            &declaration,
            &ParamEnv::default(),
            &ClaimBackend {
                module_name: "LweFixture",
                context_name: "LweFixture.backend",
                layouts: &[],
            },
            &ClaimSemantics {
                imports: &["LweFixture"],
                hash_model_type: "MxxRuntime.HashModel",
                centered_lift: "Mxx.Primitives.centeredLift",
                message_center: "LweFixture.messageCenter",
                decoder_radius: "LweFixture.decoderRadius",
            },
            &manifests,
            &directory,
        )
        .expect("generic export accepts the centered-residual decoder");
        fs::write(
            directory.join("Certificate.lean"),
            mxx_ir_core::lean::claim::assemble_certificate("LweProof", "LweProof.correctness")
                .unwrap(),
        )
        .unwrap();

        let source = fs::read_to_string(directory.join("Claim.lean")).unwrap();
        let (premises, conclusion) = source.split_once("def CorrectnessClaim").unwrap();
        assert!(premises.contains("(index : Fin 1)"));
        assert!(premises.contains(": Int) : ZMod 64)"));
        assert!(premises.contains("LweFixture.messageCenter 64"));
        assert!(!premises.contains(".natAbs <"));
        assert!(conclusion.contains(
            "(∀ index, (observedResidual execution index).natAbs < LweFixture.decoderRadius 64)"
        ));
        assert!(conclusion.contains(" = execution.«ideal»"));
        assert!(
            premises.contains("(execution.«stage_0».2.1, execution.«stage_0».1, external.input_1")
        );
        let encryption = fs::read_to_string(directory.join("Stage_encrypt.lean")).unwrap();
        assert!(encryption.contains("MxxRuntime.hashIntFamily (hashModel) (64)"));
        let decryption = fs::read_to_string(directory.join("Stage_decrypt.lean")).unwrap();
        assert!(decryption.contains("MxxRuntime.intMatrixVectorProduct false"));
    }

    #[test]
    fn centered_residual_decoder_must_depend_on_the_residual() {
        let Err(error) = lwe_protocol(false) else { panic!("an unrelated decoder was accepted") };
        assert!(matches!(
            error.downcast_ref::<mxx_ir_core::protocol::ProtocolError>(),
            Some(mxx_ir_core::protocol::ProtocolError::InvalidBundle(
                BundleValidationError::InvalidOperationalDecoderTarget
            ))
        ));
    }

    #[test]
    fn centered_residual_target_requires_a_reduced_residual_and_matching_kind() {
        let protocol = lwe_protocol(true).unwrap();
        let mut unreduced = protocol.bundle.clone();
        unreduced.operational_decoder_targets[0].residual.output = "decoded".to_owned();
        assert_eq!(
            unreduced.validate(),
            Err(BundleValidationError::InvalidOperationalDecoderTarget)
        );
        let mut mismatched = protocol.bundle.clone();
        mismatched.operational_decoder_targets[0].kind = OperationalDecoderKind::BooleanInterval;
        assert_eq!(
            mismatched.validate(),
            Err(BundleValidationError::OperationalDecoderTargetKindMismatch)
        );
        let mut wrong_semantics = protocol.bundle;
        wrong_semantics.endpoints.entries[0].semantics = EndpointSemanticBinding::ThresholdDecode;
        assert_eq!(
            wrong_semantics.validate(),
            Err(BundleValidationError::InvalidEndpointSemantics)
        );
    }

    const LWE_SEMANTICS: &str = "import MxxRuntime
namespace LweFixture
/-- The encoded bit `32 m - 16 mod 64`, subtracted from the phase. -/
def messageCenter (_ : Nat) (message : Int) (_ : Fin 1) : Int := if message = 1 then 16 else 48
/-- Phases strictly within 16 of their center decode correctly and keep the gate margin. -/
def decoderRadius (_ : Nat) : Nat := 16
end LweFixture
";

    const LWE_PROOF: &str = r#"import Claim

namespace LweProof

open GeneratedClaim

/-- The mask is an arbitrary hash output, transferred to decryption as an integer-family
artifact, so the phase cancels it exactly for every hash model. -/
theorem correctness : CorrectnessClaim := by
  intro hashModel external execution hruns
  obtain ⟨⟨_, _, hlow, hhigh⟩, ⟨encrypt, _, ⟨encryptPosition, _, hencryptDot⟩, _, hciphertext⟩,
    ⟨decrypt, ⟨decryptPosition, _, hdecryptDot⟩, _, hdecoded⟩, ⟨_, hideal⟩⟩ := hruns
  have hposition : decryptPosition = encryptPosition := Subsingleton.elim _ _
  simp only at hciphertext hdecoded hencryptDot hdecryptDot hideal
  rw [hciphertext] at hdecryptDot hdecoded
  simp only at hdecryptDot hdecoded
  rw [hposition, ← hencryptDot] at hdecryptDot
  rw [hdecryptDot] at hdecoded
  have hm : external.input_2 = 0 ∨ external.input_2 = 1 := by omega
  have hphase : ((encrypt.w_4_0 + external.input_2 * 32 + 48) % 64 - encrypt.w_4_0) % 64 =
      if external.input_2 = 1 then 16 else 48 := by
    rcases hm with hm | hm
    · rw [hm, if_neg (by decide)]
      omega
    · rw [hm, if_pos rfl]
      omega
  refine ⟨fun index ↦ ?_, ?_⟩
  · unfold observedResidual
    rw [hdecoded, hideal]
    simp only [hphase, LweFixture.messageCenter, LweFixture.decoderRadius, sub_self]
    simp [Mxx.Primitives.centeredLift]
  · rw [hdecoded, hideal]
    simp only [hphase]
    rcases hm with hm | hm <;> simp [hm]

end LweProof
"#;
}
