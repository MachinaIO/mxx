//! Small integer-LWE reference protocol used to test exact endpoints, integer hash families, and
//! integer matrix-vector products in the shared correctness machinery.

use crate::{DslContext, Family, GraphValue, IdealSpec, Int, Ring, artifact_bindings};
use mxx_ir_core::{
    artifact::{ArtifactAvailability, ProductionId, SpecHash},
    protocol::{
        ClosedProtocolBundle, ComparatorEndpointBinding, ComparatorSpec, EndpointBinding,
        EndpointBindings, EndpointSemanticBinding, EndpointSpecId, InputContract,
        InputContractEntry, InputValueContract, OutputRef, ProtocolDecl, ProtocolInputBinding,
        ProtocolInputDestination, ProtocolInputId, ProtocolPreconditionSpec, ProtocolStage,
        StageId, StageInputName, Workflow,
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
pub fn lwe_protocol() -> Result<ProtocolDecl, Box<dyn std::error::Error>> {
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
    let decoded = phase.less(Int::constant(MODULUS / 2)).to_int();
    let decryption = decryption.output("decoded", decoded)?.build()?;

    let ideal = DslContext::new("toy-lwe-ideal");
    let ideal_message: Int = ideal.input("message", crate::IntType)?;
    let ideal = IdealSpec::new(ideal.output("result", ideal_message)?.build()?.graph)?;

    let encrypt = StageId("encrypt".to_owned());
    let decrypt = StageId("decrypt".to_owned());
    let stage_id = decrypt.clone();
    let endpoint = EndpointSpecId::Exact;
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
        bindings: Default::default(),
        failure_probability_log2: None,
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
                    workflow_output: OutputRef { stage: stage_id, output: "decoded".to_owned() },
                    ideal_output: "result".to_owned(),
                }],
            },
            operational_decoder_targets: Vec::new(),
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
    use mxx_ir_core::protocol::BundleValidationError;
    use std::{fs, path::Path};

    #[test]
    fn test_exact_lwe_protocol_export() {
        let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../test_data/lean_ir_fixtures/lwe_protocol");
        // The stage layout is part of the fixture; drop modules of an earlier layout.
        let _ = fs::remove_dir_all(&directory);
        mxx_ir_core::lean::protocol::export(&lwe_protocol().unwrap(), &directory)
            .expect("generic export accepts the exact endpoint");
        fs::write(directory.join("LweProof.lean"), LWE_PROOF).unwrap();
        fs::write(directory.join("Certificate.lean"), LWE_CERTIFICATE).unwrap();

        let source = fs::read_to_string(directory.join("Claim.lean")).unwrap();
        let (premises, conclusion) = source.split_once("def CorrectnessClaim").unwrap();
        assert!(conclusion.contains("execution.«stage_1» = execution.«ideal»"));
        assert!(
            premises.contains("(execution.«stage_0».2.1, execution.«stage_0».1, external.input_1")
        );
        let encryption = fs::read_to_string(directory.join("Stage_encrypt.lean")).unwrap();
        assert!(encryption.contains("MxxRuntime.hashIntFamily (hashModel) (64)"));
        let decryption = fs::read_to_string(directory.join("Stage_decrypt.lean")).unwrap();
        assert!(decryption.contains("MxxRuntime.intMatrixVectorProduct false"));
    }

    #[test]
    fn exact_endpoint_requires_exact_semantics() {
        let mut bundle = lwe_protocol().unwrap().bundle;
        bundle.endpoints.entries[0].semantics = EndpointSemanticBinding::ThresholdDecode;
        assert_eq!(bundle.validate(), Err(BundleValidationError::InvalidEndpointSemantics));
    }

    /// Checks that the proof's theorem has exactly the generated statement.
    const LWE_CERTIFICATE: &str = "import Claim
import LweProof

theorem certificate : GeneratedClaim.CorrectnessClaim := LweProof.correctness

#print axioms certificate
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
  rw [hdecoded, hideal]
  simp only [hphase]
  rcases hm with hm | hm <;> simp [hm]

end LweProof
"#;
}
