//! Small two-stage reference protocol used to test the shared correctness machinery.

use crate::{
    ArtifactBinding, ArtifactName, Comparator, CorrectnessDecl, OutputRef, ParameterDecl,
    ParameterKind, ProtoInputName, ProtocolDecl, ProtocolStage, StageId, StageInputName,
};
use mxx_dsl::{DslContext, IdealSpec, Ring};
use mxx_ir_core::{
    IntExpr, RealExpr,
    artifact::{ArtifactConfidentiality, ProductionId, SpecHash},
};

pub const PROTOCOL_NAME: &str = "toy-example";

pub fn protocol() -> ProtocolDecl {
    let ring = Ring::new(256, 1);
    let message = ring.bool_input("message");
    let encoded = message
        .clone()
        .to_int()
        .select(vec![ring.zero((1, 1)), ring.polynomial([IntExpr::constant(128)])])
        .expect("two equally typed encoding branches");
    let ciphertext = encoded +
        ring.gaussian((1, 1), RealExpr::from_integer(1), IntExpr::Var("cutoff".to_owned()));
    let encrypt = DslContext::new("toy-example-encrypt")
        .int_parameter("cutoff")
        .public_output("ciphertext", ciphertext)
        .expect("unique output")
        .build()
        .expect("toy encryption graph");
    let placeholder = ProductionId { spec_hash: SpecHash([0; 32]), execution_nonce: [0; 32] };
    let ciphertext =
        ring.artifact_input(placeholder, "ciphertext", (1, 1), ArtifactConfidentiality::Public);
    let decrypt = DslContext::new("toy-example-decrypt")
        .int_parameter("cutoff")
        .output("ciphertext", ciphertext)
        .expect("unique output")
        .build()
        .expect("toy decryption graph");
    let decode_input = ring.input("ciphertext", (1, 1));
    let decoded = decode_input
        .threshold_decode_bools(IntExpr::constant(2), 1)
        .into_iter()
        .next()
        .expect("one decoded bit");
    let decode = IdealSpec::new(
        DslContext::new("toy-example-decode")
            .int_parameter("cutoff")
            .bool_output("result", decoded)
            .expect("unique output")
            .build()
            .expect("toy decode graph"),
    )
    .expect("sampler-free decode");
    let ideal = IdealSpec::new(
        DslContext::new("toy-example-ideal")
            .int_parameter("cutoff")
            .bool_output("result", ring.bool_input("message"))
            .expect("unique output")
            .build()
            .expect("toy ideal graph"),
    )
    .expect("sampler-free ideal");
    ProtocolDecl::new(ProtocolDecl {
        params: vec![ParameterDecl { name: "cutoff".to_owned(), kind: ParameterKind::Dimension }],
        stages: vec![
            ProtocolStage {
                id: StageId("encrypt".to_owned()),
                graph: encrypt.graph,
                bindings: Vec::new(),
            },
            ProtocolStage {
                id: StageId("decrypt".to_owned()),
                graph: decrypt.graph,
                bindings: vec![ArtifactBinding {
                    consumer_input: StageInputName("ciphertext".to_owned()),
                    producer_stage: StageId("encrypt".to_owned()),
                    producer_output: ArtifactName("ciphertext".to_owned()),
                }],
            },
        ],
        entrypoint: StageId("decrypt".to_owned()),
        semantic_certificate: Default::default(),
        correctness: CorrectnessDecl {
            protocol_inputs: vec![(
                ProtoInputName("message".to_owned()),
                vec![(StageId("encrypt".to_owned()), StageInputName("message".to_owned()))],
            )],
            requires: Vec::new(),
            ideal,
            compared_outputs: vec![OutputRef {
                stage: StageId("decrypt".to_owned()),
                output: "ciphertext".to_owned(),
            }],
            comparator: Comparator::EqualAfterMap { map: decode },
        },
    })
    .expect("toy example protocol is valid")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cutoff_is_the_same_natural_parameter_in_ir_and_lean() {
        let emitted =
            crate::emit_protocol_for(PROTOCOL_NAME, &protocol(), "MxxCorrectness").unwrap();
        assert!(emitted.ir.contains(".gaussianSample { modulus :="));
        assert!(emitted.ir.contains("(.parameter \"cutoff\")"));
        assert!(emitted.statement.contains("cutoff : Nat"));
        assert!(emitted.statement.contains("(\"cutoff\", .integer (Int.ofNat p.cutoff))"));
    }
}
