//! Perfect-correctness declaration for the shared Diamond input injector.

use super::{DiamondInputConfig, DiamondInputInjector};
use mxx_correctness::{
    ArtifactBinding, ArtifactName, Comparator, CorrectnessDecl, OutputRef, ProtoInputName,
    ProtocolDecl, ProtocolStage, StageId, StageInputName,
};
use mxx_dsl::{DslContext, IdealSpec, Int, PurePredicateSpec};
use mxx_ir_core::{
    IntExpr, RealExpr,
    artifact::{ArtifactConfidentiality, ProductionId, SpecHash},
    node::ConcatAxis,
};
use num_bigint::BigInt;

pub const PROTOCOL_NAME: &str = "diamond-input-injector";

pub fn protocol() -> ProtocolDecl {
    const INITIAL: &str = "initial";
    const PROJECTION: &str = "projection";
    let config = config();
    let injector = DiamondInputInjector::new(config.clone()).expect("fixed Diamond configuration");
    let ring = config.ring();
    let message = ring.bool_input("message");
    let message_matrix = message
        .clone()
        .to_int()
        .select(vec![ring.zero((1, 1)), ring.identity(1)])
        .expect("two message branches");
    let mut preprocessing = injector.preprocess(message_matrix).expect("Diamond preprocessing");
    let half_modulus = &config.modulus / BigInt::from(2);
    let projection_target = mxx_dsl::Mat::concat(
        ConcatAxis::Rows,
        vec![ring.zero((1, 1)), ring.polynomial([half_modulus.into()])],
    );
    let state_columns = config.state_columns().expect("fixed layout");
    let projection = preprocessing.final_trapdoors[0]
        .sample_preimage(projection_target, (state_columns, 1))
        .as_mat();
    let mut preprocess_context = DslContext::new("diamond-input-injector-preprocess")
        .public_output(INITIAL, preprocessing.p)
        .expect("initial output")
        .public_output(PROJECTION, projection)
        .expect("projection output");
    for (digit, states) in preprocessing.transitions.remove(0).into_iter().enumerate() {
        for (state, transition) in states.into_iter().enumerate() {
            preprocess_context = preprocess_context
                .public_output(format!("transition-{digit}-{state}"), transition)
                .expect("transition output");
        }
    }
    let preprocess = preprocess_context.build().expect("preprocessing graph");

    let placeholder = ProductionId { spec_hash: SpecHash([0; 32]), execution_nonce: [0; 32] };
    let initial = ring.artifact_input(
        placeholder.clone(),
        INITIAL,
        (1, state_columns),
        ArtifactConfidentiality::Public,
    );
    let projection = ring.artifact_input(
        placeholder.clone(),
        PROJECTION,
        (state_columns, 1),
        ArtifactConfidentiality::Public,
    );
    let state_count = config.state_count_at_level(1).expect("fixed layout");
    let transitions = vec![
        (0..config.digit_base)
            .map(|digit| {
                (0..state_count)
                    .map(|state| {
                        ring.artifact_input(
                            placeholder.clone(),
                            format!("transition-{digit}-{state}"),
                            (state_columns, state_columns),
                            ArtifactConfidentiality::Public,
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>(),
    ];
    let digit = ring.input("digit", (1, 1)).extract_coefficient(0);
    let states =
        injector.evaluate(initial, &[digit], &transitions).expect("online injection").states;
    let projected = states[0].clone() * projection;
    let decoded = projected
        .threshold_decode_bools(IntExpr::constant(2), 1)
        .into_iter()
        .next()
        .expect("one decoded bit");
    let evaluate = DslContext::new("diamond-input-injector-evaluate")
        .bool_output("result", decoded)
        .expect("result output")
        .build()
        .expect("evaluation graph");

    let ideal = IdealSpec::new(
        DslContext::new("diamond-input-injector-ideal")
            .bool_output("result", ring.bool_input("message"))
            .expect("ideal output")
            .build()
            .expect("ideal graph"),
    )
    .expect("sampler-free ideal");
    let digit_precondition = PurePredicateSpec::new(
        DslContext::new("diamond-input-injector-digit-precondition")
            .bool_output(
                "valid",
                ring.input("digit", (1, 1)).extract_coefficient(0).less_equal(Int::constant(1)),
            )
            .expect("precondition output")
            .build()
            .expect("precondition graph"),
    )
    .expect("sampler-free precondition");
    let preprocess_id = StageId("preprocess".to_owned());
    let evaluate_id = StageId("evaluate".to_owned());
    let mut bindings = vec![
        ArtifactBinding {
            consumer_input: StageInputName(INITIAL.to_owned()),
            producer_stage: preprocess_id.clone(),
            producer_output: ArtifactName(INITIAL.to_owned()),
        },
        ArtifactBinding {
            consumer_input: StageInputName(PROJECTION.to_owned()),
            producer_stage: preprocess_id.clone(),
            producer_output: ArtifactName(PROJECTION.to_owned()),
        },
    ];
    for digit in 0..config.digit_base {
        for state in 0..state_count {
            let name = format!("transition-{digit}-{state}");
            bindings.push(ArtifactBinding {
                consumer_input: StageInputName(name.clone()),
                producer_stage: preprocess_id.clone(),
                producer_output: ArtifactName(name),
            });
        }
    }
    ProtocolDecl::new(ProtocolDecl {
        params: Vec::new(),
        stages: vec![
            ProtocolStage {
                id: preprocess_id.clone(),
                graph: preprocess.graph,
                bindings: Vec::new(),
            },
            ProtocolStage { id: evaluate_id.clone(), graph: evaluate.graph, bindings },
        ],
        entrypoint: evaluate_id.clone(),
        correctness: CorrectnessDecl {
            protocol_inputs: vec![
                (
                    ProtoInputName("message".to_owned()),
                    vec![(preprocess_id, StageInputName("message".to_owned()))],
                ),
                (
                    ProtoInputName("digit".to_owned()),
                    vec![(evaluate_id.clone(), StageInputName("digit".to_owned()))],
                ),
            ],
            requires: vec![digit_precondition],
            ideal,
            compared_outputs: vec![OutputRef { stage: evaluate_id, output: "result".to_owned() }],
            comparator: Comparator::Equal,
        },
    })
    .expect("Diamond input injector protocol is valid")
}

fn config() -> DiamondInputConfig {
    DiamondInputConfig {
        modulus: 65_537.into(),
        ring_dimension: 1,
        input_count: 1,
        digit_base: 2,
        batch_bits: 1,
        gadget_base: 4.into(),
        digit_count: 1,
        trapdoor_sigma: RealExpr::from_integer(4),
        error_sigma: RealExpr::from_integer(1),
        error_max_coefficient_bound: 1.into(),
        preimage_max_coefficient_bound: 2.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn declaration_is_built_from_the_shared_gadget_graph() {
        let emitted =
            mxx_correctness::emit_protocol_for(PROTOCOL_NAME, &protocol(), "MxxGadgets").unwrap();
        assert!(emitted.ir.contains(".parallelLoop"));
        assert!(emitted.ir.contains(".preimageSample"));
        assert!(emitted.ir.contains(".matrixMultiply"));
    }
}
