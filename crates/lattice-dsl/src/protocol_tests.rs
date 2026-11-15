#[cfg(test)]
mod tests {
    use mxx_ir_core::protocol::*;

    fn valid_protocol() -> ProtocolDecl {
        crate::test_protocol::protocol()
    }

    #[test]
    fn artifact_binding_totality_errors_are_distinct() {
        let mut missing = valid_protocol();
        missing.bundle.workflow.stages[1].bindings.clear();
        assert_eq!(missing.validate(), Err(ProtocolError::MissingArtifactBinding));

        let mut duplicate = valid_protocol();
        let repeated = duplicate.bundle.workflow.stages[1].bindings[0].clone();
        duplicate.bundle.workflow.stages[1].bindings.push(repeated);
        assert_eq!(duplicate.validate(), Err(ProtocolError::DuplicateArtifactBinding));

        let mut producer = valid_protocol();
        producer.bundle.workflow.stages[1].bindings[0].producer_stage =
            StageId("absent".to_owned());
        assert_eq!(producer.validate(), Err(ProtocolError::MissingProducerStage));
    }
}
