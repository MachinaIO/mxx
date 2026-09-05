//! Witness-encryption protocols expressed as validated declarative graphs.

pub mod diamond;
pub mod lean;

use mxx_correctness::{
    ComparatorSpec, ProtocolDecl, ProtocolInputDestination, ProtocolInputId, StageId,
    StageInputName,
};
use mxx_gadgets::circuit::{BooleanCircuitData, BooleanCircuitShape};
use mxx_ir_core::node::NodeKind;
use std::collections::BTreeSet;
use thiserror::Error;

#[derive(Clone)]
pub struct WitnessEncryptionProtocolDecl {
    protocol: ProtocolDecl,
    interface: WitnessEncryptionInterface,
}

#[derive(Clone, Debug)]
pub struct WitnessEncryptionInterface {
    pub encryption_stage: StageId,
    pub decryption_stage: StageId,
    pub message: ProtocolInputId,
    pub instance: ProtocolInputId,
    pub witness: ProtocolInputId,
}

#[derive(Debug, Error)]
pub enum WitnessEncryptionDeclError {
    #[error("a declared witness-encryption stage does not exist in the protocol")]
    MissingStage,
    #[error("a circuit-data family is not mapped to every consuming stage")]
    MissingCircuitInputMapping,
    #[error("an instance, witness, or message input has the wrong stage mapping")]
    InvalidInterfaceInputMapping,
    #[error("the declaration does not contain the canonical validity and satisfaction predicates")]
    InvalidCorrectnessPredicates,
    #[error("the ideal graph is not the Boolean message identity")]
    InvalidIdeal,
    #[error("witness-encryption correctness must compare the decrypted message by equality")]
    InvalidComparator,
}

impl WitnessEncryptionProtocolDecl {
    pub fn new(
        protocol: ProtocolDecl,
        interface: WitnessEncryptionInterface,
    ) -> Result<Self, WitnessEncryptionDeclError> {
        if !protocol.stages().iter().any(|stage| stage.id == interface.encryption_stage) ||
            !protocol.stages().iter().any(|stage| stage.id == interface.decryption_stage)
        {
            return Err(WitnessEncryptionDeclError::MissingStage);
        }
        let circuit_consumers =
            [interface.encryption_stage.clone(), interface.decryption_stage.clone()];
        let circuit_names = [
            "circuit-active-gate-count",
            "circuit-gate-kind",
            "circuit-left-source",
            "circuit-right-source",
            "circuit-output-source",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
        for name in &circuit_names {
            let Some(binding) =
                protocol.bundle.input_bindings.iter().find(|binding| binding.input.0 == *name)
            else {
                return Err(WitnessEncryptionDeclError::MissingCircuitInputMapping);
            };
            if circuit_consumers.iter().any(|consumer| {
                !binding.destinations.iter().any(|destination| {
                    matches!(destination,
                        ProtocolInputDestination::WorkflowStage { stage, input }
                            if stage == consumer && input.0 == *name)
                })
            }) {
                return Err(WitnessEncryptionDeclError::MissingCircuitInputMapping);
            }
        }
        let destinations = |name: &ProtocolInputId| {
            protocol.bundle.input_bindings.iter().find(|binding| &binding.input == name).map(
                |binding| {
                    binding
                        .destinations
                        .iter()
                        .filter_map(|destination| match destination {
                            ProtocolInputDestination::WorkflowStage { stage, input } => {
                                Some((stage.clone(), input.clone()))
                            }
                            ProtocolInputDestination::Requirement { .. } |
                            ProtocolInputDestination::Ideal { .. } => None,
                        })
                        .collect::<BTreeSet<_>>()
                },
            )
        };
        let both = |name: &ProtocolInputId| {
            BTreeSet::from([
                (interface.encryption_stage.clone(), StageInputName(name.0.clone())),
                (interface.decryption_stage.clone(), StageInputName(name.0.clone())),
            ])
        };
        if destinations(&interface.instance) != Some(both(&interface.instance)) ||
            destinations(&interface.witness) !=
                Some(BTreeSet::from([(
                    interface.decryption_stage.clone(),
                    StageInputName(interface.witness.0.clone()),
                )])) ||
            destinations(&interface.message) !=
                Some(BTreeSet::from([(
                    interface.encryption_stage.clone(),
                    StageInputName(interface.message.0.clone()),
                )]))
        {
            return Err(WitnessEncryptionDeclError::InvalidInterfaceInputMapping);
        }
        let graph_inputs = |graph: &mxx_ir_core::Graph| {
            graph
                .root_scope()
                .nodes()
                .iter()
                .filter_map(|node| match node.kind() {
                    NodeKind::Input { name, artifact: None, .. } => Some(name.clone()),
                    _ => None,
                })
                .collect::<BTreeSet<_>>()
        };
        let mut satisfaction_names = circuit_names.clone();
        satisfaction_names.insert(interface.instance.0.clone());
        satisfaction_names.insert(interface.witness.0.clone());
        let requirement_inputs = protocol
            .bundle
            .requirements
            .iter()
            .map(|requirement| graph_inputs(&requirement.graph))
            .collect::<Vec<_>>();
        if requirement_inputs.len() != 3 ||
            !requirement_inputs.contains(&BTreeSet::new()) ||
            !requirement_inputs.contains(&circuit_names) ||
            !requirement_inputs.contains(&satisfaction_names)
        {
            return Err(WitnessEncryptionDeclError::InvalidCorrectnessPredicates);
        }
        if graph_inputs(&protocol.bundle.ideal.graph) !=
            BTreeSet::from([interface.message.0.clone()]) ||
            protocol.bundle.ideal.graph.outputs().len() != 1
        {
            return Err(WitnessEncryptionDeclError::InvalidIdeal);
        }
        if !matches!(protocol.bundle.comparator, ComparatorSpec::Equality { .. }) {
            return Err(WitnessEncryptionDeclError::InvalidComparator);
        }
        Ok(Self { protocol, interface })
    }

    pub fn protocol(&self) -> &ProtocolDecl {
        &self.protocol
    }

    pub fn interface(&self) -> &WitnessEncryptionInterface {
        &self.interface
    }
}

pub trait WitnessEncryptionProtocol {
    type Error;

    fn protocol_decl(&self) -> Result<WitnessEncryptionProtocolDecl, Self::Error>;
}

pub trait WitnessEncryptionRuntime {
    type Ciphertext;
    type Message;
    type Error;

    fn shape(&self) -> &BooleanCircuitShape;

    fn encrypt(
        &mut self,
        circuit: &BooleanCircuitData,
        instance: &[bool],
        message: &Self::Message,
    ) -> Result<Self::Ciphertext, Self::Error>;

    fn decrypt(
        &mut self,
        circuit: &BooleanCircuitData,
        instance: &[bool],
        witness: &[bool],
        ciphertext: &Self::Ciphertext,
    ) -> Result<Self::Message, Self::Error>;
}
