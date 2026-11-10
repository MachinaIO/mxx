//! The root boundary of an exported graph, read by role.

use super::BindingError;
use mxx_ir_core::lean::{BoundaryValue, RootBoundary, claim::project};
use std::collections::BTreeMap;

/// One exported graph: its module name and root boundary.
pub(super) struct Root<'a> {
    pub(super) module: &'a str,
    pub(super) root: &'a RootBoundary,
}

impl Root<'_> {
    /// `M.generatedRoot.comp` applied to its context and parameters, without the inputs.
    pub(super) fn comp(&self) -> String {
        let backend = if self.root.requires_backend { " Backend.backend" } else { "" };
        format!("{}.generatedRoot.comp{backend} {}", self.module, self.params_name())
    }

    pub(super) fn params_name(&self) -> String {
        format!("{}_params", self.module)
    }

    pub(super) fn params_definition(&self) -> String {
        let fields = self
            .root
            .parameters
            .iter()
            .map(|(name, field)| {
                let value = field.root_value.clone().unwrap_or_else(|| "(0 : Int)".into());
                format!("«{name}» := {value}")
            })
            .collect::<Vec<_>>()
            .join(", ");
        format!("def {} : {} := {{ {fields} }}\n", self.params_name(), self.root.parameter_type)
    }

    pub(super) fn expect_inputs(&self, expected: &[String]) -> Result<(), BindingError> {
        let found = roles_of(&self.root.inputs);
        let mut sorted = expected.to_vec();
        sorted.sort();
        if found != sorted {
            return Err(BindingError::Inputs { graph: self.module.into(), expected: sorted, found });
        }
        Ok(())
    }

    pub(super) fn expect_outputs(&self, expected: &[String]) -> Result<(), BindingError> {
        let found = roles_of(&self.root.outputs);
        let mut sorted = expected.to_vec();
        sorted.sort();
        if found != sorted {
            return Err(BindingError::Outputs {
                graph: self.module.into(),
                expected: sorted,
                found,
            });
        }
        Ok(())
    }

    /// Whether the graph reads the value of `role`.
    pub(super) fn reads(&self, role: &str) -> bool {
        !leaves(&self.root.inputs, role).is_empty()
    }

    /// The input tuple, with the value of every role named in `values`. A composite value
    /// supplies each of its leaves by projection.
    pub(super) fn inputs(&self, values: &BTreeMap<String, String>) -> Result<String, String> {
        let mut ordered = vec![String::new(); self.root.input_count];
        for (name, value) in &self.root.inputs {
            let role = role_of(name);
            let supplied =
                values.get(role).ok_or(format!("{} input {role} unbound", self.module))?;
            let siblings = leaves(&self.root.inputs, role);
            let index = siblings.iter().position(|(leaf, _)| *leaf == name).unwrap_or(0);
            ordered[value.tuple_index] = component(supplied, index, siblings.len());
        }
        // A single input is passed bare, like the root's input type.
        if let [value] = ordered.as_slice() {
            return Ok(value.clone());
        }
        Ok(ordered.iter().rev().fold("()".into(), |tail, value| format!("({value}, {tail})")))
    }

    pub(super) fn input_type(&self, role: &str) -> Result<String, String> {
        product_type(&leaves(&self.root.inputs, role))
            .ok_or(format!("{} has no input {role}", self.module))
    }

    pub(super) fn output_type(&self, role: &str) -> Result<String, String> {
        product_type(&leaves(&self.root.outputs, role))
            .ok_or(format!("{} has no output {role}", self.module))
    }

    /// The value of output `role` from the binder `binder`: the tuple of its leaves for a
    /// composite value.
    pub(super) fn output(&self, role: &str, binder: &str) -> Result<String, String> {
        let parts = leaves(&self.root.outputs, role)
            .into_iter()
            .map(|(_, value)| project(value, binder))
            .collect::<Vec<_>>();
        match parts.as_slice() {
            [] => Err(format!("{} has no output {role}", self.module)),
            [part] => Ok(part.clone()),
            parts => Ok(format!("({})", parts.join(", "))),
        }
    }
}

/// The role of a root port: a composite value `role` has the leaves `role.0`, `role.1`, ….
fn role_of(name: &str) -> &str {
    name.split_once('.').map_or(name, |(role, _)| role)
}

/// The roles of the root ports, sorted.
fn roles_of(ports: &BTreeMap<String, BoundaryValue>) -> Vec<String> {
    let mut roles = ports.keys().map(|name| role_of(name).to_owned()).collect::<Vec<_>>();
    roles.dedup();
    roles
}

/// The leaves of `role`, in their order in the value.
fn leaves<'a>(
    ports: &'a BTreeMap<String, BoundaryValue>,
    role: &str,
) -> Vec<(&'a String, &'a BoundaryValue)> {
    let mut leaves = ports.iter().filter(|(name, _)| role_of(name) == role).collect::<Vec<_>>();
    leaves.sort_by_key(|(name, _)| {
        name.split_once('.').and_then(|(_, index)| index.parse::<usize>().ok()).unwrap_or(0)
    });
    leaves
}

/// The Lean type of a value with these leaves: their product, right-nested.
fn product_type(leaves: &[(&String, &BoundaryValue)]) -> Option<String> {
    match leaves {
        [] => None,
        [(_, value)] => Some(value.lean_type.clone()),
        leaves => Some(
            leaves
                .iter()
                .map(|(_, value)| format!("({})", value.lean_type))
                .collect::<Vec<_>>()
                .join(" × "),
        ),
    }
}

/// Leaf `index` of the `count` leaves of the value `value`.
fn component(value: &str, index: usize, count: usize) -> String {
    if count <= 1 {
        return format!("({value})");
    }
    let seconds = ".2".repeat(index);
    if index + 1 == count { format!("({value}){seconds}") } else { format!("({value}){seconds}.1") }
}

pub(super) fn names(names: &[&str]) -> Vec<String> {
    names.iter().map(|name| (*name).to_owned()).collect()
}

/// Whether an exported graph is a simulator graph, which only the proof modules read.
pub(super) fn is_simulator(module: &str) -> bool {
    module.starts_with("Sim_")
}
