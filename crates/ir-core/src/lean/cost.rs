//! Compositional execution instrumentation, using opaque application-owned work labels.
//!
//! Observed calls and loops retain the ordinary export's witnesses and primitive conditions.
//! A scope without observed descendants is an atomic base case: its local producers execute
//! once per invocation. Its erasure is checked by Lean's type checker, rather than asserted by
//! attaching a precomputed whole-graph total to the ordinary root relation.

use super::*;
use crate::cost::WorkSite;

/// Export the ordinary relation and its counted companion from one frozen graph traversal.
/// The companion is `<module_name>_Cost` and imports the unmodified ordinary module.
pub fn export(
    validated: &ValidatedGraph,
    options: &ExportOptions,
    sites: &[WorkSite],
) -> Result<LeanArtifact, ExportError> {
    validated
        .source
        .execution_work(&validated.bindings, sites)
        .map_err(|error| ExportError::Encoding(error.to_string()))?;
    let mut emitter = Emitter::new(validated, options);
    emitter.work_sites = Some(sites);
    fn collect(
        graph: &Graph,
        scope: &FrozenGraphScopeId,
        sites: &[WorkSite],
        memo: &mut BTreeMap<FrozenGraphScopeId, bool>,
    ) -> bool {
        if let Some(result) = memo.get(scope) {
            return *result;
        }
        let mut present = sites.iter().any(|site| &site.scope == scope && site.units != 0);
        for i in 0..graph.scope(scope).expect("scope").nodes().len() {
            if let Some(child) = graph.child_scope_id(scope, NodeId(i as u64)) {
                present |= collect(graph, &child, sites, memo);
            }
        }
        memo.insert(scope.clone(), present);
        present
    }
    let mut needed = BTreeMap::new();
    collect(&validated.source, &FrozenGraphScopeId::Root, sites, &mut needed);
    emitter.counted_scopes =
        needed.into_iter().filter_map(|(scope, needed)| needed.then_some(scope)).collect();
    emitter.counted_scopes.insert(FrozenGraphScopeId::Root);
    emitter.emit()
}

#[derive(Clone)]
pub(super) struct Counted {
    pub expression: String,
    pub work: String,
    pub proof: String,
}

impl Counted {
    pub fn atom(condition: &str) -> Self {
        Self {
            expression: format!("(MxxIR.Costed.atom ({condition}) 0)"),
            work: "0".into(),
            proof: format!("(MxxIR.Costed.atom_cost ({condition}) 0)"),
        }
    }

    pub fn both(self, right: Self) -> Self {
        Self {
            expression: format!("(MxxIR.Costed.both {} {})", self.expression, right.expression),
            work: format!("({} + {})", self.work, right.work),
            proof: format!("(MxxIR.Costed.both_cost {} {})", self.proof, right.proof),
        }
    }

    pub fn witness(self, binder: &str) -> Self {
        Self {
            expression: format!("(MxxIR.Costed.witness (fun {binder} => {}))", self.expression),
            work: self.work,
            proof: format!("(MxxIR.Costed.witness_cost (fun {binder} => {}))", self.proof),
        }
    }

    pub fn all(self, count: u128) -> Self {
        Self {
            expression: format!(
                "(MxxIR.Costed.all (fun (i : Fin {count}) => {}))",
                self.expression
            ),
            work: format!("(∑ i : Fin {count}, {})", self.work),
            proof: format!("(MxxIR.Costed.all_cost (fun (i : Fin {count}) => {}))", self.proof),
        }
    }
}

impl Emitter<'_> {
    pub(super) fn counted_child(
        &self,
        child: &FrozenGraphScopeId,
        child_name: &str,
        params: &str,
        indices: &str,
        inputs: &str,
        outputs: &str,
    ) -> Counted {
        let name = &self.scopes[child];
        if !self.counted_scopes.contains(child) {
            return Counted::atom(&format!("{child_name} {params} {indices} {inputs} {outputs}"));
        }
        let arguments = &child_name[name.len()..];
        Counted {
            expression: format!(
                "({name}.costed{arguments} {params} {indices} {inputs} {outputs} event)"
            ),
            work: format!("({name}.work {params} {indices} event)"),
            proof: format!(
                "({name}.costed_cost{arguments} {params} {indices} {inputs} {outputs} event)"
            ),
        }
    }

    fn counted_scope_body(
        &self,
        scope_id: &FrozenGraphScopeId,
        bindings: &str,
        relations: &[String],
        conclusion: &str,
        lifted: bool,
    ) -> Counted {
        let name = &self.scopes[scope_id];
        let relation = |index: usize| {
            self.current_counted
                .get(&index)
                .cloned()
                .unwrap_or_else(|| Counted::atom(&relations[index]))
        };
        let mut counted = if self.options.facts {
            // Match the native facts export's fragment association, including its
            // balanced conjunction for large lifted records. Association is part
            // of the indexed proposition even though additive work is associative.
            let count = relations.len().max(1).div_ceil(32);
            let mut fragments = Vec::new();
            for part in 0..count {
                let start = (part * 32).min(relations.len());
                let end = ((part + 1) * 32).min(relations.len());
                let mut fragment =
                    if part + 1 == count { Counted::atom(conclusion) } else { relation(end - 1) };
                let before = if part + 1 == count { end } else { end - 1 };
                for index in (start..before).rev() {
                    fragment = relation(index).both(fragment);
                }
                fragments.push(fragment);
            }
            fn combine(fragments: &[Counted], balanced: bool) -> Counted {
                if let [fragment] = fragments {
                    return fragment.clone();
                }
                let middle = if balanced { fragments.len() / 2 } else { 1 };
                combine(&fragments[..middle], balanced)
                    .both(combine(&fragments[middle..], balanced))
            }
            combine(&fragments, lifted)
        } else {
            let mut counted = Counted::atom(conclusion);
            for index in (0..relations.len()).rev() {
                counted = relation(index).both(counted);
            }
            counted
        };
        if lifted {
            // Use exactly the context projections used by the ordinary large-scope
            // export. Repeating its flat let telescope here exhausts Lean's stack
            // when checking definitional equality against the factored native body.
            let wires = self
                .current_scope_values
                .iter()
                .map(|(wire, _)| wire.as_str())
                .collect::<BTreeSet<_>>();
            let constants = |text: &str| {
                facts::substitute(text, |token| {
                    if token == "inputs" || token == "witness" {
                        Some(format!("ctx.{token}"))
                    } else if wires.contains(token) {
                        Some(format!("({name}.{token} ctx)"))
                    } else {
                        None
                    }
                })
            };
            let mut environment = Vec::new();
            if self.requires_backend {
                environment.push("backend".to_owned());
            }
            if self.requires_hash_model {
                environment.push("hashModel".to_owned());
            }
            if self.requires_tape {
                environment.extend(["tape".to_owned(), "path".to_owned()]);
            }
            environment.push("params".to_owned());
            environment.extend(
                scope_loop_slots(self.graph, scope_id).iter().map(|slot| format!("i_{slot}")),
            );
            let input = if scope_input_wires(self.graph.scope(scope_id).expect("scope")).is_empty()
            {
                "()"
            } else {
                "inputs"
            };
            let prefix =
                format!("let ctx : {name}.Ctx {} := ⟨{input}, witness⟩\n", environment.join(" "));
            counted.expression = format!("({prefix}{})", constants(&counted.expression));
            counted.proof = format!("({prefix}{})", constants(&counted.proof));
        } else {
            for line in bindings.lines().rev() {
                let line = line.trim();
                if let Some(binder) = line.strip_prefix("∃ ").and_then(|s| s.strip_suffix(',')) {
                    counted = counted.witness(binder);
                } else {
                    counted.expression = format!("({line}\n{})", counted.expression);
                    counted.proof = format!("({line}\n{})", counted.proof);
                }
            }
        }
        if self.current_record {
            counted.expression = format!("(let _ := witness\n{})", counted.expression);
            counted.proof = format!("(let _ := witness\n{})", counted.proof);
            counted = counted.witness(&format!("(witness : {name}.Witness)"));
        }
        counted
    }

    pub(super) fn emit_counted_scope(
        &mut self,
        scope_id: &FrozenGraphScopeId,
        header: &str,
        bindings: &str,
        relations: &[String],
        conclusion: &str,
        lifted: bool,
    ) {
        let Some(sites) = self.work_sites else { return };
        if !self.counted_scopes.contains(scope_id) {
            return;
        }
        let name = &self.scopes[scope_id];
        let local_work = sites
            .iter()
            .filter(|site| &site.scope == scope_id)
            .map(|site| {
                format!(
                    "(if event = {} then {} else 0)",
                    serde_json::to_string(&site.event).expect("string encoding"),
                    site.units
                )
            })
            .collect::<Vec<_>>();
        let local_work =
            if local_work.is_empty() { "0".to_owned() } else { local_work.join(" + ") };
        let slots = scope_loop_slots(self.graph, scope_id);
        let loop_binders =
            slots.iter().map(|slot| format!(" (i_{slot} : Nat)")).collect::<String>();
        let loop_args = slots.iter().map(|slot| format!(" i_{slot}")).collect::<String>();
        let work_uses =
            slots.iter().map(|slot| format!("  let _ := i_{slot}\n")).collect::<String>();
        let mut header = header
            .to_owned()
            .replace("(_ : MxxRuntime.HashModel)", "(hashModel : MxxRuntime.HashModel)")
            .replace(
                "(_ : MxxRuntime.SampleTape) (_ : List Nat)",
                "(tape : MxxRuntime.SampleTape) (path : List Nat)",
            );
        // Replace positional loop binders, including selectively named slots, as one suffix.
        if !slots.is_empty() {
            let start = header.find("(params : Params)").expect("parameter binder") +
                "(params : Params)".len();
            let after = &header[start..];
            let mut position = 0;
            for _ in &slots {
                position += after[position..].find(')').expect("loop binder") + 1;
            }
            header.replace_range(start..start + position, &loop_binders);
        }
        if scope_input_wires(self.graph.scope(scope_id).expect("scope")).is_empty() {
            header = header.replace(" (_ : ", " (inputs : ");
        }
        let mut args = String::new();
        if self.requires_backend {
            args.push_str("backend ");
        }
        if self.requires_hash_model {
            args.push_str("hashModel ");
        }
        if self.requires_tape {
            args.push_str("tape path ");
        }
        args.push_str(&format!("params{loop_args} inputs outputs"));
        // A scope with no observed descendants is the base case: every local
        // producer executes once in a scope invocation. Retain its native predicate
        // atomically instead of elaborating thousands of zero-work conditions again.
        // Calls and loops containing observed sites still use compositional rules.
        let mut counted = if self.current_counted.is_empty() {
            Counted::atom(&format!("{name} {args}"))
        } else {
            self.counted_scope_body(scope_id, bindings, relations, conclusion, lifted)
        };
        counted.expression =
            format!("(MxxIR.Costed.addWork {} ({local_work}))", counted.expression);
        counted.proof = format!("(MxxIR.Costed.addWork_cost {} ({local_work}))", counted.proof);
        counted.work = format!("({} + ({local_work}))", counted.work);
        let binders = header
            .strip_prefix(&format!("def {name} "))
            .expect("scope header")
            .strip_suffix(": Prop :=\n")
            .expect("relation result");
        self.cost_source.push_str(&format!(
            "noncomputable def {name}.work (params : Params){loop_binders} (event : String) : Nat :=\n  let _ := params\n  let _ := event\n{work_uses}  {}\n\n\
             noncomputable def {name}.costed {binders}(event : String) :\n    MxxIR.Costed ({name} {args}) :=\n  let _ := event\n  {}\n\n\
             theorem {name}.costed_cost {binders}(event : String) :\n    ({name}.costed {args} event).HasCost ({name}.work params{loop_args} event) := by\n  exact {}\n\n",
            counted.work, counted.expression, counted.proof));
    }
}

#[cfg(test)]
pub(crate) fn write_test_fixture(validated: &ValidatedGraph, name: &str) {
    let sites = validated
        .source
        .scopes()
        .iter()
        .flat_map(|(id, scope)| {
            (0..scope.nodes().len()).map(|i| WorkSite {
                scope: id.clone(),
                node: NodeId(i as u64),
                event: "example-work".into(),
                units: 7,
            })
        })
        .collect::<Vec<_>>();
    let options = ExportOptions {
        module_name: name.into(),
        namespace: name.into(),
        sampling_tape: true,
        facts: true,
        ..Default::default()
    };
    let plain = super::export(validated, &options).unwrap();
    let counted = export(validated, &options, &sites).unwrap();
    assert_eq!(plain.source, counted.source);
    let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_data/lean_execution_cost");
    std::fs::create_dir_all(&directory).unwrap();
    std::fs::write(directory.join(format!("{name}.lean")), counted.source).unwrap();
    std::fs::write(directory.join(format!("{name}_Cost.lean")), counted.cost_source.unwrap())
        .unwrap();
}
