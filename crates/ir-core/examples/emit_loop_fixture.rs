//! Emit nested parallel scopes and an identity loop, including an empty outer family.
//! The inner count must be positive for its real static gather; an empty outer family
//! omits the impossible root gather. No primitive relation is overridden.
use mxx_ir_core::{
    Graph, GraphOutput, IntExpr, NodeHandle, ParamEnv,
    graph::{CompileParameter, SubgraphHandle, with_new_construction_scope},
    lean::{ExportOptions, export},
    node::{NodeKind, ParallelLoop, SequentialLoop},
    types::WireType,
    validate,
};
use std::{collections::BTreeMap, env, fs, time::Instant};

fn main() {
    let started = Instant::now();
    let mut args = env::args().skip(1);
    let output = args.next().expect("output Lean path");
    let n: usize = args.next().expect("outer N").parse().expect("N is usize");
    let l: usize = args.next().expect("sequential L").parse().expect("L is usize");
    let m: usize = args.next().map(|s| s.parse().expect("inner M is usize")).unwrap_or(n.max(1));
    assert!(m > 0, "inner M must be positive for the inner static gather");
    let inner_type = WireType::IndexedFamily { element: Box::new(WireType::Int), count: m.into() };
    let body = with_new_construction_scope(|outer_scope| {
        let inner_body = with_new_construction_scope(|inner_scope| {
            let indices = [7, 11].map(|slot| {
                NodeHandle::new(
                    NodeKind::EvaluateInt(IntExpr::LoopIndex(slot)),
                    vec![],
                    vec![WireType::ConstantInt],
                )
                .output(0)
                .unwrap()
            });
            let value = NodeHandle::new(
                NodeKind::IntBinary(mxx_ir_core::node::IntBinaryOp::Add),
                indices.to_vec(),
                vec![WireType::Int],
            )
            .output(0)
            .unwrap();
            SubgraphHandle::new("inner", inner_scope, vec![], vec![value]).unwrap()
        });
        let inner = NodeHandle::parallel_loop(
            inner_body,
            vec![],
            vec![inner_type.clone()],
            ParallelLoop {
                count: m.into(),
                minimum_count: 0,
                index_slot: 11,
                bindings: vec![],
                input_modes: vec![],
            },
        )
        .output(0)
        .unwrap();
        let gathered = NodeHandle::new(
            NodeKind::FamilyGetStatic { index: 0.into() },
            vec![inner],
            vec![WireType::Int],
        )
        .output(0)
        .unwrap();
        SubgraphHandle::new("outer", outer_scope, vec![], vec![gathered]).unwrap()
    });
    let family = NodeHandle::parallel_loop(
        body,
        vec![],
        vec![WireType::IndexedFamily { element: Box::new(WireType::Int), count: n.into() }],
        ParallelLoop {
            count: n.into(),
            minimum_count: 0,
            index_slot: 7,
            bindings: vec![],
            input_modes: vec![],
        },
    )
    .output(0)
    .unwrap();
    let initial = NodeHandle::new(
        NodeKind::Input { name: "initial".into(), wire_type: WireType::Int, artifact: None },
        vec![],
        vec![WireType::Int],
    )
    .output(0)
    .unwrap();
    let step = with_new_construction_scope(|scope| {
        let current = NodeHandle::new(
            NodeKind::Input { name: "current".into(), wire_type: WireType::Int, artifact: None },
            vec![],
            vec![WireType::Int],
        )
        .output(0)
        .unwrap();
        SubgraphHandle::new("step", scope, vec![current.clone()], vec![current]).unwrap()
    });
    let sequence = NodeHandle::sequential_loop(
        step,
        vec![initial],
        vec![WireType::Int],
        SequentialLoop { count: l.into(), index_slot: 19, bindings: vec![], carried_count: 1 },
    )
    .output(0)
    .unwrap();
    let mut outputs = BTreeMap::from([
        ("family".into(), GraphOutput { value: family.clone(), confidentiality: None }),
        ("output".into(), GraphOutput { value: sequence, confidentiality: None }),
    ]);
    if n > 0 {
        let selected = NodeHandle::new(
            NodeKind::FamilyGetStatic { index: 0.into() },
            vec![family],
            vec![WireType::Int],
        )
        .output(0)
        .unwrap();
        outputs.insert("gathered".into(), GraphOutput { value: selected, confidentiality: None });
    }
    let graph = Graph::freeze(
        format!("structural-loops-{n}-{m}-{l}"),
        Vec::<CompileParameter>::new(),
        outputs,
        vec![],
        vec![],
        BTreeMap::new(),
    )
    .unwrap()
    .0;
    let checked = validate(&graph, &ParamEnv::default()).unwrap();
    let artifact = export(
        &checked,
        &ExportOptions {
            namespace: "GeneratedLoops".into(),
            module_name: format!("GeneratedLoops{n}_{m}_{l}"),
            ..ExportOptions::default()
        },
    )
    .unwrap();
    let sequence_relation = artifact
        .source
        .lines()
        .find_map(|line| {
            line.strip_prefix("def sequential_").map(|tail| {
                format!("GeneratedLoops.sequential_{}", tail.split_whitespace().next().unwrap())
            })
        })
        .expect("actual sequential scope relation");
    let output_type = &artifact.root.output_type;
    let family_output = &artifact.root.outputs["family"].projection;
    let final_output = &artifact.root.outputs["output"].projection;
    let (unpack, extra_claim, extra_proof) = if n > 0 {
        (
            "⟨family, gathered, final, _, hfamily, hget, _, hloop, hout⟩",
            format!(" ∧ {} = 0", artifact.root.outputs["gathered"].projection),
            "\n  · obtain ⟨position, hposition, hvalue⟩ := hget\n    change gathered = 0\n    rw [hvalue, generated_parallel_value position (hfamily position)]\n    exact hposition",
        )
    } else {
        ("⟨family, final, _, hfamily, _, hloop, hout⟩", String::new(), "")
    };
    let proof = format!(
        r#"
theorem generated_parallel_value (i : Nat) {{output : Int}}
    (h : GeneratedLoops.parallel_generatedRoot_0 {{ unit := () }} i () output) :
    output = (i : Int) := by
  obtain ⟨family, gathered, _, hfamily, ⟨position, hposition, hvalue⟩, hout⟩ := h
  have hv : family position = (i : Int) + (position.val : Int) := hfamily position
  rw [hout, hvalue, hv, hposition, add_zero]

theorem generated_loop_preserves_initial {{initial : Int}} {{outputs : {output_type}}}
    (h : GeneratedLoops.generatedRoot {{ unit := () }} initial outputs) :
    {final_output} = initial ∧ (∀ i : Fin {n}, {family_output} i = (i.val : Int)){extra_claim} := by
  rcases h with {unpack}
  have hfinal : final = initial := by
    apply MxxIR.IterRuns.invariant
      (body := fun i current next => {sequence_relation} {{ unit := () }} i current next)
      (Invariant := fun _ state => state = initial)
    · rfl
    · intro _ current next hcurrent hstep
      change next = current at hstep
      exact hstep.trans hcurrent
    · exact hloop
  rw [hout]
  refine ⟨hfinal, ?_⟩
{family_proof}{extra_proof}

#print axioms generated_parallel_value
#print axioms generated_loop_preserves_initial
"#,
        family_proof = if n > 0 {
            "  constructor\n  · intro i\n    exact generated_parallel_value i (hfamily i)"
        } else {
            "  intro i\n  exact generated_parallel_value i (hfamily i)"
        }
    );
    let source = format!("{}\n{}", artifact.source, proof);
    println!(
        "N={n} M={m} L={l} source_bytes={} relation_declarations={} static_node_visits={} proof_declarations=2 generation_us={}",
        source.len(),
        artifact.source.lines().filter(|s| s.starts_with("def ")).count(),
        artifact.static_node_visits,
        started.elapsed().as_micros()
    );
    fs::write(output, source).unwrap();
}
