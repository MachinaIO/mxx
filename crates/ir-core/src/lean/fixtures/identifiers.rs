//! Lean keywords used as declared parameters and inferred subgraph bindings.
use crate::{
    Graph, GraphOutput, IntExpr, NodeHandle, ParamEnv, WireType,
    graph::{CompileParameter, CompileParameterKind, SubgraphHandle, with_new_construction_scope},
    lean::{ExportOptions, export},
    node::NodeKind,
    validate,
};
use std::collections::BTreeMap;

#[test]
fn export_keyword_identifier_fixture() {
    let child = with_new_construction_scope(|scope| {
        let value = NodeHandle::new(
            NodeKind::EvaluateInt(IntExpr::Var("namespace".into())),
            vec![],
            vec![WireType::ConstantInt],
        )
        .output(0)
        .unwrap();
        SubgraphHandle::new("keyword", scope, vec![], vec![value]).unwrap()
    });
    let value = NodeHandle::subgraph_call(
        child,
        vec![],
        vec![("namespace".into(), IntExpr::Var("match".into()))],
        vec![],
    )
    .output(0)
    .unwrap();
    let (graph, _) = Graph::freeze(
        "keyword-parameters",
        vec![CompileParameter { name: "match".into(), kind: CompileParameterKind::Integer }],
        BTreeMap::from([("value".into(), GraphOutput { value, confidentiality: None })]),
        vec![],
        vec![],
        BTreeMap::new(),
    )
    .unwrap();
    let mut bindings = ParamEnv::default();
    bindings.integers.insert("match".into(), 3.into());
    let artifact =
        export(&validate(&graph, &bindings).unwrap(), &ExportOptions::default()).unwrap();
    assert!(artifact.source.contains("«match» : Int"));
    assert!(artifact.source.contains("«namespace» : Int"));
    assert!(artifact.source.contains("params.«namespace»"));
    assert!(artifact.source.contains("{ params with «namespace» := params.«match» }"));
    super::write_fixture(
        "identifiers",
        format!(
            "{}\ndef keywordParams : Generated.Params := {{ «match» := 3, «namespace» := 0 }}\n",
            artifact.source,
        ),
    );
}
