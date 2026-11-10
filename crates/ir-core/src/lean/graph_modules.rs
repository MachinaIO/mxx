//! Write a set of named graphs as Lean modules with their sampling plans.
//!
//! [`export_sampled_graphs`] validates each graph at the given bindings, exports it with
//! [`ExportOptions::sampling_plan`](super::ExportOptions::sampling_plan), so the module defines
//! the relation `generatedRoot` and the computation `generatedRoot.comp`, and writes one module
//! per graph plus `Backend.lean` with the gadget layouts the graphs share. Protocol crates use it
//! to build statements about computations of their own syntax, such as `mxx-fhe`'s
//! implementation-agnostic FHE statements.
use super::{BackendLayout, ExportOptions, LeanArtifact, export};
use crate::{Graph, ParamEnv};
use std::{collections::BTreeMap, fs, path::Path};

/// Validates and exports `graphs` (module name, graph) into `directory` and returns each module's
/// artifact by name. The module names must be valid Lean identifiers.
pub fn export_sampled_graphs(
    graphs: &[(String, &Graph)],
    bindings: &ParamEnv,
    directory: &Path,
) -> Result<BTreeMap<String, LeanArtifact>, Box<dyn std::error::Error>> {
    fs::create_dir_all(directory)?;
    let mut artifacts = BTreeMap::new();
    let mut layouts = BTreeMap::<_, BackendLayout>::new();
    for (name, graph) in graphs {
        let validated = crate::validate(graph, bindings)?;
        let artifact = export(
            &validated,
            &ExportOptions {
                namespace: name.clone(),
                module_name: name.clone(),
                sampling_plan: true,
                ..ExportOptions::default()
            },
        )?;
        for layout in &artifact.backend_layouts {
            let key = (layout.modulus.clone(), layout.ring_dimension);
            let merged = match layouts.get(&key) {
                Some(previous) => previous.merge(layout).ok_or("graphs disagree on a layout")?,
                None => layout.clone(),
            };
            layouts.insert(key, merged);
        }
        for (module, source) in
            super::modules::partition(name, &artifact.source, super::modules::MODULE_BYTES)?
        {
            fs::write(directory.join(format!("{module}.lean")), source)?;
        }
        artifacts.insert(name.clone(), artifact);
    }
    fs::write(
        directory.join(format!("{}.lean", super::claim::BACKEND_MODULE)),
        super::backend::render_backend(&layouts.into_values().collect::<Vec<_>>()),
    )?;
    Ok(artifacts)
}
