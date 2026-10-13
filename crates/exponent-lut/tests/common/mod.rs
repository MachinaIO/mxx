//! Helpers of the integration targets that export the Lean correctness claim of the workflow
//! they execute.
//!
//! A target that runs a recorded parameter set reads its row from
//! `lean/<target>/parameters.csv` and writes the claim of the executed graphs to
//! `lean/<target>/<parameter id>/generated`. The claim itself is built with
//! [`mxx_exponent_lut::claim`].

use std::{collections::BTreeMap, path::PathBuf};

fn lean_directory(target: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("lean").join(target)
}

/// Row `id` of `lean/<target>/parameters.csv`, as column name to value.
pub fn parameter_row(target: &str, id: &str) -> BTreeMap<String, String> {
    let path = lean_directory(target).join("parameters.csv");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
    let mut lines = text.lines();
    let header = lines.next().expect("parameter CSV header").split(',').collect::<Vec<_>>();
    let rows = lines
        .map(|line| {
            header.iter().map(|name| name.to_string()).zip(line.split(',').map(str::to_owned))
        })
        .map(BTreeMap::from_iter)
        .filter(|row: &BTreeMap<String, String>| row["parameter_id"] == id)
        .collect::<Vec<_>>();
    let [row] = <[_; 1]>::try_from(rows)
        .unwrap_or_else(|_| panic!("{} must have exactly one row {id}", path.display()));
    row
}

/// A numeric column of a parameter row.
pub fn column<T: std::str::FromStr>(row: &BTreeMap<String, String>, name: &str) -> T
where
    T::Err: std::fmt::Debug,
{
    row.get(name)
        .unwrap_or_else(|| panic!("parameter column {name} is missing"))
        .parse()
        .unwrap_or_else(|error| panic!("parameter column {name} is malformed: {error:?}"))
}

/// The directory receiving the generated modules of parameter set `id`.
pub fn generated_directory(target: &str, id: &str) -> PathBuf {
    lean_directory(target).join(id).join("generated")
}
