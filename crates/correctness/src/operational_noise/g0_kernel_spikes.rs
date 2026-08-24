use std::{
    collections::BTreeSet,
    fmt::Write as _,
    fs::{self, OpenOptions},
    io::{self, Write as _},
    path::{Path, PathBuf},
    process::Command,
};

const BALANCED_ROW_COUNT: usize = 5_000;
const FUEL_HAVE_COUNT: usize = 1_000;
const FUEL_HAVE_GROUP_MAX: usize = 64;
const INLINE_TREE_DEPTH: usize = 7;
const ALLOWED_AXIOMS: [&str; 2] = ["propext", "Quot.sound"];

#[derive(Debug)]
struct BalancedRow {
    id: usize,
    value: usize,
    left: Option<Box<Self>>,
    right: Option<Box<Self>>,
}

impl BalancedRow {
    fn median_range(start: usize, end: usize) -> Option<Box<Self>> {
        if start == end {
            return None;
        }
        let middle = (start + end) / 2;
        Some(Box::new(Self {
            id: middle,
            value: middle,
            left: Self::median_range(start, middle),
            right: Self::median_range(middle + 1, end),
        }))
    }

    fn assert_order_and_balance(
        row: Option<&Self>,
        lower: Option<usize>,
        upper: Option<usize>,
        inorder: &mut Vec<usize>,
    ) -> usize {
        let Some(row) = row else {
            return 0;
        };
        assert_eq!(row.id, row.value, "row ID/value mismatch");
        assert!(lower.is_none_or(|lower| lower < row.id), "row violates lower bound");
        assert!(upper.is_none_or(|upper| row.id < upper), "row violates upper bound");
        let left_height =
            Self::assert_order_and_balance(row.left.as_deref(), lower, Some(row.id), inorder);
        inorder.push(row.id);
        let right_height =
            Self::assert_order_and_balance(row.right.as_deref(), Some(row.id), upper, inorder);
        assert!(left_height.abs_diff(right_height) <= 1, "row table is not height-balanced");
        left_height.max(right_height) + 1
    }
}

struct LakeManifestGuard {
    path: PathBuf,
    snapshot: Option<Option<Vec<u8>>>,
}

impl LakeManifestGuard {
    fn capture(path: PathBuf) -> io::Result<Self> {
        let snapshot = if path.exists() { Some(fs::read(&path)?) } else { None };
        Ok(Self { path, snapshot: Some(snapshot) })
    }

    fn restore(&mut self) -> io::Result<()> {
        let Some(snapshot) = self.snapshot.take() else {
            return Ok(());
        };
        match snapshot {
            Some(bytes) => fs::write(&self.path, bytes),
            None if self.path.exists() => fs::remove_file(&self.path),
            None => Ok(()),
        }
    }
}

impl Drop for LakeManifestGuard {
    fn drop(&mut self) {
        let _ = self.restore();
    }
}

fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("correctness crate must be under crates/")
        .to_owned()
}

fn assert_exact_tree(row: &BalancedRow, count: usize) {
    let mut inorder = Vec::with_capacity(count);
    BalancedRow::assert_order_and_balance(Some(row), None, None, &mut inorder);
    assert_eq!(inorder, (0..count).collect::<Vec<_>>(), "row IDs are not exact and ordered");
}

fn render_named_rows(
    row: Option<&BalancedRow>,
    prefix: &str,
    row_type: &str,
    value: &dyn Fn(usize) -> String,
    output: &mut String,
) -> String {
    let Some(row) = row else {
        return ".empty".to_owned();
    };
    let name = format!("{prefix}{}", row.id);
    let expression =
        render_inline_rows(Some(row), INLINE_TREE_DEPTH, prefix, row_type, value, output);
    writeln!(output, "def {name} : RowTable {row_type} := {expression}")
        .expect("writing to String cannot fail");
    name
}

fn render_inline_rows(
    row: Option<&BalancedRow>,
    remaining_depth: usize,
    prefix: &str,
    row_type: &str,
    value: &dyn Fn(usize) -> String,
    output: &mut String,
) -> String {
    let Some(row) = row else {
        return ".empty".to_owned();
    };
    if remaining_depth == 0 {
        return render_named_rows(Some(row), prefix, row_type, value, output);
    }
    let left = render_inline_rows(
        row.left.as_deref(),
        remaining_depth - 1,
        prefix,
        row_type,
        value,
        output,
    );
    let right = render_inline_rows(
        row.right.as_deref(),
        remaining_depth - 1,
        prefix,
        row_type,
        value,
        output,
    );
    format!("(.node {} {} {left} {right})", row.id, value(row.value))
}

fn render_all_from_proof(row: Option<&BalancedRow>) -> String {
    let Some(row) = row else {
        return "by trivial".to_owned();
    };
    format!(
        "⟨rfl, {}, {}⟩",
        render_all_from_proof(row.left.as_deref()),
        render_all_from_proof(row.right.as_deref()),
    )
}

fn render_balanced_rows_module(row: &BalancedRow) -> String {
    let mut definitions = String::new();
    let root =
        render_named_rows(Some(row), "row", "Nat", &|value| value.to_string(), &mut definitions);
    let proof = render_all_from_proof(Some(row));
    format!(
        "import Mxx.Certificate.OperationalNoise.Core\n\
         \n\
         set_option autoImplicit false\n\
         set_option relaxedAutoImplicit false\n\
         \n\
         namespace Mxx.Certificate.OperationalNoise.G0BalancedSpike\n\
         \n\
         {definitions}\n\
         def rows : RowTable Nat := {root}\n\
         \n\
         theorem rows_wellFormed : rows.wellFormed = true := by\n\
         \x20 rfl\n\
         \n\
         theorem rows_allFrom : rows.AllFrom (fun id value => id = value) := by\n\
         \x20 exact {proof}\n\
         \n\
         #print axioms rows_wellFormed\n\
         #print axioms rows_allFrom\n\
         \n\
         end Mxx.Certificate.OperationalNoise.G0BalancedSpike\n"
    )
}

fn evaluation_claim(index: usize) -> String {
    format!("∀ fuel, evalExpr (fuel + 1) cert none samplers inputs ⟨{index}⟩ = some matrix")
}

fn render_balanced_conjunction(items: &[String]) -> String {
    if let [item] = items {
        return format!("({item})");
    }
    let middle = items.len() / 2;
    format!(
        "({} ∧ {})",
        render_balanced_conjunction(&items[..middle]),
        render_balanced_conjunction(&items[middle..]),
    )
}

fn render_balanced_have_proof(ids: &[usize], consumed: &mut BTreeSet<usize>) -> String {
    if let [id] = ids {
        assert!(consumed.insert(*id), "fuel have consumed more than once");
        return format!("h{id}");
    }
    let middle = ids.len() / 2;
    format!(
        "⟨{}, {}⟩",
        render_balanced_have_proof(&ids[..middle], consumed),
        render_balanced_have_proof(&ids[middle..], consumed),
    )
}

fn collect_fuel_groups(start: usize, end: usize, groups: &mut Vec<(usize, usize)>) {
    if end - start <= FUEL_HAVE_GROUP_MAX {
        groups.push((start, end));
        return;
    }
    let middle = (start + end) / 2;
    collect_fuel_groups(start, middle, groups);
    collect_fuel_groups(middle, end, groups);
}

fn render_fuel_group_proof(
    start: usize,
    end: usize,
    groups: &BTreeSet<(usize, usize)>,
    used_groups: &mut BTreeSet<(usize, usize)>,
) -> String {
    if groups.contains(&(start, end)) {
        assert!(used_groups.insert((start, end)), "fuel group consumed more than once");
        return format!("fuel_group_{start}_{end} samplers inputs");
    }
    let middle = (start + end) / 2;
    format!(
        "⟨{}, {}⟩",
        render_fuel_group_proof(start, middle, groups, used_groups),
        render_fuel_group_proof(middle, end, groups, used_groups),
    )
}

fn render_fuel_haves_module(row: &BalancedRow) -> String {
    let mut definitions = String::new();
    let root = render_named_rows(
        Some(row),
        "expressionRow",
        "ExprRow",
        &|_| "(.constant matrix)".to_owned(),
        &mut definitions,
    );
    let have_ids = (0..FUEL_HAVE_COUNT).collect::<BTreeSet<_>>();
    let claims = have_ids.iter().copied().map(evaluation_claim).collect::<Vec<_>>();
    let mut group_ranges = Vec::new();
    collect_fuel_groups(0, FUEL_HAVE_COUNT, &mut group_ranges);
    let mut declared = BTreeSet::new();
    let mut consumed = BTreeSet::new();
    let mut group_theorems = String::new();
    for &(start, end) in &group_ranges {
        let group_claims = &claims[start..end];
        writeln!(
            group_theorems,
            "theorem fuel_group_{start}_{end} (samplers : SamplerAssignment) \
             (inputs : InputAssignment) :\n  {} := by",
            render_balanced_conjunction(group_claims),
        )
        .expect("writing to String cannot fail");
        for (offset, claim) in group_claims.iter().enumerate() {
            let index = start + offset;
            assert!(declared.insert(index), "fuel have declared more than once");
            writeln!(
                group_theorems,
                "  have h{index} : {claim} := by\n    intro fuel\n    cases fuel <;> rfl"
            )
            .expect("writing to String cannot fail");
        }
        let group_ids = (start..end).collect::<Vec<_>>();
        writeln!(
            group_theorems,
            "  exact {}\n",
            render_balanced_have_proof(&group_ids, &mut consumed),
        )
        .expect("writing to String cannot fail");
    }
    assert_eq!(declared, have_ids, "generated fuel haves are not exact");
    assert_eq!(consumed, have_ids, "final conjunction does not consume every fuel have once");
    let conclusion = render_balanced_conjunction(&claims);
    let group_set = group_ranges.into_iter().collect::<BTreeSet<_>>();
    let mut used_groups = BTreeSet::new();
    let proof = render_fuel_group_proof(0, FUEL_HAVE_COUNT, &group_set, &mut used_groups);
    assert_eq!(used_groups, group_set, "final conjunction does not consume every fuel group once");
    format!(
        "import Mxx.Certificate.OperationalNoise.Core\n\
         \n\
         set_option autoImplicit false\n\
         set_option relaxedAutoImplicit false\n\
         \n\
         namespace Mxx.Certificate.OperationalNoise.G0FuelSpike\n\
         \n\
         def shape : MatrixShape :=\n\
         \x20 {{ modulus := 17, ringDimension := 1, rows := 1, columns := 1 }}\n\
         \n\
         def matrix : Matrix := {{ shape := shape, coefficients := [0] }}\n\
         \n\
         {definitions}\n\
         def expressionRows : RowTable ExprRow := {root}\n\
         \n\
         def cert : Cert :=\n\
         \x20 {{ plaintextModulus := 2\n\
         \x20   ciphertextModulus := 17\n\
         \x20   ringDimension := 1\n\
         \x20   expressions := expressionRows\n\
         \x20   programs := .empty\n\
         \x20   sources := .empty\n\
         \x20   events := .empty\n\
         \x20   residualRoot := .closed ⟨0⟩ }}\n\
         \n\
         {group_theorems}\n\
         theorem fuel_haves (samplers : SamplerAssignment) (inputs : InputAssignment) :\n\
         \x20 {conclusion} := by\n\
         \x20 exact {proof}\n\
         \n\
         #print axioms fuel_haves\n\
         \n\
         end Mxx.Certificate.OperationalNoise.G0FuelSpike\n"
    )
}

fn lean_identifier_tokens(source: &str) -> Vec<String> {
    let characters = source.chars().collect::<Vec<_>>();
    let mut tokens = Vec::new();
    let mut index = 0;
    let mut block_depth = 0_u32;
    while index < characters.len() {
        if block_depth != 0 {
            if characters.get(index..index + 2) == Some(&['/', '-']) {
                block_depth += 1;
                index += 2;
            } else if characters.get(index..index + 2) == Some(&['-', '/']) {
                block_depth -= 1;
                index += 2;
            } else {
                index += 1;
            }
            continue;
        }
        if characters.get(index..index + 2) == Some(&['-', '-']) {
            index += 2;
            while index < characters.len() && characters[index] != '\n' {
                index += 1;
            }
            continue;
        }
        if characters.get(index..index + 2) == Some(&['/', '-']) {
            block_depth = 1;
            index += 2;
            continue;
        }
        if characters[index] == '"' {
            index += 1;
            while index < characters.len() {
                if characters[index] == '\\' {
                    index += 2;
                } else if characters[index] == '"' {
                    index += 1;
                    break;
                } else {
                    index += 1;
                }
            }
            continue;
        }
        if characters[index].is_ascii_alphabetic() || characters[index] == '_' {
            let start = index;
            index += 1;
            while index < characters.len() &&
                (characters[index].is_ascii_alphanumeric() ||
                    matches!(characters[index], '_' | '\''))
            {
                index += 1;
            }
            tokens.push(characters[start..index].iter().collect());
        } else {
            index += 1;
        }
    }
    assert_eq!(block_depth, 0, "unterminated generated Lean block comment");
    tokens
}

fn assert_generated_source(source: &str) {
    let forbidden = [
        "True",
        "sorry",
        "admit",
        "native_decide",
        "axiom",
        "unsafe",
        "maxRecDepth",
        "maxHeartbeats",
    ];
    let tokens = lean_identifier_tokens(source).into_iter().collect::<BTreeSet<_>>();
    for token in forbidden {
        assert!(!tokens.contains(token), "generated Lean contains forbidden token {token}");
    }
}

fn write_new(path: &Path, contents: &str) -> io::Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(contents.as_bytes())
}

fn parse_axiom_report(output: &str, declaration: &str) {
    let marker = format!("'{declaration}' ");
    let reports = output.lines().filter(|line| line.starts_with(&marker)).collect::<Vec<_>>();
    assert_eq!(reports.len(), 1, "missing or duplicate axiom report for {declaration}:\n{output}");
    let report = reports[0].strip_prefix(&marker).expect("matched prefix");
    if report == "does not depend on any axioms" {
        return;
    }
    let encoded = report
        .strip_prefix("depends on axioms: [")
        .and_then(|report| report.strip_suffix(']'))
        .unwrap_or_else(|| panic!("malformed axiom report for {declaration}: {report}"));
    for axiom in encoded.split(',').map(str::trim).filter(|axiom| !axiom.is_empty()) {
        assert!(ALLOWED_AXIOMS.contains(&axiom), "unexpected axiom {axiom} for {declaration}");
    }
}

fn compile_lean_module(lean_root: &Path, path: &Path, declarations: &[&str]) {
    let output = Command::new("lake")
        .args(["env", "lean"])
        .arg(path)
        .current_dir(lean_root)
        .output()
        .expect("lake env lean must start");
    let stdout = String::from_utf8(output.stdout).expect("Lean stdout must be UTF-8");
    let stderr = String::from_utf8(output.stderr).expect("Lean stderr must be UTF-8");
    let combined = format!("{stdout}{stderr}");
    assert!(output.status.success(), "Lean failed for {}:\n{combined}", path.display());
    assert!(
        !combined.lines().any(|line| {
            let lowercase = line.to_ascii_lowercase();
            lowercase.contains("warning:") || lowercase.contains("error:")
        }),
        "Lean emitted a warning or error for {}:\n{combined}",
        path.display(),
    );
    let report_count =
        combined.lines().filter(|line| line.contains("depend") && line.contains("axiom")).count();
    assert_eq!(report_count, declarations.len(), "unexpected axiom report count:\n{combined}");
    for declaration in declarations {
        parse_axiom_report(&combined, declaration);
    }
}

#[test]
#[ignore = "exact fixed-size Lean kernel scalability gate"]
fn g0_kernel_spikes_compile_exact_sizes() {
    let balanced_rows = BalancedRow::median_range(0, BALANCED_ROW_COUNT).expect("nonempty table");
    assert_exact_tree(&balanced_rows, BALANCED_ROW_COUNT);
    let fuel_rows = BalancedRow::median_range(0, FUEL_HAVE_COUNT).expect("nonempty table");
    assert_exact_tree(&fuel_rows, FUEL_HAVE_COUNT);

    let balanced_source = render_balanced_rows_module(&balanced_rows);
    let fuel_source = render_fuel_haves_module(&fuel_rows);
    assert_generated_source(&balanced_source);
    assert_generated_source(&fuel_source);

    let repository_root = repository_root();
    let lean_root = repository_root.join("lean");
    let mut manifest_guard = LakeManifestGuard::capture(lean_root.join("lake-manifest.json"))
        .expect("manifest snapshot");
    let temporary = tempfile::Builder::new()
        .prefix("mxx-g0-kernel-spikes-")
        .tempdir()
        .expect("temporary kernel-spike directory");
    let balanced_path = temporary.path().join("BalancedRows.lean");
    let fuel_path = temporary.path().join("FuelHaves.lean");
    write_new(&balanced_path, &balanced_source).expect("new balanced Lean module");
    write_new(&fuel_path, &fuel_source).expect("new fuel Lean module");

    compile_lean_module(
        &lean_root,
        &balanced_path,
        &[
            "Mxx.Certificate.OperationalNoise.G0BalancedSpike.rows_wellFormed",
            "Mxx.Certificate.OperationalNoise.G0BalancedSpike.rows_allFrom",
        ],
    );
    compile_lean_module(
        &lean_root,
        &fuel_path,
        &["Mxx.Certificate.OperationalNoise.G0FuelSpike.fuel_haves"],
    );
    manifest_guard.restore().expect("restore lake manifest");
    println!("balanced_rows={BALANCED_ROW_COUNT}");
    println!("fuel_haves={FUEL_HAVE_COUNT}");
    println!("PASS");
}
