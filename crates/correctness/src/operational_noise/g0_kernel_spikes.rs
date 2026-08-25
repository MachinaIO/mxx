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
const SECURITY0_STATEMENT_ROW_COUNT: usize = 35_975;
const SECURITY0_EVENT_COUNT: usize = 107_567;
const SECURITY0_EVENT_CHUNK_SIZE: usize = 256;
const SECURITY0_EVENT_LEAF_SIZE: usize = 16;
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

fn render_security0_statement_row(index: usize) -> String {
    let inputs = if index == 0 { "[⟨1⟩]" } else { "[]" };
    format!(
        "{{ descriptor := .operation (.stable (.argument {index} statementType)) statementType, \
         inputs := {inputs}, program := none }}"
    )
}

fn render_security0_event(index: usize, event_count: usize) -> String {
    let final_result = event_count - 3;
    let final_pre_fold = event_count - 2;
    let final_end = event_count - 1;
    let event = match index {
        0 => ".invocationStart outerOwner".to_owned(),
        1 => ".resultCoefficient sourceOwner (.finite 1)".to_owned(),
        2 => ".predecessor outerOwner 0 ⟨1⟩ 1".to_owned(),
        10 => ".invocationStart nestedOwner".to_owned(),
        11 => ".resultExact nestedOwner [] .exactZero".to_owned(),
        12 => ".preFoldPolynomial [] .exactZero none".to_owned(),
        13 => ".invocationEndExact nestedOwner [] .exactZero".to_owned(),
        14 => ".specializationComputed outerOwner dispatch ⟨10, 14⟩".to_owned(),
        index if index == final_result => ".resultExact outerOwner [] .exactZero".to_owned(),
        index if index == final_pre_fold => ".preFoldPolynomial [] .exactZero none".to_owned(),
        index if index == final_end => ".invocationEndExact outerOwner [] .exactZero".to_owned(),
        index if index % SECURITY0_EVENT_LEAF_SIZE == 0 => {
            format!(".boundTransfer outerOwner (.identity (.result {} .coefficient))", index - 1,)
        }
        index if index > 1 && index % SECURITY0_EVENT_LEAF_SIZE == 1 => {
            ".boundTransfer outerOwner (.identity (.result 1 .coefficient))".to_owned()
        }
        _ => ".resultCoefficient sourceOwner (.finite 1)".to_owned(),
    };
    let frame_start = if (10..=13).contains(&index) { 10 } else { 0 };
    format!("{{ event := {event}, frameStart := {frame_start} }}")
}

fn render_security0_events(
    event_count: usize,
    chunk_size: usize,
) -> (String, String, Vec<usize>, Vec<usize>, usize) {
    assert!(event_count > 17, "Security0 structural trace must include fixed lifecycle events");
    assert!(chunk_size > 15, "first shard must contain the nested frame range");
    assert_eq!(
        chunk_size % SECURITY0_EVENT_LEAF_SIZE,
        0,
        "logical shards must contain whole event leaves",
    );
    let mut definitions = String::new();
    let mut names = Vec::new();
    let leaf_count = event_count.div_ceil(SECURITY0_EVENT_LEAF_SIZE);
    for leaf in 0..leaf_count {
        let start = leaf * SECURITY0_EVENT_LEAF_SIZE;
        let end = (start + SECURITY0_EVENT_LEAF_SIZE).min(event_count);
        let events = (start..end)
            .map(|index| render_security0_event(index, event_count))
            .collect::<Vec<_>>()
            .join(",\n  ");
        let chunk = start / chunk_size;
        let local = (start % chunk_size) / SECURITY0_EVENT_LEAF_SIZE;
        let name = format!("eventShard{chunk}Leaf{local}");
        writeln!(definitions, "def {name} : Array AnnotatedEvent := #[\n  {events}\n]")
            .expect("writing to String cannot fail");
        names.push(name);
    }
    let leaf_rows = BalancedRow::median_range(0, leaf_count).expect("nonempty event leaf table");
    assert_exact_tree(&leaf_rows, leaf_count);
    let root = render_named_rows(
        Some(&leaf_rows),
        "eventLeafRow",
        "(Array AnnotatedEvent)",
        &|leaf| names[leaf].clone(),
        &mut definitions,
    );
    let leaf_ends = (SECURITY0_EVENT_LEAF_SIZE..event_count)
        .step_by(SECURITY0_EVENT_LEAF_SIZE)
        .chain(std::iter::once(event_count))
        .collect::<Vec<_>>();
    assert_eq!(leaf_ends.len(), leaf_count);
    let shard_ends = (chunk_size..event_count)
        .step_by(chunk_size)
        .chain(std::iter::once(event_count))
        .collect::<Vec<_>>();
    assert_eq!(shard_ends.len(), event_count.div_ceil(chunk_size));
    (definitions, root, leaf_ends, shard_ends, leaf_count)
}

fn render_balanced_replay_chain(prefix: &str, start: usize, end: usize) -> String {
    if end - start == 1 {
        return format!("{prefix}{start}");
    }
    let middle = (start + end) / 2;
    format!(
        "(.trans {} {})",
        render_balanced_replay_chain(prefix, start, middle),
        render_balanced_replay_chain(prefix, middle, end),
    )
}

fn render_security0_replay_chain(
    leaf_ends: &[usize],
    shard_ends: &[usize],
    event_count: usize,
    chunk_size: usize,
) -> String {
    let mut output = "def replayState0 : ReplayState := initialState\n\n".to_owned();
    for (leaf, end) in leaf_ends.iter().copied().enumerate() {
        let next = leaf + 1;
        let frames = if end == event_count {
            "[]"
        } else {
            "[⟨outerOwner, 0, #[some ⟨⟨1⟩, 1⟩], none, false⟩]"
        };
        writeln!(
            output,
            "def replayState{next} : ReplayState := ⟨{end}, {frames}⟩\n\n\
             theorem replayLeaf{leaf} : ReplayChain document history replayState{leaf} \
             replayState{next} :=\n  .chunk {end} (by rfl)\n"
        )
        .expect("writing to String cannot fail");
    }
    let leaves_per_shard = chunk_size / SECURITY0_EVENT_LEAF_SIZE;
    for (shard, end) in shard_ends.iter().copied().enumerate() {
        let start_leaf = shard * leaves_per_shard;
        let end_leaf = (start_leaf + leaves_per_shard).min(leaf_ends.len());
        let proof = render_balanced_replay_chain("replayLeaf", start_leaf, end_leaf);
        writeln!(
            output,
            "theorem replayShard{shard} : ReplayChain document history replayState{start_leaf} \
             replayState{end_leaf} :=\n  {proof}\n"
        )
        .expect("writing to String cannot fail");
        assert_eq!(leaf_ends[end_leaf - 1], end, "shard boundary does not match leaf boundary");
    }
    let proof = render_balanced_replay_chain("replayShard", 0, shard_ends.len());
    writeln!(
        output,
        "theorem replayChain : ReplayChain document history replayState0 \
         replayState{} := by\n  exact {proof}\n",
        leaf_ends.len(),
    )
    .expect("writing to String cannot fail");
    output
}

fn render_security0_structure_module(
    row: &BalancedRow,
    statement_row_count: usize,
    event_count: usize,
    event_chunk_size: usize,
) -> String {
    let mut row_definitions = String::new();
    let row_root = render_named_rows(
        Some(row),
        "statementRow",
        "ExpressionRow",
        &render_security0_statement_row,
        &mut row_definitions,
    );
    let (event_definitions, event_root, leaf_ends, shard_ends, event_leaf_count) =
        render_security0_events(event_count, event_chunk_size);
    let replay_chain =
        render_security0_replay_chain(&leaf_ends, &shard_ends, event_count, event_chunk_size);
    let middle = statement_row_count / 2;
    let last = statement_row_count - 1;
    let middle_event = event_count / 2;
    let last_event = event_count - 1;
    let final_leaf_size = event_count - (event_leaf_count - 1) * SECURITY0_EVENT_LEAF_SIZE;
    let final_state = leaf_ends.len();
    format!(
        "import Mxx.Certificate.OperationalNoise.TallSecurity0ABI\n\
         \n\
         set_option autoImplicit false\n\
         set_option relaxedAutoImplicit false\n\
         \n\
         namespace Mxx.Certificate.OperationalNoise.G0Security0StructuralSpike\n\
         \n\
         open SchemaV1 TallSecurity0ABI\n\
         \n\
         def statementType : ValueType := .matrix \"257\" 1 1 1\n\
         \n\
         {row_definitions}\n\
         def expressionRows : RowTable ExpressionRow := {row_root}\n\
         \n\
         def programRow : SchemaV1.ProgramRow :=\n\
           {{ signature := [], output := statementType, family := none, root := ⟨2⟩ }}\n\
         \n\
         def document : TallDocument :=\n\
         \x20 {{ schemaId := \"mxx-operational-noise-certificate\"\n\
         \x20   schemaVersion := 1\n\
         \x20   plaintextModulus := \"2\"\n\
         \x20   ciphertextModulus := \"257\"\n\
         \x20   ringDimension := 1\n\
         \x20   expressions := expressionRows\n\
         \x20   programs := .node 0 programRow .empty .empty\n\
         \x20   sources := .empty\n\
         \x20   events := .empty\n\
         \x20   indexUses := .empty\n\
         \x20   sliceGroups := .empty\n\
         \x20   residualRoot := .closed ⟨0⟩ }}\n\
         \n\
         def closedOwner (expression : Nat) : Owner := ⟨.closed ⟨0⟩, ⟨expression⟩⟩\n\
         def outerOwner : Owner := closedOwner 0\n\
         def sourceOwner : Owner := closedOwner 1\n\
         def nestedOwner : Owner := closedOwner 2\n\
         def dispatch : UniversalDispatch := ⟨⟨0⟩, ⟨3⟩, ⟨4⟩⟩\n\
         \n\
         {event_definitions}\n\
         def historyLeaves : RowTable (Array AnnotatedEvent) := {event_root}\n\
         def history : EventHistory := {{ leaves := historyLeaves, size := {event_count} }}\n\
         \n\
         {replay_chain}\n\
         theorem firstRowLookup :\n\
             expressionRows.lookup 0 = some ({}) := by\n\
         \x20 rfl\n\
         \n\
         theorem middleRowLookup :\n\
             expressionRows.lookup {middle} = some ({}) := by\n\
         \x20 rfl\n\
         \n\
         theorem lastRowLookup :\n\
             expressionRows.lookup {last} = some ({}) := by\n\
         \x20 rfl\n\
         \n\
         theorem historyHasExactSize : history.size = {event_count} := by\n\
         \x20 decide\n\
         \n\
         theorem historyHasExactLeafCount : history.leafCount = {event_leaf_count} := by\n\
         \x20 decide\n\
         \n\
         theorem historyNodeCount : rowTableNodeCount history.leaves = {event_leaf_count} := by\n\
         \x20 decide\n\
         \n\
         theorem firstEventLookup : history.lookup 0 = some ({}) := by\n\
         \x20 rfl\n\
         \n\
         theorem middleEventLookup : history.lookup {middle_event} = some ({}) := by\n\
         \x20 rfl\n\
         \n\
         theorem lastEventLookup : history.lookup {last_event} = some ({}) := by\n\
         \x20 rfl\n\
         \n\
         theorem finalLeafHasExactSize :\n\
             (history.leaves.lookup {}).map Array.size = some {final_leaf_size} := by\n\
         \x20 rfl\n\
         \n\
         theorem historyIsWellFormed : history.wellFormed = true := by\n\
         \x20 decide\n\
         \n\
         theorem replayCloses :\n\
             replayState{final_state}.cursor = {event_count} ∧\n\
               replayState{final_state}.frames = [] := by\n\
         \x20 exact ⟨rfl, rfl⟩\n\
         \n\
         #print axioms firstRowLookup\n\
         #print axioms middleRowLookup\n\
         #print axioms lastRowLookup\n\
         #print axioms historyHasExactSize\n\
         #print axioms historyHasExactLeafCount\n\
         #print axioms historyNodeCount\n\
         #print axioms firstEventLookup\n\
         #print axioms middleEventLookup\n\
         #print axioms lastEventLookup\n\
         #print axioms finalLeafHasExactSize\n\
         #print axioms historyIsWellFormed\n\
         #print axioms replayChain\n\
         #print axioms replayCloses\n\
         \n\
         end Mxx.Certificate.OperationalNoise.G0Security0StructuralSpike\n",
        render_security0_statement_row(0),
        render_security0_statement_row(middle),
        render_security0_statement_row(last),
        render_security0_event(0, event_count),
        render_security0_event(middle_event, event_count),
        render_security0_event(last_event, event_count),
        event_leaf_count - 1,
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

#[test]
#[ignore = "actual-cardinality Security0 Lean structural compile-feasibility gate"]
fn g0_kernel_spike_compiles_security0_actual_structure() {
    let statement_rows = BalancedRow::median_range(0, SECURITY0_STATEMENT_ROW_COUNT)
        .expect("nonempty Security0 statement table");
    assert_exact_tree(&statement_rows, SECURITY0_STATEMENT_ROW_COUNT);
    let source = render_security0_structure_module(
        &statement_rows,
        SECURITY0_STATEMENT_ROW_COUNT,
        SECURITY0_EVENT_COUNT,
        SECURITY0_EVENT_CHUNK_SIZE,
    );
    assert_generated_source(&source);

    let repository_root = repository_root();
    let lean_root = repository_root.join("lean");
    let mut manifest_guard = LakeManifestGuard::capture(lean_root.join("lake-manifest.json"))
        .expect("manifest snapshot");
    let temporary = tempfile::Builder::new()
        .prefix("mxx-g0-security0-structural-spike-")
        .tempdir()
        .expect("temporary Security0 structural-spike directory");
    let path = temporary.path().join("Security0ActualStructure.lean");
    write_new(&path, &source).expect("new Security0 structural-spike Lean module");

    compile_lean_module(
        &lean_root,
        &path,
        &[
            "Mxx.Certificate.OperationalNoise.G0Security0StructuralSpike.firstRowLookup",
            "Mxx.Certificate.OperationalNoise.G0Security0StructuralSpike.middleRowLookup",
            "Mxx.Certificate.OperationalNoise.G0Security0StructuralSpike.lastRowLookup",
            "Mxx.Certificate.OperationalNoise.G0Security0StructuralSpike.historyHasExactSize",
            "Mxx.Certificate.OperationalNoise.G0Security0StructuralSpike.historyHasExactLeafCount",
            "Mxx.Certificate.OperationalNoise.G0Security0StructuralSpike.historyNodeCount",
            "Mxx.Certificate.OperationalNoise.G0Security0StructuralSpike.firstEventLookup",
            "Mxx.Certificate.OperationalNoise.G0Security0StructuralSpike.middleEventLookup",
            "Mxx.Certificate.OperationalNoise.G0Security0StructuralSpike.lastEventLookup",
            "Mxx.Certificate.OperationalNoise.G0Security0StructuralSpike.finalLeafHasExactSize",
            "Mxx.Certificate.OperationalNoise.G0Security0StructuralSpike.historyIsWellFormed",
            "Mxx.Certificate.OperationalNoise.G0Security0StructuralSpike.replayChain",
            "Mxx.Certificate.OperationalNoise.G0Security0StructuralSpike.replayCloses",
        ],
    );
    manifest_guard.restore().expect("restore lake manifest");
    println!("statement_rows={SECURITY0_STATEMENT_ROW_COUNT}");
    println!("events={SECURITY0_EVENT_COUNT}");
    println!("event_chunks={}", SECURITY0_EVENT_COUNT.div_ceil(SECURITY0_EVENT_CHUNK_SIZE),);
    println!("PASS");
}

#[test]
#[ignore = "small generated Lean syntax gate for the Security0 structural spike renderer"]
fn g0_kernel_spike_renderer_compiles_tiny_structure() {
    const TINY_ROW_COUNT: usize = 5;
    const TINY_EVENT_COUNT: usize = 32;
    const TINY_CHUNK_SIZE: usize = 16;
    let statement_rows =
        BalancedRow::median_range(0, TINY_ROW_COUNT).expect("nonempty tiny statement table");
    assert_exact_tree(&statement_rows, TINY_ROW_COUNT);
    let source = render_security0_structure_module(
        &statement_rows,
        TINY_ROW_COUNT,
        TINY_EVENT_COUNT,
        TINY_CHUNK_SIZE,
    );
    assert_generated_source(&source);

    let repository_root = repository_root();
    let lean_root = repository_root.join("lean");
    let mut manifest_guard = LakeManifestGuard::capture(lean_root.join("lake-manifest.json"))
        .expect("manifest snapshot");
    let temporary = tempfile::Builder::new()
        .prefix("mxx-g0-security0-structural-renderer-")
        .tempdir()
        .expect("temporary tiny structural-renderer directory");
    let path = temporary.path().join("Security0TinyStructure.lean");
    write_new(&path, &source).expect("new tiny Security0 structural Lean module");
    compile_lean_module(
        &lean_root,
        &path,
        &[
            "Mxx.Certificate.OperationalNoise.G0Security0StructuralSpike.firstRowLookup",
            "Mxx.Certificate.OperationalNoise.G0Security0StructuralSpike.middleRowLookup",
            "Mxx.Certificate.OperationalNoise.G0Security0StructuralSpike.lastRowLookup",
            "Mxx.Certificate.OperationalNoise.G0Security0StructuralSpike.historyHasExactSize",
            "Mxx.Certificate.OperationalNoise.G0Security0StructuralSpike.historyHasExactLeafCount",
            "Mxx.Certificate.OperationalNoise.G0Security0StructuralSpike.historyNodeCount",
            "Mxx.Certificate.OperationalNoise.G0Security0StructuralSpike.firstEventLookup",
            "Mxx.Certificate.OperationalNoise.G0Security0StructuralSpike.middleEventLookup",
            "Mxx.Certificate.OperationalNoise.G0Security0StructuralSpike.lastEventLookup",
            "Mxx.Certificate.OperationalNoise.G0Security0StructuralSpike.finalLeafHasExactSize",
            "Mxx.Certificate.OperationalNoise.G0Security0StructuralSpike.historyIsWellFormed",
            "Mxx.Certificate.OperationalNoise.G0Security0StructuralSpike.replayChain",
            "Mxx.Certificate.OperationalNoise.G0Security0StructuralSpike.replayCloses",
        ],
    );
    manifest_guard.restore().expect("restore lake manifest");
}
