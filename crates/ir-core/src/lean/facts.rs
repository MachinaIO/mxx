//! Facts modules: the atoms of a record scope as named projections over wire constants.
//!
//! A record scope `S` (a root, a large scope, or a body containing a loop) is
//! `∃ witness, S.body … witness`. Small bodies use a chain of `let`s ending in the
//! `S.constraints_k` fragments; large bodies use the context-indexed constants directly.
//! Destructuring a long let body in a proof (`extract_lets`, `obtain`) puts thousands of variables
//! into one context, and every later step is abstracted over them. The facts module of `S` instead
//! names everything once, at the top level:
//!
//! - `S.Ctx env` holds the scope's `inputs` and `witness`, over the arguments `env` the body takes
//!   before its inputs (backend, hash model, sampling tape and path, parameters, loop indices);
//! - `S.w_<node>_<port> ctx` is each named wire, defined by its `let` right-hand side over the
//!   other wire constants (a `let` equation of the body is `rfl`), and `S.outs ctx` the outputs;
//! - `S.Facts_k ctx` has one field per top-level conjunct of fragment `S.constraints_k` (the last
//!   also takes `outputs`), named `h<node>_<port>` after the last wire the conjunct mentions
//!   (`_<n>` for the `n`-th repetition), `t<position>` when it mentions none, and `hout` for the
//!   output equation `outputs = S.outs ctx`; `S.Facts ctx outputs` collects them as `c<k>`;
//! - `S.facts ctx h : S.Facts ctx outputs` for `h : S.body … ctx.inputs outputs ctx.witness`,
//!   proven fragment by fragment (`S.facts_k`), each conjunct a projection;
//! - `S.<field>_at F i : ∃ w, Child.Facts ⟨inputs_i, w⟩ outputs_i` for every parallel-loop conjunct
//!   `∀ i : Fin n, Child … i inputs_i outputs_i` whose body `Child` is a record scope.
//!
//! A proof then holds `F : S.Facts ctx outputs` and reads an atom as `F.c<k>.<field>`, whose
//! statement mentions only wire constants.
//! Each scope shares its environment binders and context notation in a local section, keeping
//! source size proportional to its atoms rather than repeating environment types per atom.
use std::collections::{BTreeMap, BTreeSet};

/// A parallel-loop conjunct over a child scope, as the parent renders it.
pub(super) struct ChildLoop {
    /// The rendered conjunct, `(∀ i : Fin n, …)`.
    pub conjunct: String,
    /// Symbolic cardinality, rendered directly as a Lean natural without iteration.
    pub count: u128,
    /// The child relation (unqualified) and its arguments before its inputs.
    pub child: String,
    pub env_arguments: String,
    pub inputs: String,
    pub outputs: String,
}

/// What [`render`] needs to know about one record scope.
pub(super) struct RecordScope<'a> {
    pub name: &'a str,
    /// The arguments of `S.body` before its inputs, with their Lean types.
    pub env: Vec<(String, String)>,
    pub input_type: &'a str,
    pub output_type: &'a str,
    /// The named wires in definition order: name, Lean type and defining term.
    pub wires: Vec<(String, String, String)>,
    /// Per fragment `S.constraints_k`: its call and its relations. The output equation follows
    /// the relations of the last fragment.
    pub fragments: Vec<(String, Vec<String>)>,
    /// Whether the fragments are independent (the body is their conjunction) rather than each
    /// ending in the call of the next (see [`super::ExportOptions::facts`]).
    pub independent: bool,
    /// Independent fragments arranged in a balanced conjunction tree.
    pub balanced: bool,
    pub output_tuple: String,
    pub loops: Vec<ChildLoop>,
}

pub(super) enum Declarations {
    All,
    Bindings,
    Facts,
}

pub(super) fn balanced_conjunction(parts: &[String]) -> String {
    match parts {
        [] => "True".to_owned(),
        [part] => part.clone(),
        _ => {
            let middle = parts.len() / 2;
            format!(
                "({} ∧ {})",
                balanced_conjunction(&parts[..middle]),
                balanced_conjunction(&parts[middle..])
            )
        }
    }
}

fn balanced_projection(root: &str, index: usize, count: usize) -> String {
    assert!(index < count);
    if count == 1 {
        root.to_owned()
    } else {
        let middle = count / 2;
        if index < middle {
            balanced_projection(&format!("{root}.1"), index, middle)
        } else {
            balanced_projection(&format!("{root}.2"), index - middle, count - middle)
        }
    }
}

/// Replaces identifiers for which `replace` gives a term; keeps «…», strings, numerals, fields.
pub(super) fn substitute(text: &str, replace: impl Fn(&str) -> Option<String>) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.char_indices().peekable();
    let mut previous = None::<char>;
    while let Some((start, character)) = chars.next() {
        if character == '«' || character == '"' {
            let end = if character == '«' { '»' } else { '"' };
            out.push(character);
            let mut escaped = false;
            for (_, inner) in chars.by_ref() {
                out.push(inner);
                if escaped {
                    escaped = false;
                } else if end == '"' && inner == '\\' {
                    escaped = true;
                } else if inner == end {
                    break;
                }
            }
            previous = Some(end);
            continue;
        }
        if character.is_ascii_alphanumeric() || character == '_' {
            let mut stop = start + character.len_utf8();
            while let Some(&(position, next)) = chars.peek() {
                if next.is_ascii_alphanumeric() || next == '_' || next == '\'' {
                    stop = position + next.len_utf8();
                    chars.next();
                } else {
                    break;
                }
            }
            let token = &text[start..stop];
            let identifier = !character.is_ascii_digit() && previous != Some('.');
            match identifier.then(|| replace(token)).flatten() {
                Some(term) => out.push_str(&term),
                None => out.push_str(token),
            }
            previous = token.chars().last();
            continue;
        }
        out.push(character);
        previous = Some(character);
    }
    out
}

/// The top-level conjuncts (split at ` ∧ ` outside brackets and quotes).
fn conjuncts(text: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut quote = None::<char>;
    let mut start = 0;
    let mut previous = ' ';
    for (position, character) in text.char_indices() {
        if let Some(end) = quote {
            if character == end && !(end == '"' && previous == '\\') {
                quote = None;
            }
        } else {
            match character {
                '«' => quote = Some('»'),
                '"' => quote = Some('"'),
                '(' | '[' | '{' | '⟨' => depth += 1,
                ')' | ']' | '}' | '⟩' => depth -= 1,
                '∧' if depth == 0 &&
                    text[..position].ends_with(' ') &&
                    text[position + '∧'.len_utf8()..].starts_with(' ') =>
                {
                    parts.push(text[start..position - 1].trim());
                    start = position + '∧'.len_utf8() + 1;
                }
                _ => {}
            }
        }
        previous = character;
    }
    parts.push(text[start..].trim());
    parts
}

/// The port wires `w_<node>_<port>` of a conjunct in order (not the intermediate
/// `w_<node>_concat_<k>`), as `generator/patterns.py` reads them.
fn wire_tokens(text: &str) -> Vec<String> {
    let tokens = std::cell::RefCell::new(Vec::new());
    substitute(text, |token| {
        let parts = token.split('_').collect::<Vec<_>>();
        let numeric = |s: &&str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
        let wire = parts.len() == 3 && parts[0] == "w" && parts[1..].iter().all(numeric);
        if wire {
            tokens.borrow_mut().push(token.to_owned());
        }
        None
    });
    tokens.into_inner()
}

/// The field names of the conjuncts of a scope, in order.
fn field_names(conjuncts: &[&str]) -> Vec<String> {
    let mut used = BTreeMap::<String, usize>::new();
    conjuncts
        .iter()
        .enumerate()
        .map(|(position, conjunct)| {
            if conjunct.starts_with("outputs = ") {
                return "hout".into();
            }
            let Some(last) = wire_tokens(conjunct).pop() else {
                return format!("t{position}");
            };
            let base = format!("h{}", &last[2..]);
            match used.get_mut(&base) {
                Some(count) => {
                    *count += 1;
                    format!("{base}_{count}")
                }
                None => {
                    used.insert(base.clone(), 0);
                    base
                }
            }
        })
        .collect()
}

/// The facts declarations of one record scope; `records` names the record scopes emitted so far.
pub(super) fn render(
    scope: &RecordScope<'_>,
    records: &BTreeSet<String>,
    declarations: Declarations,
) -> String {
    let name = scope.name;
    let wires = scope.wires.iter().map(|(wire, _, _)| wire.as_str()).collect::<BTreeSet<_>>();
    let constants = |text: &str| {
        substitute(text, |token| match token {
            "inputs" => Some("ctx.inputs".into()),
            "witness" => Some("ctx.witness".into()),
            _ if wires.contains(token) => Some(format!("({name}.{token} ctx)")),
            _ => None,
        })
    };
    let variables = scope
        .env
        .iter()
        .map(|(binder, ty)| format!("{{{binder} : {ty}}}"))
        .collect::<Vec<_>>()
        .join(" ");
    let env_names =
        scope.env.iter().map(|(binder, _)| binder.as_str()).collect::<Vec<_>>().join(" ");
    let implicit = "";
    let ctx_type = format!("{name}Ctx");
    let mut out = if matches!(declarations, Declarations::Facts) {
        String::new()
    } else {
        format!(
            "set_option genInjectivity false in\nset_option genSizeOf false in\nstructure {name}.Ctx {} where\n  inputs : {}\n  witness : {name}.Witness\n\n",
            scope.env.iter().map(|(b, t)| format!("({b} : {t})")).collect::<Vec<_>>().join(" "),
            scope.input_type
        )
    };
    let section = if matches!(declarations, Declarations::Bindings) {
        "noncomputable section\nset_option linter.unusedVariables false"
    } else {
        "section"
    };
    out.push_str(&format!("{section}\nvariable {variables}\nlocal notation \"{ctx_type}\" => ({name}.Ctx {env_names})\n\n"));
    if !matches!(declarations, Declarations::Facts) {
        let definition = if matches!(declarations, Declarations::Bindings) {
            "noncomputable def"
        } else {
            "def"
        };
        for (wire, ty, term) in &scope.wires {
            out.push_str(&format!(
                "{definition} {name}.{wire} {implicit} (ctx : {ctx_type}) : {} :=\n  {}\n\n",
                constants(ty),
                constants(term)
            ));
        }
        out.push_str(&format!(
            "{definition} {name}.outs {implicit} (ctx : {ctx_type}) : {} :=\n  {}\n\n",
            scope.output_type,
            constants(&scope.output_tuple)
        ));
    }
    if matches!(declarations, Declarations::Bindings) {
        out.push_str("end\n\n");
        return out;
    }
    let last = scope.fragments.len() - 1;
    let output_equation = format!("outputs = {}", scope.output_tuple);
    let mut fragment_conjuncts = scope
        .fragments
        .iter()
        .map(|(_, relations)| relations.iter().flat_map(|r| conjuncts(r)).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    fragment_conjuncts[last].push(&output_equation);
    let all_names = field_names(&fragment_conjuncts.concat());
    let mut names = all_names.clone().into_iter();
    let outputs_binder = format!("(outputs : {})", scope.output_type);
    let mut fields = Vec::new();
    for (k, conjuncts) in fragment_conjuncts.iter().enumerate() {
        let is_last = k == last;
        let fragment_fields = conjuncts
            .iter()
            .map(|conjunct| {
                let field = names.next().expect("one name per conjunct");
                let statement = if field == "hout" {
                    format!("outputs = {name}.outs ctx")
                } else {
                    constants(conjunct)
                };
                (field, statement)
            })
            .collect::<Vec<_>>();
        out.push_str(&format!(
            "structure {name}.Facts_{k} {implicit} (ctx : {ctx_type}){} : Prop where\n{}\n",
            if is_last { format!(" {outputs_binder}") } else { String::new() },
            fragment_fields.iter().map(|(f, s)| format!("  {f} : {s}\n")).collect::<String>()
        ));
        let count = fragment_fields.len();
        // An independent fragment holds its conjuncts only, so its last one is no `.1`.
        let ends = is_last || scope.independent;
        let projection = |j: usize| {
            let mut term = "h".to_owned();
            for _ in 0..j {
                term.push_str(".2");
            }
            if !(ends && j + 1 == count) {
                term.push_str(".1");
            }
            term
        };
        let record = format!("⟨{}⟩", (0..count).map(projection).collect::<Vec<_>>().join(", "));
        let (conclusion, proof) = if is_last {
            (format!("{name}.Facts_{k} ctx outputs"), record)
        } else if scope.independent {
            (format!("{name}.Facts_{k} ctx"), record)
        } else {
            (
                format!("{name}.Facts_{k} ctx ∧ {}", constants(&scope.fragments[k + 1].0)),
                format!("⟨{record}, h{}⟩", ".2".repeat(count)),
            )
        };
        out.push_str(&format!(
            "theorem {name}.facts_{k} {implicit} {{ctx : {ctx_type}}} {{outputs : {}}}\n    (h : {}) :\n    {conclusion} :=\n  {proof}\n\n",
            scope.output_type,
            constants(&scope.fragments[k].0)
        ));
        fields.push(format!(
            "  c{k} : {name}.Facts_{k} ctx{}\n",
            if is_last { " outputs" } else { "" }
        ));
    }
    out.push_str(&format!(
        "structure {name}.Facts {implicit} (ctx : {ctx_type}) {outputs_binder} : Prop where\n{}\n",
        fields.concat()
    ));
    let mut proof = String::new();
    let parts = if scope.balanced {
        (0..=last)
            .map(|part| {
                format!(
                    "({name}.facts_{part} (ctx := ctx) (outputs := outputs) {})",
                    balanced_projection("h", part, last + 1)
                )
            })
            .collect::<Vec<_>>()
    } else if scope.independent {
        // The body is `constraints_0 … ∧ … ∧ constraints_last …`: fragment `k` is its `k`-th
        // conjunct. Each tail `t<k>` is named once, so no projection chain grows with `k`.
        proof.push_str("  have t0 := h\n");
        for k in 0..=last {
            let conjunct = if k == last { format!("t{k}") } else { format!("t{k}.1") };
            proof.push_str(&format!(
                "  have r{k} := {name}.facts_{k} (ctx := ctx) (outputs := outputs) {conjunct}\n"
            ));
            if k != last {
                proof.push_str(&format!("  have t{} := t{k}.2\n", k + 1));
            }
        }
        (0..=last).map(|k| format!("r{k}")).collect::<Vec<_>>()
    } else {
        proof.push_str(&format!(
            "  have r0 := {name}.facts_0 (ctx := ctx) (outputs := outputs) h\n"
        ));
        for k in 1..=last {
            proof.push_str(&format!(
                "  have r{k} := {name}.facts_{k} (ctx := ctx) (outputs := outputs) r{}.2\n",
                k - 1
            ));
        }
        (0..=last)
            .map(|k| if k == last { format!("r{k}") } else { format!("r{k}.1") })
            .collect::<Vec<_>>()
    };
    proof.push_str(&format!("  ⟨{}⟩", parts.join(", ")));
    out.push_str(&format!(
        "theorem {name}.facts {implicit} (ctx : {ctx_type}) {{outputs : {}}}\n    (h : {name}.body {env_names} ctx.inputs outputs ctx.witness) :\n    {name}.Facts ctx outputs :=\n{proof}\n\n",
        scope.output_type
    ));
    let fields = fragment_conjuncts
        .iter()
        .enumerate()
        .flat_map(|(k, cs)| cs.iter().map(move |c| (k, *c)))
        .zip(all_names)
        .collect::<Vec<_>>();
    for child in &scope.loops {
        if !records.contains(&child.child) {
            continue;
        }
        let Some(((k, _), field)) = fields.iter().find(|((_, c), _)| *c == child.conjunct.as_str())
        else {
            continue;
        };
        let c = &child.child;
        out.push_str(&format!(
            "theorem {name}.{field}_at {implicit} {{ctx : {ctx_type}}} {{outputs : {}}}\n    (F : {name}.Facts ctx outputs) (i : Fin {}) :\n    ∃ w : {c}.Witness, {c}.Facts (⟨{}, w⟩ : {c}.Ctx {}) {} :=\n  let ⟨w, h⟩ := F.c{k}.{field} i\n  ⟨w, {c}.facts _ h⟩\n\n",
            scope.output_type,
            child.count,
            constants(&child.inputs),
            constants(&child.env_arguments),
            constants(&child.outputs),
        ));
    }
    out.push_str("end\n\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn balanced_fragments_have_matching_logarithmic_projections() {
        let parts = ["a", "b", "c", "d", "e"].map(str::to_owned);
        assert_eq!(balanced_conjunction(&parts), "((a ∧ b) ∧ (c ∧ (d ∧ e)))");
        assert_eq!(balanced_projection("h", 0, 5), "h.1.1");
        assert_eq!(balanced_projection("h", 2, 5), "h.2.1");
        assert_eq!(balanced_projection("h", 4, 5), "h.2.2.2");
        for index in 0..1025 {
            assert!(balanced_projection("h", index, 1025).matches('.').count() <= 11);
        }
    }

    #[test]
    fn substitution_keeps_fields_quotes_and_numerals() {
        let text = "f w_1_0 witness.w_1_0 params.«w_1_0» \"w_1_0\" 12w_1_0 (w_1_0 i)";
        let out = substitute(text, |token| (token == "w_1_0").then(|| "X".to_owned()));
        assert_eq!(out, "f X witness.w_1_0 params.«w_1_0» \"w_1_0\" 12w_1_0 (X i)");
    }

    #[test]
    fn conjuncts_split_only_at_the_top_level() {
        assert_eq!(
            conjuncts("a ∧ (b ∧ c) ∧ f «x ∧ y» ∧ (if p ∧ q then r else s)"),
            vec!["a", "(b ∧ c)", "f «x ∧ y»", "(if p ∧ q then r else s)"]
        );
    }

    #[test]
    fn field_names_follow_the_last_wire() {
        let names = field_names(&[
            "f w_1_0 w_2_0",
            "g w_2_0",
            "(5) = 5",
            "h w_3_concat_0",
            "outputs = (w_2_0, ())",
        ]);
        assert_eq!(names, vec!["h2_0", "h2_0_1", "t2", "t3", "hout"]);
    }
}
