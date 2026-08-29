use super::{MODULE_ROOT, NAMESPACE, generated_file, list, option, quoted};
use crate::operational_noise::{
    certificate_schema::{
        CertificateDocumentV1, CertificateEventRowV1, CertificateExpressionRow,
        CertificateIndexUse, CertificateProgramRow, CertificateRange, CertificateResidualRootV1,
        CertificateSliceGroup, CertificateSourceRowV1, RawCoefficientClassV1, RawValueContractV1,
    },
    g0::{
        CanonicalEventOperator, CanonicalExpressionDescriptor, CanonicalExpressionOperator,
        CanonicalExpressionSource, CanonicalStatementScope, IndexLutRow, IndexUseKind, SliceLutRow,
        SliceMemberRole, StableArtifact, StableConstant, StableConstantValue,
        StableFamilySourceIdentity, StableFrontierAxis, StableHashDefinition, StableHashVariant,
        StableLayout, StableMatrixConstantKind, StableMatrixOperation, StableObservedOccurrence,
        StableObservedProducer, StableObservedSourceAccess, StableObservedWire, StableOperator,
        StablePlanRef, StableSampleDescriptor, StableSamplerOperation, StableScalarOperation,
        StableScope, StableTransformOperation, StableTrapdoorOperation, StableValueType,
    },
};
use num_bigint::BigInt;
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write as _,
};

const PACKAGE_SIZE: usize = 256;
const EXPRESSION_INPUT_LEAF_SIZE: usize = 16;
const SLICE_LUT_ROW_LEAF_SIZE: usize = 256;
const SLICE_LUT_ROW_FANOUT: usize = 16;

pub(super) fn render(
    document: &CertificateDocumentV1,
) -> Result<Vec<super::GeneratedLeanFile>, String> {
    let mut files = Vec::new();
    render_expression_packages(&document.expressions, &mut files)?;
    render_packages("Program", "ProgramRow", &document.programs, program_row, &mut files)?;
    render_packages("Source", "SourceRow", &document.sources, source_row, &mut files)?;
    render_packages("Event", "EventRow", &document.events, event_row, &mut files)?;
    render_index_use_packages(document, &mut files)?;
    render_slice_group_packages(&document.slice_groups, &mut files)?;
    files.push(generated_file("Cert/Cert.lean", render_top(document)?));
    Ok(files)
}

fn render_expression_packages(
    rows: &[CertificateExpressionRow],
    files: &mut Vec<super::GeneratedLeanFile>,
) -> Result<(), String> {
    for (package, chunk) in rows.chunks(PACKAGE_SIZE).enumerate() {
        let module = format!("Expression{package:03}");
        let mut source = header("Mxx.Certificate.OperationalNoise.CertificateABI", &module);
        let start = package * PACKAGE_SIZE;
        for (offset, row) in chunk.iter().enumerate() {
            let row_id = start + offset;
            render_expression_inputs(row_id, &row.inputs, &mut source);
            writeln!(
                source,
                "def ExpressionRow{row_id} : CertificateABI.ExpressionRow :=\n  {}\n",
                expression_row(row, row_id)?
            )
            .expect("writing to String cannot fail");
        }
        writeln!(source, "end {NAMESPACE}.Cert.{module}").expect("writing to String cannot fail");
        files.push(generated_file(format!("Cert/{module}.lean"), source));
    }
    Ok(())
}

fn render_expression_inputs(row: usize, inputs: &[u64], out: &mut String) {
    let leaf_count = inputs.len().div_ceil(EXPRESSION_INPUT_LEAF_SIZE);
    if leaf_count == 0 {
        writeln!(out, "def ExpressionInputs{row} : ExpressionInputs := ⟨.empty, 0⟩\n")
            .expect("writing to String cannot fail");
        return;
    }

    if leaf_count == 1 {
        writeln!(
            out,
            "def ExpressionInputs{row} : ExpressionInputs :=\n  ⟨(.node 0 {} .empty .empty), {}⟩\n",
            array_refs(inputs),
            inputs.len()
        )
        .expect("writing to String cannot fail");
        return;
    }

    for (leaf, values) in inputs.chunks(EXPRESSION_INPUT_LEAF_SIZE).enumerate() {
        writeln!(
            out,
            "def ExpressionInputLeaf{row}_{leaf} : Array ExpressionRef := {}",
            array_refs(values)
        )
        .expect("writing to String cannot fail");
    }
    let leaves = balanced_expression_input_leaves(row, 0, leaf_count);
    writeln!(
        out,
        "def ExpressionInputs{row} : ExpressionInputs :=\n  ⟨{leaves}, {}⟩\n",
        inputs.len()
    )
    .expect("writing to String cannot fail");
}

fn balanced_expression_input_leaves(row: usize, start: usize, end: usize) -> String {
    if start == end {
        return ".empty".to_owned();
    }
    let middle = (start + end) / 2;
    let left = balanced_expression_input_leaves(row, start, middle);
    let right = balanced_expression_input_leaves(row, middle + 1, end);
    format!("(.node {middle} ExpressionInputLeaf{row}_{middle} {left} {right})")
}

fn array_refs(values: &[u64]) -> String {
    format!("#[{}]", values.iter().map(|value| event_ref(*value)).collect::<Vec<_>>().join(", "))
}

fn render_index_use_packages(
    document: &CertificateDocumentV1,
    files: &mut Vec<super::GeneratedLeanFile>,
) -> Result<(), String> {
    let rows = &document.index_uses;
    for (package, chunk) in rows.chunks(PACKAGE_SIZE).enumerate() {
        let module = format!("IndexUse{package:03}");
        let mut source = header("Mxx.Certificate.OperationalNoise.CertificateABI", &module);
        let start = package * PACKAGE_SIZE;
        for (offset, row) in chunk.iter().enumerate() {
            let row_id = start + offset;
            let row_value = project_index_lut_rows(document, row).map_err(|error| {
                format!("index use row {row_id} has unsupported typed index expression: {error}")
            })?;
            writeln!(
                source,
                "def IndexUseRow{row_id} : CertificateABI.IndexUseRow :=\n  {}\n",
                index_use_row(row, &row_value)?
            )
            .expect("writing to String cannot fail");
        }
        writeln!(source, "end {NAMESPACE}.Cert.{module}").expect("writing to String cannot fail");
        files.push(generated_file(format!("Cert/{module}.lean"), source));
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ProjectedIndexExpression {
    Binding { expression: u64, position: usize },
    Constant { expression: u64, value: BigInt },
    Add { expression: u64, left: Box<Self>, right: Box<Self> },
    Subtract { expression: u64, left: Box<Self>, right: Box<Self> },
    Multiply { expression: u64, left: Box<Self>, right: Box<Self> },
    Divide { expression: u64, left: Box<Self>, right: Box<Self> },
    Remainder { expression: u64, left: Box<Self>, right: Box<Self> },
    Negate { expression: u64, value: Box<Self> },
}

impl ProjectedIndexExpression {
    fn evaluate(&self, bindings: &[BigInt]) -> Result<BigInt, String> {
        let pair = |left: &Self, right: &Self| {
            Ok::<_, String>((left.evaluate(bindings)?, right.evaluate(bindings)?))
        };
        match self {
            Self::Binding { position, .. } => bindings
                .get(*position)
                .cloned()
                .ok_or_else(|| "typed index binding position is missing".to_owned()),
            Self::Constant { value, .. } => Ok(value.clone()),
            Self::Add { left, right, .. } => {
                let (left, right) = pair(left, right)?;
                Ok(left + right)
            }
            Self::Subtract { left, right, .. } => {
                let (left, right) = pair(left, right)?;
                Ok(left - right)
            }
            Self::Multiply { left, right, .. } => {
                let (left, right) = pair(left, right)?;
                Ok(left * right)
            }
            Self::Divide { left, right, .. } => {
                let (left, right) = pair(left, right)?;
                mxx_ir_core::expr::euclidean_div_rem(&left, &right)
                    .map(|pair| pair.0)
                    .map_err(|_| "typed index division has a zero divisor".to_owned())
            }
            Self::Remainder { left, right, .. } => {
                let (left, right) = pair(left, right)?;
                mxx_ir_core::expr::euclidean_div_rem(&left, &right)
                    .map(|pair| pair.1)
                    .map_err(|_| "typed index remainder has a zero divisor".to_owned())
            }
            Self::Negate { value, .. } => Ok(-value.evaluate(bindings)?),
        }
    }

    fn render(&self) -> String {
        let binary = |constructor: &str, expression: u64, left: &Self, right: &Self| {
            format!(".{constructor} ⟨{expression}⟩ ({}) ({})", left.render(), right.render())
        };
        match self {
            Self::Binding { expression, .. } => format!(".binding ⟨{expression}⟩"),
            Self::Constant { expression, value } => {
                format!(".constant ⟨{expression}⟩ ({value})")
            }
            Self::Add { expression, left, right } => binary("add", *expression, left, right),
            Self::Subtract { expression, left, right } => {
                binary("subtract", *expression, left, right)
            }
            Self::Multiply { expression, left, right } => {
                binary("multiply", *expression, left, right)
            }
            Self::Divide { expression, left, right } => binary("divide", *expression, left, right),
            Self::Remainder { expression, left, right } => {
                binary("remainder", *expression, left, right)
            }
            Self::Negate { expression, value } => {
                format!(".negate ⟨{expression}⟩ ({})", value.render())
            }
        }
    }
}

fn project_index_lut_rows(
    document: &CertificateDocumentV1,
    use_row: &CertificateIndexUse,
) -> Result<String, String> {
    let mut bindings = BTreeMap::new();
    for (position, axis) in use_row.frontier.iter().enumerate() {
        let expression = validate_index_frontier_axis(document, axis)?;
        if bindings.insert(expression, position).is_some() {
            return Err(format!("frontier expression {expression} is duplicated"));
        }
    }
    let StablePlanRef::Expression { row: root } = use_row.index else {
        return Err("typed index root is not an expression reference".to_owned());
    };
    let expression = project_index_expression(document, root, &bindings, &mut BTreeSet::new())?;
    validate_projected_index_rows(&expression, &use_row.frontier, &use_row.rows)?;
    Ok(format!("⟨{}⟩", expression.render()))
}

fn validate_index_frontier_axis(
    document: &CertificateDocumentV1,
    axis: &StableFrontierAxis,
) -> Result<u64, String> {
    let reference = match axis {
        StableFrontierAxis::Argument { expression, .. } |
        StableFrontierAxis::ExtractedCoefficient { expression, .. } => expression,
    };
    let StablePlanRef::Expression { row } = reference else {
        return Err("frontier binding is not an expression reference".to_owned());
    };
    let node = document
        .expressions
        .get(usize::try_from(*row).map_err(|_| "frontier expression overflows usize")?)
        .ok_or_else(|| format!("frontier expression {row} is missing"))?;
    let CanonicalExpressionDescriptor::Operation { operator, value_type } = &node.descriptor else {
        return Err(format!("frontier expression {row} is not an operation"));
    };
    if value_type != &StableValueType::Int {
        return Err(format!("frontier expression {row} is not integer typed"));
    }
    match (axis, operator) {
        (
            StableFrontierAxis::Argument { position: expected, .. },
            CanonicalExpressionOperator::Stable(StableOperator::Argument {
                position,
                value_type: StableValueType::Int,
            }),
        ) if position == expected => {}
        (
            StableFrontierAxis::ExtractedCoefficient { domain, .. },
            CanonicalExpressionOperator::Stable(StableOperator::ExtractCoefficient {
                canonical_input_exclusive_upper: Some(upper),
                ..
            }),
        ) if domain.0 == 0 && *upper == domain.1.to_string() => {}
        _ => return Err(format!("frontier expression {row} descriptor does not match its axis")),
    }
    Ok(*row)
}

fn project_index_expression(
    document: &CertificateDocumentV1,
    expression: u64,
    bindings: &BTreeMap<u64, usize>,
    visiting: &mut BTreeSet<u64>,
) -> Result<ProjectedIndexExpression, String> {
    if let Some(position) = bindings.get(&expression) {
        return Ok(ProjectedIndexExpression::Binding { expression, position: *position });
    }
    if !visiting.insert(expression) {
        return Err(format!("typed index expression cycle at {expression}"));
    }
    let result = (|| {
        let node = document
            .expressions
            .get(usize::try_from(expression).map_err(|_| "index expression overflows usize")?)
            .ok_or_else(|| format!("typed index expression {expression} is missing"))?;
        if let CanonicalExpressionDescriptor::Source {
            source: CanonicalExpressionSource::Direct { source },
        } = &node.descriptor
        {
            if !node.inputs.is_empty() {
                return Err(format!("typed index constant source {expression} has inputs"));
            }
            let source = document
                .sources
                .get(usize::try_from(*source).map_err(|_| "constant source overflows usize")?)
                .ok_or_else(|| format!("typed index constant source {source} is missing"))?;
            let CertificateSourceRowV1::Constant { value } = source else {
                return Err(format!("typed index source expression {expression} is not constant"));
            };
            if value.value_type != StableValueType::Int {
                return Err(format!(
                    "typed index constant source {expression} is not integer typed"
                ));
            }
            let StableConstantValue::Int { value } = &value.value else {
                return Err(format!("typed index constant source {expression} is not an integer"));
            };
            let parsed = value
                .parse::<BigInt>()
                .map_err(|_| format!("typed index constant source {expression} is not decimal"))?;
            if parsed.to_string() != *value {
                return Err(format!("typed index constant source {expression} is not canonical"));
            }
            return Ok(ProjectedIndexExpression::Constant { expression, value: parsed });
        }
        let CanonicalExpressionDescriptor::Operation { operator, value_type } = &node.descriptor
        else {
            return Err(format!("typed index expression {expression} is not an operation"));
        };
        if value_type != &StableValueType::Int {
            return Err(format!("typed index expression {expression} is not integer typed"));
        }
        let project_input = |position: usize, visiting: &mut BTreeSet<u64>| {
            let input = node.inputs.get(position).copied().ok_or_else(|| {
                format!("typed index expression {expression} is missing input {position}")
            })?;
            project_index_expression(document, input, bindings, visiting)
        };
        match operator {
            CanonicalExpressionOperator::Stable(StableOperator::Constant { value }) => {
                if !node.inputs.is_empty() || value.value_type != StableValueType::Int {
                    return Err(format!(
                        "typed index constant {expression} has invalid type or arity"
                    ));
                }
                let StableConstantValue::Int { value } = &value.value else {
                    return Err(format!("typed index constant {expression} is not an integer"));
                };
                let parsed = value
                    .parse::<BigInt>()
                    .map_err(|_| format!("typed index constant {expression} is not decimal"))?;
                if parsed.to_string() != *value {
                    return Err(format!("typed index constant {expression} is not canonical"));
                }
                Ok(ProjectedIndexExpression::Constant { expression, value: parsed })
            }
            CanonicalExpressionOperator::Stable(StableOperator::Scalar { operation }) => {
                let binary = |constructor: fn(
                    u64,
                    Box<ProjectedIndexExpression>,
                    Box<ProjectedIndexExpression>,
                ) -> ProjectedIndexExpression,
                              visiting: &mut BTreeSet<u64>| {
                    if node.inputs.len() != 2 {
                        return Err(format!(
                            "typed index expression {expression} has invalid binary arity"
                        ));
                    }
                    Ok(constructor(
                        expression,
                        Box::new(project_input(0, visiting)?),
                        Box::new(project_input(1, visiting)?),
                    ))
                };
                match operation {
                    StableScalarOperation::Add => binary(
                        |expression, left, right| ProjectedIndexExpression::Add {
                            expression,
                            left,
                            right,
                        },
                        visiting,
                    ),
                    StableScalarOperation::Subtract => binary(
                        |expression, left, right| ProjectedIndexExpression::Subtract {
                            expression,
                            left,
                            right,
                        },
                        visiting,
                    ),
                    StableScalarOperation::Multiply => binary(
                        |expression, left, right| ProjectedIndexExpression::Multiply {
                            expression,
                            left,
                            right,
                        },
                        visiting,
                    ),
                    StableScalarOperation::Divide => binary(
                        |expression, left, right| ProjectedIndexExpression::Divide {
                            expression,
                            left,
                            right,
                        },
                        visiting,
                    ),
                    StableScalarOperation::Remainder => binary(
                        |expression, left, right| ProjectedIndexExpression::Remainder {
                            expression,
                            left,
                            right,
                        },
                        visiting,
                    ),
                    StableScalarOperation::Negate => {
                        if node.inputs.len() != 1 {
                            return Err(format!(
                                "typed index expression {expression} has invalid unary arity"
                            ));
                        }
                        Ok(ProjectedIndexExpression::Negate {
                            expression,
                            value: Box::new(project_input(0, visiting)?),
                        })
                    }
                    _ => Err(format!(
                        "typed index expression {expression} uses unsupported scalar operation {operation:?}"
                    )),
                }
            }
            CanonicalExpressionOperator::Stable(
                StableOperator::Argument { .. } | StableOperator::ExtractCoefficient { .. },
            ) => Err(format!("typed index expression {expression} has no frontier binding")),
            _ => Err(format!("typed index expression {expression} has an unsupported descriptor")),
        }
    })();
    visiting.remove(&expression);
    result
}

fn validate_projected_index_rows(
    expression: &ProjectedIndexExpression,
    frontier: &[StableFrontierAxis],
    rows: &[IndexLutRow],
) -> Result<(), String> {
    let widths = frontier
        .iter()
        .map(|axis| {
            let (minimum, maximum) = match axis {
                StableFrontierAxis::Argument { domain, .. } |
                StableFrontierAxis::ExtractedCoefficient { domain, .. } => *domain,
            };
            maximum
                .checked_sub(minimum)
                .and_then(|width| usize::try_from(width).ok())
                .ok_or_else(|| "frontier domain is invalid or too wide".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let row_count = widths.iter().try_fold(1_usize, |count, width| {
        count.checked_mul(*width).ok_or_else(|| "frontier product overflows usize".to_owned())
    })?;
    if rows.len() != row_count {
        return Err(format!("raw LUT has {} rows but frontier requires {row_count}", rows.len()));
    }
    for (row_position, row) in rows.iter().enumerate() {
        let mut remainder = row_position;
        let mut tuple = Vec::with_capacity(frontier.len());
        for (axis_position, axis) in frontier.iter().enumerate() {
            let tail = widths[axis_position + 1..]
                .iter()
                .try_fold(1_usize, |count, width| count.checked_mul(*width))
                .ok_or_else(|| "frontier suffix product overflows usize".to_owned())?;
            let digit = if tail == 0 { 0 } else { remainder / tail };
            if tail != 0 {
                remainder %= tail;
            }
            let minimum = match axis {
                StableFrontierAxis::Argument { domain, .. } |
                StableFrontierAxis::ExtractedCoefficient { domain, .. } => domain.0,
            };
            tuple.push(BigInt::from(minimum) + BigInt::from(digit));
        }
        let expected_tuple = tuple.iter().map(ToString::to_string).collect::<Vec<_>>();
        if row.tuple != expected_tuple {
            return Err(format!("raw LUT tuple mismatch at row {row_position}"));
        }
        let output = expression.evaluate(&tuple)?;
        if row.output != output.to_string() {
            return Err(format!("raw LUT output mismatch at row {row_position}"));
        }
    }
    Ok(())
}

fn render_slice_group_packages(
    rows: &[CertificateSliceGroup],
    files: &mut Vec<super::GeneratedLeanFile>,
) -> Result<(), String> {
    let row_values = rows
        .iter()
        .enumerate()
        .map(|(row, value)| render_slice_lut_row_packages(row, &value.rows, files))
        .collect::<Result<Vec<_>, _>>()?;
    for (package, chunk) in rows.chunks(PACKAGE_SIZE).enumerate() {
        let module = format!("SliceGroup{package:03}");
        let start = package * PACKAGE_SIZE;
        let mut imports = vec!["Mxx.Certificate.OperationalNoise.CertificateABI".to_owned()];
        imports.extend(row_values[start..start + chunk.len()].iter().filter_map(|(_, root)| {
            root.as_ref().map(|root| format!("{MODULE_ROOT}.Cert.{root}"))
        }));
        let mut source =
            imports.iter().map(|import| format!("import {import}")).collect::<Vec<_>>().join("\n");
        write!(
            source,
            "\n\nset_option autoImplicit false\nset_option relaxedAutoImplicit false\n\nnamespace {NAMESPACE}.Cert.{module}\n\nopen Mxx.Certificate.OperationalNoise\nopen SchemaV1\nopen CertificateABI\n\n"
        )
        .expect("writing to String cannot fail");
        for (offset, row) in chunk.iter().enumerate() {
            let row_id = start + offset;
            writeln!(
                source,
                "def SliceGroupRow{row_id} : SchemaV1.SliceGroupRow :=\n  {}\n",
                slice_group_row(row, &row_values[row_id].0)?
            )
            .expect("writing to String cannot fail");
        }
        writeln!(source, "end {NAMESPACE}.Cert.{module}").expect("writing to String cannot fail");
        files.push(generated_file(format!("Cert/{module}.lean"), source));
    }
    Ok(())
}

fn render_slice_lut_row_packages(
    row: usize,
    rows: &[SliceLutRow],
    files: &mut Vec<super::GeneratedLeanFile>,
) -> Result<(String, Option<String>), String> {
    if rows.len() <= SLICE_LUT_ROW_LEAF_SIZE {
        return Ok((list(rows, slice_lut_row)?, None));
    }

    let mut children = Vec::new();
    for (leaf, values) in rows.chunks(SLICE_LUT_ROW_LEAF_SIZE).enumerate() {
        let module = format!("SliceGroupRows{row}Leaf{leaf:04}");
        let mut source = header("Mxx.Certificate.OperationalNoise.CertificateABI", &module);
        writeln!(
            source,
            "def rows : List SchemaV1.SliceLutRow := {}\n\nend {NAMESPACE}.Cert.{module}",
            list(values, slice_lut_row)?
        )
        .expect("writing to String cannot fail");
        files.push(generated_file(format!("Cert/{module}.lean"), source));
        children.push(module);
    }

    let mut level = 0;
    while children.len() > 1 {
        let mut parents = Vec::new();
        for (node, group) in children.chunks(SLICE_LUT_ROW_FANOUT).enumerate() {
            let module = format!("SliceGroupRows{row}Level{level}_{node:04}");
            let mut source = group
                .iter()
                .map(|child| format!("import {MODULE_ROOT}.Cert.{child}"))
                .collect::<Vec<_>>()
                .join("\n");
            write!(
                source,
                "\n\nset_option autoImplicit false\nset_option relaxedAutoImplicit false\n\nnamespace {NAMESPACE}.Cert.{module}\n\nopen Mxx.Certificate.OperationalNoise\nopen SchemaV1\n\ndef rows : List SchemaV1.SliceLutRow :=\n  {}\n\nend {NAMESPACE}.Cert.{module}",
                group
                    .iter()
                    .map(|child| format!("{NAMESPACE}.Cert.{child}.rows"))
                    .collect::<Vec<_>>()
                    .join(" ++\n    ")
            )
            .expect("writing to String cannot fail");
            files.push(generated_file(format!("Cert/{module}.lean"), source));
            parents.push(module);
        }
        children = parents;
        level += 1;
    }
    let root = children
        .into_iter()
        .next()
        .ok_or_else(|| "large slice LUT row collection had no leaves".to_owned())?;
    Ok((format!("{NAMESPACE}.Cert.{root}.rows"), Some(root)))
}

fn header(import: &str, suffix: &str) -> String {
    format!(
        "import {import}\n\nset_option autoImplicit false\nset_option relaxedAutoImplicit false\n\nnamespace {NAMESPACE}.Cert.{suffix}\n\nopen Mxx.Certificate.OperationalNoise\nopen SchemaV1\nopen CertificateABI\n\n"
    )
}

fn render_packages<T>(
    label: &str,
    row_type: &str,
    rows: &[T],
    render: fn(&T) -> Result<String, String>,
    files: &mut Vec<super::GeneratedLeanFile>,
) -> Result<(), String> {
    for (package, chunk) in rows.chunks(PACKAGE_SIZE).enumerate() {
        let module = format!("{label}{package:03}");
        let mut source = header("Mxx.Certificate.OperationalNoise.CertificateABI", &module);
        let start = package * PACKAGE_SIZE;
        for (offset, row) in chunk.iter().enumerate() {
            writeln!(
                source,
                "def {label}Row{} : SchemaV1.{row_type} :=\n  {}\n",
                start + offset,
                render(row)?
            )
            .expect("writing to String cannot fail");
        }
        writeln!(source, "end {NAMESPACE}.Cert.{module}").expect("writing to String cannot fail");
        files.push(generated_file(format!("Cert/{module}.lean"), source));
    }
    Ok(())
}

fn render_top(document: &CertificateDocumentV1) -> Result<String, String> {
    let ciphertext_modulus = super::semantics::ciphertext_modulus_text(document)?;
    let tables = [
        ("Expression", "CertificateABI.ExpressionRow", document.expressions.len()),
        ("Program", "SchemaV1.ProgramRow", document.programs.len()),
        ("Source", "SchemaV1.SourceRow", document.sources.len()),
        ("Event", "SchemaV1.EventRow", document.events.len()),
        ("IndexUse", "CertificateABI.IndexUseRow", document.index_uses.len()),
        ("SliceGroup", "SchemaV1.SliceGroupRow", document.slice_groups.len()),
    ];
    let mut source = String::new();
    for (label, _, count) in tables {
        for package in 0..count.div_ceil(PACKAGE_SIZE) {
            writeln!(source, "import {MODULE_ROOT}.Cert.{label}{package:03}")
                .expect("writing to String cannot fail");
        }
    }
    source.push_str("\nset_option autoImplicit false\nset_option relaxedAutoImplicit false\n\n");
    writeln!(source, "namespace {NAMESPACE}\n").expect("writing to String cannot fail");
    source.push_str("open Mxx.Certificate.OperationalNoise\nopen SchemaV1\nopen CertificateABI\n");
    for (label, _, _) in tables {
        for package in 0..match label {
            "Expression" => document.expressions.len(),
            "Program" => document.programs.len(),
            "Source" => document.sources.len(),
            "Event" => document.events.len(),
            "IndexUse" => document.index_uses.len(),
            _ => document.slice_groups.len(),
        }
        .div_ceil(PACKAGE_SIZE)
        {
            writeln!(source, "open Cert.{label}{package:03}")
                .expect("writing to String cannot fail");
        }
    }
    source.push('\n');
    for (label, row_type, count) in tables {
        let root = render_balanced_table(label, row_type, count, &mut source);
        writeln!(source, "def {label}Rows : RowTable {row_type} := {root}\n")
            .expect("writing to String cannot fail");
    }
    let residual = residual_root(&document.residual_root);
    let CertificateResidualRootV1::Family { domain, .. } = document.residual_root else {
        return Err(
            "certificate reached statement rendering requires a family residual root".to_owned()
        );
    };
    writeln!(source, "def selectorMinimum : Nat := {}", domain.minimum)
        .expect("writing to String cannot fail");
    writeln!(source, "def selectorMaximum : Nat := {}", domain.maximum_exclusive)
        .expect("writing to String cannot fail");
    writeln!(source, "def ringDimension : Nat := {}\n", document.ring_dimension)
        .expect("writing to String cannot fail");
    writeln!(
        source,
        "def document : CertificateDocument :=\n  {{ schemaId := {}\n    schemaVersion := {}\n    plaintextModulus := {}\n    ciphertextModulus := toString {}\n    ringDimension := {}\n    expressions := ExpressionRows\n    programs := ProgramRows\n    sources := SourceRows\n    events := EventRows\n    indexUses := IndexUseRows\n    sliceGroups := SliceGroupRows\n    residualRoot := {residual} }}\n\nend {NAMESPACE}",
        quoted(document.schema_id)?,
        document.schema_version,
        quoted(&document.plaintext_modulus)?,
        ciphertext_modulus,
        document.ring_dimension,
    )
    .expect("writing to String cannot fail");
    Ok(source)
}

fn render_balanced_table(label: &str, row_type: &str, count: usize, out: &mut String) -> String {
    fn node(
        label: &str,
        row_type: &str,
        start: usize,
        end: usize,
        depth: usize,
        out: &mut String,
    ) -> String {
        if start == end {
            return ".empty".to_owned();
        }
        let middle = (start + end) / 2;
        if depth == 4 {
            let name = format!("{label}Tree{middle}");
            let value = node(label, row_type, start, end, 0, out);
            writeln!(out, "def {name} : RowTable {row_type} := {value}")
                .expect("writing to String cannot fail");
            return name;
        }
        let left = node(label, row_type, start, middle, depth + 1, out);
        let right = node(label, row_type, middle + 1, end, depth + 1, out);
        format!("(.node {middle} {label}Row{middle} {left} {right})")
    }
    node(label, row_type, 0, count, 0, out)
}

fn range(value: &CertificateRange) -> String {
    format!("⟨{}, {}⟩", value.minimum, value.maximum_exclusive)
}

fn value_type(value: &StableValueType) -> Result<String, String> {
    Ok(match value {
        StableValueType::Bool => ".bool".to_owned(),
        StableValueType::Int => ".int".to_owned(),
        StableValueType::Real => ".real".to_owned(),
        StableValueType::Bytes => ".bytes".to_owned(),
        StableValueType::Trapdoor => ".trapdoor".to_owned(),
        StableValueType::Matrix { modulus, ring_dimension, rows, columns } => {
            format!(".matrix {} {ring_dimension} {rows} {columns}", quoted(modulus)?)
        }
    })
}

fn constant(value: &StableConstant) -> Result<String, String> {
    let item = match &value.value {
        StableConstantValue::Bool { value } => format!(".bool {}", super::bool_text(*value)),
        StableConstantValue::Int { value } => format!(".int {}", quoted(value)?),
        StableConstantValue::Real { value } => format!(".real {}", quoted(value)?),
        StableConstantValue::Bytes { value } => {
            format!(".bytes {}", list(value, |item| Ok(item.to_string()))?)
        }
    };
    Ok(format!("⟨{}, {item}⟩", value_type(&value.value_type)?))
}

fn event_ref(row: u64) -> String {
    format!("⟨{row}⟩")
}

fn artifact(value: &StableArtifact) -> Result<String, String> {
    let domain =
        option(value.domain.as_ref(), |&(minimum, maximum)| Ok(format!("({minimum}, {maximum})")))?;
    Ok(format!(
        "⟨{}, {}, {}, {}, {}, {domain}⟩",
        quoted(&value.definition)?,
        value.version,
        value.confidentiality,
        value_type(&value.value_type)?,
        quoted(&value.layout)?,
    ))
}

fn matrix_constant(value: &StableMatrixConstantKind) -> Result<String, String> {
    Ok(match value {
        StableMatrixConstantKind::Zero => ".zero".to_owned(),
        StableMatrixConstantKind::Identity => ".identity".to_owned(),
        StableMatrixConstantKind::UnitRow { index } => format!(".unitRow {index}"),
        StableMatrixConstantKind::UnitColumn { index } => format!(".unitColumn {index}"),
        StableMatrixConstantKind::Gadget { base, small } => {
            format!(".gadget {base} {}", super::bool_text(*small))
        }
        StableMatrixConstantKind::PowerOfBase { base, exponent } => {
            format!(".powerOfBase {} {}", quoted(base)?, quoted(exponent)?)
        }
        StableMatrixConstantKind::Rotation { exponent } => format!(".rotation {exponent}"),
        StableMatrixConstantKind::Polynomial { coefficients } => {
            format!(".polynomial {}", list(coefficients, |item| quoted(item))?)
        }
    })
}

fn sample_descriptor(value: &StableSampleDescriptor) -> Result<String, String> {
    Ok(format!(
        "⟨{}, {}, {}, {}, {}, {}⟩",
        quoted(&value.definition)?,
        list(&value.parameters, |item| Ok(item.to_string()))?,
        value_type(&value.output_type)?,
        option(value.gadget_base.as_ref(), |item| quoted(item))?,
        option(value.digit_count.as_ref(), |item| Ok(item.to_string()))?,
        option(value.decomposition.as_ref(), |item| quoted(item))?,
    ))
}

fn source_identity(
    value: &crate::operational_noise::g0::StableSourceIdentity,
) -> Result<String, String> {
    Ok(format!(
        "⟨{}, {}, {}, {}, {}, {}, {}⟩",
        quoted(&value.definition)?,
        option(value.sample_event.as_ref(), |event| Ok(event_ref(event.row)))?,
        quoted(&value.output_role)?,
        option(value.artifact.as_ref(), artifact)?,
        value_type(&value.value_type)?,
        list(&value.coordinates, |item| Ok(item.to_string()))?,
        option(value.matrix_constant.as_ref(), matrix_constant)?,
    ))
}

fn family_identity(value: &StableFamilySourceIdentity) -> Result<String, String> {
    Ok(format!(
        "⟨{}, {}, {}, ({}, {}), {}⟩",
        quoted(&value.definition)?,
        quoted(&value.invocation)?,
        value_type(&value.element_type)?,
        value.domain.0,
        value.domain.1,
        option(value.artifact.as_ref(), artifact)?,
    ))
}

fn scope(value: &StableScope) -> Result<String, String> {
    Ok(match value {
        StableScope::Root => ".root".to_owned(),
        StableScope::Subgraph { canonical_name } => {
            format!(".subgraph {}", quoted(canonical_name)?)
        }
        StableScope::ParallelBody { parent, owner } => {
            format!(".parallelBody ({}) {owner}", scope(parent)?)
        }
        StableScope::SequentialBody { parent, owner } => {
            format!(".sequentialBody ({}) {owner}", scope(parent)?)
        }
    })
}

fn observed_wire(value: &StableObservedWire) -> Result<String, String> {
    Ok(format!(
        "⟨{}, {}, {}, {}, {}⟩",
        quoted(&value.stage)?,
        scope(&value.definition)?,
        value.path,
        value.node,
        value.port,
    ))
}

fn observed_producer(value: &StableObservedProducer) -> Result<String, String> {
    Ok(format!(
        "⟨{}, {}, {}, {}, {}⟩",
        observed_wire(&value.consumer)?,
        quoted(&value.consumer_input)?,
        quoted(&value.producer_stage)?,
        quoted(&value.producer_output)?,
        observed_wire(&value.producer)?,
    ))
}

fn source_access(value: &StableObservedSourceAccess) -> Result<String, String> {
    Ok(match value {
        StableObservedSourceAccess::DeclaredProtocolInput { owner, input } => {
            format!(".declaredProtocolInput ({}) {}", observed_wire(owner)?, quoted(input)?)
        }
        StableObservedSourceAccess::UnboundOccurrenceInput { owner } => {
            format!(".unboundOccurrenceInput ({})", observed_wire(owner)?)
        }
        StableObservedSourceAccess::ProducerArtifact { producer } => {
            format!(".producerArtifact ({})", observed_producer(producer)?)
        }
    })
}

fn raw_contract(value: &RawValueContractV1) -> Result<String, String> {
    let signed = option(value.signed_range.as_ref(), |range| {
        Ok(format!("⟨{}, {}⟩", quoted(&range.minimum)?, quoted(&range.max_exclusive)?))
    })?;
    let coefficient = option(value.coefficient_class.as_ref(), |class| {
        Ok(match class {
            RawCoefficientClassV1::ExactZero => ".exactZero".to_owned(),
            RawCoefficientClassV1::Finite { maximum_absolute_coefficient } => {
                format!(".finite {}", quoted(maximum_absolute_coefficient)?)
            }
            RawCoefficientClassV1::Large => ".large".to_owned(),
        })
    })?;
    Ok(format!(
        "⟨{signed}, {coefficient}, {}, {}⟩",
        option(value.canonical_coefficient_exclusive_upper.as_ref(), |item| quoted(item))?,
        option(value.polynomial_support_upper.as_ref(), |item| Ok(item.to_string()))?,
    ))
}

fn source_row(value: &CertificateSourceRowV1) -> Result<String, String> {
    Ok(match value {
        CertificateSourceRowV1::Constant { value } => format!(".constant ({})", constant(value)?),
        CertificateSourceRowV1::Direct { identity, access, contract } => format!(
            ".direct ({}) ({}) ({})",
            source_identity(identity)?,
            option(access.as_ref(), source_access)?,
            option(contract.as_ref(), raw_contract)?,
        ),
        CertificateSourceRowV1::Family { identity, contract } => format!(
            ".family ({}) ({})",
            family_identity(identity)?,
            option(contract.as_ref(), raw_contract)?,
        ),
    })
}

fn sampler(value: &StableSamplerOperation) -> Result<String, String> {
    Ok(match value {
        StableSamplerOperation::UniformResidue { output } => {
            format!(".uniformResidue ({})", value_type(output)?)
        }
        StableSamplerOperation::UniformInterval { output, minimum, maximum } => format!(
            ".uniformInterval ({}) {} {}",
            value_type(output)?,
            quoted(minimum)?,
            quoted(maximum)?
        ),
        StableSamplerOperation::Gaussian { output, sigma, max_coefficient_bound } => format!(
            ".gaussian ({}) {} {}",
            value_type(output)?,
            quoted(sigma)?,
            quoted(max_coefficient_bound)?
        ),
        StableSamplerOperation::Hash {
            output,
            variant,
            tag_prefix,
            tag_expressions,
            tag_decimal_expressions,
            tag_u64_le_expressions,
            base,
            digit_count,
        } => format!(
            ".hash ({}) {} {} {} {} {} {} {}",
            value_type(output)?,
            hash_variant(*variant),
            list(tag_prefix, |item| Ok(item.to_string()))?,
            ref_list(tag_expressions),
            ref_list(tag_decimal_expressions),
            ref_list(tag_u64_le_expressions),
            option(base.as_ref(), |item| Ok(item.to_string()))?,
            option(digit_count.as_ref(), |item| Ok(item.to_string()))?,
        ),
        StableSamplerOperation::Trapdoor {
            output,
            sigma,
            gadget_base,
            digit_count,
            preimage_max_coefficient_bound,
        } => format!(
            ".trapdoor ({}) {} {gadget_base} {digit_count} {}",
            value_type(output)?,
            quoted(sigma)?,
            quoted(preimage_max_coefficient_bound)?
        ),
        StableSamplerOperation::Preimage { output, max_coefficient_bound } => {
            format!(".preimage ({}) {}", value_type(output)?, quoted(max_coefficient_bound)?)
        }
    })
}

fn hash_variant(value: StableHashVariant) -> &'static str {
    match value {
        StableHashVariant::Plain => ".plain",
        StableHashVariant::Decomposed => ".decomposed",
        StableHashVariant::SmallDecomposed => ".smallDecomposed",
    }
}

fn event_row(value: &CertificateEventRowV1) -> Result<String, String> {
    Ok(match value {
        CertificateEventRowV1::Sample { owner, descriptor, contract } => format!(
            ".sample ({}) ({}) ({})",
            observed_wire(owner)?,
            sample_descriptor(descriptor)?,
            option(contract.as_ref(), raw_contract)?
        ),
        CertificateEventRowV1::Sampler { owner, operation, contract } => format!(
            ".sampler ({}) ({}) ({})",
            observed_wire(owner)?,
            sampler(operation)?,
            option(contract.as_ref(), raw_contract)?
        ),
        CertificateEventRowV1::GadgetDecompose {
            scope: owner_scope,
            expression,
            output,
            base,
            small,
            digit_count,
            input,
            contract,
        } => format!(
            ".gadgetDecompose ({}) ⟨{expression}⟩ ({}) {base} {} {digit_count} ⟨{input}⟩ ({})",
            statement_scope(*owner_scope),
            value_type(output)?,
            super::bool_text(*small),
            option(contract.as_ref(), raw_contract)?,
        ),
    })
}

fn statement_scope(value: CanonicalStatementScope) -> String {
    match value {
        CanonicalStatementScope::Closed { root } => format!(".closed ⟨{root}⟩"),
        CanonicalStatementScope::Program { program } => format!(".program ⟨{program}⟩"),
    }
}

fn ref_list(values: &[u64]) -> String {
    format!("[{}]", values.iter().map(|value| event_ref(*value)).collect::<Vec<_>>().join(", "))
}

fn expression_row(value: &CertificateExpressionRow, row: usize) -> Result<String, String> {
    Ok(format!(
        "⟨{}, {}, {}⟩",
        expression_descriptor(&value.descriptor)?,
        format!("ExpressionInputs{row}"),
        value.program.map_or_else(|| "none".to_owned(), |row| format!("some ⟨{row}⟩")),
    ))
}

fn expression_descriptor(value: &CanonicalExpressionDescriptor) -> Result<String, String> {
    Ok(match value {
        CanonicalExpressionDescriptor::Source { source } => match source {
            CanonicalExpressionSource::Direct { source } => {
                format!(".source (.direct ⟨{source}⟩)")
            }
            CanonicalExpressionSource::Family { source, selector } => {
                format!(".source (.family ⟨{source}⟩ ⟨{selector}⟩)")
            }
        },
        CanonicalExpressionDescriptor::Event { operator } => {
            format!(".event ({})", event_operator(operator))
        }
        CanonicalExpressionDescriptor::Operation { operator, value_type: output } => {
            format!(".operation ({}) ({})", expression_operator(operator)?, value_type(output)?)
        }
    })
}

fn event_operator(value: &CanonicalEventOperator) -> String {
    match value {
        CanonicalEventOperator::Sample { event } => format!(".sample ⟨{}⟩", event.row),
        CanonicalEventOperator::Sampler { event } => format!(".sampler ⟨{}⟩", event.row),
        CanonicalEventOperator::GadgetDecompose { events } => format!(
            ".gadgetDecompose [{}]",
            events.iter().map(|event| event_ref(event.row)).collect::<Vec<_>>().join(", ")
        ),
    }
}

fn expression_operator(value: &CanonicalExpressionOperator) -> Result<String, String> {
    match value {
        CanonicalExpressionOperator::Stable(value) => {
            Ok(format!(".stable ({})", stable_operator(value)?))
        }
        CanonicalExpressionOperator::Event(value) => {
            Ok(format!(".event ({})", event_operator(value)))
        }
    }
}

fn stable_operator(value: &crate::operational_noise::g0::StableOperator) -> Result<String, String> {
    use crate::operational_noise::g0::StableOperator as O;
    Ok(match value {
        O::Argument { position, value_type: output } => {
            format!(".argument {position} ({})", value_type(output)?)
        }
        O::Constant { value } => format!(".constant ({})", constant(value)?),
        O::Source { identity } => format!(".source ({})", source_identity(identity)?),
        O::Sample { event, descriptor } => format!(
            ".sample ({}) ({})",
            option(event.as_ref(), |event| Ok(event_ref(event.row)))?,
            sample_descriptor(descriptor)?
        ),
        O::Sampler { event, operation } => format!(
            ".sampler ({}) ({})",
            option(event.as_ref(), |event| Ok(event_ref(event.row)))?,
            sampler(operation)?
        ),
        O::DeterministicHash {
            definition,
            version,
            key_byte_length,
            output,
            tag_prefix,
            binary_tag_count,
            decimal_tag_count,
            u64_le_tag_count,
            dynamic_tag_count,
        } => format!(
            ".deterministicHash {} {version} {key_byte_length} ({}) {} {binary_tag_count} {decimal_tag_count} {u64_le_tag_count} {dynamic_tag_count}",
            match definition {
                StableHashDefinition::MxxPolynomialHash => ".mxxPolynomialHash",
            },
            value_type(output)?,
            list(tag_prefix, |item| Ok(item.to_string()))?,
        ),
        O::OpaqueFamilyElement { identity } => {
            format!(".opaqueFamilyElement ({})", family_identity(identity)?)
        }
        O::IndexMap { definition, parameters } => {
            format!(".indexMap {definition} {}", list(parameters, |item| Ok(item.to_string()))?)
        }
        O::ExplicitElement { domain, element_type } => {
            format!(".explicitElement ({}, {}) ({})", domain.0, domain.1, value_type(element_type)?)
        }
        O::ProgramCall => ".programCall".to_owned(),
        O::Transform { operation } => format!(".transform ({})", transform(operation)?),
        O::ExtractCoefficient { position, canonical_input_exclusive_upper } => format!(
            ".extractCoefficient {position} ({})",
            option(canonical_input_exclusive_upper.as_ref(), |item| quoted(item))?
        ),
        O::Scalar { operation } => format!(".scalar ({})", scalar(operation)?),
        O::Matrix { operation } => format!(".matrix ({})", matrix(operation)?),
        O::Trapdoor { operation } => format!(".trapdoor ({})", trapdoor(operation)?),
    })
}

fn scalar(value: &StableScalarOperation) -> Result<String, String> {
    use StableScalarOperation as O;
    Ok(match value {
        O::Add => ".add".to_owned(),
        O::Subtract => ".subtract".to_owned(),
        O::Multiply => ".multiply".to_owned(),
        O::Divide => ".divide".to_owned(),
        O::Remainder => ".remainder".to_owned(),
        O::Negate => ".negate".to_owned(),
        O::Equal => ".equal".to_owned(),
        O::Less => ".less".to_owned(),
        O::LessEqual => ".lessEqual".to_owned(),
        O::BoolToInt => ".boolToInt".to_owned(),
        O::IntToReal => ".intToReal".to_owned(),
        O::RealAdd => ".realAdd".to_owned(),
        O::RealSubtract => ".realSubtract".to_owned(),
        O::RealMultiply => ".realMultiply".to_owned(),
        O::RealDivide => ".realDivide".to_owned(),
        O::RealSqrt => ".realSqrt".to_owned(),
        O::ThresholdDecode { plaintext_modulus, length, output_bool } => format!(
            ".thresholdDecode {} {length} {}",
            quoted(plaintext_modulus)?,
            super::bool_text(*output_bool)
        ),
        O::Bit { position } => format!(".bit {position}"),
        O::Slice { start, end_exclusive } => format!(".slice {start} {end_exclusive}"),
        O::Hash { tag, dynamic_tags } => {
            format!(".hash {} {}", quoted(tag)?, list(dynamic_tags, |item| Ok(item.to_string()))?)
        }
        O::ExtractCoefficient { row, column } => format!(".extractCoefficient {row} {column}"),
        O::LiftConstantPolynomial { output, coefficient_bits } => {
            format!(".liftConstantPolynomial ({}) {coefficient_bits}", value_type(output)?)
        }
    })
}

fn layout(value: &StableLayout) -> Result<String, String> {
    Ok(format!("⟨{}, {}, {}⟩", quoted(&value.name)?, value.row_stride, value.column_stride))
}

fn matrix(value: &StableMatrixOperation) -> Result<String, String> {
    use StableMatrixOperation as O;
    Ok(match value {
        O::Add => ".add".to_owned(),
        O::Subtract => ".subtract".to_owned(),
        O::Multiply => ".multiply".to_owned(),
        O::Negate => ".negate".to_owned(),
        O::Scale => ".scale".to_owned(),
        O::RingAutomorphism { index } => format!(".ringAutomorphism {index}"),
        O::Transpose => ".transpose".to_owned(),
        O::Slice {
            row_start,
            row_end_exclusive,
            column_start,
            column_end_exclusive,
            layout: value,
        } => format!(
            ".slice {row_start} {row_end_exclusive} {column_start} {column_end_exclusive} ({})",
            layout(value)?
        ),
        O::IndexedSlice { output, layout: value } => {
            format!(".indexedSlice ({}) ({})", value_type(output)?, layout(value)?)
        }
        O::View { output, layout: value } => {
            format!(".view ({}) ({})", value_type(output)?, layout(value)?)
        }
        O::Concat { axis, output, layout: value } => {
            format!(".concat {axis} ({}) ({})", value_type(output)?, layout(value)?)
        }
        O::Tensor { output, left_layout, right_layout, output_layout } => format!(
            ".tensor ({}) ({}) ({}) ({})",
            value_type(output)?,
            layout(left_layout)?,
            layout(right_layout)?,
            layout(output_layout)?
        ),
        O::CrtRecompose { plaintext_moduli, reconstruction_coefficients, output } => format!(
            ".crtRecompose {} {} ({})",
            list(plaintext_moduli, |item| quoted(item))?,
            list(reconstruction_coefficients, |item| quoted(item))?,
            value_type(output)?
        ),
        O::ExtractCoefficient { row, column } => format!(".extractCoefficient {row} {column}"),
        O::LiftConstantPolynomial { output, coefficient_bits } => {
            format!(".liftConstantPolynomial ({}) {coefficient_bits}", value_type(output)?)
        }
    })
}

fn transform(value: &StableTransformOperation) -> Result<String, String> {
    Ok(match value {
        StableTransformOperation::GadgetDecompose { output, base, small, digit_count } => format!(
            ".gadgetDecompose ({}) {base} {} {digit_count}",
            value_type(output)?,
            super::bool_text(*small)
        ),
        StableTransformOperation::PackPolynomialCoefficients { output, coefficient_bits } => {
            format!(".packPolynomialCoefficients ({}) {coefficient_bits}", value_type(output)?)
        }
    })
}

fn trapdoor(value: &StableTrapdoorOperation) -> Result<String, String> {
    Ok(match value {
        StableTrapdoorOperation::Generate {
            descriptor,
            parameters,
            paired_public_event,
            paired_public_output_role,
        } => format!(
            ".generate {} {} ({}) {}",
            quoted(descriptor)?,
            list(parameters, |item| Ok(item.to_string()))?,
            option(paired_public_event.as_ref(), |event| Ok(event_ref(event.row)))?,
            quoted(paired_public_output_role)?
        ),
        StableTrapdoorOperation::Transform { descriptor, output, parameters } => format!(
            ".transform {} ({}) {}",
            quoted(descriptor)?,
            value_type(output)?,
            list(parameters, |item| Ok(item.to_string()))?
        ),
    })
}

fn program_row(value: &CertificateProgramRow) -> Result<String, String> {
    let signature = list(&value.signature, |input| {
        Ok(format!(
            "⟨{}, {}⟩",
            value_type(&input.value_type)?,
            option(input.trusted_index_range.as_ref(), |item| Ok(range(item)))?
        ))
    })?;
    let family = option(value.family.as_ref(), |family| {
        Ok(format!(
            "⟨{}, {}, {}, {}⟩",
            range(&family.domain),
            value_type(&family.element_type)?,
            super::bool_text(family.reducible),
            option(family.artifact.as_ref(), artifact)?,
        ))
    })?;
    Ok(format!("⟨{signature}, {}, {family}, ⟨{}⟩⟩", value_type(&value.output)?, value.root))
}

fn plan_ref(value: &StablePlanRef) -> String {
    match value {
        StablePlanRef::Expression { row } => format!(".expression ⟨{row}⟩"),
        StablePlanRef::Family { row } => format!(".family ⟨{row}⟩"),
    }
}

fn occurrence(value: &StableObservedOccurrence) -> Result<String, String> {
    Ok(format!("⟨{}, {}⟩", scope(&value.definition)?, value.path))
}

fn frontier(value: &StableFrontierAxis) -> Result<String, String> {
    Ok(match value {
        StableFrontierAxis::Argument { owner, expression, position, domain } => format!(
            ".argument ({}) ({}) {position} ({}, {})",
            occurrence(owner)?,
            plan_ref(expression),
            domain.0,
            domain.1
        ),
        StableFrontierAxis::ExtractedCoefficient { owner, expression, domain } => format!(
            ".extractedCoefficient ({}) ({}) ({}, {})",
            occurrence(owner)?,
            plan_ref(expression),
            domain.0,
            domain.1
        ),
    })
}

fn index_use_row(value: &CertificateIndexUse, rows: &str) -> Result<String, String> {
    let kind = match value.kind {
        IndexUseKind::IntegerExpression => ".integerExpression",
        IndexUseKind::FamilyGetStatic => ".familyGetStatic",
        IndexUseKind::FamilyGetDynamic => ".familyGetDynamic",
        IndexUseKind::Select => ".select",
        IndexUseKind::IndexedSlice => ".indexedSlice",
    };
    Ok(format!(
        "⟨{}, {}, {}, {kind}, {}, {}, {}, {rows}⟩",
        observed_wire(&value.owner)?,
        option(value.result.as_ref(), |item| Ok(plan_ref(item)))?,
        option(value.consumed.as_ref(), |item| Ok(plan_ref(item)))?,
        option(value.output_range.as_ref(), |item| Ok(range(item)))?,
        value_type(&value.output_type)?,
        list(&value.frontier, frontier)?,
    ))
}

fn slice_group_row(value: &CertificateSliceGroup, rows: &str) -> Result<String, String> {
    Ok(format!(
        "⟨{}, {}, {}, {}, {}, {}, {}, {}, {}⟩",
        observed_wire(&value.owner)?,
        option(value.result.as_ref(), |item| Ok(plan_ref(item)))?,
        option(value.consumed.as_ref(), |item| Ok(plan_ref(item)))?,
        value_type(&value.output_type)?,
        list(&value.frontier, frontier)?,
        option(value.row_span.as_ref(), |item| Ok(item.to_string()))?,
        option(value.column_span.as_ref(), |item| Ok(item.to_string()))?,
        list(&value.members, |member| {
            let role = match member.role {
                SliceMemberRole::RowStart => ".rowStart",
                SliceMemberRole::RowEndExclusive => ".rowEndExclusive",
                SliceMemberRole::ColumnStart => ".columnStart",
                SliceMemberRole::ColumnEndExclusive => ".columnEndExclusive",
            };
            Ok(format!("⟨{role}, {}, {}⟩", plan_ref(&member.expression), range(&member.range)))
        })?,
        rows,
    ))
}

fn slice_lut_row(row: &SliceLutRow) -> Result<String, String> {
    Ok(format!(
        "⟨{}, {}, {}, {}, {}⟩",
        list(&row.tuple, |item| quoted(item))?,
        quoted(&row.row_start)?,
        quoted(&row.row_end_exclusive)?,
        quoted(&row.column_start)?,
        quoted(&row.column_end_exclusive)?
    ))
}

fn residual_root(value: &CertificateResidualRootV1) -> String {
    match value {
        CertificateResidualRootV1::Closed { expression } => format!(".closed ⟨{expression}⟩"),
        CertificateResidualRootV1::Family { program, domain } => {
            format!(".family ⟨{program}⟩ {}", range(domain))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_document(ciphertext_modulus: &str) -> CertificateDocumentV1 {
        CertificateDocumentV1 {
            schema_id: "mxx.operational-noise.certificate",
            schema_version: 1,
            plaintext_modulus: "2".to_owned(),
            ciphertext_modulus: ciphertext_modulus.to_owned(),
            ring_dimension: 1,
            expressions: Vec::new(),
            programs: Vec::new(),
            sources: Vec::new(),
            events: Vec::new(),
            index_uses: Vec::new(),
            slice_groups: Vec::new(),
            residual_root: CertificateResidualRootV1::Family {
                program: 0,
                domain: CertificateRange { minimum: 0, maximum_exclusive: 1 },
            },
        }
    }

    #[test]
    fn render_top_rejects_noncanonical_ciphertext_modulus() {
        let source = render_top(&minimal_document("257")).expect("canonical modulus renders");
        assert!(source.contains("ciphertextModulus := toString 257"));
        for value in ["0257", "0", "not-a-decimal"] {
            assert!(render_top(&minimal_document(value)).is_err(), "accepted {value:?}");
        }
    }

    fn lut_row(tuple: &[&str], output: &str) -> IndexLutRow {
        IndexLutRow {
            tuple: tuple.iter().map(|item| (*item).to_owned()).collect(),
            output: output.to_owned(),
        }
    }

    fn slice_row(value: usize) -> SliceLutRow {
        let value = value.to_string();
        SliceLutRow {
            tuple: vec![value.clone()],
            row_start: value.clone(),
            row_end_exclusive: value.clone(),
            column_start: value.clone(),
            column_end_exclusive: value,
        }
    }

    #[test]
    fn expression_inputs_use_one_fixed_type_at_zero_and_leaf_boundaries() {
        let mut zero = String::new();
        render_expression_inputs(7, &[], &mut zero);
        assert_eq!(zero, "def ExpressionInputs7 : ExpressionInputs := ⟨.empty, 0⟩\n\n");

        let values = (0_u64..16).collect::<Vec<_>>();
        let mut one_leaf = String::new();
        render_expression_inputs(8, &values, &mut one_leaf);
        assert!(one_leaf.starts_with("def ExpressionInputs8 : ExpressionInputs :=\n"));
        assert!(one_leaf.contains("(.node 0 #[⟨0⟩, ⟨1⟩"));
        assert!(one_leaf.contains("⟨14⟩, ⟨15⟩] .empty .empty), 16⟩"));
        assert!(!one_leaf.contains("ExpressionInputLeaf"));
    }

    #[test]
    fn expression_inputs_preserve_duplicates_order_and_fixed_sixteen_element_leaves() {
        let values = vec![9, 4, 9, 3, 2, 1, 0, 8, 7, 6, 5, 11, 10, 13, 12, 14, 9];
        let mut source = String::new();
        render_expression_inputs(12, &values, &mut source);
        assert!(source.contains(
            "ExpressionInputLeaf12_0 : Array ExpressionRef := #[⟨9⟩, ⟨4⟩, ⟨9⟩, ⟨3⟩, ⟨2⟩, ⟨1⟩, ⟨0⟩, ⟨8⟩, ⟨7⟩, ⟨6⟩, ⟨5⟩, ⟨11⟩, ⟨10⟩, ⟨13⟩, ⟨12⟩, ⟨14⟩]"
        ));
        assert!(source.contains("ExpressionInputLeaf12_1 : Array ExpressionRef := #[⟨9⟩]"));
        assert!(source.contains("ExpressionInputs12 : ExpressionInputs"));
        assert!(source.contains(", 17⟩"));
    }

    #[test]
    fn actual_scale_explicit_element_inputs_are_balanced_without_remapping() {
        let values = (0_u64..1_153).collect::<Vec<_>>();
        let mut source = String::new();
        render_expression_inputs(9_307, &values, &mut source);
        assert_eq!(source.matches("def ExpressionInputLeaf9307_").count(), 73);
        assert!(source.contains("ExpressionInputLeaf9307_0 : Array ExpressionRef := #[⟨0⟩, ⟨1⟩"));
        assert!(source.contains("⟨14⟩, ⟨15⟩]"));
        assert!(source.contains("ExpressionInputLeaf9307_1 : Array ExpressionRef := #[⟨16⟩, ⟨17⟩"));
        assert!(source.contains("ExpressionInputLeaf9307_72 : Array ExpressionRef := #[⟨1152⟩]"));
        assert!(source.contains("def ExpressionInputs9307 : ExpressionInputs :="));
        assert!(source.contains(", 1153⟩"));
        assert!(!source.contains("[⟨0⟩, ⟨1⟩, ⟨2⟩, ⟨3⟩, ⟨4⟩, ⟨5⟩, ⟨6⟩, ⟨7⟩, ⟨8⟩, ⟨9⟩, ⟨10⟩, ⟨11⟩, ⟨12⟩, ⟨13⟩, ⟨14⟩, ⟨15⟩, ⟨16⟩"));
    }

    fn argument_axis(expression: u64, position: u32, domain: (u64, u64)) -> StableFrontierAxis {
        StableFrontierAxis::Argument {
            owner: StableObservedOccurrence { definition: StableScope::Root, path: 0 },
            expression: StablePlanRef::Expression { row: expression },
            position,
            domain,
        }
    }

    #[test]
    fn typed_index_projection_preserves_mixed_radix_order_and_exact_raw_rows() {
        let expression = ProjectedIndexExpression::Subtract {
            expression: 12,
            left: Box::new(ProjectedIndexExpression::Binding { expression: 10, position: 0 }),
            right: Box::new(ProjectedIndexExpression::Binding { expression: 11, position: 1 }),
        };
        let frontier = vec![argument_axis(10, 0, (3, 5)), argument_axis(11, 1, (7, 10))];
        let raw = [("3", "7"), ("3", "8"), ("3", "9"), ("4", "7"), ("4", "8"), ("4", "9")].map(
            |(left, right)| {
                let output = left.parse::<i64>().unwrap() - right.parse::<i64>().unwrap();
                lut_row(&[left, right], &output.to_string())
            },
        );
        validate_projected_index_rows(&expression, &frontier, &raw)
            .expect("last frontier axis changes fastest and nonzero domains are preserved");
        let mut changed = raw.clone();
        changed[1].tuple[1] = "08".to_owned();
        assert!(validate_projected_index_rows(&expression, &frontier, &changed).is_err());
        let mut changed = raw.clone();
        changed[2].output = "-05".to_owned();
        assert!(validate_projected_index_rows(&expression, &frontier, &changed).is_err());
        let mut changed = raw.clone();
        changed.swap(1, 2);
        assert!(validate_projected_index_rows(&expression, &frontier, &changed).is_err());
        assert!(validate_projected_index_rows(&expression, &frontier, &raw[..5]).is_err());
    }

    #[test]
    fn typed_index_projection_matches_euclidean_division_and_rejects_zero_divisor() {
        let constant = |expression, value| ProjectedIndexExpression::Constant {
            expression,
            value: BigInt::from(value),
        };
        let divide = ProjectedIndexExpression::Divide {
            expression: 2,
            left: Box::new(constant(0, -7)),
            right: Box::new(constant(1, -3)),
        };
        let remainder = ProjectedIndexExpression::Remainder {
            expression: 3,
            left: Box::new(constant(0, -7)),
            right: Box::new(constant(1, -3)),
        };
        assert_eq!(divide.evaluate(&[]).unwrap(), BigInt::from(-3));
        assert_eq!(remainder.evaluate(&[]).unwrap(), BigInt::from(2));
        let zero = ProjectedIndexExpression::Divide {
            expression: 4,
            left: Box::new(constant(0, -7)),
            right: Box::new(constant(1, 0)),
        };
        assert!(zero.evaluate(&[]).is_err());
    }

    #[test]
    fn large_slice_lut_rows_are_deterministically_sharded_without_remapping() {
        let rows = (0..257).map(slice_row).collect::<Vec<_>>();
        let mut first = Vec::new();
        let first_root = render_slice_lut_row_packages(7, &rows, &mut first).expect("first render");
        let mut second = Vec::new();
        let second_root =
            render_slice_lut_row_packages(7, &rows, &mut second).expect("second render");

        assert_eq!(first_root, second_root);
        assert_eq!(first, second);
        assert_eq!(first.len(), 3);
        assert_eq!(first[0].relative_path, "Cert/SliceGroupRows7Leaf0000.lean");
        assert_eq!(first[1].relative_path, "Cert/SliceGroupRows7Leaf0001.lean");
        assert_eq!(first[2].relative_path, "Cert/SliceGroupRows7Level0_0000.lean");
        let first_leaf = String::from_utf8(first[0].bytes.clone()).expect("UTF-8 leaf");
        let second_leaf = String::from_utf8(first[1].bytes.clone()).expect("UTF-8 leaf");
        assert!(first_leaf.contains("⟨[\"255\"], \"255\", \"255\", \"255\", \"255\"⟩"));
        assert!(second_leaf.contains("⟨[\"256\"], \"256\", \"256\", \"256\", \"256\"⟩"));
        assert_eq!(first_root.0, format!("{NAMESPACE}.Cert.SliceGroupRows7Level0_0000.rows"));
    }
}
