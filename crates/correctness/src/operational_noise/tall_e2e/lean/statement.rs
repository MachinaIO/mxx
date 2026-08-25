use super::{MODULE_ROOT, NAMESPACE, generated_file, list, option, quoted};
use crate::operational_noise::{
    certificate_schema::{
        CertificateDocumentV1, CertificateEventRowV1, CertificateExpressionRow,
        CertificateIndexUse, CertificateProgramRow, CertificateRange, CertificateResidualRootV1,
        CertificateSliceGroup, CertificateSourceRowV1, RawCoefficientClassV1, RawValueContractV1,
    },
    g0::{
        CanonicalEventOperator, CanonicalExpressionDescriptor, CanonicalExpressionOperator,
        CanonicalExpressionSource, CanonicalStatementScope, IndexUseKind, SliceMemberRole,
        StableArtifact, StableConstant, StableConstantValue, StableFamilySourceIdentity,
        StableFrontierAxis, StableHashDefinition, StableHashVariant, StableLayout,
        StableMatrixConstantKind, StableMatrixOperation, StableObservedOccurrence,
        StableObservedProducer, StableObservedSourceAccess, StableObservedWire, StablePlanRef,
        StableSampleDescriptor, StableSamplerOperation, StableScalarOperation, StableScope,
        StableTransformOperation, StableTrapdoorOperation, StableValueType,
    },
};
use std::fmt::Write as _;

const PACKAGE_SIZE: usize = 256;
const EXPRESSION_INPUT_LEAF_SIZE: usize = 16;

pub(super) fn render(
    document: &CertificateDocumentV1,
) -> Result<Vec<super::TallSecurity0GeneratedFile>, String> {
    let mut files = Vec::new();
    render_expression_packages(&document.expressions, &mut files)?;
    render_packages("Program", "ProgramRow", &document.programs, program_row, &mut files)?;
    render_packages("Source", "SourceRow", &document.sources, source_row, &mut files)?;
    render_packages("Event", "EventRow", &document.events, event_row, &mut files)?;
    render_packages("IndexUse", "IndexUseRow", &document.index_uses, index_use_row, &mut files)?;
    render_packages(
        "SliceGroup",
        "SliceGroupRow",
        &document.slice_groups,
        slice_group_row,
        &mut files,
    )?;
    files.push(generated_file("Cert/Cert.lean", render_top(document)?));
    Ok(files)
}

fn render_expression_packages(
    rows: &[CertificateExpressionRow],
    files: &mut Vec<super::TallSecurity0GeneratedFile>,
) -> Result<(), String> {
    for (package, chunk) in rows.chunks(PACKAGE_SIZE).enumerate() {
        let module = format!("Expression{package:03}");
        let mut source = header("Mxx.Certificate.OperationalNoise.TallSecurity0ABI", &module);
        let start = package * PACKAGE_SIZE;
        for (offset, row) in chunk.iter().enumerate() {
            let row_id = start + offset;
            render_expression_inputs(row_id, &row.inputs, &mut source);
            writeln!(
                source,
                "def ExpressionRow{row_id} : TallSecurity0ABI.ExpressionRow :=\n  {}\n",
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

fn header(import: &str, suffix: &str) -> String {
    format!(
        "import {import}\n\nset_option autoImplicit false\nset_option relaxedAutoImplicit false\n\nnamespace {NAMESPACE}.Cert.{suffix}\n\nopen Mxx.Certificate.OperationalNoise\nopen SchemaV1\nopen TallSecurity0ABI\n\n"
    )
}

fn render_packages<T>(
    label: &str,
    row_type: &str,
    rows: &[T],
    render: fn(&T) -> Result<String, String>,
    files: &mut Vec<super::TallSecurity0GeneratedFile>,
) -> Result<(), String> {
    for (package, chunk) in rows.chunks(PACKAGE_SIZE).enumerate() {
        let module = format!("{label}{package:03}");
        let mut source = header("Mxx.Certificate.OperationalNoise.TallSecurity0ABI", &module);
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
    let tables = [
        ("Expression", "ExpressionRow", document.expressions.len()),
        ("Program", "ProgramRow", document.programs.len()),
        ("Source", "SourceRow", document.sources.len()),
        ("Event", "EventRow", document.events.len()),
        ("IndexUse", "IndexUseRow", document.index_uses.len()),
        ("SliceGroup", "SliceGroupRow", document.slice_groups.len()),
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
    source
        .push_str("open Mxx.Certificate.OperationalNoise\nopen SchemaV1\nopen TallSecurity0ABI\n");
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
        writeln!(source, "def {label}Rows : RowTable SchemaV1.{row_type} := {root}\n")
            .expect("writing to String cannot fail");
    }
    let residual = residual_root(&document.residual_root);
    writeln!(
        source,
        "def document : TallDocument :=\n  {{ schemaId := {}\n    schemaVersion := {}\n    plaintextModulus := {}\n    ciphertextModulus := {}\n    ringDimension := {}\n    expressions := ExpressionRows\n    programs := ProgramRows\n    sources := SourceRows\n    events := EventRows\n    indexUses := IndexUseRows\n    sliceGroups := SliceGroupRows\n    residualRoot := {residual} }}\n\nend {NAMESPACE}",
        quoted(document.schema_id)?,
        document.schema_version,
        quoted(&document.plaintext_modulus)?,
        quoted(&document.ciphertext_modulus)?,
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
            writeln!(out, "def {name} : RowTable SchemaV1.{row_type} := {value}")
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

fn index_use_row(value: &CertificateIndexUse) -> Result<String, String> {
    let kind = match value.kind {
        IndexUseKind::IntegerExpression => ".integerExpression",
        IndexUseKind::FamilyGetStatic => ".familyGetStatic",
        IndexUseKind::FamilyGetDynamic => ".familyGetDynamic",
        IndexUseKind::Select => ".select",
        IndexUseKind::IndexedSlice => ".indexedSlice",
    };
    Ok(format!(
        "⟨{}, {}, {}, {kind}, {}, {}, {}, {}, {}⟩",
        observed_wire(&value.owner)?,
        option(value.result.as_ref(), |item| Ok(plan_ref(item)))?,
        option(value.consumed.as_ref(), |item| Ok(plan_ref(item)))?,
        plan_ref(&value.index),
        option(value.output_range.as_ref(), |item| Ok(range(item)))?,
        value_type(&value.output_type)?,
        list(&value.frontier, frontier)?,
        list(&value.rows, |row| Ok(format!(
            "⟨{}, {}⟩",
            list(&row.tuple, |item| quoted(item))?,
            quoted(&row.output)?
        )))?,
    ))
}

fn slice_group_row(value: &CertificateSliceGroup) -> Result<String, String> {
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
        list(&value.rows, |row| Ok(format!(
            "⟨{}, {}, {}, {}, {}⟩",
            list(&row.tuple, |item| quoted(item))?,
            quoted(&row.row_start)?,
            quoted(&row.row_end_exclusive)?,
            quoted(&row.column_start)?,
            quoted(&row.column_end_exclusive)?
        )))?,
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
}
