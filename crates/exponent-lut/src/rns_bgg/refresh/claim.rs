//! The claim of a recorded refresh run ([`RefreshRun::protocol`]).

use super::run::{RefreshRun, RnsBggRefresh, SIGMA_ROW_OUTPUT};
use crate::{
    api::ApiError,
    claim::{external, inputs_ending_with, stage, to_stage},
    program::{RhsId, WireId},
    program_gadgets::prf::SparseLwrPrf,
    refresh::{RefreshDeclaration, bit_flip_source},
    rns_bgg::{REFRESH_MATRIX_INPUT, REFRESH_TERMS_INPUT, data_input_name, plaintext_input_prefix},
};
use mxx_dsl::{DslContext, Family, FamilyType, IdealSpec, Int, IntType};
use mxx_ir_core::{
    Graph, IntExpr, ParamEnv,
    artifact::ProductionId,
    merge::{MergeInterface, MergeNames, merge_instances},
    node::{ArtifactInput, NodeKind},
    protocol::{
        ArtifactBinding, ArtifactName, ClosedProtocolBundle, ComparatorEndpointBinding,
        ComparatorSpec, EndpointBinding, EndpointBindings, EndpointSemanticBinding, EndpointSpecId,
        InputContract, InputContractEntry, InputValueContract, OutputRef, ProtocolDecl,
        ProtocolInputBinding, ProtocolInputDestination, ProtocolPreconditionSpec, ProtocolStage,
        PurePredicateSpec, StageId, StageInputName, Workflow,
    },
};
use num_traits::ToPrimitive;
use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
};

/// The phases of the caller's encoding of `sigma_0 G`, one instance per chain (stage
/// `<phase>_<chain>`), each merged into one stage of the claim over the chains; both parts of a
/// chain read it (the part's preprocessing its key, its online evaluation its row).
const CHAIN_PHASES: [&str; 3] = ["sigma_preprocess", "sigma_encrypt", "sigma_online_eval"];

/// The phases every part runs (stage `<phase>_<chain>_<component>`), each merged into one stage
/// of the claim over the parts.
const PART_PHASES: [&str; 3] = ["preprocess", "encrypt", "online_eval"];

/// The phase of a merged stage name, its instance and the phase's instance count, for a refresh
/// of `chains` chains: chain `c` of a chain phase, part `(c, j)` at `2 c + j` of a part phase.
fn stage_instance(stage: &str, chains: usize) -> Option<(&'static str, usize, usize)> {
    let rest = |phase: &str| stage.strip_prefix(phase)?.strip_prefix('_');
    for phase in CHAIN_PHASES {
        if let Some(chain) = rest(phase).and_then(|rest| rest.parse::<usize>().ok()) {
            return (chain < chains).then_some((phase, chain, chains));
        }
    }
    PART_PHASES.iter().find_map(|phase| {
        let (chain, component) = rest(phase)?.split_once('_')?;
        let (chain, component) = (chain.parse::<usize>().ok()?, component.parse::<usize>().ok()?);
        (chain < chains && component < 2).then_some((*phase, 2 * chain + component, 2 * chains))
    })
}

impl RefreshRun {
    /// The recorded run of `refresh` as a closed protocol, with failure probability
    /// `2^failure_probability_log2` (`None` for a worst-case claim). `a_prime` is the recorded
    /// graph of the caller's stage that produced `A'` and its output name: the protocol then has
    /// it as stage `a_prime` (its byte inputs external) and the claim is about that `A'`;
    /// without it `A'` is an external matrix of the reference ring (any `A'`), which the
    /// protocol states but [`mxx_ir_core::lean::protocol::export`] does not support. The run must
    /// be complete (its decode graph recorded by [`Self::decode`]) and the refresh's source
    /// must be [`bit_flip_source`].
    ///
    /// The protocol has the payload and mask sampling, every part's stages of the caller's
    /// encoding of `sigma_0 G` (recorded through [`super::RefreshSigmaRequest::record`]; the
    /// claim needs them, and their Program must be a binary [`crate::refresh::FreshNoiseProgram`]
    /// on the invocation's label), every part's preprocessing, encryption and online evaluation
    /// on its chain, and the decode stage, which counts the coefficients of
    /// `y_j - (sigma A'_(j) - Delta_0 r_j x G^st_0)` that decode to one (an error of at least
    /// `P / 4`, `P` the product of the refreshed primes); the ideal count is zero. External
    /// inputs are the hash keys, the caller's `A'` (any matrix), the source input `X^b` and the
    /// expected `X^(1 - b)`, the label `a` and its monomials, the selector bits and the
    /// coordinate tables (the cell coordinates of the key layouts, one per PBC bucket, which
    /// every preprocessing stage receives and its online evaluation reads) of every PRF (the
    /// sigma keys, the fresh-noise keys of component `j` shared by every chain, each decoder's
    /// own), the unit offsets, and the mask polynomial `sigma_0`. The requirement ties the
    /// monomials to `a`, makes every selector row one-hot, the expected exponent the LUT of the
    /// input, `sigma_0` the bits of the sigma keys on `a` read through the sigma tables, and `a`
    /// the invocation's label.
    ///
    /// The phases of the parts are merged into one stage each (by `merge_instances`), and every
    /// per-part input is a member of a family external over the parts (e.g. one family of each
    /// bucket's smudge selector rows over the parts); the requirement checks these families in
    /// loops over the parts, ties the members of parts that read one key set (the sigma keys, a
    /// component's fresh keys, the label and the source input) and reads the sigma keys of part
    /// 0. The decode stage reads the family of the parts' residue terms. The stages and the claim
    /// thus do not grow with the number of parts or columns.
    pub fn protocol(
        &self,
        refresh: &RnsBggRefresh,
        failure_probability_log2: Option<u32>,
        a_prime: Option<(&Graph, &str)>,
    ) -> Result<ProtocolDecl, ApiError> {
        let (source, input, _) = bit_flip_source().map_err(ApiError::scheme)?;
        if refresh.source.id() != source.id() {
            return Err(ApiError::input("the refresh claim is stated for the bit-flip source"));
        }
        let top = &refresh.top;
        // The stage that produced the caller's `A'`, if given.
        let mut graphs = self.graphs.clone();
        if let Some((producer, _)) = a_prime {
            graphs.insert("a_prime".to_owned(), producer.clone());
        }
        let graphs = &graphs;
        let graph = |name: &str| {
            graphs
                .get(name)
                .ok_or_else(|| ApiError::input(format!("the run has no recorded stage {name}")))
        };
        let id = |name: &str| StageId(name.to_owned());
        let productions = self
            .productions
            .iter()
            .map(|(production, stage)| (production.clone(), id(stage)))
            .collect::<BTreeMap<_, _>>();
        let secret = |stage: &str| (id(stage), "secret".to_owned());
        let sampled = |name: Option<&str>| {
            name.map(str::to_owned).ok_or_else(|| ApiError::input("a secret is not sampled"))
        };
        let mut links = BTreeMap::from([
            (sampled(self.payload.input_name())?, secret("payload")),
            (sampled(self.mask.input_name())?, secret("mask")),
        ]);
        let parts =
            (0..top.chains()).flat_map(|chain| [(chain, 0), (chain, 1)]).collect::<Vec<_>>();
        if self.sigmas.len() != top.chains() {
            return Err(ApiError::input("the run has no sigma encoding per chain"));
        }
        // Each part's online evaluation reads the row of its chain's sigma encoding.
        for &(chain, component) in &parts {
            let part = top
                .part((chain, component))
                .ok_or_else(|| ApiError::input("the refresh has no such part"))?;
            links.insert(
                super::refresh_sigma_input_name(*part.spec()),
                (id(&format!("sigma_online_eval_{chain}")), SIGMA_ROW_OUTPUT.to_owned()),
            );
        }
        // The decode stage reads the family of the parts' residue terms: the merged online
        // evaluation's output family `refresh` (part `(chain, j)` at `2 chain + j`).
        links.insert(REFRESH_TERMS_INPUT.to_owned(), (id("online_eval_0_0"), "refresh".to_owned()));
        // Online evaluation reads each child data family as a resident input named after the
        // preprocessing output holding it.
        for &(chain, component) in &parts {
            let preprocess = id(&format!("preprocess_{chain}_{component}"));
            for name in inputs_ending_with(graph(&format!("online_eval_{chain}_{component}"))?, "")
            {
                if name.contains("data/DataId(") {
                    links.insert(name.clone(), (preprocess.clone(), name));
                }
            }
        }
        let mut names = vec!["payload".to_owned(), "mask".to_owned()];
        if a_prime.is_some() {
            names.push("a_prime".to_owned());
        }
        for chain in 0..top.chains() {
            for stage in CHAIN_PHASES {
                names.push(format!("{stage}_{chain}"));
            }
        }
        for &(chain, component) in &parts {
            for stage in PART_PHASES {
                names.push(format!("{stage}_{chain}_{component}"));
            }
        }
        names.push("decode".to_owned());

        let range = |lower: i64, upper: i64| InputValueContract::IntegerRange {
            lower: lower.into(),
            upper: upper.into(),
        };
        let family_range = |count: usize, lower: i64, upper: i64| InputValueContract::Family {
            count: IntExpr::constant(count),
            element: Box::new(range(lower, upper)),
        };
        let requirement_input = |input: &str| ProtocolInputDestination::Requirement {
            requirement: 0,
            input: input.into(),
        };
        let mut externals = Vec::new();

        // The caller's `A'`, read by the parts and the decode stage: the output of its producer
        // stage `a_prime`, whose byte inputs are externals, or else any matrix of the reference
        // ring (an external the Lean export does not support).
        match a_prime {
            Some((producer, output)) => {
                links.insert(REFRESH_MATRIX_INPUT.to_owned(), (id("a_prime"), output.to_owned()));
                for node in producer.root_scope().nodes() {
                    let NodeKind::Input { name, wire_type, artifact: None } = node.kind() else {
                        continue;
                    };
                    let mxx_ir_core::types::WireType::Bytes { length } = wire_type else {
                        return Err(ApiError::input("the A' stage reads a non-byte input"));
                    };
                    if !name.starts_with("rns-bgg/hash-key") {
                        externals.push(external(
                            &format!("a_prime_{name}"),
                            InputValueContract::Bytes { length: length.clone() },
                            vec![to_stage("a_prime", name)],
                        ));
                    }
                }
            }
            None => {
                let destinations = names
                    .iter()
                    .map(|name| {
                        Ok(inputs_ending_with(graph(name)?, REFRESH_MATRIX_INPUT)
                            .iter()
                            .filter(|input| input.as_str() == REFRESH_MATRIX_INPUT)
                            .map(|input| to_stage(name, input))
                            .collect::<Vec<_>>())
                    })
                    .collect::<Result<Vec<_>, ApiError>>()?
                    .concat();
                externals.push(external(
                    "a_prime",
                    InputValueContract::MatrixLarge {
                        matrix_type: super::top_reference_ring(top.reference())
                            .matrix_type((2, 2 * top.reference().columns())),
                    },
                    destinations,
                ));
            }
        }

        // Hash keys: one external per derived key name.
        let mut hash_keys = BTreeMap::<String, Vec<ProtocolInputDestination>>::new();
        for name in &names {
            for graph_input in inputs_ending_with(graph(name)?, "") {
                if graph_input.starts_with("rns-bgg/hash-key") {
                    hash_keys
                        .entry(graph_input.clone())
                        .or_default()
                        .push(to_stage(name, &graph_input));
                }
            }
        }
        for (index, destinations) in hash_keys.into_values().enumerate() {
            externals.push(external(
                &format!("hash_key_{index}"),
                InputValueContract::Bytes { length: 32.into() },
                destinations,
            ));
        }

        // The source input `X^b` of every chain and the expected `X^(1 - b)` of the decode stage.
        let source_input = |field: &str| {
            parts
                .iter()
                .map(|&(chain, component)| {
                    let prefix = plaintext_input_prefix(Some(&top.chain_namespace(chain)), "input");
                    to_stage(
                        &format!("encrypt_{chain}_{component}"),
                        &format!("{prefix}/input/{input:?}.{field}"),
                    )
                })
                .collect::<Vec<_>>()
        };
        externals.push(external("x_coefficient", range(1, 1), source_input("coefficient")));
        let mut destinations = source_input("exponent");
        destinations.push(requirement_input("x_exponent"));
        destinations.push(ProtocolInputDestination::Ideal { input: "x_exponent".into() });
        externals.push(external("x_exponent", range(0, 1), destinations));
        let n = top.reference().ring_dimension as i64;
        externals.push(external(
            "expected_coefficient",
            range(1, 1),
            vec![to_stage("decode", "top-refresh-expected.coefficient")],
        ));
        externals.push(external(
            "expected_exponent",
            range(0, n - 1),
            vec![
                to_stage("decode", "top-refresh-expected.exponent"),
                requirement_input("expected_exponent"),
            ],
        ));
        externals.push(external(
            "sigma",
            family_range(n as usize, 0, 1),
            vec![to_stage("decode", "top-refresh-sigma.coefficients"), requirement_input("sigma")],
        ));

        // The PRF children of every part: inputs (the state `X^0` and the label monomials) and
        // RHS.
        let profile = *top.profile();
        let q = profile.q().map_err(ApiError::scheme)?;
        let members = profile.lwr_dimension + 1;
        let mut state_coefficient = Vec::new();
        let mut state_exponent = Vec::new();
        let mut label_coefficient = Vec::new();
        let mut label_exponent = Vec::new();
        let mut selectors =
            BTreeMap::<String, (usize, usize, Vec<ProtocolInputDestination>)>::new();
        let mut tables = BTreeMap::<String, (usize, Vec<ProtocolInputDestination>)>::new();
        let mut offset_coefficient = Vec::new();
        let mut zero_exponents = BTreeMap::<usize, Vec<ProtocolInputDestination>>::new();
        let mismatch = || ApiError::input("the parts' PRF children differ in shape");
        for &(chain, component) in &parts {
            // The chain's sigma Program is read once, with the chain's first part.
            let sigma = (component == 0).then(|| &self.sigmas[chain]);
            for child in children(top, (chain, component), sigma)? {
                let stage = |phase: &str| match child.per_chain {
                    true => format!("sigma_{phase}_{chain}"),
                    false => format!("{phase}_{chain}_{component}"),
                };
                let (encrypt, preprocess) = (stage("encrypt"), stage("preprocess"));
                let prefix = plaintext_input_prefix(Some(&child.namespace), "input");
                let wire = |wire: WireId, field: &str| {
                    to_stage(&encrypt, &format!("{prefix}/input/{wire:?}.{field}"))
                };
                state_coefficient.push(wire(child.prf.initial_input(), "coefficient"));
                state_exponent.push(wire(child.prf.initial_input(), "exponent"));
                label_coefficient.push(wire(child.prf.input_vector(), "coefficient"));
                label_exponent.push(wire(child.prf.input_vector(), "exponent"));
                let prefix = plaintext_input_prefix(Some(&child.namespace), "rhs");
                let rhs = |rhs: RhsId, field: &str| {
                    to_stage(&preprocess, &format!("{prefix}/rhs/{rhs:?}.{field}"))
                };
                for (bucket, family) in child.prf.selector_families().iter().enumerate() {
                    let row_length = family.row_length.unwrap_or(family.members);
                    let entry = selectors
                        .entry(format!("{}_selector_{bucket}", child.selectors))
                        .or_insert_with(|| (family.members, row_length, Vec::new()));
                    if (entry.0, entry.1) != (family.members, row_length) {
                        return Err(mismatch());
                    }
                    entry.2.push(rhs(family.rhs, "coefficient"));
                    zero_exponents
                        .entry(family.members)
                        .or_default()
                        .push(rhs(family.rhs, "exponent"));
                }
                for offset in child.offsets {
                    offset_coefficient.push(rhs(*offset, "coefficient"));
                    zero_exponents.entry(0).or_default().push(rhs(*offset, "exponent"));
                }
                // Bucket `b`'s coordinate table: the cell coordinates of every key's layout, a
                // row of `W` entries per key, as long as the bucket's selector family.
                let families = child.prf.selector_families();
                for (bucket, table) in child.prf.tables().iter().enumerate() {
                    let entry = tables
                        .entry(format!("{}_table_{bucket}", child.selectors))
                        .or_insert_with(|| (families[bucket].members, Vec::new()));
                    if entry.0 != families[bucket].members {
                        return Err(mismatch());
                    }
                    let name = data_input_name(Some(&child.namespace), *table);
                    entry.1.push(to_stage(&preprocess, &name));
                    // A Program evaluated on its own reads its tables as plain inputs of its
                    // online evaluation (a child reads them from the part's preprocessing).
                    if child.per_chain {
                        entry.1.push(to_stage(&stage("online_eval"), &name));
                    }
                }
            }
        }
        externals.push(external("state_coefficient", range(1, 1), state_coefficient));
        externals.push(external("state_exponent", range(0, 0), state_exponent));
        label_coefficient.push(requirement_input("a_coefficient"));
        label_exponent.push(requirement_input("a_exponent"));
        externals.push(external("a_coefficient", family_range(members, -1, 1), label_coefficient));
        externals.push(external("a_exponent", family_range(members, 0, n - 1), label_exponent));
        externals.push(external(
            "a",
            family_range(members, 0, q as i64 - 1),
            vec![requirement_input("a")],
        ));
        for (name, (count, _, destinations)) in &selectors {
            externals.push(external(name, family_range(*count, 0, 1), destinations.clone()));
        }
        if !offset_coefficient.is_empty() {
            externals.push(external("offset_coefficient", range(1, 1), offset_coefficient));
        }
        // Every table entry is a member of the label family (`nu` the zero member).
        for (name, (count, destinations)) in &tables {
            externals.push(external(
                name,
                family_range(*count, 0, profile.lwr_dimension as i64),
                destinations.clone(),
            ));
        }
        // Every RHS exponent is zero: one external per family size (`0` for a scalar).
        for (count, destinations) in zero_exponents {
            let contract = if count == 0 { range(0, 0) } else { family_range(count, 0, 0) };
            externals.push(external(&format!("rhs_exponent_{count}"), contract, destinations));
        }

        // Every per-part input is a member of a family input of the merged stages, fed by a
        // family external over the parts.
        let stages = names
            .iter()
            .map(|name| stage(name, graph(name)?, &productions, &links))
            .collect::<Result<Vec<_>, _>>()?;
        let (stages, merged) = merged_stages(stages, top.chains(), &productions, &links)?;
        let (mut externals, families) = pack_externals(externals, &merged, top.chains())?;

        // The requirement.
        let dsl = |error: mxx_dsl::DslError| ApiError::scheme(error);
        let family = |count: usize| FamilyType {
            element: IntType,
            count: IntExpr::constant(count),
            member_slot: None,
        };
        let requirement = DslContext::new("refresh-top-inputs");
        let a = requirement.input::<Family<Int>>("a", family(members)).map_err(dsl)?;
        let a_coefficient =
            requirement.input::<Family<Int>>("a_coefficient", family(members)).map_err(dsl)?;
        let a_exponent =
            requirement.input::<Family<Int>>("a_exponent", family(members)).map_err(dsl)?;
        let stride = profile.layout.stride() as i64;
        // The label is public and fixed by the invocation (the online stage reads its public
        // payload as constants); the last member is the zero member.
        let mut label = top.label(self.hash_key, &self.label_tag).map_err(ApiError::scheme)?;
        label.push(0);
        let mut valid = mxx_dsl::iterate(members, Int::constant(0).equal(0), |member, valid| {
            let fixed = mxx_dsl::select(
                member.clone(),
                label.iter().map(|value| Int::constant(*value as i64)).collect(),
            )?;
            let sign = a_coefficient.at(member.clone());
            // `2 e + n (1 - c) = 2 stride a` with `c = ±1`.
            Ok(valid &
                a.at(member.clone()).equal(fixed) &
                sign.clone().mul(sign.clone()).equal(1) &
                a_exponent
                    .at(member.clone())
                    .mul(2)
                    .add(Int::constant(1).sub(sign).mul(n))
                    .equal(a.at(member).mul(2 * stride)))
        })
        .map_err(dsl)?;
        let x_exponent = requirement.input::<Int>("x_exponent", IntType).map_err(dsl)?;
        let expected_exponent =
            requirement.input::<Int>("expected_exponent", IntType).map_err(dsl)?;
        valid = valid & expected_exponent.equal(Int::constant(1).sub(x_exponent.clone()));
        // The family externals over the parts: parts reading one original external read equal
        // members, a member standing for a kept external equals it, and every selector row is
        // one-hot. A pinned element (one admissible value) needs no tie, and byte strings (hash
        // keys) are not tied. Integer families are read as families of members.
        let pinned = |contract: &InputValueContract| match contract {
            InputValueContract::IntegerRange { lower, upper } => lower == upper,
            InputValueContract::Family { element, .. } => matches!(
                element.as_ref(),
                InputValueContract::IntegerRange { lower, upper } if lower == upper
            ),
            _ => false,
        };
        let scalars = BTreeMap::from([("x_exponent", x_exponent)]);
        let label_families = BTreeMap::from([
            ("a_coefficient", a_coefficient.clone()),
            ("a_exponent", a_exponent.clone()),
        ]);
        let unknown_tie = || ApiError::input("a part family is tied to an unknown external");
        let mut part_families = BTreeMap::new();
        for part_family in &families {
            let selector = selectors.get(&part_family.originals[0]).map(|(_, row, _)| *row);
            let tied = part_family.representatives.iter().enumerate().any(|(k, r)| *r != k) ||
                !part_family.ties.is_empty();
            let read = part_family.originals[0].starts_with("mask_");
            let parts_count = part_family.count;
            let bytes = matches!(part_family.element, InputValueContract::Bytes { .. });
            if bytes || pinned(&part_family.element) || (!tied && selector.is_none() && !read) {
                continue;
            }
            let representative = |member: Int| {
                mxx_dsl::select(
                    member,
                    part_family.representatives.iter().map(|r| Int::constant(*r as i64)).collect(),
                )
            };
            match &part_family.element {
                InputValueContract::IntegerRange { .. } => {
                    let input = requirement
                        .input::<Family<Int>>(part_family.name.clone(), family(parts_count))
                        .map_err(dsl)?;
                    valid = mxx_dsl::iterate(parts_count, valid, |k, valid| {
                        Ok(valid & input.at(k.clone()).equal(input.at(representative(k)?)))
                    })
                    .map_err(dsl)?;
                    for (k, original) in &part_family.ties {
                        let original = scalars.get(original.as_str()).ok_or_else(unknown_tie)?;
                        valid = valid & input.at(Int::constant(*k as i64)).equal(original.clone());
                    }
                }
                InputValueContract::Family { count, .. } => {
                    let count = count
                        .evaluate(&ParamEnv::default())
                        .ok()
                        .and_then(|count| count.to_usize())
                        .ok_or_else(|| ApiError::input("a part family has no constant count"))?;
                    let input = requirement
                        .input::<Family<Family<Int>>>(
                            part_family.name.clone(),
                            FamilyType {
                                element: family(count),
                                count: IntExpr::constant(parts_count),
                                member_slot: None,
                            },
                        )
                        .map_err(dsl)?;
                    if tied {
                        valid = mxx_dsl::iterate(parts_count, valid, |k, valid| {
                            let member = input.at(k.clone());
                            let other = input.at(representative(k)?);
                            mxx_dsl::iterate(count, valid, |entry, valid| {
                                Ok(valid & member.at(entry.clone()).equal(other.at(entry)))
                            })
                        })
                        .map_err(dsl)?;
                    }
                    for (k, original) in &part_family.ties {
                        let member = input.at(Int::constant(*k as i64));
                        let original =
                            label_families.get(original.as_str()).ok_or_else(unknown_tie)?;
                        valid = mxx_dsl::iterate(count, valid, |entry, valid| {
                            Ok(valid & member.at(entry.clone()).equal(original.at(entry)))
                        })
                        .map_err(dsl)?;
                    }
                    if let Some(row_length) = selector {
                        valid = mxx_dsl::iterate(parts_count, valid, |k, valid| {
                            let member = input.at(k);
                            mxx_dsl::iterate(count / row_length, valid, |row, valid| {
                                let total = mxx_dsl::iterate(
                                    row_length,
                                    Int::constant(0),
                                    |slot, total| {
                                        Ok(total.add(
                                            member.at(row.clone().mul(row_length as i64).add(slot)),
                                        ))
                                    },
                                )?;
                                Ok(valid & total.equal(1))
                            })
                        })
                        .map_err(dsl)?;
                    }
                    part_families.insert(part_family.originals[0].clone(), input);
                }
                other => {
                    return Err(ApiError::input(format!(
                        "{}: unexpected family element {other:?}",
                        part_family.name
                    )));
                }
            }
            let binding = &mut externals
                .iter_mut()
                .find(|(contract, _)| contract.name == part_family.name)
                .ok_or_else(|| ApiError::input("a part family has no external"))?
                .1;
            binding.destinations.push(requirement_input(&part_family.name));
        }
        // `sigma_k` is the bit of sigma member `k`: one exactly when `z_k mod Q < Q / 2` for
        // `z_k = sum_b a[coordinate of the selected cell of bucket b]`; slot `f` of bucket `b`'s
        // selector row `k` selects cell `f` of member `k`, whose coordinate is entry `k W + f` of
        // bucket `b`'s sigma table (the zero member `v` for a dummy or padding cell). Every part
        // reads the same sigma keys (its members are tied), so part 0's stand for all.
        let sigma = requirement.input::<Family<Int>>("sigma", family(n as usize)).map_err(dsl)?;
        let mask_prf = self.sigmas[0].0.sampler().prf();
        // A sigma input over several chains is a family external (chain 0's member read); with
        // one chain it stays the original external, read as is.
        let mut mask_family = |name: String| -> Result<Family<Int>, ApiError> {
            if let Some(family) = part_families.get(&name) {
                return Ok(family.at(Int::constant(0)));
            }
            let count = selectors
                .get(&name)
                .map(|(count, _, _)| *count)
                .or_else(|| tables.get(&name).map(|(count, _)| *count))
                .ok_or_else(|| ApiError::input(format!("the requirement does not read {name}")))?;
            let binding = &mut externals
                .iter_mut()
                .find(|(contract, _)| contract.name == name)
                .ok_or_else(|| ApiError::input(format!("{name} has no external")))?
                .1;
            binding.destinations.push(requirement_input(&name));
            requirement.input::<Family<Int>>(name, family(count)).map_err(dsl)
        };
        let mask_buckets = (0..mask_prf.tables().len())
            .map(|bucket| {
                let width = selectors
                    .get(&format!("mask_selector_{bucket}"))
                    .ok_or_else(|| ApiError::input("a mask bucket has no selector"))?
                    .1 as i64;
                Ok((
                    width,
                    mask_family(format!("mask_selector_{bucket}"))?,
                    mask_family(format!("mask_table_{bucket}"))?,
                ))
            })
            .collect::<Result<Vec<_>, ApiError>>()?;
        valid = mxx_dsl::iterate(n as usize, valid, |member, valid| {
            let mut z = Int::constant(0);
            for (width, selector, table) in &mask_buckets {
                z = z.add(mxx_dsl::iterate(*width as usize, Int::constant(0), |slot, total| {
                    let cell = member.clone().mul(*width).add(slot);
                    Ok(total.add(selector.at(cell.clone()).mul(a.at(table.at(cell)))))
                })?);
            }
            let bit = Int::constant(1).sub(z.rem(q as i64).div(q as i64 / 2));
            Ok(valid & sigma.at(member).equal(bit))
        })
        .map_err(dsl)?;
        let requirement = requirement.output("valid", valid).map_err(dsl)?.build().map_err(dsl)?;
        let requirement = PurePredicateSpec::new(requirement.graph).map_err(ApiError::scheme)?;
        // The ideal count of failed coefficients is zero (an `Int` read from the input, as the
        // endpoint compares two runtime integers).
        let ideal = DslContext::new("refresh-top-ideal");
        let x = ideal.input::<Int>("x_exponent", IntType).map_err(dsl)?;
        let ideal = ideal.output("value", x.mul(0)).map_err(dsl)?.build().map_err(dsl)?;
        let ideal = IdealSpec::new(ideal.graph).map_err(ApiError::scheme)?;

        let (contracts, bindings): (Vec<_>, Vec<_>) = externals.into_iter().unzip();
        let endpoint = EndpointSpecId::Exact;
        ProtocolDecl::new(ProtocolDecl {
            params: Vec::new(),
            bindings: Default::default(),
            failure_probability_log2,
            bundle: ClosedProtocolBundle {
                workflow: Workflow { stages, entrypoint: id("decode") },
                ideal,
                requirements: vec![requirement],
                comparator: ComparatorSpec::Equality {
                    endpoints: vec![ComparatorEndpointBinding {
                        endpoint,
                        actual_input: "decoded-residual".to_owned(),
                        ideal_input: "value".to_owned(),
                        result_output: "failure".to_owned(),
                        failure_value: true,
                    }],
                },
                endpoints: EndpointBindings {
                    entries: vec![EndpointBinding {
                        spec: endpoint,
                        semantics: EndpointSemanticBinding::Exact,
                        workflow_output: OutputRef {
                            stage: id("decode"),
                            output: "decoded-residual".to_owned(),
                        },
                        ideal_output: "value".to_owned(),
                    }],
                },
                operational_decoder_targets: Vec::new(),
                endpoint_specs: vec![endpoint],
                input_contract: InputContract { inputs: contracts },
                input_bindings: bindings,
                precondition_spec: ProtocolPreconditionSpec {
                    requirement_outputs: vec!["valid".to_owned()],
                },
            },
        })
        .map_err(ApiError::scheme)
    }
}

/// One PRF Program of a part: its hash namespace, its PRF, its offset RHS (bound to one), the
/// external name of its selector families, and whether it runs in its own stages once per chain
/// (`sigma_<phase>_<chain>`, the caller's encoding of `sigma_0 G`) rather than in the part's.
struct Child<'a> {
    namespace: [u8; 32],
    prf: &'a SparseLwrPrf,
    offsets: &'a [RhsId],
    selectors: String,
    per_chain: bool,
}

/// The PRF Programs of part `(chain, component)` of `top`: the chain's encoding of `sigma_0 G`
/// when given (`sigma`, its Program and namespace), the fresh-noise child and the decoder's
/// smudge child.
fn children<'a>(
    top: &'a RefreshDeclaration,
    (chain, component): (usize, usize),
    sigma: Option<&'a (crate::refresh::FreshNoiseProgram, [u8; 32])>,
) -> Result<Vec<Child<'a>>, ApiError> {
    let part = (chain, component);
    let missing = || ApiError::input("the refresh has no such part");
    let fresh = top.fresh(part).ok_or_else(missing)?;
    let decoder = top.decoder(part).ok_or_else(missing)?;
    let sigma = sigma.map(|(program, namespace)| Child {
        namespace: *namespace,
        prf: program.sampler().prf(),
        offsets: program.sampler().offset_rhs(),
        selectors: "mask".to_owned(),
        per_chain: true,
    });
    Ok(sigma
        .into_iter()
        .chain([
            Child {
                namespace: top.fresh_namespace(part).ok_or_else(missing)?,
                prf: fresh.sampler().prf(),
                offsets: fresh.sampler().offset_rhs(),
                selectors: format!("fresh_{component}"),
                per_chain: false,
            },
            Child {
                namespace: *decoder.commitment(),
                prf: decoder.smudge().prf(),
                offsets: &[],
                selectors: format!("smudge_{chain}_{component}"),
                per_chain: false,
            },
        ])
        .collect())
}

/// Where the merged stages read each per-instance stage input: per-instance `(stage, input)` to
/// the merged `(stage, input)`, and the merged inputs that are families over the instances.
#[derive(Default)]
struct MergedInputs {
    targets: BTreeMap<(String, String), (String, String)>,
    families: BTreeSet<(String, String)>,
}

/// The protocol's stages with the per-instance stages of each phase replaced by one stage running
/// every instance in a parallel loop ([`merge_instances`]): the chain phases ([`CHAIN_PHASES`])
/// over the chains, the part phases ([`PART_PHASES`]) over the parts (part `(chain, component)`
/// at `2 chain + component`), and where the merged stages read the per-instance inputs. Every
/// input that differs between the instances is one family input over the producer's instances:
/// instance `k` reading an artifact (or a linked output) of instance `floor(k / d)` of a producer
/// merged over `1 / d` as many instances reads that member of the producer's output family (`d =
/// 2` for a part reading its chain's sigma encoding, else `1`), and a plain input is member `k`
/// of a family input fed by a family external. Stages outside the merged phases read whole output
/// families, so no stage's interface grows with the number of chains.
fn merged_stages(
    stages: Vec<ProtocolStage>,
    chains: usize,
    productions: &BTreeMap<ProductionId, StageId>,
    links: &BTreeMap<String, (StageId, String)>,
) -> Result<(Vec<ProtocolStage>, MergedInputs), ApiError> {
    let instance_of = |stage: &str| stage_instance(stage, chains);
    let instance_names = |phase: &str| -> Vec<String> {
        match CHAIN_PHASES.contains(&phase) {
            true => (0..chains).map(|chain| format!("{phase}_{chain}")).collect(),
            false => (0..chains)
                .flat_map(|chain| [0, 1].map(|component| format!("{phase}_{chain}_{component}")))
                .collect(),
        }
    };
    let graphs = stages
        .iter()
        .map(|stage| (stage.id.0.clone(), stage.graph.clone()))
        .collect::<BTreeMap<_, _>>();
    // Phases are merged in order; instance `k` reading output `o` of instance `k / d` of a merged
    // producer reads that member of the producer's family named `merged[..].outputs[k / d][o]`.
    let mut merged: BTreeMap<&str, (Graph, MergeNames)> = BTreeMap::new();
    // The family inputs of the merged stages, by phase and name.
    let family_inputs = RefCell::new(BTreeSet::new());
    for phase in CHAIN_PHASES.into_iter().chain(PART_PHASES) {
        let stage_names = instance_names(phase);
        let count = stage_names.len();
        let instances = stage_names
            .iter()
            .map(|name| {
                graphs
                    .get(name)
                    .cloned()
                    .ok_or_else(|| ApiError::input(format!("no recorded stage {name}")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        // The producer instance every consumer instance reads (from a merged phase), and the
        // group size `d` with `instance = k / d`, the same for every `k`.
        let grouped = |producers: Vec<Option<(&'static str, usize, usize)>>| {
            let (phase, _, producer_count) = producers[0]?;
            (count % producer_count == 0).then_some(())?;
            let group = count / producer_count;
            producers
                .iter()
                .enumerate()
                .all(|(k, producer)| {
                    matches!(producer, Some((other, instance, _))
                        if *other == phase && *instance == k / group)
                })
                .then_some((phase, group))
        };
        let family_input = |kinds: &[&NodeKind]| {
            let NodeKind::Input { name, artifact, .. } = kinds[0] else {
                return None;
            };
            // An input equal in every instance is passed once unless it reads a merged
            // producer (whose output is a family).
            let equal = kinds.iter().all(|kind| *kind == kinds[0]);
            let Some(artifact) = artifact else {
                // A plain input linked to a merged producer's output (a part reading its
                // chain's sigma row) reads its member; another differing one is member `k` of a
                // family external.
                let producers = kinds
                    .iter()
                    .map(|kind| match kind {
                        NodeKind::Input { name, artifact: None, .. } => {
                            links.get(name).and_then(|(stage, _)| instance_of(&stage.0))
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                let group = match producers[0] {
                    Some(_) => grouped(producers)?.1,
                    None if equal => return None,
                    None => 1,
                };
                family_inputs.borrow_mut().insert((phase, name.clone()));
                return Some((name.clone(), None, group));
            };
            let producers = kinds
                .iter()
                .map(|kind| match kind {
                    NodeKind::Input { artifact: Some(artifact), .. } => {
                        instance_of(&productions.get(&artifact.production_id)?.0)
                    }
                    _ => None,
                })
                .collect::<Vec<_>>();
            let (producer_phase, group) = grouped(producers.clone())?;
            let mut families = kinds.iter().zip(&producers).map(|(kind, producer)| {
                let NodeKind::Input { artifact: Some(artifact), .. } = kind else {
                    return None;
                };
                let (_, instance, _) = (*producer)?;
                merged.get(producer_phase)?.1.outputs[instance]
                    .get(&artifact.artifact_name)
                    .cloned()
            });
            let family = families.next()??;
            for other in families {
                if other? != family {
                    return None;
                }
            }
            family_inputs.borrow_mut().insert((phase, name.clone()));
            Some((
                name.clone(),
                Some(ArtifactInput { artifact_name: family, ..artifact.clone() }),
                group,
            ))
        };
        let interface = MergeInterface { family_input: &family_input, unpacked: &|_| false };
        let result = merge_instances(phase, &instances, &interface)
            .map_err(|error| ApiError::input(format!("merging the {phase} stages: {error}")))?;
        merged.insert(phase, result);
    }
    // A producer (per-instance stage, output) as a merged stage and its output family.
    let producer = |stage: &StageId, output: &str| match instance_of(&stage.0) {
        Some((phase, instance, _)) => (
            StageId(phase.to_owned()),
            merged[phase].1.outputs[instance]
                .get(output)
                .cloned()
                .unwrap_or_else(|| output.to_owned()),
        ),
        None => (stage.clone(), output.to_owned()),
    };
    let mut inputs = MergedInputs::default();
    for (&phase, (_, names)) in &merged {
        for (instance, stage) in instance_names(phase).into_iter().enumerate() {
            for (original, name) in &names.inputs[instance] {
                inputs
                    .targets
                    .insert((stage.clone(), original.clone()), (phase.to_owned(), name.clone()));
            }
        }
    }
    inputs.families = family_inputs
        .into_inner()
        .into_iter()
        .map(|(phase, name)| (phase.to_owned(), name))
        .collect();
    let merged_stage = |phase: &str| -> Result<ProtocolStage, ApiError> {
        let (graph, names) = &merged[phase];
        let mut bindings = Vec::new();
        for node in graph.root_scope().nodes() {
            let NodeKind::Input { name, artifact, .. } = node.kind() else {
                continue;
            };
            let production = |artifact: &ArtifactInput| {
                productions.get(&artifact.production_id).ok_or_else(|| {
                    ApiError::input(format!("input {name} reads an unknown production"))
                })
            };
            let (stage, output) = match artifact {
                // A family input reads the producer's output family itself.
                Some(artifact) if inputs.families.contains(&(phase.to_owned(), name.clone())) => {
                    let Some((producer_phase, _, _)) = instance_of(&production(artifact)?.0) else {
                        continue;
                    };
                    (StageId(producer_phase.to_owned()), artifact.artifact_name.clone())
                }
                Some(artifact) => producer(production(artifact)?, &artifact.artifact_name),
                None => {
                    // The original name of a plain input, in the first instance having it.
                    let original = names
                        .inputs
                        .iter()
                        .find_map(|instance| {
                            instance
                                .iter()
                                .find(|(_, merged)| *merged == name)
                                .map(|(original, _)| original)
                        })
                        .cloned()
                        .unwrap_or_else(|| name.clone());
                    let Some((stage, output)) = links.get(&original) else {
                        continue;
                    };
                    producer(stage, output)
                }
            };
            bindings.push(ArtifactBinding {
                consumer_input: StageInputName(name.clone()),
                producer_stage: stage,
                producer_output: ArtifactName(output),
            });
        }
        Ok(ProtocolStage { id: StageId(phase.to_owned()), graph: graph.clone(), bindings })
    };
    let mut result = Vec::new();
    for stage in stages {
        match instance_of(&stage.id.0) {
            // The merged stage takes the place of its first instance.
            Some((phase, 0, _)) => result.push(merged_stage(phase)?),
            Some(_) => {}
            None => {
                let bindings = stage
                    .bindings
                    .iter()
                    .map(|binding| {
                        let (producer_stage, output) =
                            producer(&binding.producer_stage, &binding.producer_output.0);
                        ArtifactBinding {
                            consumer_input: binding.consumer_input.clone(),
                            producer_stage,
                            producer_output: ArtifactName(output),
                        }
                    })
                    .collect();
                result.push(ProtocolStage { bindings, ..stage });
            }
        }
    }
    Ok((result, inputs))
}

/// A family external over the instances of a merged phase (the parts, or the chains): member `k`
/// feeds member `k` of every family input in `inputs` and stands for the external `originals[k]`
/// the per-instance input of instance `k` read.
/// Parts that read one original external read equal members (`representatives[k]` the first
/// part reading `originals[k]`), and a member standing for an original external that other
/// destinations still read equals it; [`RefreshRun::protocol`]'s requirement ties both (byte
/// strings excepted: every part may use its own hash keys).
struct PartFamily {
    name: String,
    /// The number of instances.
    count: usize,
    element: InputValueContract,
    originals: Vec<String>,
    representatives: Vec<usize>,
    /// `(k, original)`: member `k` equals the kept external `original`.
    ties: Vec<(usize, String)>,
}

/// The externals of the merged protocol: each external's destinations in the merged stages
/// (deduplicated), with the per-part inputs merged into family inputs fed by one [`PartFamily`]
/// per pattern of original externals over a phase's instances; an original external whose every
/// destination was such a member is dropped.
#[allow(clippy::type_complexity)]
fn pack_externals(
    externals: Vec<(InputContractEntry, ProtocolInputBinding)>,
    merged: &MergedInputs,
    chains: usize,
) -> Result<(Vec<(InputContractEntry, ProtocolInputBinding)>, Vec<PartFamily>), ApiError> {
    // For every family input, the original external feeding each member.
    let mut feeds = BTreeMap::<(String, String), Vec<Option<usize>>>::new();
    let mut kept = Vec::new();
    for (index, (contract, binding)) in externals.iter().enumerate() {
        let mut destinations = Vec::new();
        for destination in &binding.destinations {
            let ProtocolInputDestination::WorkflowStage { stage, input } = destination else {
                destinations.push(destination.clone());
                continue;
            };
            let Some((phase, name)) = merged.targets.get(&(stage.0.clone(), input.0.clone()))
            else {
                destinations.push(destination.clone());
                continue;
            };
            if merged.families.contains(&(phase.clone(), name.clone())) {
                let (_, member, count) = stage_instance(&stage.0, chains).ok_or_else(|| {
                    ApiError::input(format!("{} is not a merged instance stage", stage.0))
                })?;
                let slot = &mut feeds
                    .entry((phase.clone(), name.clone()))
                    .or_insert_with(|| vec![None; count])[member];
                if slot.replace(index).is_some() {
                    return Err(ApiError::input(format!("{phase}/{name}: one external per member")));
                }
            } else {
                let destination = to_stage(phase, name);
                if !destinations.contains(&destination) {
                    destinations.push(destination);
                }
            }
        }
        if !destinations.is_empty() {
            kept.push((contract.clone(), ProtocolInputBinding { destinations, ..binding.clone() }));
        }
    }
    let kept_names = kept.iter().map(|(contract, _)| contract.name.clone()).collect::<Vec<_>>();
    // Family inputs fed by one pattern of originals share one family external.
    let mut patterns = BTreeMap::<Vec<usize>, Vec<ProtocolInputDestination>>::new();
    for ((phase, name), members) in feeds {
        let members = members
            .into_iter()
            .map(|member| {
                member.ok_or_else(|| ApiError::input(format!("{phase}/{name}: a member is unfed")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        patterns.entry(members).or_default().push(to_stage(&phase, &name));
    }
    let mut families = Vec::new();
    for (members, destinations) in patterns {
        let element = externals[members[0]].0.value.clone();
        if !members.iter().all(|member| externals[*member].0.value == element) {
            return Err(ApiError::input("a family external has one element contract"));
        }
        let originals =
            members.iter().map(|member| externals[*member].0.name.clone()).collect::<Vec<_>>();
        // A family over the parts, or over the chains (a chain phase's input).
        let count = members.len();
        let representatives = (0..count)
            .map(|k| members.iter().position(|member| *member == members[k]).unwrap_or(k))
            .collect::<Vec<_>>();
        let ties = (0..count)
            .filter(|k| representatives[*k] == *k && kept_names.contains(&originals[*k]))
            .map(|k| (k, originals[k].clone()))
            .collect();
        let prefix = if count == 2 * chains { "parts" } else { "chains" };
        let name = format!("{prefix}_{}", originals[0]);
        kept.push(external(
            &name,
            InputValueContract::Family {
                count: IntExpr::constant(count),
                element: Box::new(element.clone()),
            },
            destinations,
        ));
        families.push(PartFamily { name, count, element, originals, representatives, ties });
    }
    Ok((kept, families))
}
