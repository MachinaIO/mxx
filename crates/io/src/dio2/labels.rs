//! The label transition `A_(i+1) = A_i[:, 1..] D_x` (spec §3, §6.4) in the clear and on
//! encodings.
//!
//! `D_x` is column block `x` of one fixed-seed SHA-256 bit matrix of `m_v` rows and
//! `2^c (m_v + 1)` columns. [`LabelProducts`] records only the seed and the dimensions: one
//! `Sha256BitColumns` node packs each column's ascending selected row indices into binary fields,
//! and the evaluator's runtime chunk `x` selects the column block. [`LabelProducts::next`] is the
//! clear transition of the label values; [`LabelProducts::product`] multiplies the label encodings
//! of one new label, for either carrier ([`super::carrier`]).
//!
//! The product of a label is the right-deep chain `l_0 (l_1 (... l_(p-1)))` of its selected
//! leaves in ascending order, padded with the one-encoding to two leaves. A public product
//! `MulPub(P, Q)` scales the left operand's error by the gadget action and adds the right one's
//! (`E = A E_P + E_Q` for a monomial `P`), so every leaf but the innermost gains one action and
//! the chain's error is `A (p - 1) e + e`: linear in the leaf count. The chain is one loop over
//! the leaf slots (the graph has the same size for every label width); the noise-only scalar
//! Programs of [`super::step`] multiply in the same order.

use super::{Dio2Error, Dio2Parameters, carrier::Carrier};
use mxx_dsl::{DslError, Family, Int};
use mxx_exponent_lut::rns_bgg::RnsBggParameters;
use mxx_ir_core::IntExpr;
use sha2::{Digest, Sha256};

/// The fixed-seed selector recipe of the irregular leaf sets of `D_x`.
pub struct LabelProducts {
    seed: [u8; 32],
    width: usize,
    chunks: usize,
    /// The leaf slots of every column: the label rows `m_v` (at least two), a bound on every
    /// column's selected-row count that holds by construction, so the chain multiplies exactly
    /// the selected rows without a bound that depends on the hashed selector table.
    pub max_leaves: usize,
    /// Binary field width of an input row index.
    pub index_bits: usize,
}

/// The packed columns of every chunk's `D_x` and their selected-row counts
/// ([`LabelProducts::columns`]).
pub type LabelColumns = (Family<Int>, Family<Int>);

impl LabelProducts {
    /// Retain the fixed seed and the dimensions of `parameters`' labels for chunks of
    /// `eval_chunk_bits` bits, without embedding the selector table.
    pub fn new(parameters: &Dio2Parameters, eval_chunk_bits: usize) -> Self {
        let width = parameters.label_width();
        let index_bits = (usize::BITS - (width - 1).leading_zeros()) as usize;
        Self {
            seed: label_transition_seed(),
            width,
            chunks: 1usize.checked_shl(eval_chunk_bits as u32).expect("addressable chunks"),
            max_leaves: width.max(2),
            index_bits,
        }
    }

    /// One shared public computation of all packed columns and their unpadded leaf counts. A
    /// graph computes it once, outside its loops.
    pub fn columns(&self) -> LabelColumns {
        mxx_dsl::DslContext::new("dio2/label-columns").sha256_bit_columns(
            self.seed,
            self.width,
            self.chunks * (self.width + 1),
            self.index_bits,
        )
    }

    /// The selected input row of slot `leaf` of `column` of chunk `chunk`, read from its
    /// packed column by one loop over the index bits.
    pub fn index(
        &self,
        columns: &Family<Int>,
        chunk: Int,
        column: Int,
        leaf: &IntExpr,
    ) -> Result<Int, DslError> {
        let packed = columns.at(chunk.mul(self.width + 1).add(column));
        let (index, _) = mxx_dsl::iterate(
            self.index_bits,
            (Int::constant(0), Int::constant(1)),
            |bit, (index, power)| {
                let position = IntExpr::Add(
                    Box::new(IntExpr::Mul(
                        Box::new(leaf.clone()),
                        Box::new(IntExpr::constant(self.index_bits)),
                    )),
                    Box::new(bit.expression()?),
                );
                let set = packed.clone().bit(Int::evaluate(position))?.to_int();
                Ok((index.add(set.mul(power.clone())), power.mul(2usize)))
            },
        )?;
        Ok(index)
    }

    /// The clear transition of the row-major label values `labels` (entry `(r, c)` at
    /// `r (m_v + 1) + c`) on chunk `chunk`: the same leaves as [`Self::product`], summed modulo
    /// the label modulus.
    pub fn next(
        &self,
        parameters: &Dio2Parameters,
        columns: &LabelColumns,
        labels: &Family<Int>,
        chunk: Int,
    ) -> Result<Family<Int>, DslError> {
        let width = self.width;
        let (packed, counts) = columns;
        mxx_dsl::parallel(parameters.label_rows() * (width + 1), |member| {
            let row = member.clone().div(width + 1);
            let column = member.rem(width + 1);
            let count = counts.at(chunk.clone().mul(width + 1).add(column.clone()));
            let total = mxx_dsl::iterate(self.max_leaves, Int::constant(0), |leaf, total| {
                let index =
                    self.index(packed, chunk.clone(), column.clone(), &leaf.expression()?)?;
                let value = labels.at(row.clone().mul(width + 1).add(index).add(1usize));
                Ok(total.add(mxx_dsl::select(
                    leaf.less(count.clone()).to_int(),
                    vec![Int::constant(0), value],
                )?))
            })?;
            Ok(total.rem(parameters.label_modulus()))
        })
    }

    /// The product of the new label `(r, c)` of member `member = r (m_v + 1) + c` at level 0, as
    /// a public encoding: the right-deep chain of the leaves `leaf(r m_v + k, r (m_v + 1) + 1 + k)`
    /// of the rows `k` that column `c` of chunk `chunk`'s `D_x` selects (the label input member
    /// and the label value index of `(r, 1 + k)`), padded with `one` to two leaves. One loop
    /// over the leaf slots, so its graph does not depend on the label width; a refresh part
    /// computes the product of its own label where it decodes it.
    #[allow(clippy::too_many_arguments)]
    pub fn product<C: Carrier>(
        &self,
        chain: &RnsBggParameters,
        columns: &LabelColumns,
        chunk: Int,
        member: Int,
        leaf: impl Fn(Int, Int) -> Result<C::Public, Dio2Error>,
        one: C::Public,
    ) -> Result<C::Public, Dio2Error> {
        let width = self.width;
        let (packed, counts) = columns;
        let last = self.max_leaves - 1;
        let mut failure = None;
        let mut guard = |result: Result<C::Public, Dio2Error>| {
            result.map_err(|error| {
                failure = Some(error);
                DslError::Schema
            })
        };
        let row = member.clone().div(width + 1);
        let column = member.rem(width + 1);
        let count = counts.at(chunk.clone().mul(width + 1).add(column.clone()));
        let padded =
            mxx_dsl::select(count.clone().less(2).to_int(), vec![count.clone(), Int::constant(2)]);
        let product = padded.and_then(|padded| {
            // Slot `max_leaves - 1 - i` at iteration `i`: from the innermost leaf outwards.
            mxx_dsl::iterate(self.max_leaves, one.clone(), |iteration, product| {
                let slot = IntExpr::Sub(
                    Box::new(IntExpr::constant(last)),
                    Box::new(iteration.expression()?),
                )
                .canonicalize();
                let index = self.index(packed, chunk.clone(), column.clone(), &slot)?;
                let input = row.clone().mul(width).add(index.clone());
                let value = row.clone().mul(width + 1).add(index).add(1usize);
                let slot = Int::evaluate(slot);
                let leaf = mxx_dsl::select(
                    slot.clone().less(count.clone()).to_int(),
                    vec![one.clone(), guard(leaf(input, value))?],
                )?;
                let chained =
                    guard(C::public_product(chain, 0, &leaf, &product).map_err(Dio2Error::from))?;
                let innermost = slot.clone().equal(padded.clone().sub(1usize)).to_int();
                let current = mxx_dsl::select(innermost, vec![chained, leaf])?;
                mxx_dsl::select(slot.less(padded.clone()).to_int(), vec![product, current])
            })
        });
        if let Some(error) = failure {
            return Err(error);
        }
        Ok(product?)
    }
}

/// The label transition `D_x` (spec §3): column block `val(x)` of the common row-major
/// `m_v` by `2^c (m_v + 1)` bit matrix, shared by every step and selected little-endian.
pub fn label_transition(parameters: &Dio2Parameters, x: &[bool]) -> Vec<Vec<bool>> {
    let width = parameters.label_width();
    let seed = label_transition_seed();
    let columns = (1usize << x.len()) * (width + 1);
    let base = super::function::chunk_value(x) * (width + 1);
    let mut cached_index = usize::MAX;
    let mut block = [0u8; 32];
    let mut bit = |index: usize| {
        let block_index = index / 256;
        if block_index != cached_index {
            let mut digest = Sha256::new();
            digest.update(seed);
            digest.update((block_index as u64).to_le_bytes());
            block = digest.finalize().into();
            cached_index = block_index;
        }
        let offset = index % 256;
        block[offset / 8] >> (offset % 8) & 1 == 1
    };
    (0..width).map(|k| (0..=width).map(|c| bit(k * columns + base + c)).collect()).collect()
}

/// Fixed domain of the single row-major label bit matrix; chunks select column blocks.
fn label_transition_seed() -> [u8; 32] {
    Sha256::digest(b"mxx-io/dio2/label-transition/column-blocks/v1").into()
}

/// The next label matrix `A_(i+1) = A_i[:, 1..] D_x mod Q` in the clear.
pub fn next_labels(
    parameters: &Dio2Parameters,
    labels: &[Vec<usize>],
    x: &[bool],
) -> Vec<Vec<usize>> {
    let q = parameters.label_modulus();
    let transition = label_transition(parameters, x);
    labels
        .iter()
        .map(|row| {
            (0..transition[0].len())
                .map(|column| {
                    (0..transition.len())
                        .filter(|k| transition[*k][column])
                        .map(|k| row[1 + k])
                        .sum::<usize>() %
                        q
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dio2::test_support::toy;
    use std::collections::BTreeMap;

    /// The symbolic selector and integer transition agree with the clear transition, including
    /// arbitrary column leaf counts and multiple rows, and the graph has the same size for every
    /// label height.
    #[test]
    fn symbolic_label_transition_matches_clear_labels() {
        use mxx_backends::{
            MemoryArtifactStore, RuntimeValue, authority::CpuExecution, backend::poly::cpu_backend,
        };
        use mxx_dsl::DslContext;
        use mxx_exponent_lut::api::{StageExecution, StageGraph, execute_graph};
        use num_bigint::BigInt;
        use rand::Rng;
        let mut node_counts = Vec::new();
        for height in [1, 2, 8] {
            let (mut parameters, _) = toy();
            parameters.prf.lwr_dimension = height;
            let mut execution = CpuExecution::new(cpu_backend(parameters.rings().unwrap()));
            for chunk in [false, true] {
                let labels = (0..height)
                    .map(|_| {
                        (0..=parameters.label_width())
                            .map(|_| rand::rng().random_range(0..parameters.label_modulus()))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let products = LabelProducts::new(&parameters, 1);
                let graph = DslContext::new("dio2-symbolic-labels");
                let input =
                    graph.int_family_input("labels", height * (parameters.label_width() + 1));
                let graph = graph
                    .output(
                        "next",
                        products
                            .next(
                                &parameters,
                                &products.columns(),
                                &input,
                                mxx_dsl::Int::constant(usize::from(chunk)),
                            )
                            .unwrap(),
                    )
                    .unwrap()
                    .build()
                    .unwrap();
                node_counts.push(
                    graph.graph.scopes().values().map(|scope| scope.nodes().len()).sum::<usize>(),
                );
                let runtime = BTreeMap::from([(
                    "labels".to_owned(),
                    RuntimeValue::integer_values(
                        labels.iter().flatten().copied().map(BigInt::from).collect(),
                    ),
                )]);
                let output = execute_graph(
                    StageGraph::new(graph, runtime, BTreeMap::new()),
                    &mxx_ir_core::ParamEnv::default(),
                    &mut execution,
                    &mut MemoryArtifactStore::default(),
                    |_| Ok(()),
                )
                .unwrap();
                let actual =
                    <CpuExecution as StageExecution<MemoryArtifactStore>>::download_integers(
                        &execution,
                        &output.outputs["next"],
                    )
                    .unwrap();
                let expected = next_labels(&parameters, &labels, &[chunk])
                    .into_iter()
                    .flatten()
                    .map(BigInt::from)
                    .collect::<Vec<_>>();
                assert_eq!(actual, expected);
            }
        }
        assert!(node_counts.iter().all(|count| *count == node_counts[0]), "{node_counts:?}");
    }

    /// One unchanged graph selects random multi-bit chunks from one SHA-256 column recipe.
    #[test]
    fn chunk_label_columns_match_clear_transition() {
        use mxx_backends::{
            MemoryArtifactStore, RuntimeValue, authority::CpuExecution, backend::poly::cpu_backend,
        };
        use mxx_dsl::DslContext;
        use mxx_exponent_lut::api::{StageExecution, StageGraph, execute_graph};
        use rand::Rng;
        let bits = std::env::var("MXX_IO_DIO2_LABEL_CHUNK_BITS")
            .map_or(3usize, |value| value.parse().unwrap());
        let (parameters, _) = toy();
        let labels = (0..parameters.label_rows())
            .map(|_| {
                (0..=parameters.label_width())
                    .map(|_| rand::rng().random_range(0..parameters.label_modulus()))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let graph = DslContext::new("chunk-label-columns");
        let chunk = graph.input::<mxx_dsl::Int>("chunk", mxx_dsl::IntType).unwrap();
        let input = graph
            .int_family_input("labels", parameters.label_rows() * (parameters.label_width() + 1));
        let products = LabelProducts::new(&parameters, bits);
        let graph = graph
            .output("next", products.next(&parameters, &products.columns(), &input, chunk).unwrap())
            .unwrap()
            .build()
            .unwrap();
        let recipes = graph
            .graph
            .scopes()
            .values()
            .flat_map(|scope| scope.nodes())
            .filter(|node| {
                matches!(node.kind(), mxx_ir_core::node::NodeKind::Sha256BitColumns { .. })
            })
            .count();
        assert_eq!(recipes, 1);
        let mut execution = CpuExecution::new(cpu_backend(parameters.rings().unwrap()));
        for chunk in [0, (1usize << bits) - 1, rand::rng().random_range(0..1usize << bits)] {
            let runtime = BTreeMap::from([
                (
                    "labels".to_owned(),
                    RuntimeValue::integer_values(
                        labels.iter().flatten().copied().map(num_bigint::BigInt::from).collect(),
                    ),
                ),
                ("chunk".to_owned(), RuntimeValue::Int(chunk.into())),
            ]);
            let output = execute_graph(
                StageGraph::new(
                    mxx_dsl::BuiltGraph { graph: graph.graph.clone() },
                    runtime,
                    BTreeMap::new(),
                ),
                &mxx_ir_core::ParamEnv::default(),
                &mut execution,
                &mut MemoryArtifactStore::default(),
                |_| Ok(()),
            )
            .unwrap();
            let actual = <CpuExecution as StageExecution<MemoryArtifactStore>>::download_integers(
                &execution,
                &output.outputs["next"],
            )
            .unwrap();
            let x = (0..bits).map(|bit| (chunk >> bit) & 1 == 1).collect::<Vec<_>>();
            let expected = next_labels(&parameters, &labels, &x)
                .into_iter()
                .flatten()
                .map(num_bigint::BigInt::from)
                .collect::<Vec<_>>();
            assert_eq!(actual, expected);
        }
    }

    /// Column blocks share a single row-major hash expansion, including block boundaries.
    #[test]
    fn label_transition_preserves_hash_expansion() {
        let (mut parameters, _) = toy();
        parameters.prf.lwr_dimension = 8;
        let x = [true];
        let width = parameters.label_width();
        let seed = label_transition_seed();
        for (row, values) in label_transition(&parameters, &x).iter().enumerate() {
            for (column, value) in values.iter().enumerate() {
                let index = row * (2 * (width + 1)) + width + 1 + column;
                let mut digest = Sha256::new();
                digest.update(seed);
                digest.update(((index / 256) as u64).to_le_bytes());
                let block: [u8; 32] = digest.finalize().into();
                let offset = index % 256;
                assert_eq!(*value, block[offset / 8] >> (offset % 8) & 1 == 1);
            }
        }
    }

    /// The label products of every label height have the same graph: the leaf chain and the
    /// index decoding are loops.
    #[test]
    fn label_products_do_not_grow_with_the_label_width() {
        use mxx_dsl::DslContext;
        let counts = [1, 2, 8].map(|height| {
            let (mut parameters, _) = toy();
            parameters.prf.lwr_dimension = height;
            let root = parameters.root(rand::random()).unwrap();
            let chain = parameters.chain(&root).unwrap();
            let products = LabelProducts::new(&parameters, 1);
            let hash_key = root.hash_key_input();
            let key = |member: Int| {
                root.key(
                    hash_key.clone(),
                    mxx_exponent_lut::rns_bgg::RnsBggRoot::input_tag("label", Some(member)),
                )
            };
            let chunk = DslContext::new("chunk").input::<Int>("chunk", mxx_dsl::IntType).unwrap();
            let columns = products.columns();
            let labels = mxx_dsl::parallel(
                parameters.label_rows() * (parameters.label_width() + 1),
                |member| {
                    products
                        .product::<super::super::carrier::Key>(
                            &chain,
                            &columns,
                            chunk.clone(),
                            member,
                            |input, _| Ok(key(input)),
                            key(Int::constant(0)),
                        )
                        .map_err(|_| DslError::Schema)
                },
            )
            .unwrap();
            let graph = DslContext::new("label-products")
                .output("labels", labels)
                .unwrap()
                .build()
                .unwrap();
            mxx_ir_core::validate(&graph.graph, &mxx_ir_core::ParamEnv::default()).unwrap();
            graph.graph.scopes().values().map(|scope| scope.nodes().len()).sum::<usize>()
        });
        assert!(counts.iter().all(|count| *count == counts[0]), "{counts:?}");
    }
}
