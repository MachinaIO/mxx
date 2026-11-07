//! Counter-mode TurboSHAKE128 over public sign-exponent encodings, packed into label encodings.
//!
//! [`TurboShake128LabelHash`] evaluates the random-oracle stream
//! `G(M) = Y_0 || Y_1 || ...`, `Y_i = TurboSHAKE128(T || BE32(i) || M, D = 0x1F, 1344 bits)`
//! (RFC 9861), for a fixed-length public tag `T` and a public message `M` that fits one block
//! (`|T| + 4 + |M| <= 167` bytes). Every instance needs one call of the 12-round permutation
//! Keccak-p[1600, 12], and all instances run in parallel, so the noise depth is 12.
//!
//! **Encodings.** A bit `b` is held as the public sign `mu = (-1)^b = X^(n b)`. XOR is a product
//! of signs, and the chi step is the exact identity
//! `mu' = 1/2 eps (nu_0 - nu_0 nu_1 + nu_0 nu_2 + nu_0 nu_1 nu_2)` over the signs of its three
//! inputs (`eps` is the iota sign). The theta and rho/pi outputs are never materialized: each chi
//! input sign is a chain of [`ExponentLutProgramBuilder::mul_public_scaled`] products whose
//! gadget-side operand is always a round input (a leaf), so a round costs one decomposition gain.
//! The `1/2` is not executed: the round-`r` leaves carry the public scale `2^(r-1)`, which each
//! product removes inside its decomposition.
//!
//! **Shared parity blocks.** From round 2 on, the ten leaves of each theta column-parity block
//! `D[x, z]` are chained once per round into a prefix, and the first chi input of an output is
//! its prefix times the output's own leaf; five outputs share each prefix. A prefix is only ever
//! a chain start (accumulator side), so every chain keeps its length and noise bound.
//!
//! **Inputs.** One public family of `8 |M| + 1` members: member `8 k + t` is the sign of bit `t`
//! (least significant first) of message byte `k`, and the last member is the unit encoding
//! (payload `+1`). Tag, counter, domain byte, padding and capacity bits are public constants. In
//! round 1 every chi input's leaf list is padded with the unit member to the round's longest
//! list, so that every round is one uniform family body.
//!
//! **Outputs.** The first `w v` bits of `G(M)` form `v` labels of `w` bits (label `r` is
//! `sum_j 2^j b_(w r + j)`), and the next `c` bits are sign outputs (chain bits). A label is the
//! public exponent encoding of `X^(stride a_r)`, `stride = 2n / 2^w`, with the public scale
//! `4^w` ([`TurboShake128LabelHash::label_scale_log2`]); a consumer that reads it as the left
//! operand of a public product removes the scale with
//! [`ExponentLutProgramBuilder::mul_public_scaled`]. The packing is fused into the last round:
//! from the unit, each bit applies `4 X^(c_j b_j) acc = 2 (1 + X^(c_j)) acc + (1 - X^(c_j)) eps
//! (C_1 - C_2 + C_3 + C_4)`, where `c_j = stride 2^j` and the `C_k` are the chi chains started at
//! `acc` instead of a leaf (so they cannot start at a shared prefix). A chain bit has payload
//! `2^rounds (-1)^b`.
//!
//! **Graph size.** The state after a round is one family of `K * 1600` members and its prefixes
//! one family of `K * 320` members (instance-major, `K` instances), and the labels and chain bits
//! are families of `v` and `c` members. Every family is materialized
//! ([`ExponentLutProgramBuilder::parallel_materialized`]): later rounds read many of its members,
//! so a caller that lowers sampler families where they are read (`lower_program_deferred`)
//! still lowers each round once, and can export the label and chain families. Member indices, signs
//! and round constants are integer expressions of the loop index over constant tables, so the
//! Program structure is bounded independently of `v`, `c` and `K`; executed work and family storage
//! still scale with them.
//!
//! The Program depends on the ring dimension `n` (signs are `X^n = -1` shifts, and `stride` is
//! `2n / 2^w`). This module only assembles the Program and prepares its plaintext inputs.

use crate::{
    api::PlaintextInputs,
    program::{
        AttributeKnowledge, ExponentLutProgram, ExponentLutProgramBuilder, FamilyRange,
        InputBinding, Operand, ProgramValidationError, PublicScale, WireId,
    },
};
use mxx_backends::poly::Poly;
use mxx_ir_core::IntExpr;
use std::collections::BTreeMap;

/// TurboSHAKE128 rate in bytes.
pub const RATE_BYTES: usize = 168;
/// TurboSHAKE128 rate in bits: the output bits of one instance.
pub const RATE_BITS: usize = 8 * RATE_BYTES;
/// Keccak state bits.
pub const STATE_BITS: usize = 1600;
/// Rounds of Keccak-f[1600], whose last rounds form Keccak-p[1600, n_r].
pub const KECCAK_ROUNDS: usize = 24;
/// Rounds of Keccak-p[1600, 12], the TurboSHAKE permutation.
pub const TURBOSHAKE_ROUNDS: usize = 12;
/// The TurboSHAKE domain-separation byte of the hash, followed by the padding.
pub const DOMAIN_BYTE: u8 = 0x1f;
const COUNTER_BYTES: usize = 4;
/// Theta column-parity blocks per instance: one per `(x, z)`.
const PARITY_BLOCKS: usize = 5 * 64;

/// Round constants of Keccak-f[1600]; Keccak-p[1600, n_r] uses the last `n_r`.
pub const ROUND_CONSTANTS: [u64; KECCAK_ROUNDS] = [
    0x0000_0000_0000_0001,
    0x0000_0000_0000_8082,
    0x8000_0000_0000_808a,
    0x8000_0000_8000_8000,
    0x0000_0000_0000_808b,
    0x0000_0000_8000_0001,
    0x8000_0000_8000_8081,
    0x8000_0000_0000_8009,
    0x0000_0000_0000_008a,
    0x0000_0000_0000_0088,
    0x0000_0000_8000_8009,
    0x0000_0000_8000_000a,
    0x0000_0000_8000_808b,
    0x8000_0000_0000_008b,
    0x8000_0000_0000_8089,
    0x8000_0000_0000_8003,
    0x8000_0000_0000_8002,
    0x8000_0000_0000_0080,
    0x0000_0000_0000_800a,
    0x8000_0000_8000_000a,
    0x8000_0000_8000_8081,
    0x8000_0000_0000_8080,
    0x0000_0000_8000_0001,
    0x8000_0000_8000_8008,
];

/// Rho offsets `r[x, y]`, indexed `[y][x]`.
pub const RHO_OFFSETS: [[u32; 5]; 5] = [
    [0, 1, 62, 28, 27],
    [36, 44, 6, 55, 20],
    [3, 10, 43, 25, 39],
    [41, 45, 15, 21, 8],
    [18, 2, 61, 56, 14],
];

/// Errors in a TurboSHAKE128 label-hash declaration.
#[derive(Debug, thiserror::Error, Eq, PartialEq)]
pub enum TurboShake128Error {
    #[error("invalid TurboSHAKE128 label-hash declaration: {0}")]
    InvalidDeclaration(&'static str),
    #[error(transparent)]
    Program(#[from] ProgramValidationError),
}

/// The declaration of a TurboSHAKE128 label hash (see the module documentation).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TurboShake128LabelHash {
    tag: Vec<u8>,
    message_bytes: usize,
    labels: usize,
    label_bits: u32,
    chain_bits: usize,
    ring_dimension: usize,
    rounds: usize,
    /// The instances of the state and prefix families, when not the hash's own
    /// ([`TurboShake128LabelHash::over_instances`]).
    instances: Option<usize>,
}

/// The output families of [`TurboShake128LabelHash::append`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TurboShake128Wires {
    /// `v` label encodings of payload `4^w X^(stride a_r)`, or `None` when `v = 0`.
    pub labels: Option<WireId>,
    /// `c` chain-bit encodings of payload `2^rounds (-1)^b`, or `None` when `c = 0`.
    pub chain: Option<WireId>,
}

impl TurboShake128LabelHash {
    /// Declares the hash of a `message_bytes`-byte message under `tag` into `labels` labels of
    /// `label_bits` bits followed by `chain_bits` chain bits, over ring dimension
    /// `ring_dimension` and the permutation Keccak-p[1600, `rounds`] ([`TURBOSHAKE_ROUNDS`] for
    /// TurboSHAKE128; fewer only to test the schedule, and [`KECCAK_ROUNDS`] gives SHAKE128).
    pub fn new(
        tag: Vec<u8>,
        message_bytes: usize,
        labels: usize,
        label_bits: u32,
        chain_bits: usize,
        ring_dimension: usize,
        rounds: usize,
    ) -> Result<Self, TurboShake128Error> {
        if tag.len() + COUNTER_BYTES + message_bytes > RATE_BYTES - 1 {
            return Err(TurboShake128Error::InvalidDeclaration("the input does not fit one block"));
        }
        if labels * label_bits as usize + chain_bits == 0 {
            return Err(TurboShake128Error::InvalidDeclaration("the hash has no output"));
        }
        if !ring_dimension.is_power_of_two() ||
            label_bits == 0 && labels > 0 ||
            1usize.checked_shl(label_bits).is_none_or(|modulus| modulus > 2 * ring_dimension)
        {
            return Err(TurboShake128Error::InvalidDeclaration("labels do not fit the ring"));
        }
        if rounds == 0 || rounds > KECCAK_ROUNDS {
            return Err(TurboShake128Error::InvalidDeclaration("invalid round count"));
        }
        Ok(Self {
            tag,
            message_bytes,
            labels,
            label_bits,
            chain_bits,
            ring_dimension,
            rounds,
            instances: None,
        })
    }

    /// This hash with state and prefix families of `instances` instances side by side: the later
    /// rounds and prefixes of many hashes evaluated at once, which read no message and no
    /// per-instance constant. Only [`Self::append_variable_round`], [`Self::append_prefixes`]
    /// and [`Self::append_variable_prefixes`] may use it.
    pub fn over_instances(&self, instances: usize) -> Self {
        Self { instances: Some(instances), ..self.clone() }
    }

    pub fn tag(&self) -> &[u8] {
        &self.tag
    }

    pub const fn message_bytes(&self) -> usize {
        self.message_bytes
    }

    pub const fn labels(&self) -> usize {
        self.labels
    }

    pub const fn label_bits(&self) -> u32 {
        self.label_bits
    }

    pub const fn chain_bits(&self) -> usize {
        self.chain_bits
    }

    pub const fn ring_dimension(&self) -> usize {
        self.ring_dimension
    }

    pub const fn rounds(&self) -> usize {
        self.rounds
    }

    /// The TurboSHAKE128 instances `K = ceil((w v + c) / 1344)`.
    pub fn instances(&self) -> usize {
        self.instances.unwrap_or_else(|| self.output_bits().div_ceil(RATE_BITS))
    }

    /// The stream bits `w v + c` the hash outputs.
    pub fn output_bits(&self) -> usize {
        self.labels * self.label_bits as usize + self.chain_bits
    }

    /// The exponent stride `2n / 2^w` of a label.
    pub fn stride(&self) -> usize {
        2 * self.ring_dimension >> self.label_bits
    }

    /// The public scale `4^w` of a label payload, as `2w`.
    pub const fn label_scale_log2(&self) -> u32 {
        2 * self.label_bits
    }

    /// The public scale `2^rounds` of a chain-bit payload.
    pub const fn chain_scale_log2(&self) -> u32 {
        self.rounds as u32
    }

    /// Members of the input family: the message bits and the unit.
    pub fn input_members(&self) -> usize {
        8 * self.message_bytes + 1
    }

    /// Append the hash to `builder`, reading the public input family `message` of
    /// [`Self::input_members`] members (message bit signs, then the unit).
    ///
    /// This is [`Self::append_round`] for every round but the last, each later round reading
    /// the parity-block prefixes of [`Self::append_prefixes`], and then [`Self::append_outputs`].
    /// A caller can instead append these parts to separate Programs, binding the state and
    /// prefix families of one Program as public input families of the next.
    pub fn append(
        &self,
        builder: &mut ExponentLutProgramBuilder,
        message: WireId,
    ) -> Result<TurboShake128Wires, TurboShake128Error> {
        let last = self.rounds - 1;
        let unit =
            Operand::Member { family: message, index: IntExpr::constant(self.input_members() - 1) };
        let mut input = TurboShake128RoundInput::Message(message);
        for round in 0..last {
            input = self.with_prefixes(builder, input, round)?;
            let state = self.append_round(builder, input, round)?;
            input = TurboShake128RoundInput::State { state, prefixes: None };
        }
        // The packed labels start their chains at an accumulator, so only the chain bits read
        // the last round's prefixes.
        if self.output_reads_prefixes() {
            input = self.with_prefixes(builder, input, last)?;
        }
        self.append_outputs(builder, input, unit)
    }

    /// `input` with the prefixes of a later round `round` appended.
    fn with_prefixes(
        &self,
        builder: &mut ExponentLutProgramBuilder,
        input: TurboShake128RoundInput,
        round: usize,
    ) -> Result<TurboShake128RoundInput, TurboShake128Error> {
        Ok(match input {
            TurboShake128RoundInput::State { state, prefixes: None } if round > 0 => {
                let prefixes = Some(self.append_prefixes(builder, state, round)?);
                TurboShake128RoundInput::State { state, prefixes }
            }
            input => input,
        })
    }

    /// Members `K * 1600` of a state family (instance-major).
    pub fn state_members(&self) -> usize {
        self.instances() * STATE_BITS
    }

    /// Members `K * 320` of a parity-block prefix family (instance-major).
    pub fn prefix_members(&self) -> usize {
        self.instances() * PARITY_BLOCKS
    }

    /// The public scale `2^r` of the state read by round `r > 0` (the output of round `r - 1`),
    /// as `r`; its prefixes carry the same scale.
    pub const fn state_scale_log2(&self, round: usize) -> u32 {
        round as u32
    }

    /// [`Self::state_scale_log2`] of round `round`, an integer expression.
    fn state_scale(&self, round: &IntExpr) -> PublicScale {
        PublicScale { log2: round.clone(), max: self.rounds as u32 - 1 }
    }

    /// Whether [`Self::append_outputs`] reads the last round's prefixes when they are given: the
    /// chain bits do when the last round is a later round.
    pub const fn output_reads_prefixes(&self) -> bool {
        self.chain_bits > 0 && self.rounds > 1
    }

    /// Append the theta parity-block prefixes of a later round `round` (`0 < round`) of the
    /// state family `state` read by that round: member `i * 320 + 64 x + z` is the chain of the
    /// ten leaves of `D[x, z]` (columns `x - 1` and `x + 1`) of instance `i`.
    pub fn append_prefixes(
        &self,
        builder: &mut ExponentLutProgramBuilder,
        state: WireId,
        round: usize,
    ) -> Result<WireId, TurboShake128Error> {
        if round == 0 || round >= self.rounds {
            return Err(TurboShake128Error::InvalidDeclaration("prefixes need a later round"));
        }
        self.append_variable_prefixes(builder, state, &IntExpr::constant(round))
    }

    /// [`Self::append_prefixes`] of the round `round`, an integer expression of variables the
    /// lowering binds (`IntExpr::Var`) whose value is a later round (`0 < round < rounds`): one
    /// Program serves every later round.
    pub fn append_variable_prefixes(
        &self,
        builder: &mut ExponentLutProgramBuilder,
        state: WireId,
        round: &IntExpr,
    ) -> Result<WireId, TurboShake128Error> {
        let mut source = Source::Round { state, scale: self.state_scale(round), prefixes: None };
        source.share_prefixes(builder, self.instances())?;
        let Source::Round { prefixes: Some(prefixes), .. } = source else {
            unreachable!("a later round shares its prefixes")
        };
        Ok(prefixes)
    }

    /// Append round `round` but the last (see [`Self::append_outputs`]): the state family after
    /// that round, of [`Self::state_members`] members with payload `2^(round + 1) (-1)^bit`.
    /// Round 0 reads the message, a later round the state of round `round - 1` and, when given,
    /// its prefixes (without them every chi input chains all eleven of its leaves).
    pub fn append_round(
        &self,
        builder: &mut ExponentLutProgramBuilder,
        input: TurboShake128RoundInput,
        round: usize,
    ) -> Result<WireId, TurboShake128Error> {
        if round + 1 >= self.rounds {
            return Err(TurboShake128Error::InvalidDeclaration("the last round outputs the hash"));
        }
        let source = self.source(input, round)?;
        self.append_source_round(builder, &source, &IntExpr::constant(round))
    }

    /// [`Self::append_round`] of the round `round`, an integer expression of variables the
    /// lowering binds (`IntExpr::Var`) whose value is a later round but the last
    /// (`0 < round < rounds - 1`), reading the state and prefixes `input`: one Program serves
    /// every such round.
    pub fn append_variable_round(
        &self,
        builder: &mut ExponentLutProgramBuilder,
        input: TurboShake128RoundInput,
        round: &IntExpr,
    ) -> Result<WireId, TurboShake128Error> {
        let TurboShake128RoundInput::State { state, prefixes } = input else {
            return Err(TurboShake128Error::InvalidDeclaration("a variable round reads a state"));
        };
        let source = Source::Round { state, scale: self.state_scale(round), prefixes };
        self.append_source_round(builder, &source, round)
    }

    /// The state family after round `round` of `source`.
    fn append_source_round(
        &self,
        builder: &mut ExponentLutProgramBuilder,
        source: &Source,
        round: &IntExpr,
    ) -> Result<WireId, TurboShake128Error> {
        let range = FamilyRange::full(IntExpr::constant(self.state_members()))?;
        Ok(builder.parallel_materialized(range, |builder, slot| {
            let instance = floor_div(slot.clone(), STATE_BITS);
            let position = slot % STATE_BITS;
            self.chi_output(builder, source, &instance, &position, round)
        })?)
    }

    /// Append the last round, which outputs the labels and chain bits; `unit` is the unit
    /// encoding (payload `+1`) the packed labels start at.
    pub fn append_outputs(
        &self,
        builder: &mut ExponentLutProgramBuilder,
        input: TurboShake128RoundInput,
        unit: Operand<WireId>,
    ) -> Result<TurboShake128Wires, TurboShake128Error> {
        let last = self.rounds - 1;
        let source = self.source(input, last)?;
        let labels = if self.labels == 0 {
            None
        } else {
            let range = FamilyRange::full(IntExpr::constant(self.labels))?;
            Some(builder.parallel_materialized(range, |builder, label| {
                self.packed_label(builder, &source, &label, &IntExpr::constant(last), &unit)
            })?)
        };
        let chain = if self.chain_bits == 0 {
            None
        } else {
            let range = FamilyRange::full(IntExpr::constant(self.chain_bits))?;
            Some(builder.parallel_materialized(range, |builder, bit| {
                let stream = bit + self.labels * self.label_bits as usize;
                let (instance, position) = stream_position(stream);
                self.chi_output(builder, &source, &instance, &position, &IntExpr::constant(last))
            })?)
        };
        Ok(TurboShake128Wires { labels, chain })
    }

    /// The chi source of round `round` reading `input`.
    fn source(
        &self,
        input: TurboShake128RoundInput,
        round: usize,
    ) -> Result<Source, TurboShake128Error> {
        match input {
            TurboShake128RoundInput::Message(message) if round == 0 => {
                Ok(Source::Input { message, tables: InputTables::new(self) })
            }
            TurboShake128RoundInput::State { state, prefixes } if round > 0 => {
                let scale = self.state_scale(&IntExpr::constant(round));
                Ok(Source::Round { state, scale, prefixes })
            }
            _ => Err(TurboShake128Error::InvalidDeclaration(
                "round 0 reads the message and a later round the state",
            )),
        }
    }

    /// The public plaintexts of the input family for `message`: `(-1)^bit` per message bit and
    /// `+1` for the unit.
    pub fn input_plaintexts<P: Poly>(&self, parameters: &P::Params, message: &[u8]) -> Vec<P> {
        assert_eq!(message.len(), self.message_bytes, "the message length is fixed");
        let one = P::from_usize_to_constant(parameters, 1);
        (0..8 * self.message_bytes)
            .map(
                |bit| {
                    if message[bit / 8] >> (bit % 8) & 1 == 1 { -one.clone() } else { one.clone() }
                },
            )
            .chain(std::iter::once(one.clone()))
            .collect()
    }

    /// The payload `4^w X^(stride a)` of a label with value `a`.
    pub fn label_payload<P: Poly>(&self, parameters: &P::Params, value: usize) -> P {
        signed_monomial(
            parameters,
            self.ring_dimension,
            1i64 << self.label_scale_log2(),
            value * self.stride(),
        )
    }

    /// The payload `2^rounds (-1)^bit` of a chain bit.
    pub fn chain_payload<P: Poly>(&self, parameters: &P::Params, bit: bool) -> P {
        let scale = 1i64 << self.chain_scale_log2();
        signed_monomial(parameters, self.ring_dimension, if bit { -scale } else { scale }, 0)
    }

    /// The labels and chain bits of `message` from the stream bits `stream` (at least
    /// [`Self::output_bits`] bits, least significant first in each byte).
    pub fn split_stream(&self, stream: &[u8]) -> (Vec<usize>, Vec<bool>) {
        let bit = |index: usize| stream[index / 8] >> (index % 8) & 1 == 1;
        let width = self.label_bits as usize;
        let labels = (0..self.labels)
            .map(|label| (0..width).map(|j| usize::from(bit(width * label + j)) << j).sum())
            .collect();
        let chain = (0..self.chain_bits).map(|index| bit(width * self.labels + index)).collect();
        (labels, chain)
    }

    /// The padded block of instance `instance` with every message bit zero.
    fn constant_block(&self, instance: usize) -> [u8; RATE_BYTES] {
        let mut block = [0u8; RATE_BYTES];
        block[..self.tag.len()].copy_from_slice(&self.tag);
        let counter = self.tag.len();
        block[counter..counter + COUNTER_BYTES].copy_from_slice(&(instance as u32).to_be_bytes());
        block[counter + COUNTER_BYTES + self.message_bytes] ^= DOMAIN_BYTE;
        block[RATE_BYTES - 1] ^= 0x80;
        block
    }

    /// The index in [`ROUND_CONSTANTS`] of round `round` of Keccak-p[1600, `rounds`].
    fn round_constant(&self, round: usize) -> usize {
        KECCAK_ROUNDS - self.rounds + round
    }

    /// The iota bit of round `round` at `position`: a table read of the round's constant for a
    /// constant round, and of the constants of every round of the permutation otherwise.
    fn iota(&self, position: &IntExpr, round: &IntExpr) -> IntExpr {
        if let IntExpr::Const(round) = round.canonicalize() {
            let round = num_traits::ToPrimitive::to_usize(&round).expect("a round index");
            return iota_bit(position, self.round_constant(round));
        }
        let bits = (0..self.rounds)
            .flat_map(|round| {
                let constant = ROUND_CONSTANTS[self.round_constant(round)];
                (0..64).map(move |z| (constant >> z & 1) as usize)
            })
            .collect::<Vec<_>>();
        let mut lane = vec![0; 25];
        lane[0] = 1;
        select(&floor_div(position.clone(), 64), &lane) *
            select(&(round.clone() * 64usize + position % 64usize), &bits)
    }

    /// The first message bit's state position.
    fn message_offset(&self) -> usize {
        8 * (self.tag.len() + COUNTER_BYTES)
    }

    /// The round-`round` chi output at `(instance, position)`: payload `2^(scale + 1) mu'`.
    fn chi_output(
        &self,
        builder: &mut ExponentLutProgramBuilder,
        source: &Source,
        instance: &IntExpr,
        position: &IntExpr,
        round: &IntExpr,
    ) -> Result<Operand<WireId>, ProgramValidationError> {
        let terms = match source.prefix(instance, position) {
            // `T_1` is the shared theta parity block times the own leaf (block, then self).
            Some(prefix) => {
                let own = source.leaf(instance, position, 0, 0);
                let first = builder.mul_public_by_scale(own, prefix, source.scale())?;
                let first = Operand::Scalar(first);
                self.chi_terms(builder, source, (instance, position), first, None, None)?
            }
            None => {
                let first = source.leaf(instance, position, 0, 0);
                self.chi_terms(builder, source, (instance, position), first, Some(1), None)?
            }
        };
        self.combine(builder, terms, position, round)
    }

    /// The chi chains `T_1 .. T_4` from `start`, multiplying the leaves of the first chi input
    /// from leaf `skip` (1 when `start` is that input's first leaf, 0 for an accumulator, and
    /// `None` when `start` already holds them all).
    fn chi_terms(
        &self,
        builder: &mut ExponentLutProgramBuilder,
        source: &Source,
        (instance, position): (&IntExpr, &IntExpr),
        start: Operand<WireId>,
        skip: Option<usize>,
        leaves: Option<&[Vec<Operand<WireId>>; 3]>,
    ) -> Result<[Operand<WireId>; 4], ProgramValidationError> {
        let scale = source.scale();
        // A chain is one loop over its leaves: iteration `k` multiplies by leaf `from + k`, the
        // leaves read once before the loop.
        let chain = |builder: &mut ExponentLutProgramBuilder,
                     accumulator: Operand<WireId>,
                     input: usize,
                     from: usize|
         -> Result<Operand<WireId>, ProgramValidationError> {
            let leaves = (from..source.width())
                .map(|leaf| match leaves {
                    Some(leaves) => leaves[input][leaf].clone(),
                    None => source.leaf(instance, position, input, leaf),
                })
                .collect::<Vec<_>>();
            let accumulator = public_chain(builder, accumulator, leaves, &scale)?;
            let sign = source.sign(instance, position, input);
            self.signed(builder, accumulator, sign)
        };
        let t1 = chain(builder, start, 0, skip.unwrap_or(source.width()))?;
        let t2 = chain(builder, t1.clone(), 1, 0)?;
        // `T_3` and `T_4` multiply the same leaves (of the third chi input): one step reads
        // each leaf for both chains, so a leaf is needed only in that step.
        let (mut t3, mut t4) = (t1.clone(), t2.clone());
        for leaf in 0..source.width() {
            let leaf = match leaves {
                Some(leaves) => leaves[2][leaf].clone(),
                None => source.leaf(instance, position, 2, leaf),
            };
            t3 = Operand::Scalar(builder.mul_public_by_scale(leaf.clone(), t3, scale.clone())?);
            t4 = Operand::Scalar(builder.mul_public_by_scale(leaf, t4, scale.clone())?);
        }
        let sign = source.sign(instance, position, 2);
        let t3 = self.signed(builder, t3, sign.clone())?;
        let t4 = self.signed(builder, t4, sign)?;
        Ok([t1, t2, t3, t4])
    }

    /// `eps (T_1 - T_2 + T_3 + T_4)` with the iota sign of round `round` at `position`.
    fn combine(
        &self,
        builder: &mut ExponentLutProgramBuilder,
        [t1, t2, t3, t4]: [Operand<WireId>; 4],
        position: &IntExpr,
        round: &IntExpr,
    ) -> Result<Operand<WireId>, ProgramValidationError> {
        let negated = builder.mul_small_constant(t2, vec![-1], IntExpr::constant(0))?;
        let sum = builder.add(t1, Operand::Scalar(negated))?;
        let sum = builder.add(Operand::Scalar(sum), t3)?;
        let sum = builder.add(Operand::Scalar(sum), t4)?;
        self.signed(builder, Operand::Scalar(sum), self.iota(position, round))
    }

    /// Label `label`: from the unit, each bit `j` maps `acc` to `4 X^(c_j b_j) acc`.
    fn packed_label(
        &self,
        builder: &mut ExponentLutProgramBuilder,
        source: &Source,
        label: &IntExpr,
        round: &IntExpr,
        unit: &Operand<WireId>,
    ) -> Result<Operand<WireId>, ProgramValidationError> {
        // The bits are unrolled, for the reason [`public_chain`] gives.
        let width = self.label_bits as usize;
        let mut accumulator = unit.clone();
        for bit in 0..width {
            let (instance, position) = stream_position(label * width + bit);
            let terms = self.chi_terms(
                builder,
                source,
                (&instance, &position),
                accumulator.clone(),
                Some(0),
                None,
            )?;
            // `chi = eps (C_1 - C_2 + C_3 + C_4) = 2 (-1)^b acc`.
            let chi = self.combine(builder, terms, &position, round)?;
            let exponent = IntExpr::constant(self.stride() << bit);
            let shifted =
                builder.mul_small_constant(accumulator.clone(), vec![1], exponent.clone())?;
            let even = builder.add(accumulator, Operand::Scalar(shifted))?;
            let even =
                builder.mul_small_constant(Operand::Scalar(even), vec![2], IntExpr::constant(0))?;
            let shifted = builder.mul_small_constant(chi.clone(), vec![-1], exponent)?;
            let odd = builder.add(chi, Operand::Scalar(shifted))?;
            accumulator =
                Operand::Scalar(builder.add(Operand::Scalar(even), Operand::Scalar(odd))?);
        }
        Ok(accumulator)
    }

    /// `value` times `X^(n bit)`, or `value` itself when `bit` is the constant zero.
    fn signed(
        &self,
        builder: &mut ExponentLutProgramBuilder,
        value: Operand<WireId>,
        bit: IntExpr,
    ) -> Result<Operand<WireId>, ProgramValidationError> {
        let bit = bit.canonicalize();
        if bit == IntExpr::constant(0) {
            return Ok(value);
        }
        let shift = (bit % 2usize) * self.ring_dimension;
        Ok(Operand::Scalar(builder.mul_small_constant(value, vec![1], shift)?))
    }
}

/// The input of one round of [`TurboShake128LabelHash`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TurboShake128RoundInput {
    /// Round 0 reads the message family.
    Message(WireId),
    /// A later round `r` reads the public state family after round `r - 1` (payload
    /// `2^r (-1)^bit`) and, when given, the family of its parity-block prefixes
    /// ([`TurboShake128LabelHash::append_prefixes`]).
    State { state: WireId, prefixes: Option<WireId> },
}

/// The round input read by a chi output.
enum Source {
    /// Round 1: the input family and the padded leaf lists.
    Input { message: WireId, tables: InputTables },
    /// A later round: the previous round's state family, of public scale `scale`, and once
    /// shared, its family of theta parity-block prefixes.
    Round { state: WireId, scale: PublicScale, prefixes: Option<WireId> },
}

impl Source {
    /// Materialize the theta parity-block prefixes of a later round: member
    /// `i * 320 + 64 x + z` is the chain of the ten leaves of `D[x, z]` (columns `x - 1` and
    /// `x + 1`), started at its first leaf. Each chi output's first input reads the prefix of its
    /// theta coordinates, which five outputs share; prefixes stay on the accumulator side. Round
    /// 1 keeps unshared padded lists.
    fn share_prefixes(
        &mut self,
        builder: &mut ExponentLutProgramBuilder,
        instances: usize,
    ) -> Result<(), ProgramValidationError> {
        let Self::Round { state, scale, prefixes } = self else {
            return Ok(());
        };
        let (state, scale) = (*state, scale.clone());
        let range = FamilyRange::full(IntExpr::constant(instances * PARITY_BLOCKS))?;
        *prefixes = Some(builder.parallel_materialized(range, |builder, slot| {
            let instance = floor_div(slot.clone(), PARITY_BLOCKS);
            let block = slot % PARITY_BLOCKS;
            let (x, z) = (floor_div(block.clone(), 64), block % 64usize);
            let leaf = |index: usize| Operand::Member {
                family: state,
                index: &instance * STATE_BITS + parity_source(&x, &z, index),
            };
            let leaves = (1..THETA_SOURCES - 1).map(leaf).collect();
            public_chain(builder, leaf(0), leaves, &scale)
        })?);
        Ok(())
    }

    /// The shared parity-block prefix of the first chi input of the output at
    /// `(instance, position)`, once materialized.
    fn prefix(&self, instance: &IntExpr, position: &IntExpr) -> Option<Operand<WireId>> {
        let Self::Round { prefixes: Some(prefixes), .. } = self else {
            return None;
        };
        let (x, _, z) = theta_coordinates(position, 0);
        Some(Operand::Member {
            family: *prefixes,
            index: instance * PARITY_BLOCKS + x * 64usize + z,
        })
    }

    /// The leaves multiplied per chi input.
    fn width(&self) -> usize {
        match self {
            Self::Input { tables, .. } => tables.width,
            Self::Round { .. } => THETA_SOURCES,
        }
    }

    fn scale(&self) -> PublicScale {
        match self {
            Self::Input { .. } => PublicScale::constant(0),
            Self::Round { scale, .. } => scale.clone(),
        }
    }

    /// Leaf `leaf` of chi input `input` of the output at `(instance, position)`.
    fn leaf(
        &self,
        instance: &IntExpr,
        position: &IntExpr,
        input: usize,
        leaf: usize,
    ) -> Operand<WireId> {
        match self {
            Self::Input { message, tables } => Operand::Member {
                family: *message,
                index: select(position, &tables.leaves[input][leaf]),
            },
            Self::Round { state, .. } => Operand::Member {
                family: *state,
                index: instance * STATE_BITS + theta_source(position, input, leaf),
            },
        }
    }

    /// The parity of the constants of chi input `input` at `(instance, position)`.
    fn sign(&self, instance: &IntExpr, position: &IntExpr, input: usize) -> IntExpr {
        match self {
            // Every table has constant entries, so each is one table read: a Select whose
            // entries depend on the loop index would branch once per entry.
            Self::Input { tables, .. } => (0..tables.counter_bits).fold(
                select(position, &tables.parities[input]),
                |sum, shift| {
                    let occurs = &tables.counter_occurrences[input][shift];
                    if occurs.iter().all(|occurs| *occurs == 0) {
                        sum
                    } else {
                        sum + select(position, occurs) *
                            (floor_div(instance.clone(), 1usize << shift) % 2usize)
                    }
                },
            ),
            Self::Round { .. } => IntExpr::constant(0),
        }
    }
}

/// Theta sources of one state bit: itself and the two neighboring columns.
const THETA_SOURCES: usize = 11;

/// Round-1 tables: per chi input and leaf slot, the input member of every state position; per
/// chi input the parity of the constants of every position with a zero counter, and per counter
/// bit `s < counter_bits` whether bit `s` of the instance flips that parity.
struct InputTables {
    width: usize,
    leaves: [Vec<Vec<usize>>; 3],
    parities: [Vec<usize>; 3],
    counter_bits: usize,
    counter_occurrences: [Vec<Vec<usize>>; 3],
}

impl InputTables {
    fn new(hash: &TurboShake128LabelHash) -> Self {
        let base = hash.constant_block(0);
        let offset = hash.message_offset();
        let message_bits = 8 * hash.message_bytes;
        let counter = 8 * hash.tag.len()..8 * (hash.tag.len() + COUNTER_BYTES);
        let unit = message_bits;
        let constant_bit = |bit: usize| bit < RATE_BITS && base[bit / 8] >> (bit % 8) & 1 == 1;
        let mut lists = [vec![], vec![], vec![]];
        let mut signs: [Vec<(usize, Vec<usize>)>; 3] = [vec![], vec![], vec![]];
        for position in 0..STATE_BITS {
            for input in 0..3 {
                let mut leaves = Vec::new();
                let mut parity = 0usize;
                let mut shifts = Vec::new();
                for source in chi_input_sources(position, input) {
                    if (offset..offset + message_bits).contains(&source) {
                        leaves.push(source - offset);
                    } else {
                        parity += usize::from(constant_bit(source));
                        if counter.contains(&source) {
                            // Bit `t` of counter byte `j` is bit `8 (3 - j) + t` of BE32(i).
                            let byte = source / 8 - hash.tag.len();
                            shifts.push(8 * (COUNTER_BYTES - 1 - byte) + source % 8);
                        }
                    }
                }
                lists[input].push(leaves);
                signs[input].push((parity % 2, shifts));
            }
        }
        let width = lists.iter().flatten().map(Vec::len).max().unwrap_or(0).max(1);
        let leaves = lists.map(|lists| {
            (0..width)
                .map(|slot| {
                    lists.iter().map(|leaves| leaves.get(slot).copied().unwrap_or(unit)).collect()
                })
                .collect()
        });
        // Instances are below `K`, so only the counter bits below `bits(K - 1)` vary.
        let counter_bits = (usize::BITS - (hash.instances() - 1).leading_zeros()) as usize;
        let parities = signs.clone().map(|signs| signs.iter().map(|(parity, _)| *parity).collect());
        let counter_occurrences = signs.map(|signs| {
            (0..counter_bits)
                .map(|bit| {
                    signs
                        .iter()
                        .map(|(_, shifts)| shifts.iter().filter(|shift| **shift == bit).count() % 2)
                        .collect()
                })
                .collect()
        });
        Self { width, leaves, parities, counter_bits, counter_occurrences }
    }
}

/// `numerator / denominator` rounded down.
fn floor_div(numerator: IntExpr, denominator: usize) -> IntExpr {
    IntExpr::FloorDiv(Box::new(numerator), Box::new(IntExpr::constant(denominator))).canonicalize()
}

/// The constant `table[selector]`.
fn select(selector: &IntExpr, table: &[usize]) -> IntExpr {
    IntExpr::Select {
        selector: Box::new(selector.clone()),
        branches: table.iter().copied().map(IntExpr::constant).collect(),
    }
}

/// `(instance, position)` of stream bit `stream`.
fn stream_position(stream: IntExpr) -> (IntExpr, IntExpr) {
    (floor_div(stream.clone(), RATE_BITS), stream % RATE_BITS)
}

/// The iota sign bit of `round` at `position`: bit `z` of the round constant in lane `(0, 0)`.
fn iota_bit(position: &IntExpr, round: usize) -> IntExpr {
    let mut lane = vec![0; 25];
    lane[0] = 1;
    let bits = (0..64).map(|z| (ROUND_CONSTANTS[round] >> z & 1) as usize).collect::<Vec<_>>();
    select(&floor_div(position.clone(), 64), &lane) * select(&(position % 64usize), &bits)
}

/// `accumulator` multiplied in turn by each of `leaves` (each the public left operand, of scale
/// `scale`). The chain is unrolled rather than one loop over its leaves: a loop would read every
/// leaf before its first iteration and hold them all, while an unrolled chain reads each leaf
/// where it is multiplied, so a backend reads the leaves of an artifact or spilled state family
/// one at a time.
fn public_chain(
    builder: &mut ExponentLutProgramBuilder,
    accumulator: Operand<WireId>,
    leaves: Vec<Operand<WireId>>,
    scale: &PublicScale,
) -> Result<Operand<WireId>, ProgramValidationError> {
    leaves.into_iter().try_fold(accumulator, |accumulator, leaf| {
        Ok(Operand::Scalar(builder.mul_public_by_scale(leaf, accumulator, scale.clone())?))
    })
}

/// The theta coordinates `(x_s, y_s, z_s)` of chi input `input` of the output at `position`, as
/// integer expressions (see [`chi_input_sources`]).
fn theta_coordinates(position: &IntExpr, input: usize) -> (IntExpr, IntExpr, IntExpr) {
    let x = floor_div(position.clone(), 64) % 5usize;
    let y = floor_div(position.clone(), 320);
    let z = position % 64usize;
    let source_x = (x.clone() + input + y * 3usize) % 5usize;
    let source_y = (x + input) % 5usize;
    let offsets = RHO_OFFSETS.iter().flatten().map(|offset| *offset as usize).collect::<Vec<_>>();
    let offset = select(&(source_y.clone() * 5usize + source_x.clone()), &offsets);
    let source_z = (z + 64usize - offset) % 64usize;
    (source_x, source_y, source_z)
}

/// The state position of `(x, y, z)`.
fn place(x: IntExpr, y: IntExpr, z: IntExpr) -> IntExpr {
    (y * 5usize + x) * 64usize + z
}

/// The state position of leaf `index < 10` of the theta parity block `D[x, z]`: `(x - 1, y', z)`
/// and then `(x + 1, y', z - 1)` for every `y'`.
fn parity_source(x: &IntExpr, z: &IntExpr, index: usize) -> IntExpr {
    if index < 5 {
        place((x + 4usize) % 5usize, IntExpr::constant(index), z.clone())
    } else {
        place((x + 1usize) % 5usize, IntExpr::constant(index - 5), (z + 63usize) % 64usize)
    }
}

/// The state position of theta source `leaf` of chi input `input` of the output at `position`:
/// the own bit, then the parity block (see [`chi_input_sources`]).
fn theta_source(position: &IntExpr, input: usize, leaf: usize) -> IntExpr {
    let (x, y, z) = theta_coordinates(position, input);
    match leaf {
        0 => place(x, y, z),
        _ => parity_source(&x, &z, leaf - 1),
    }
}

/// The theta sources of chi input `input` (`B[x + input, y, z]`) of the output at `position`:
/// the theta output at `(x_s, y_s, z_s) = ((x + input + 3 y) mod 5, x + input,
/// z - r[x_s, y_s])` is the XOR of the state bits at `(x_s, y_s, z_s)`, `(x_s - 1, y', z_s)`
/// and `(x_s + 1, y', z_s - 1)` for every `y'`, in this order.
pub fn chi_input_sources(position: usize, input: usize) -> [usize; THETA_SOURCES] {
    let (x, y, z) = (position / 64 % 5, position / 320, position % 64);
    let source_x = (x + input + 3 * y) % 5;
    let source_y = (x + input) % 5;
    let source_z = (z + 64 - RHO_OFFSETS[source_y][source_x] as usize) % 64;
    let place = |x: usize, y: usize, z: usize| 64 * (5 * y + x) + z;
    let mut sources = [0; THETA_SOURCES];
    sources[0] = place(source_x, source_y, source_z);
    for row in 0..5 {
        sources[1 + row] = place((source_x + 4) % 5, row, source_z);
        sources[6 + row] = place((source_x + 1) % 5, row, (source_z + 63) % 64);
    }
    sources
}

/// The signed monomial `coefficient X^exponent` in the negacyclic ring of `ring_dimension`.
fn signed_monomial<P: Poly>(
    parameters: &P::Params,
    ring_dimension: usize,
    coefficient: i64,
    exponent: usize,
) -> P {
    let exponent = exponent % (2 * ring_dimension);
    let magnitude = P::from_usize_to_constant(parameters, coefficient.unsigned_abs() as usize);
    let rotation = P::const_rotate_poly(parameters, exponent % ring_dimension) * magnitude;
    if (exponent >= ring_dimension) != (coefficient < 0) { -rotation } else { rotation }
}

/// Keccak-p[1600, `rounds`] (the last `rounds` rounds of Keccak-f[1600]) on lanes
/// `state[x + 5 y]`.
pub fn keccak_p(state: &mut [u64; 25], rounds: usize) {
    for constant in &ROUND_CONSTANTS[KECCAK_ROUNDS - rounds..] {
        let columns: [u64; 5] =
            std::array::from_fn(|x| (0..5).fold(0, |parity, y| parity ^ state[x + 5 * y]));
        for x in 0..5 {
            let mix = columns[(x + 4) % 5] ^ columns[(x + 1) % 5].rotate_left(1);
            for y in 0..5 {
                state[x + 5 * y] ^= mix;
            }
        }
        let mut permuted = [0u64; 25];
        for x in 0..5 {
            for y in 0..5 {
                permuted[y + 5 * ((2 * x + 3 * y) % 5)] =
                    state[x + 5 * y].rotate_left(RHO_OFFSETS[y][x]);
            }
        }
        for y in 0..5 {
            for x in 0..5 {
                state[x + 5 * y] = permuted[x + 5 * y] ^
                    (!permuted[(x + 1) % 5 + 5 * y] & permuted[(x + 2) % 5 + 5 * y]);
            }
        }
        state[0] ^= constant;
    }
}

impl TurboShake128LabelHash {
    /// The stream of [`Self::output_bits`] bits computed in the clear with Keccak-p[1600,
    /// [`Self::rounds`]] (for 12 rounds TurboSHAKE128, for 24 rounds SHAKE128), bytes least
    /// significant bit first.
    pub fn reference_stream(&self, message: &[u8]) -> Vec<u8> {
        assert_eq!(message.len(), self.message_bytes, "the message length is fixed");
        let mut stream = Vec::new();
        for instance in 0..self.instances() {
            let mut block = self.constant_block(instance);
            let start = self.tag.len() + COUNTER_BYTES;
            block[start..start + message.len()].copy_from_slice(message);
            let mut state = [0u64; 25];
            for (lane, bytes) in block.chunks(8).enumerate() {
                state[lane] = u64::from_le_bytes(bytes.try_into().expect("eight bytes"));
            }
            keccak_p(&mut state, self.rounds);
            stream.extend(state[..RATE_BYTES / 8].iter().flat_map(|lane| lane.to_le_bytes()));
        }
        stream.truncate(self.output_bits().div_ceil(8));
        stream
    }
}

/// A standalone Program around [`TurboShake128LabelHash::append`] whose root outputs are the labels
/// and then the chain bits, one output per member, so its outputs (not its hash) grow with
/// `v + c`. Tests verify encodings with it; applications append the hash to their own Program.
#[derive(Clone, Debug)]
pub struct TurboShake128LabelProgram {
    hash: TurboShake128LabelHash,
    program: ExponentLutProgram,
    message: WireId,
}

impl TurboShake128LabelProgram {
    pub fn new(hash: TurboShake128LabelHash) -> Result<Self, TurboShake128Error> {
        let mut builder = ExponentLutProgramBuilder::new();
        let message = builder.input_family(
            IntExpr::constant(hash.input_members()),
            AttributeKnowledge::PublicPayload,
        )?;
        let wires = hash.append(&mut builder, message)?;
        let members = |family: Option<WireId>, count: usize| {
            family.into_iter().flat_map(move |family| {
                (0..count)
                    .map(move |index| Operand::Member { family, index: IntExpr::constant(index) })
            })
        };
        let outputs = members(wires.labels, hash.labels)
            .chain(members(wires.chain, hash.chain_bits))
            .collect();
        let program = builder.finish(outputs)?;
        Ok(Self { hash, program, message })
    }

    pub const fn hash(&self) -> &TurboShake128LabelHash {
        &self.hash
    }

    pub const fn program(&self) -> &ExponentLutProgram {
        &self.program
    }

    pub const fn message_input(&self) -> WireId {
        self.message
    }

    /// The plaintext binding of the input family for `message`.
    pub fn plaintext_inputs<P: Poly>(
        &self,
        parameters: &P::Params,
        message: &[u8],
    ) -> PlaintextInputs<WireId, P> {
        BTreeMap::from([(
            self.message,
            InputBinding::Family(self.hash.input_plaintexts(parameters, message)),
        )])
    }

    /// The expected output payloads, in output order, of the labels and chain bits `labels`
    /// and `chain` (for example from [`TurboShake128LabelHash::split_stream`]).
    pub fn expected_outputs<P: Poly>(
        &self,
        parameters: &P::Params,
        labels: &[usize],
        chain: &[bool],
    ) -> Vec<P> {
        labels
            .iter()
            .map(|label| self.hash.label_payload(parameters, *label))
            .chain(chain.iter().map(|bit| self.hash.chain_payload(parameters, *bit)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::program::{Body, Step};
    use mxx_ir_core::ParamEnv;
    use num_bigint::BigInt;
    use num_traits::ToPrimitive;
    use sha3::{
        Shake128, TurboShake128, TurboShake128Core,
        digest::{ExtendableOutput, Update, XofReader},
    };

    const TAG: &[u8; 16] = b"mxx-turboshake-t";

    /// `G(M)` from RustCrypto: `TurboSHAKE128(T || BE32(i) || M, D = 0x1F)` per instance for 12
    /// rounds, or `SHAKE128` (the same padding) for 24, concatenated.
    fn rustcrypto_stream(hash: &TurboShake128LabelHash, message: &[u8]) -> Vec<u8> {
        let mut stream = Vec::new();
        for instance in 0..hash.instances() {
            let input = [hash.tag(), &(instance as u32).to_be_bytes(), message].concat();
            let mut block = [0u8; RATE_BYTES];
            match hash.rounds() {
                TURBOSHAKE_ROUNDS => {
                    let mut hasher = TurboShake128::from_core(TurboShake128Core::new(DOMAIN_BYTE));
                    hasher.update(&input);
                    hasher.finalize_xof().read(&mut block);
                }
                KECCAK_ROUNDS => {
                    let mut hasher = Shake128::default();
                    hasher.update(&input);
                    hasher.finalize_xof().read(&mut block);
                }
                rounds => panic!("RustCrypto has no {rounds}-round reference"),
            }
            stream.extend_from_slice(&block);
        }
        stream.truncate(hash.output_bits().div_ceil(8));
        stream
    }

    fn message(length: usize, seed: u8) -> Vec<u8> {
        (0..length).map(|index| (index as u8).wrapping_mul(29).wrapping_add(seed)).collect()
    }

    #[test]
    fn reference_stream_is_counter_mode_turboshake128() {
        for rounds in [TURBOSHAKE_ROUNDS, KECCAK_ROUNDS] {
            for length in [0, 1, 12, 13, 32, 64, 100, 147] {
                let hash =
                    TurboShake128LabelHash::new(TAG.to_vec(), length, 1024, 8, 0, 128, rounds)
                        .unwrap();
                assert_eq!(hash.instances(), 7);
                let message = message(length, 3);
                assert_eq!(
                    hash.reference_stream(&message),
                    rustcrypto_stream(&hash, &message),
                    "{rounds} rounds, message length {length}"
                );
            }
        }
    }

    #[test]
    fn index_expressions_match_the_host_schedule() {
        let mut env = ParamEnv::default();
        let slot = IntExpr::LoopIndex(0);
        let expressions: Vec<Vec<IntExpr>> = (0..3)
            .map(|input| (0..THETA_SOURCES).map(|leaf| theta_source(&slot, input, leaf)).collect())
            .collect();
        let iota: Vec<IntExpr> = (0..KECCAK_ROUNDS).map(|round| iota_bit(&slot, round)).collect();
        for position in 0..STATE_BITS {
            env.loop_indices.insert(0, position.into());
            for input in 0..3 {
                let sources = chi_input_sources(position, input);
                for leaf in 0..THETA_SOURCES {
                    assert_eq!(
                        expressions[input][leaf].evaluate(&env).unwrap(),
                        sources[leaf].into()
                    );
                }
            }
            for (round, expression) in iota.iter().enumerate() {
                let expected = position < 64 && ROUND_CONSTANTS[round] >> position & 1 == 1;
                assert_eq!(expression.evaluate(&env).unwrap(), usize::from(expected).into());
            }
        }
    }

    /// Every table lookup of the Program reads constant entries: the GPU lowers such a
    /// `Select` to one table read, but branches once per entry when an entry depends on the loop
    /// index.
    #[test]
    fn every_select_reads_a_constant_table() {
        fn check(expression: &IntExpr) {
            match expression {
                IntExpr::Select { selector, branches } => {
                    assert!(branches.iter().all(|branch| !branch.contains_loop_index()));
                    check(selector);
                }
                IntExpr::Add(lhs, rhs) |
                IntExpr::Sub(lhs, rhs) |
                IntExpr::Mul(lhs, rhs) |
                IntExpr::Div(lhs, rhs) |
                IntExpr::FloorDiv(lhs, rhs) |
                IntExpr::Rem(lhs, rhs) |
                IntExpr::RoundDiv(lhs, rhs) => {
                    check(lhs);
                    check(rhs);
                }
                IntExpr::Log2Ceil(value) => check(value),
                _ => {}
            }
        }
        fn operand(operand: &Operand<WireId>) {
            if let Operand::Member { index, .. } = operand {
                check(index);
            }
        }
        fn walk(body: &Body) {
            for step in &body.steps {
                match step {
                    Step::Add { lhs, rhs, .. } | Step::MulPublic { lhs, rhs, .. } => {
                        operand(lhs);
                        operand(rhs);
                    }
                    Step::MulSmallConstant { input, shift, .. } => {
                        operand(input);
                        check(shift);
                    }
                    Step::Parallel { body, .. } => walk(body),
                    Step::Fold { initial, body, .. } => {
                        operand(initial);
                        walk(body);
                    }
                    Step::Select { selector, candidates, .. } => {
                        check(selector);
                        candidates.iter().for_each(operand);
                    }
                    other => panic!("unexpected step in the hash: {other:?}"),
                }
            }
            body.outputs.iter().for_each(operand);
        }
        for rounds in [1, 2, TURBOSHAKE_ROUNDS] {
            let hash =
                TurboShake128LabelHash::new(TAG.to_vec(), 32, 1024, 8, 64, 128, rounds).unwrap();
            walk(TurboShake128LabelProgram::new(hash).unwrap().program().body());
        }
    }

    /// Public products per family member, scaled by the family counts.
    fn products(body: &Body) -> usize {
        body.steps
            .iter()
            .map(|step| match step {
                Step::MulPublic { .. } => 1,
                Step::Parallel { range, body, .. } => {
                    range.count().evaluate(&ParamEnv::default()).unwrap().to_usize().unwrap() *
                        products(body)
                }
                Step::Fold { count, body, .. } => count * products(body),
                _ => 0,
            })
            .sum()
    }

    /// A later round costs `320 * 9` prefix products, one own-leaf product and the `11 + 22`
    /// products of `T_2 .. T_4` per output (57,280 per instance); the production shape needs
    /// 4,493,248 products.
    #[test]
    fn shared_prefixes_cut_the_public_products() {
        let count = |rounds| {
            let hash =
                TurboShake128LabelHash::new(TAG.to_vec(), 32, 1024, 8, 0, 128, rounds).unwrap();
            products(TurboShake128LabelProgram::new(hash).unwrap().program().body())
        };
        assert_eq!(count(3) - count(2), 7 * 57_280);
        // Round 1 pads to three leaves per list, a later round costs 57,280, and the fused
        // last round 44 per label bit plus its packing.
        assert_eq!(count(TURBOSHAKE_ROUNDS), 4_493_248);
    }

    #[test]
    fn round_one_pads_to_the_longest_message_list() {
        // The widths of the payload simulator: the unpadded alpha_1 of 32 bytes is 18 and its
        // longest list has 3 leaves; an empty message starts every chain from the unit.
        for (length, width) in [(0, 1), (16, 3), (32, 3), (48, 4), (147, 9)] {
            let hash =
                TurboShake128LabelHash::new(TAG.to_vec(), length, 1, 8, 0, 128, TURBOSHAKE_ROUNDS)
                    .unwrap();
            assert_eq!(InputTables::new(&hash).width, width, "message length {length}");
        }
    }

    /// A public payload: coefficients of `X^0 .. X^(n-1)` modulo `X^n + 1`.
    type Clear = BTreeMap<usize, i128>;

    /// Evaluates a public-only Program in the clear, following its steps directly, and checks
    /// the scaled-product contract (the LHS payload is `2^k` times a signed monomial).
    struct ClearInterpreter {
        ring_dimension: usize,
        scalars: BTreeMap<WireId, Clear>,
        families: BTreeMap<WireId, Vec<Clear>>,
        env: ParamEnv,
    }

    impl ClearInterpreter {
        fn index(&self, expression: &IntExpr) -> usize {
            expression.evaluate(&self.env).unwrap().to_usize().unwrap()
        }

        fn operand(&self, operand: &Operand<WireId>) -> Clear {
            match operand {
                Operand::Scalar(wire) => self.scalars[wire].clone(),
                Operand::Member { family, index } => {
                    self.families[family][self.index(index)].clone()
                }
                Operand::Selected { .. } => panic!("no data in the hash"),
            }
        }

        fn multiply(&self, lhs: &Clear, rhs: &Clear) -> Clear {
            let n = self.ring_dimension;
            let mut product = Clear::new();
            for (left, a) in lhs {
                for (right, b) in rhs {
                    let exponent = left + right;
                    let (exponent, sign) =
                        if exponent >= n { (exponent - n, -1) } else { (exponent, 1) };
                    *product.entry(exponent).or_default() += sign * a * b;
                }
            }
            product.retain(|_, value| *value != 0);
            product
        }

        fn monomial(&self, coefficient: i128, shift: usize) -> Clear {
            let shift = shift % (2 * self.ring_dimension);
            let (exponent, sign) = if shift >= self.ring_dimension {
                (shift - self.ring_dimension, -1)
            } else {
                (shift, 1)
            };
            Clear::from([(exponent, sign * coefficient)])
        }

        fn run(&mut self, body: &Body, depth: u32) -> Vec<Clear> {
            for step in &body.steps {
                let (output, value) = match step {
                    Step::Add { lhs, rhs, output } => {
                        let mut sum = self.operand(lhs);
                        for (exponent, value) in self.operand(rhs) {
                            *sum.entry(exponent).or_default() += value;
                        }
                        sum.retain(|_, value| *value != 0);
                        (*output, sum)
                    }
                    Step::MulPublic { lhs, rhs, output, lhs_scale_log2, lhs_scale } => {
                        let log2 = lhs_scale
                            .as_ref()
                            .map_or(*lhs_scale_log2 as usize, |log2| self.index(log2));
                        assert!(log2 <= *lhs_scale_log2 as usize, "the largest scale");
                        let scale = 1i128 << log2;
                        let lhs = self.operand(lhs);
                        // The scaled-product contract: `2^k` times a signed monomial.
                        assert!(
                            lhs.len() == 1 && lhs.values().all(|value| value.abs() == scale),
                            "LHS scale contract"
                        );
                        let lhs = lhs
                            .into_iter()
                            .map(|(exponent, value)| (exponent, value / scale))
                            .collect();
                        (*output, self.multiply(&lhs, &self.operand(rhs)))
                    }
                    Step::MulSmallConstant { input, coefficients, shift, output } => {
                        let shift = self.index(shift);
                        let mut constant = Clear::new();
                        for (power, coefficient) in coefficients.iter().enumerate() {
                            for (exponent, value) in
                                self.monomial(i128::from(*coefficient), shift + power)
                            {
                                *constant.entry(exponent).or_default() += value;
                            }
                        }
                        (*output, self.multiply(&self.operand(input), &constant))
                    }
                    Step::Parallel { range, body, output, .. } => {
                        let count = self.index(range.count());
                        let members = (0..count)
                            .map(|member| {
                                self.env.loop_indices.insert(depth, BigInt::from(member));
                                let mut outputs = self.run(body, depth + 1);
                                outputs.remove(0)
                            })
                            .collect();
                        self.env.loop_indices.remove(&depth);
                        self.families.insert(*output, members);
                        continue;
                    }
                    Step::Fold { count, initial, accumulator, body, output } => {
                        let mut value = self.operand(initial);
                        for iteration in 0..*count {
                            self.env.loop_indices.insert(depth, BigInt::from(iteration));
                            self.scalars.insert(*accumulator, value);
                            value = self.run(body, depth + 1).remove(0);
                        }
                        self.env.loop_indices.remove(&depth);
                        (*output, value)
                    }
                    Step::Select { selector, candidates, output } => {
                        (*output, self.operand(&candidates[self.index(selector)]))
                    }
                    other => panic!("unexpected step in the hash: {other:?}"),
                };
                self.scalars.insert(output, value);
            }
            body.outputs.iter().map(|output| self.operand(output)).collect()
        }
    }

    fn interpreter(hash: &TurboShake128LabelHash) -> ClearInterpreter {
        ClearInterpreter {
            ring_dimension: hash.ring_dimension(),
            scalars: BTreeMap::new(),
            families: BTreeMap::new(),
            env: ParamEnv::default(),
        }
    }

    /// The input family payloads of `message`: the bit signs, then the unit.
    fn message_payloads(hash: &TurboShake128LabelHash, message: &[u8]) -> Vec<Clear> {
        let interpreter = interpreter(hash);
        (0..8 * message.len())
            .map(|bit| {
                interpreter.monomial(
                    1,
                    hash.ring_dimension() * usize::from(message[bit / 8] >> (bit % 8) & 1 == 1),
                )
            })
            .chain(std::iter::once(interpreter.monomial(1, 0)))
            .collect()
    }

    /// Compares output payloads with the payloads of the RustCrypto stream.
    fn check_outputs(hash: &TurboShake128LabelHash, message: &[u8], outputs: &[Clear]) {
        let interpreter = interpreter(hash);
        // Other round counts use the host Keccak-p reference, which the RustCrypto comparison
        // checks at 12 and 24 rounds.
        let stream = match hash.rounds() {
            TURBOSHAKE_ROUNDS | KECCAK_ROUNDS => rustcrypto_stream(hash, message),
            _ => hash.reference_stream(message),
        };
        let (labels, chain) = hash.split_stream(&stream);
        let expected = labels
            .iter()
            .map(|label| interpreter.monomial(1 << hash.label_scale_log2(), label * hash.stride()))
            .chain(chain.iter().map(|bit| {
                interpreter.monomial(
                    1 << hash.chain_scale_log2(),
                    hash.ring_dimension() * usize::from(*bit),
                )
            }))
            .collect::<Vec<_>>();
        assert_eq!(outputs.len(), expected.len());
        for (index, (output, expected)) in outputs.iter().zip(&expected).enumerate() {
            assert_eq!(output, expected, "output {index}");
        }
    }

    /// Runs the Program in the clear and compares every output payload with the payloads of the
    /// RustCrypto stream.
    fn check_clear(hash: TurboShake128LabelHash, message: &[u8]) {
        let program = TurboShake128LabelProgram::new(hash.clone()).unwrap();
        let mut interpreter = interpreter(&hash);
        interpreter.families.insert(program.message_input(), message_payloads(&hash, message));
        let outputs = interpreter.run(program.program().body(), 0);
        check_outputs(&hash, message, &outputs);
    }

    /// Every member of `family` as an output.
    fn members(family: WireId, count: usize) -> Vec<Operand<WireId>> {
        (0..count)
            .map(|index| Operand::Member { family, index: IntExpr::constant(index) })
            .collect()
    }

    /// The hash as one Program per part: round 0 from the message, every later round's prefixes
    /// and then the round from the previous Program's outputs bound as public input families, and
    /// the outputs from the last state, its prefixes and a unit input. With `variable`, the
    /// prefixes and the round of every later round but the last are one Program each, reading
    /// the round as a variable.
    fn check_round_by_round(hash: TurboShake128LabelHash, message: &[u8], variable: bool) {
        let public = AttributeKnowledge::PublicPayload;
        let round_variable = IntExpr::Var("round".into());
        let run_round =
            |program: ExponentLutProgram, inputs: Vec<(WireId, Vec<Clear>)>, round: usize| {
                let mut interpreter = interpreter(&hash);
                interpreter.env.integers.insert("round".into(), BigInt::from(round));
                for (wire, payloads) in inputs {
                    if matches!(program.inputs()[&wire].shape(), InputBinding::Scalar(())) {
                        interpreter.scalars.insert(wire, payloads[0].clone());
                    } else {
                        interpreter.families.insert(wire, payloads);
                    }
                }
                interpreter.run(program.body(), 0)
            };
        let run = |program, inputs| run_round(program, inputs, 0);
        let last = hash.rounds() - 1;
        let mut state = Vec::new();
        if last > 0 {
            let mut builder = ExponentLutProgramBuilder::new();
            let input =
                builder.input_family(IntExpr::constant(hash.input_members()), public).unwrap();
            let next = hash
                .append_round(&mut builder, TurboShake128RoundInput::Message(input), 0)
                .unwrap();
            let program = builder.finish(members(next, hash.state_members())).unwrap();
            state = run(program, vec![(input, message_payloads(&hash, message))]);
        }
        let prefixes = |round: usize, state: &[Clear]| {
            let mut builder = ExponentLutProgramBuilder::new();
            let input =
                builder.input_family(IntExpr::constant(hash.state_members()), public).unwrap();
            let prefixes = if variable {
                hash.append_variable_prefixes(&mut builder, input, &round_variable).unwrap()
            } else {
                hash.append_prefixes(&mut builder, input, round).unwrap()
            };
            let program = builder.finish(members(prefixes, hash.prefix_members())).unwrap();
            run_round(program, vec![(input, state.to_vec())], round)
        };
        for round in 1..last {
            let prefix_payloads = prefixes(round, &state);
            let mut builder = ExponentLutProgramBuilder::new();
            let input =
                builder.input_family(IntExpr::constant(hash.state_members()), public).unwrap();
            let prefix =
                builder.input_family(IntExpr::constant(hash.prefix_members()), public).unwrap();
            let round_input =
                TurboShake128RoundInput::State { state: input, prefixes: Some(prefix) };
            let next = if variable {
                hash.append_variable_round(&mut builder, round_input, &round_variable)
            } else {
                hash.append_round(&mut builder, round_input, round)
            }
            .unwrap();
            let program = builder.finish(members(next, hash.state_members())).unwrap();
            state = run_round(program, vec![(input, state), (prefix, prefix_payloads)], round);
        }
        let mut builder = ExponentLutProgramBuilder::new();
        let mut inputs = Vec::new();
        let round_input = if last == 0 {
            let input =
                builder.input_family(IntExpr::constant(hash.input_members()), public).unwrap();
            inputs.push((input, message_payloads(&hash, message)));
            TurboShake128RoundInput::Message(input)
        } else {
            let input =
                builder.input_family(IntExpr::constant(hash.state_members()), public).unwrap();
            let prefixes = hash.output_reads_prefixes().then(|| {
                let prefix =
                    builder.input_family(IntExpr::constant(hash.prefix_members()), public).unwrap();
                inputs.push((prefix, prefixes(last, &state)));
                prefix
            });
            inputs.push((input, state));
            TurboShake128RoundInput::State { state: input, prefixes }
        };
        let unit = builder.input(public).unwrap();
        inputs.push((unit, vec![interpreter(&hash).monomial(1, 0)]));
        let wires = hash.append_outputs(&mut builder, round_input, Operand::Scalar(unit)).unwrap();
        let outputs = wires
            .labels
            .into_iter()
            .flat_map(|labels| members(labels, hash.labels()))
            .chain(wires.chain.into_iter().flat_map(|chain| members(chain, hash.chain_bits())))
            .collect();
        let program = builder.finish(outputs).unwrap();
        let outputs = run(program, inputs);
        check_outputs(&hash, message, &outputs);
    }

    /// A chain of public products as one loop over selected leaves ([`Step::Fold`],
    /// [`Step::Select`]) computes the unrolled chain's payload.
    #[test]
    fn folded_chains_compute_the_unrolled_chain() {
        let hash = TurboShake128LabelHash::new(TAG.to_vec(), 7, 4, 5, 9, 16, 3).unwrap();
        let public = AttributeKnowledge::PublicPayload;
        let leaves = 5usize;
        let run = |fold: bool| {
            let mut builder = ExponentLutProgramBuilder::new();
            let input = builder.input_family(IntExpr::constant(leaves + 1), public).unwrap();
            let operands = (1..=leaves)
                .map(|leaf| Operand::Member { family: input, index: IntExpr::constant(leaf) });
            let start = Operand::Member { family: input, index: IntExpr::constant(0) };
            let output = if fold {
                let operands = operands.collect::<Vec<_>>();
                Operand::Scalar(
                    builder
                        .fold(leaves, start, |builder, accumulator, k| {
                            let leaf = builder.select(k, operands)?;
                            Ok(Operand::Scalar(
                                builder.mul_public(Operand::Scalar(leaf), accumulator)?,
                            ))
                        })
                        .unwrap(),
                )
            } else {
                operands.fold(start, |accumulator, leaf| {
                    Operand::Scalar(builder.mul_public(leaf, accumulator).unwrap())
                })
            };
            let program = builder.finish(vec![output]).unwrap();
            let mut interpreter = interpreter(&hash);
            let payloads = (0..=leaves)
                .map(|leaf| interpreter.monomial(1, leaf * hash.ring_dimension()))
                .collect();
            interpreter.families.insert(input, payloads);
            interpreter.run(program.body(), 0)
        };
        assert_eq!(run(true), run(false));
    }

    #[test]
    fn round_by_round_programs_compute_the_same_hash() {
        let hash =
            TurboShake128LabelHash::new(TAG.to_vec(), 32, 170, 8, 16, 128, TURBOSHAKE_ROUNDS)
                .unwrap();
        for variable in [false, true] {
            check_round_by_round(hash.clone(), &message(32, 7), variable);
            let hash = TurboShake128LabelHash::new(TAG.to_vec(), 7, 270, 5, 3, 16, 3).unwrap();
            check_round_by_round(hash, &message(7, 5), variable);
            let hash = TurboShake128LabelHash::new(TAG.to_vec(), 7, 4, 5, 9, 16, 1).unwrap();
            check_round_by_round(hash, &message(7, 9), variable);
        }
    }

    #[test]
    fn clear_payloads_are_turboshake128_across_instances() {
        // 170 labels of 8 bits and 16 chain bits span two instances; the labels of 8 bits read
        // whole bytes, the top bit's shift is `n`, and the chain bits read the last round's
        // prefixes.
        let hash =
            TurboShake128LabelHash::new(TAG.to_vec(), 32, 170, 8, 16, 128, TURBOSHAKE_ROUNDS)
                .unwrap();
        assert_eq!(hash.instances(), 2);
        check_clear(hash, &message(32, 7));
    }

    /// The production label shape: `v = 1024` labels of `Q = 256` over seven instances, plus a
    /// 256-bit chain. Minutes in a debug build.
    #[test]
    #[ignore = "seven full TurboSHAKE128 instances in the clear"]
    fn clear_payloads_of_the_production_label_vector() {
        let hash =
            TurboShake128LabelHash::new(TAG.to_vec(), 32, 1024, 8, 256, 512, TURBOSHAKE_ROUNDS)
                .unwrap();
        assert_eq!(hash.instances(), 7);
        check_clear(hash, &message(32, 11));
    }

    /// Labels of 5 bits cross the instance boundary at stream bit 1344 (label 268 reads bits
    /// 1340..1345), and three rounds of Keccak-p cover one later round with prefixes.
    #[test]
    fn clear_payloads_of_labels_across_instances_and_reduced_rounds() {
        let hash =
            TurboShake128LabelHash::new(TAG.to_vec(), 7, 270, 5, 3, 16, TURBOSHAKE_ROUNDS).unwrap();
        assert_eq!(hash.instances(), 2);
        check_clear(hash, &message(7, 5));
        let hash = TurboShake128LabelHash::new(TAG.to_vec(), 7, 4, 5, 9, 16, 3).unwrap();
        check_clear(hash, &message(7, 9));
    }

    #[test]
    fn clear_payloads_of_an_empty_message_and_odd_widths() {
        let hash =
            TurboShake128LabelHash::new(TAG.to_vec(), 0, 5, 3, 7, 16, TURBOSHAKE_ROUNDS).unwrap();
        check_clear(hash, &[]);
    }

    /// The worst-case noise bound of a production-size label shape
    /// (`n = 2^16`, 27 primes of 60 bits, base `2^6`, sigma 4), reported against the modulus.
    #[test]
    #[ignore = "noise simulation at production size"]
    fn noise_bound_at_production_parameters() {
        use crate::rns_bgg::{
            RnsBggDerivedWorkload, RnsBggNoiseBindings, RnsBggNoiseFamily, RnsBggNoiseValue,
            RnsBggParameters, RnsNoiseBuild,
        };
        let hash =
            TurboShake128LabelHash::new(TAG.to_vec(), 32, 1024, 8, 256, 1 << 16, TURBOSHAKE_ROUNDS)
                .unwrap();
        let program = TurboShake128LabelProgram::new(hash.clone()).unwrap();
        let level =
            mxx_backends::poly::dcrt::params::DCRTPolyParams::new(1 << 16, 27, 60, 6, None, None);
        let parameters = RnsBggParameters::new(vec![level], 1, 1).unwrap();
        let workload =
            RnsBggDerivedWorkload::from_program_with_parameters(&parameters, program.program())
                .unwrap();
        let input = RnsBggNoiseValue::sampled_encoding_sigma(
            &bigdecimal::BigDecimal::from(4),
            b"turboshake128-input",
            &[BigInt::from(1u8)],
        )
        .unwrap();
        let bindings = RnsBggNoiseBindings::new(
            BTreeMap::from([(
                program.message_input(),
                InputBinding::Family(RnsBggNoiseFamily {
                    element: input.independent_family_members(),
                    range: FamilyRange::full(IntExpr::constant(hash.input_members())).unwrap(),
                    max_nonzero_payloads: 1u8.into(),
                    row_length: None,
                }),
            )]),
            BTreeMap::new(),
        )
        .unwrap();
        let trace = crate::api::noise_simulate(
            program.program(),
            RnsNoiseBuild::new(&parameters, program.program(), &workload, &bindings).unwrap(),
            |_, outputs| Ok(outputs),
        )
        .unwrap();
        let modulus_bits =
            mxx_backends::poly::PolyParams::modulus(parameters.level(0).unwrap()).bits();
        let bits = |values: &[RnsBggNoiseValue]| {
            values.iter().map(|value| value.bound.error.bits()).max().unwrap()
        };
        let label_bits = bits(&trace.outputs[..hash.labels()]);
        let chain_bits = bits(&trace.outputs[hash.labels()..]);
        eprintln!(
            "TurboSHAKE128 label hash noise: labels 2^{label_bits}, chain bits 2^{chain_bits}, modulus 2^{modulus_bits}"
        );
        assert!(label_bits.max(chain_bits) + 2 < modulus_bits);
    }

    /// Counts public products as they are lowered; every family body is lowered once
    /// symbolically.
    #[derive(Default)]
    struct Counting {
        site: Option<crate::program::LoweringSite>,
        products: usize,
    }

    #[derive(Debug, thiserror::Error)]
    #[error("counting lowering error")]
    struct CountingError;

    impl crate::program::ExponentLUTProgramLowering for Counting {
        type Wire = ();
        type Rhs = ();
        type Error = CountingError;

        fn add(&mut self, _: &(), _: &()) -> Result<(), CountingError> {
            Ok(())
        }
        fn mul_public(&mut self, _: &(), _: &()) -> Result<(), CountingError> {
            self.products += 1;
            Ok(())
        }
        fn mul_private(&mut self, _: &(), _: &()) -> Result<(), CountingError> {
            Err(CountingError)
        }
        fn lut(&mut self, _: &(), _: &crate::program::LutTable) -> Result<(), CountingError> {
            Err(CountingError)
        }
        fn mul_small_constant(
            &mut self,
            _: &(),
            _: &[i64],
            _: &IntExpr,
        ) -> Result<(), CountingError> {
            Ok(())
        }
    }

    impl crate::program::ProgramLoweringContext for Counting {
        fn replace_site(
            &mut self,
            site: Option<crate::program::LoweringSite>,
        ) -> Option<crate::program::LoweringSite> {
            std::mem::replace(&mut self.site, site)
        }
    }

    struct CountingFamilies;

    impl crate::program::FamilyExecution<Counting> for CountingFamilies {
        type WireFamily = ();
        type RhsFamily = ();

        fn wire_at(
            &mut self,
            _: &mut Counting,
            _: &(),
            _: &IntExpr,
        ) -> Result<(), crate::program::ProgramLowerError<CountingError>> {
            Ok(())
        }
        fn wire_selected(
            &mut self,
            _: &mut Counting,
            _: &(),
            _: crate::program::DataId,
            _: &IntExpr,
        ) -> Result<(), crate::program::ProgramLowerError<CountingError>> {
            Ok(())
        }
        fn rhs_at(
            &mut self,
            _: &(),
            _: &IntExpr,
        ) -> Result<(), crate::program::ProgramLowerError<CountingError>> {
            Ok(())
        }
        fn rhs_selected(
            &mut self,
            _: &(),
            _: crate::program::DataId,
            _: &IntExpr,
        ) -> Result<(), crate::program::ProgramLowerError<CountingError>> {
            Ok(())
        }
        fn parallel(
            &mut self,
            evaluator: &mut Counting,
            _: &FamilyRange,
            body: impl FnOnce(
                &mut Self,
                &mut Counting,
                IntExpr,
            )
                -> Result<(), crate::program::ProgramLowerError<CountingError>>,
        ) -> Result<(), crate::program::ProgramLowerError<CountingError>> {
            body(self, evaluator, IntExpr::LoopIndex(0))
        }
        fn sum(
            &mut self,
            _: &mut Counting,
            _: &(),
            _: usize,
        ) -> Result<(), crate::program::ProgramLowerError<CountingError>> {
            Ok(())
        }
    }

    /// Lowers the hash where sampler families are deferred: a deferred row reads
    /// one label per block from a non-materialized family. Returns the products lowered by the
    /// lowering and then by `reads` deferred block reads, and whether both families exported.
    fn deferred_products(hash: &TurboShake128LabelHash, reads: usize) -> (usize, usize, bool) {
        let mut builder = ExponentLutProgramBuilder::new();
        let message = builder
            .input_family(
                IntExpr::constant(hash.input_members()),
                AttributeKnowledge::PublicPayload,
            )
            .unwrap();
        let wires = hash.append(&mut builder, message).unwrap();
        let (labels, chain) = (wires.labels.unwrap(), wires.chain.unwrap());
        let range = FamilyRange::full(IntExpr::constant(hash.labels())).unwrap();
        let read = builder
            .parallel(range, |builder, label| {
                builder
                    .mul_small_constant(
                        Operand::Member { family: labels, index: label },
                        vec![1],
                        IntExpr::constant(0),
                    )
                    .map(Operand::Scalar)
            })
            .unwrap();
        let row = builder.sum_columns(read, hash.labels()).unwrap();
        let program = builder.finish(vec![Operand::Scalar(row)]).unwrap();
        let mut environment = crate::program::Environment::new();
        environment.bind_wire(message, InputBinding::Family(())).unwrap();
        let mut evaluator = Counting::default();
        let (outputs, exported) = crate::program::lower_program_deferred_exporting(
            &program,
            &mut evaluator,
            &mut CountingFamilies,
            &mut environment,
            &[0].into(),
            &[labels, chain].into(),
        )
        .unwrap();
        let lowered = evaluator.products;
        let crate::program::LoweredOutput::Columns(columns) = &outputs[0] else {
            panic!("the label row is deferred");
        };
        for block in 0..reads {
            columns
                .block(
                    &program,
                    &mut evaluator,
                    &mut CountingFamilies,
                    &mut environment,
                    &IntExpr::constant(block),
                )
                .unwrap();
        }
        (lowered, evaluator.products - lowered, exported.len() == 2)
    }

    /// Under deferred lowering every round is lowered once: the lowered products grow by the
    /// same amount per round, do not depend on the label and chain counts, and reading deferred
    /// blocks that read the materialized labels lowers no product again.
    #[test]
    fn materialized_rounds_lower_once_under_deferred_lowering() {
        let products = |labels, chain, rounds| {
            let hash = TurboShake128LabelHash::new(TAG.to_vec(), 32, labels, 8, chain, 128, rounds)
                .unwrap();
            deferred_products(&hash, 3)
        };
        let (two, reread, exported) = products(16, 16, 2);
        assert!(exported);
        assert_eq!(reread, 0, "a deferred block read must not lower the hash again");
        let (three, ..) = products(16, 16, 3);
        let (four, ..) = products(16, 16, 4);
        assert_eq!(four - three, three - two, "the graph grows linearly with the rounds");
        assert_eq!(products(1024, 256, 3).0, three, "the graph does not grow with v or c");
    }
}
