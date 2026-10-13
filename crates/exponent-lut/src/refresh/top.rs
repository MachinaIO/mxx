//! Noise refresh to the top level.
//!
//! The target is a private Program output `c_x` at level `L` (both component
//! rows, payload `Delta_L r_j x G^st_L`, `r = (r_0, 1)`). The refresh outputs
//! the two rows of a level-0 private encoding of the same `x` under a new
//! mask `sigma = (sigma_0, 1)`:
//!
//! `y_j = sigma A'_(j) - Delta_0 r_j x G^st_0 + e'_j      (j = 0, 1)`
//!
//! in the format of a reference chain ([`TopReference`]: its level-0 primes
//! `Q_0` and level-1 primes `Q_1`, `Delta_0 = Q_0 / Q_1`, `G^st_0` the
//! gadget of `Q_1`), with `sigma_0` a uniform binary polynomial and `e'_j`
//! CBD vectors, both computed by the sparse-LWR PRF, and hash-sampled public
//! `A'_(j) in R_(Q_0)^(2 x w_0)`.
//!
//! # Chains
//!
//! Decoding `y_j mod q` needs `q` in the decoding modulus, but level `L` has
//! lost most primes of `Q_0`. The refresh therefore evaluates the source
//! Program on one chain per refreshed slot, a prime `q` of `Q_0`: level `l` of chain `q` has
//! the first `depths[l]` primes of its pool order `[q, the other primes of
//! Q_0 in canonical order, extension primes]`, so it keeps `q` to level
//! `L + 1` (the plaintext modulus of level `L`). Every ring lists its towers
//! in the canonical order, so a modulus has one tower order on every chain. On chain `q` the part
//! `(q, j)` ([`RefreshDeclaration::top_part`]) decodes `y_j mod q` with the
//! existing refresh combination, two changes aside: the one encoding is
//! `[g | c_one]` with `g` the mask child's carrier of payload
//! `(M + 1) sigma_0 G` (`= sigma_0 G` modulo `q`), and the target action
//! maps `Delta_L r_j x G^st_L` to `Q_q (Delta_0 r_j x G^st_0 mod q)`.
//! [`crate::rns_bgg`] recombines the residues of every chain modulo the
//! product `P` of the refreshed slots' primes (`Q_0` when every slot is
//! refreshed); refreshing a subset of the slots yields `y_j mod P` only.
//!
//! # Shared and chain-specific material
//!
//! The mask and the CBD children of every chain use the same keys (one mask
//! key set, one CBD key set per component) and the same label, so all chains
//! compute the same `sigma_0` and `e'_j`; every decoder has its own smudge
//! keys, one key set per column (one family over the columns). The stages of chain `q` run under
//! the hash namespace [`TopRefreshDeclaration::chain_namespace`], so no two chains share a
//! hash-derived public matrix (with a shared mask, two input rows under one
//! key would cancel it). `A'_(j)` and the label are sampled under the top
//! refresh's own namespaces, identical for every chain.
//!
//! # Caller responsibilities and preconditions
//!
//! - The target must be the output of an exponent LUT, whose plaintext is one signed table monomial
//!   (`||x||_1 = 1`) under the crate's LUT input contract (every LUT input is one signed monomial);
//!   a Program that sums before its last LUT is outside the claim.
//! - Every invocation uses a fresh label tag; a repeated label with the same keys repeats
//!   `sigma_0`, `e'` and every smudge, and two refreshes of `x` and `x~` then differ by `-Delta_0
//!   r_j (x - x~) G^st_0` exactly.
//! - Label encodings of another label under the right artifact names are outside the contract.
//!
//! The smudges hide the decoders' noise under the heuristic of the existing
//! decoder, over the joint view of every chain's hint, preimages and child
//! encodings.
use super::*;

/// The hash tag of the label of an invocation with label tag `label_tag` of
/// the top refresh `top`.
pub(super) fn label_hash_tag(top: &[u8; 32], label_tag: &[u8]) -> Vec<u8> {
    [b"refresh-top/v1/label".as_slice(), top.as_slice(), label_tag].concat()
}

/// The key shapes of a top refresh's PRF keys. Each shape commits to its
/// key set (see [`PrfKeyShape`]); the keys' layouts are the stages'
/// integer data (coordinate tables, see [`crate::program::ProgramData`]).
pub struct TopRefreshKeys<'a> {
    /// The mask child's `n` keys, shared by every chain and component.
    pub mask: PrfKeyShape,
    /// Component `j`'s CBD child keys (`w_0 n`), shared by every chain.
    pub fresh: [PrfKeyShape; 2],
    /// The `w_0 n d` smudge keys of the decoder of part `(chain,
    /// component)`, column `j` the keys from `j n d`; every call must return
    /// fresh keys.
    pub smudge: &'a mut dyn FnMut(usize, usize) -> PrfKeyShape,
}

/// Immutable, committed declaration of a refresh to the top level.
#[derive(Clone, Debug)]
pub struct TopRefreshDeclaration {
    reference: TopReference,
    depths: Vec<usize>,
    extension: Vec<u64>,
    level: usize,
    /// The refreshed slots: chain `chain` decodes prime `reference.top[slots[chain]]`.
    slots: Vec<usize>,
    /// Part `(chain, component)` at `2 chain + component`.
    parts: Vec<RefreshDeclaration>,
    spec: [u8; 32],
}

impl TopRefreshDeclaration {
    /// Check and freeze the refresh of `target_output` of `source_program`,
    /// whose level is `depths.len() - 2`, to the format `reference`: the
    /// chains have `depths[l]` primes at level `l` (`depths[0]` the primes of
    /// `reference.top`, `depths[1]` those of `reference.next`), the children
    /// extend them with primes of `extension`, the fresh noise is CBD(`eta`)
    /// and every decoder smudges with `digits` digits, on the keys of `keys`.
    /// `slots` lists the refreshed primes of `reference.top` by increasing
    /// index, one chain each. The specification commits to the mask and CBD
    /// key sets, and every part's to its smudge key sets, so refreshes of
    /// distinct key sets share no hash namespace.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        source_program: &ExponentLutProgram,
        target_output: WireId,
        profile: SparseLwrPrfProfile,
        eta: u32,
        reference: TopReference,
        depths: Vec<usize>,
        slots: Vec<usize>,
        extension: Vec<u64>,
        digits: usize,
        keys: TopRefreshKeys<'_>,
        matrix_tag: Vec<u8>,
    ) -> Result<Self, RefreshError> {
        let invalid = RefreshError::InvalidDeclaration;
        let level = depths.len().checked_sub(2).ok_or(RefreshError::InvalidDeclaration)?;
        let coefficients =
            profile.ring_dimension().map_err(|_| RefreshError::InvalidDeclaration)?;
        if reference.ring_dimension as usize != coefficients ||
            depths.first() != Some(&reference.top.len()) ||
            depths.get(1) != Some(&reference.next.len()) ||
            !reference.next.iter().all(|prime| reference.top.contains(prime)) ||
            depths.windows(2).any(|pair| pair[1] > pair[0]) ||
            depths[level] < 2 ||
            depths[level + 1] == 0 ||
            extension.iter().any(|prime| reference.top.contains(prime)) ||
            slots.is_empty() ||
            slots.windows(2).any(|pair| pair[1] <= pair[0]) ||
            slots.last().is_some_and(|slot| *slot >= reference.top.len())
        {
            return Err(invalid);
        }
        target_admission(source_program, target_output)?;
        let spec = commitment(&(
            b"mxx-exponent-lut/refresh-top/spec/v2".as_slice(),
            source_program.id(),
            target_output,
            &profile,
            eta,
            &reference,
            &depths,
            &slots,
            &extension,
            digits,
            &matrix_tag,
            &keys.mask,
            &keys.fresh,
        ))?;
        let primes = depths[level];
        let outputs = slots.iter().map(|slot| reference.top[*slot]).collect::<Vec<_>>();
        let mut parts = Vec::with_capacity(2 * slots.len());
        for (chain, prime) in slots.iter().map(|slot| &reference.top[*slot]).enumerate() {
            // The decoding level keeps the output prime and the first `primes - 1` other
            // primes of `Q_0`, in canonical tower order.
            let kept = reference.top.iter().filter(|other| *other != prime).take(primes - 1);
            let kept = kept.collect::<Vec<_>>();
            let position = reference
                .top
                .iter()
                .take_while(|other| *other != prime)
                .filter(|other| kept.contains(other))
                .count();
            for component in 0..2 {
                let smudges = vec![(keys.smudge)(chain, component)];
                parts.push(RefreshDeclaration::top_part(
                    source_program,
                    target_output,
                    profile,
                    eta,
                    primes,
                    (position, *prime),
                    outputs.clone(),
                    component,
                    digits,
                    reference.clone(),
                    keys.mask.clone(),
                    keys.fresh[component].clone(),
                    smudges,
                    matrix_tag.clone(),
                    spec,
                )?);
            }
        }
        Ok(Self { reference, depths, extension, level, slots, parts, spec })
    }

    pub const fn spec(&self) -> &[u8; 32] {
        &self.spec
    }

    pub const fn reference(&self) -> &TopReference {
        &self.reference
    }

    /// The target's level `L`.
    pub const fn level(&self) -> usize {
        self.level
    }

    /// The prime count of every level of every chain.
    pub fn depths(&self) -> &[usize] {
        &self.depths
    }

    /// The number of chains: one per refreshed slot.
    pub fn chains(&self) -> usize {
        self.slots.len()
    }

    /// The refreshed slots, indices into the reference's `Q_0` primes.
    pub fn slots(&self) -> &[usize] {
        &self.slots
    }

    /// Chain `chain`'s output prime.
    pub fn output_prime(&self, chain: usize) -> Option<u64> {
        self.slots.get(chain).map(|slot| self.reference.top[*slot])
    }

    /// The primes of the refreshed slots in canonical order; their product
    /// `P` is the modulus of the recombined rows.
    pub fn output_primes(&self) -> Vec<u64> {
        self.slots.iter().map(|slot| self.reference.top[*slot]).collect()
    }

    /// Chain `chain`'s pool order: its output prime, the other primes of
    /// `Q_0` in canonical order, then the extension primes.
    pub fn pool_order(&self, chain: usize) -> Option<Vec<u64>> {
        let prime = self.output_prime(chain)?;
        Some(
            std::iter::once(prime)
                .chain(self.reference.top.iter().copied().filter(|other| *other != prime))
                .chain(self.extension.iter().copied())
                .collect(),
        )
    }

    /// Part `(chain, component)`.
    pub fn part(&self, chain: usize, component: usize) -> Option<&RefreshDeclaration> {
        (component < 2).then(|| self.parts.get(2 * chain + component)).flatten()
    }

    /// Every part, `(chain, component)` at `2 chain + component`.
    pub fn parts(&self) -> &[RefreshDeclaration] {
        &self.parts
    }

    /// The hash namespace of chain `chain`'s stages.
    pub fn chain_namespace(&self, chain: usize) -> [u8; 32] {
        let mut digest = Sha256::new();
        digest.update(b"mxx-exponent-lut/refresh-top/chain-namespace/v1");
        digest.update(self.spec);
        digest.update((chain as u64).to_le_bytes());
        digest.finalize().into()
    }

    /// The hash namespace of `A'_(0)` and `A'_(1)`, shared by every chain.
    pub fn matrix_namespace(&self) -> [u8; 32] {
        let mut digest = Sha256::new();
        digest.update(b"mxx-exponent-lut/refresh-top/matrix-namespace/v1");
        digest.update(self.spec);
        digest.finalize().into()
    }

    /// The PRF label `a in Z_Q'^v` of an invocation with label tag
    /// `label_tag` under the stage hash key, shared by every part.
    pub fn label(&self, hash_key: [u8; 32], label_tag: &[u8]) -> Result<Vec<usize>, RefreshError> {
        self.parts[0].label(hash_key, label_tag)
    }
}

/// The admission rule of a top-refresh target: the output of a LUT with an
/// exponent representation (one signed table monomial, `||x||_1 = 1`, under
/// the LUT input contract), so `||r_j x||_inf <= 1 < q_min / 2` for the
/// ternary payload secret and the same integer represents `r_j x` modulo the
/// target's plaintext modulus and modulo `Q_1`.
fn target_admission(program: &ExponentLutProgram, target: WireId) -> Result<(), RefreshError> {
    let representation = program.lut_output_representation(target)?;
    match representation {
        Some(crate::program::LutOutputRepresentation::Exponent(_)) => Ok(()),
        _ => Err(RefreshError::InvalidDeclaration),
    }
}

#[cfg(test)]
pub(crate) mod tests;
