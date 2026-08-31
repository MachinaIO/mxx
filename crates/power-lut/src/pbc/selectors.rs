//! Trusted construction of private PBC selector material.
//!
//! The trusted producer validates a private schedule, converts it to a private
//! runtime bit family, and samples selector RHS packages in one structural
//! loop. Public projections retain only matrices and role/layout identities;
//! selected slots never enter the graph or artifact namespace.

use crate::encoding::{PowerLutEncodingSampler, PowerLutSamplingError};
use mxx_dsl::{Bytes, Family, HashTag, Mat, Ring};
use num_bigint::BigInt;

use super::{PbcError, PbcGeneratedKeyLayout, PbcLayoutId, PbcPrivateSchedule, PbcPublicLayout};

/// The trusted private selector bits used by the structural producer graph.
///
/// `family` is a single runtime family input.  `values` is the private
/// key-provider payload for that input and is intentionally kept separate from
/// the graph: neither selected slots nor support coordinates are represented
/// in the graph or in any artifact name.  The constructor is the only place
/// where a private schedule is converted into one-hot values.
pub struct PbcTrustedSelectorBits {
    layout_id: PbcLayoutId,
    key_instance_id: [u8; 32],
    input_name: String,
    family: Family<Mat>,
    values: Vec<Mat>,
    runtime_bits: Vec<u8>,
}

impl std::fmt::Debug for PbcTrustedSelectorBits {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PbcTrustedSelectorBits")
            .field("layout_id", &self.layout_id)
            .field("key_instance_id", &self.key_instance_id)
            .field("input_name", &self.input_name)
            .field("family_count", &self.values.len())
            .finish()
    }
}

impl PbcTrustedSelectorBits {
    /// Converts a validated private schedule into a private runtime family.
    ///
    /// The returned values are constant polynomials, but they are supplied to
    /// the graph as runtime input values rather than graph constants.  This is
    /// important: the graph/specification commits to the family shape and
    /// order, never to the selected slot.
    pub fn from_schedule(
        generated: &PbcGeneratedKeyLayout,
        ring: &Ring,
        key_instance_id: [u8; 32],
    ) -> Result<Self, PbcError> {
        let layout = &generated.public_layout;
        layout.validate()?;
        generated.private_schedule().validate(layout)?;
        let active = super::layout::PbcActiveCellIndex::build(layout)?;
        let expected = layout
            .parameters
            .universe_size
            .checked_mul(layout.parameters.hash_count)
            .and_then(|real| real.checked_add(layout.parameters.bucket_count))
            .ok_or(PbcError::SizeOverflow)?;
        if active.len() != expected {
            return Err(PbcError::InvalidSchedule(
                "active selector count does not equal real cells plus dummies".into(),
            ));
        }
        for bucket in 0..layout.parameters.bucket_count {
            let width = active
                .bucket_active_count(bucket)
                .ok_or_else(|| PbcError::InvalidSchedule("missing active bucket".into()))?;
            let selected = active
                .bucket_iter(bucket)
                .ok_or_else(|| PbcError::InvalidSchedule("missing active bucket".into()))?
                .filter(|(_, slot, _)| generated.private_schedule().selected_slot(bucket) == *slot)
                .count();
            if width == 0 || selected != 1 {
                return Err(PbcError::InvalidSchedule(
                    "schedule must select exactly one active cell per bucket".into(),
                ));
            }
        }

        let runtime_bits = active
            .iter()
            .map(|(bucket, slot, _)| {
                u8::from(generated.private_schedule().selected_slot(bucket) == slot)
            })
            .collect::<Vec<_>>();
        let values = runtime_bits
            .iter()
            .map(|bit| ring.polynomial([BigInt::from(*bit).into()]))
            .collect::<Vec<_>>();
        let input_name = super::compiler::selector_bit_family_name(layout, key_instance_id);
        let family = ring.input_family(input_name.clone(), expected, (1, 1));
        Ok(Self {
            layout_id: layout.layout_id,
            key_instance_id,
            input_name,
            family,
            values,
            runtime_bits,
        })
    }

    /// Builds trusted bits from a host bit vector, for key-provider tests and
    /// alternate schedule backends.  The vector is checked before any DSL
    /// value is created, so non-binary/nonconstant host values cannot cross
    /// the trusted boundary.
    pub fn from_host_bits(
        layout: &PbcPublicLayout,
        schedule: &PbcPrivateSchedule,
        ring: &Ring,
        key_instance_id: [u8; 32],
        bits: &[u8],
    ) -> Result<Self, PbcError> {
        layout.validate()?;
        schedule.validate(layout)?;
        let active = super::layout::PbcActiveCellIndex::build(layout)?;
        let expected = active.len();
        if bits.len() != expected || bits.iter().any(|bit| *bit > 1) {
            return Err(PbcError::InvalidSchedule(
                "trusted selector bits must be binary and have active-cell length".into(),
            ));
        }
        for bucket in 0..layout.parameters.bucket_count {
            let selected = active
                .bucket_iter(bucket)
                .ok_or_else(|| PbcError::InvalidSchedule("missing active bucket".into()))?
                .filter(|(_, slot, flat)| {
                    bits[*flat] == 1 && schedule.selected_slot(bucket) == *slot
                })
                .count();
            let ones =
                active.bucket_iter(bucket).unwrap().filter(|(_, _, flat)| bits[*flat] == 1).count();
            if selected != 1 || ones != 1 {
                return Err(PbcError::InvalidSchedule(
                    "trusted selector bits must contain one one-hot bit per bucket".into(),
                ));
            }
        }
        let values =
            bits.iter().map(|bit| ring.polynomial([BigInt::from(*bit).into()])).collect::<Vec<_>>();
        let input_name = super::compiler::selector_bit_family_name(layout, key_instance_id);
        let family = ring.input_family(input_name.clone(), expected, (1, 1));
        Ok(Self {
            layout_id: layout.layout_id,
            key_instance_id,
            input_name,
            family,
            values,
            runtime_bits: bits.to_vec(),
        })
    }

    /// The private runtime family consumed by the structural selector loop.
    pub fn family(&self) -> &Family<Mat> {
        &self.family
    }

    /// Returns the key-provider values to bind to [`Self::family`].
    pub fn runtime_values(&self) -> &[Mat] {
        &self.values
    }

    /// Returns the trusted binary payload for binding the runtime family to a
    /// concrete backend. The selected schedule is not part of the graph.
    pub fn runtime_bits(&self) -> &[u8] {
        &self.runtime_bits
    }

    /// Canonical runtime input name, bound to layout and key identity only.
    pub fn input_name(&self) -> &str {
        &self.input_name
    }

    /// Returns the public layout identity bound into this selector family.
    pub fn layout_id(&self) -> PbcLayoutId {
        self.layout_id
    }

    /// Returns the public key-instance identity used for family naming.
    pub fn key_instance_id(&self) -> [u8; 32] {
        self.key_instance_id
    }
}

/// Component families emitted by one structural selector loop.
pub struct PbcStructuralSelectorFamilies {
    pub(crate) gsw: Family<Mat>,
    pub(crate) companions: Vec<(Family<Mat>, Family<Mat>)>,
}

/// Samples all selector RHS packages in one reusable `ParallelLoop`.
///
/// The selector bit is the only payload-dependent input.  Public companion
/// matrices are derived from the canonical `(layout, key, flat_index)` tag and
/// therefore do not depend on that bit.  The family output is intentionally
/// not an artifact output for the bit family itself.
pub fn build_structural_selector_families(
    sampler: &PowerLutEncodingSampler,
    bits: Family<Mat>,
    source: Mat,
    target: Mat,
    hash_key: Bytes,
    layout: &PbcPublicLayout,
    key_instance_id: [u8; 32],
) -> Result<PbcStructuralSelectorFamilies, PbcError> {
    layout.validate()?;
    let active = super::layout::PbcActiveCellIndex::build(layout)?;
    let expected = active.len();
    if expected == 0 || *bits.count() != mxx_ir_core::IntExpr::constant(expected) {
        return Err(PbcError::InvalidSchedule(
            "private selector family count does not match canonical active order".into(),
        ));
    }
    if bits.element_type() != &sampler.layout.ring().matrix_type((1, 1)) {
        return Err(PbcError::InvalidSchedule(
            "private selector family must contain constant-polynomial scalar matrices".into(),
        ));
    }
    if sampler.layout.secret_dimension < 2 ||
        sampler.layout.public_key_columns() == 0 ||
        !same_secret_shape(&source, &sampler.layout) ||
        !same_secret_shape(&target, &sampler.layout) ||
        sampler.gaussian_sigma.is_some() != sampler.gaussian_max_coefficient_bound.is_some()
    {
        return Err(PbcError::InvalidSchedule("selector sampler has an empty shape".into()));
    }
    let companion_count = sampler
        .layout
        .secret_dimension
        .checked_mul(sampler.layout.public_key_columns())
        .ok_or(PbcError::SizeOverflow)?;
    let sampler = sampler.clone();
    let layout_id = layout.layout_id;
    let components: Vec<Family<Mat>> = bits
        .parallel_map_values(move |index, bit| {
            let mut tag = selector_rhs_tag(layout_id, key_instance_id);
            tag.push(index);
            let package = sampler
                .sample_cross_secret_rhs(source.clone(), target.clone(), bit, hash_key.clone(), tag)
                .unwrap_or_else(|error: PowerLutSamplingError| {
                    panic!("validated selector sampler failed while building graph: {error}")
                });
            let mut values = Vec::with_capacity(1 + 2 * companion_count);
            values.push(package.gsw_ciphertext().clone());
            for companion in 0..companion_count {
                let block = package
                    .companion_at(companion)
                    .expect("validated selector package companion count");
                values.push(block.vector.clone());
                values.push(block.public_matrix.clone());
            }
            values
        })
        .map_err(|_| PbcError::InvalidSchedule("selector RHS loop construction failed".into()))?;
    let mut components = components.into_iter();
    let gsw = components.next().ok_or(PbcError::InvalidSchedule("missing GSW family".into()))?;
    let mut companions = Vec::with_capacity(companion_count);
    for _ in 0..companion_count {
        let vector =
            components.next().ok_or(PbcError::InvalidSchedule("missing vector family".into()))?;
        let public =
            components.next().ok_or(PbcError::InvalidSchedule("missing public family".into()))?;
        companions.push((vector, public));
    }
    if components.next().is_some() {
        return Err(PbcError::InvalidSchedule("selector family component order mismatch".into()));
    }
    Ok(PbcStructuralSelectorFamilies { gsw, companions })
}

fn same_secret_shape(value: &Mat, layout: &mxx_bgg::BggSamplerLayout) -> bool {
    let expected = layout.ring().matrix_type((1, layout.secret_dimension));
    let actual = value.matrix_type();
    actual.modulus.canonicalize() == expected.modulus.canonicalize() &&
        actual.ring_dimension.canonicalize() == expected.ring_dimension.canonicalize() &&
        actual.rows.canonicalize() == expected.rows.canonicalize() &&
        actual.columns.canonicalize() == expected.columns.canonicalize()
}

fn selector_rhs_tag(layout_id: PbcLayoutId, key_instance_id: [u8; 32]) -> HashTag {
    let mut prefix = Vec::with_capacity(32 + 32 + 40);
    prefix.extend_from_slice(b"mxx-power-lut/pbc/selector-rhs/v1");
    prefix.extend_from_slice(&(layout_id.0.len() as u64).to_le_bytes());
    prefix.extend_from_slice(&layout_id.0);
    prefix.extend_from_slice(&(key_instance_id.len() as u64).to_le_bytes());
    prefix.extend_from_slice(&key_instance_id);
    HashTag::from(prefix)
}
