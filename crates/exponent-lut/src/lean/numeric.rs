//! Kernel-checked scalar instances used by the implementation soundness proof.
//! These are prerequisites for the graph theorem, not a replacement certificate.

use std::fmt::Write;

use crate::noise_refresh::simulation::{ExponentLutNoiseReport, ExponentLutNoiseSnapshot};

pub(super) fn refresh_bounds(
    snapshot: &ExponentLutNoiseSnapshot,
    report: &ExponentLutNoiseReport,
) -> Result<String, String> {
    let topology =
        snapshot.refresh.transcript_topology.as_ref().ok_or("missing transcript topology")?;
    let routes = snapshot.refresh.route_gains.as_ref().ok_or("missing refresh action gains")?;
    let mut source = "import ExponentRefreshBounds\n\nnamespace ExponentGeneratedNumeric\nopen MxxExponentLut\n\n".to_owned();
    for (i, slot) in report.refresh.slots.iter().enumerate() {
        writeln!(
            source,
            "theorem pre_switch_{i} : ({} : Nat) * {} + ({} + {}) * {} = {} := by decide",
            routes.state_scaling[i].l1,
            snapshot.initial_state_bound,
            routes.mask[i].l1,
            routes.fresh[i].l1,
            report.readout_bound,
            slot.pre_switch_bound
        )
        .unwrap();
        writeln!(
            source,
            "theorem decoder_{i} : ({} : Nat) * {} = {} := by decide",
            routes.decoder[i].l1, snapshot.helper_error_cutoff, slot.decoder_bound
        )
        .unwrap();
        writeln!(
            source,
            "theorem switched_bound_{i} : switchedErrorBound {} {} {} {} = {} := by decide",
            slot.pre_switch_bound,
            slot.lift,
            topology.coefficient_count,
            slot.decoder_bound,
            slot.residual_bound
        )
        .unwrap();
        writeln!(source, "theorem positive_lift_{i} : (0 : Nat) < {} := by decide", slot.lift)
            .unwrap();
        writeln!(
            source,
            "theorem mask_bound_{i} : ({} : Nat) / 2 = {} := by decide",
            slot.mask_modulus, slot.mask_bound
        )
        .unwrap();
        // Rejected export-only fixtures still have correct arithmetic equalities;
        // only accepted candidates obtain strict-decoding hypotheses.
        if report.refresh.accepted {
            writeln!(source, "theorem strict_rounding_{i} : 2 * ({} + switchedErrorBound {} {} {} {}) < ({} : Nat) := by decide",
                slot.mask_bound, slot.pre_switch_bound, slot.lift, topology.coefficient_count,
                slot.decoder_bound, slot.spacing).unwrap();
        }
        source.push('\n');
    }
    source.push_str("end ExponentGeneratedNumeric\n");
    Ok(source)
}
