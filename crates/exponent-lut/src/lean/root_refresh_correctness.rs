//! Closed refresh correctness assembled from every actual high-source bound.
use crate::noise_refresh::simulation::ExponentLutNoiseReport;

pub(super) fn root_refresh_correctness(report: &ExponentLutNoiseReport) -> Option<String> {
    if report.refresh.slots.is_empty() ||
        report.refresh.slots.iter().any(|slot| slot.rounding_margin <= 0.into())
    {
        return None;
    }
    let mut source = String::from(
        "import RootRefreshHighSource\n\nnamespace ExponentGeneratedRefreshCorrectness\n\ntheorem correctness : GeneratedClaim.CorrectnessClaim := by\n  intro hashModel external execution runs\n  exact ExponentGeneratedRefreshFinish.finish runs\n    (ExponentGeneratedRefreshSourceProof.readoutExponent runs)\n",
    );
    for slot in 0..report.refresh.slots.len() {
        source.push_str(&format!("    (ExponentGeneratedRefreshHighSource.source_{slot} runs)\n"));
    }
    source.push_str("\nend ExponentGeneratedRefreshCorrectness\n");
    Some(source)
}
