//! Choose mapped host backing at the actual memory-pressure intervals. Moving
//! bytes that are absent from a peak cannot reduce that peak. Recompute after
//! every choice, since several disjoint peaks can require different captures.

use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug)]
pub(super) struct ResidencySpan {
    pub device: i32,
    pub first: usize,
    pub end: usize,
    pub bytes: u64,
}

fn peaks(events: &BTreeMap<(i32, usize), i128>) -> BTreeMap<i32, (u64, usize)> {
    let mut live = BTreeMap::<i32, i128>::new();
    let mut result = BTreeMap::<i32, (u64, usize)>::new();
    for (&(device, position), &delta) in events {
        let current = live.entry(device).or_default();
        *current += delta;
        let peak = result.entry(device).or_default();
        if *current > i128::from(peak.0) {
            *peak =
                (u64::try_from(*current).expect("admitted scratch span size fits u64"), position);
        }
    }
    result
}

/// Select only arena ranges, leaving graph temporaries device-local. Account
/// for the existing layout's placement overhead as well as logical overlap;
/// the caller must still validate the final repacked device and host pools.
pub(super) fn select_host_ranges(
    graph: &[ResidencySpan],
    arena: &[ResidencySpan],
    sizes: &BTreeMap<i32, u64>,
    limits: &BTreeMap<i32, u64>,
) -> Result<BTreeSet<usize>, String> {
    let mut events = BTreeMap::new();
    for span in graph.iter().chain(arena) {
        *events.entry((span.device, span.first)).or_default() += i128::from(span.bytes);
        *events.entry((span.device, span.end)).or_default() -= i128::from(span.bytes);
    }
    let initial_peaks = peaks(&events);
    let targets = sizes
        .iter()
        .map(|(&device, &bytes)| {
            let overhead =
                bytes.saturating_sub(initial_peaks.get(&device).map_or(0, |peak| peak.0));
            (device, limits[&device].saturating_sub(overhead))
        })
        .collect::<BTreeMap<_, _>>();
    let mut selected = BTreeSet::new();
    loop {
        let Some((device, (peak, position))) = peaks(&events)
            .into_iter()
            .filter(|(device, (peak, _))| *peak > targets[device])
            .max_by_key(|(device, (peak, _))| peak.saturating_sub(targets[device]))
        else {
            return Ok(selected)
        };
        let candidate = arena
            .iter()
            .enumerate()
            .filter(|(index, span)| {
                !selected.contains(index) &&
                    span.device == device &&
                    span.first <= position &&
                    position < span.end
            })
            .max_by_key(|(index, span)| {
                (
                    u128::from(span.bytes) * (span.end - span.first) as u128,
                    span.bytes,
                    std::cmp::Reverse(*index),
                )
            });
        let Some((index, span)) = candidate else {
            return Err(format!(
                "GPU scratch has an unspillable peak of {peak} bytes on device {device} at operation {position}, exceeding placement target {}",
                targets[&device]
            ));
        };
        selected.insert(index);
        *events.entry((span.device, span.first)).or_default() -= i128::from(span.bytes);
        *events.entry((span.device, span.end)).or_default() += i128::from(span.bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(first: usize, end: usize, bytes: u64) -> ResidencySpan {
        ResidencySpan { device: 0, first, end, bytes }
    }

    #[test]
    fn selects_a_peak_contributor_instead_of_a_longer_unrelated_capture() {
        let selected = select_host_ranges(
            &[span(10, 11, 90)],
            &[span(0, 15, 20), span(20, 100, 19)],
            &BTreeMap::from([(0, 110)]),
            &BTreeMap::from([(0, 90)]),
        )
        .unwrap();
        assert_eq!(selected, BTreeSet::from([0]));
    }

    #[test]
    fn resolves_each_disjoint_peak_and_preserves_placement_headroom() {
        let selected = select_host_ranges(
            &[span(10, 11, 90), span(30, 31, 90)],
            &[span(0, 15, 20), span(25, 40, 20)],
            &BTreeMap::from([(0, 120)]),
            &BTreeMap::from([(0, 100)]),
        )
        .unwrap();
        assert_eq!(selected, BTreeSet::from([0, 1]));
    }

    #[test]
    fn leaves_fitting_devices_alone_and_rejects_an_unspillable_peak() {
        let graph = [span(0, 1, 100)];
        let sizes = BTreeMap::from([(0, 100)]);
        assert!(select_host_ranges(&graph, &[], &sizes, &sizes).unwrap().is_empty());
        assert!(
            select_host_ranges(&graph, &[], &sizes, &BTreeMap::from([(0, 99)]))
                .unwrap_err()
                .contains("unspillable peak")
        );
    }
}
