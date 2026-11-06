//! Batch containment and crossing queries for replayed host-loop bodies.
//!
//! Bodies are half-open; allocation use spans are inclusive. Two ordered sweeps
//! replace scanning every body for every allocation, without changing which
//! allocations may reuse storage across replay boundaries.

use std::{collections::BTreeSet, ops::Bound};

/// Bound host-import reservations by ordered Graph launches. A span crossing
/// a replay boundary needs its full replay lifetime instead, so leave it alone.
pub(super) fn import_region_boundaries(
    bodies: &[(usize, usize)],
    spans: &[(usize, usize)],
    operations: usize,
) -> Vec<usize> {
    let properties = replay_span_properties(bodies, spans);
    spans
        .iter()
        .zip(properties)
        .filter(|(_, (_, uncrossed))| *uncrossed)
        .flat_map(|(&(first, last), _)| [Some(first), last.checked_add(1)])
        .flatten()
        .filter(|&boundary| boundary < operations)
        .collect()
}

/// Wave variants sharing one physical body execute that body consecutively.
/// A single execution is not a replay boundary; an enclosing replay is accounted
/// for by its own interval. Saturating at two avoids overflowing logical counts.
pub(super) fn repeated_bodies(
    bodies: impl IntoIterator<Item = (usize, usize, u128)>,
) -> Vec<(usize, usize)> {
    let mut counts = std::collections::BTreeMap::<(usize, usize), u8>::new();
    for (start, end, count) in bodies {
        let total = counts.entry((start, end)).or_default();
        *total = (*total + count.min(2) as u8).min(2);
    }
    counts.into_iter().filter_map(|(body, count)| (count > 1).then_some(body)).collect()
}

/// Return `(contained_in_a_body, crosses_no_body_boundary)` for each use span.
/// A span enclosing a whole body does not cross a boundary in this sense.
pub(super) fn replay_span_properties(
    bodies: &[(usize, usize)],
    spans: &[(usize, usize)],
) -> Vec<(bool, bool)> {
    let mut result = vec![(false, true); spans.len()];
    let mut ordered_bodies = bodies.to_vec();
    ordered_bodies.sort_unstable();
    let mut queries = (0..spans.len()).collect::<Vec<_>>();
    queries.sort_unstable_by_key(|&index| spans[index].0);
    let mut ends = BTreeSet::new();
    let mut next = 0;
    for &index in &queries {
        let (first, last) = spans[index];
        debug_assert!(first <= last);
        while next < ordered_bodies.len() && ordered_bodies[next].0 <= first {
            ends.insert(ordered_bodies[next].1);
            next += 1;
        }
        // A containing body starts no later than first and ends after last.
        result[index].0 = ends.last().is_some_and(|&end| end > last);
        // A partial overlap at the left has start <= first < end <= last.
        result[index].1 =
            ends.range((Bound::Excluded(first), Bound::Included(last))).next().is_none();
    }
    ordered_bodies.sort_unstable_by_key(|&(_, end)| std::cmp::Reverse(end));
    queries.sort_unstable_by_key(|&index| std::cmp::Reverse(spans[index].1));
    let mut starts = BTreeSet::new();
    next = 0;
    for index in queries {
        let (first, last) = spans[index];
        while next < ordered_bodies.len() && ordered_bodies[next].1 > last {
            starts.insert(ordered_bodies[next].0);
            next += 1;
        }
        // A partial overlap at the right has first < start <= last < end.
        result[index].1 &=
            starts.range((Bound::Excluded(first), Bound::Included(last))).next().is_none();
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_import_boundaries_preserve_replay_captures_and_split_local_uses() {
        // Imports before and after expensive intervening work can share its
        // pool. Captures live into a replay, or results escaping one, cannot.
        assert_eq!(
            import_region_boundaries(&[(10, 100)], &[(12, 16), (80, 89), (0, 12), (80, 101)], 110),
            vec![12, 17, 80, 90],
        );
        assert_eq!(import_region_boundaries(&[], &[(0, 0), (3, 4)], 5), vec![0, 1, 3]);
        assert_eq!(
            import_region_boundaries(&[], &[(usize::MAX - 1, usize::MAX)], usize::MAX),
            vec![usize::MAX - 1]
        );
    }

    #[test]
    fn test_repeated_bodies_count_variants_without_repeating_single_children() {
        assert_eq!(
            repeated_bodies([
                (0, 100, 2),
                (10, 20, 1),
                (30, 40, 1),
                (30, 40, 1),
                (50, 60, u128::MAX),
                (50, 60, u128::MAX),
                (70, 80, 0),
            ]),
            vec![(0, 100), (30, 40), (50, 60)]
        );
        assert!(repeated_bodies([]).is_empty());
    }

    #[test]
    fn test_replay_spans_match_original_predicates_at_all_boundaries() {
        let points = [0, 1, 2, 3, usize::MAX - 1, usize::MAX];
        let intervals = points
            .iter()
            .flat_map(|&start| {
                points.iter().copied().filter(move |&end| start <= end).map(move |end| (start, end))
            })
            .collect::<Vec<_>>();
        let check = |bodies: &[(usize, usize)]| {
            let expected = intervals
                .iter()
                .map(|&(first, last)| {
                    let contained = bodies.iter().any(|&(start, end)| start <= first && last < end);
                    let nested = bodies.iter().all(|&(start, end)| {
                        last < start ||
                            first >= end ||
                            (start <= first && last < end) ||
                            (first < start && last >= end)
                    });
                    (contained, nested)
                })
                .collect::<Vec<_>>();
            assert_eq!(replay_span_properties(bodies, &intervals), expected, "{bodies:?}");
        };
        check(&[]);
        check(&intervals);
        for &left in &intervals {
            for &right in &intervals {
                // Includes duplicate, nested, disjoint and crossing bodies.
                check(&[left, right]);
            }
        }
        assert!(replay_span_properties(&intervals, &[]).is_empty());
    }
}
