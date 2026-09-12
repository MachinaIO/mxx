//! Ownership-preserving column schedules shared by runtime and measurement.
//!
//! Stored intervals are independent of compute widths. Jobs are generated lazily
//! so a schedule does not allocate one host descriptor for every future wave.

use serde::Serialize;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct GpuColumnInterval {
    pub device: usize,
    pub start: usize,
    pub end: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct GpuColumnJob {
    pub device: usize,
    pub source_interval: usize,
    pub start: usize,
    pub end: usize,
}

/// One distinct fleet wave shape of an actual schedule: the active devices with
/// their local job widths, how many waves have exactly that shape, and the
/// representative jobs of the first such wave with their global column ranges.
/// `sum(multiplicity)` over all classes equals `wave_count`.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct GpuColumnWaveClass {
    pub jobs: Vec<GpuColumnJob>,
    pub multiplicity: usize,
    pub first_wave: usize,
}

/// A run of consecutive waves on one device with one local job width.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DeviceWaveSegment {
    wave_start: usize,
    wave_end: usize,
    source_interval: usize,
    column_start: usize,
    width: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct GpuColumnSchedule {
    intervals: Vec<GpuColumnInterval>,
    widths: Vec<usize>,
    owners: Vec<Vec<usize>>,
    local_job_counts: Vec<usize>,
    wave_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum GpuScheduleError {
    #[error("GPU column schedule requires at least one configured device")]
    EmptyFleet,
    #[error("GPU column intervals must cover every column exactly once in global order")]
    InvalidCoverage,
    #[error("GPU column owner {0} is outside the configured fleet")]
    InvalidOwner(usize),
    #[error("active GPU column owner {0} has zero capacity")]
    ZeroCapacity(usize),
    #[error("GPU column job count overflow")]
    Overflow,
}

impl GpuColumnSchedule {
    /// Inputs already include the primitive's source/range boundaries. Idle
    /// devices may have zero capacity; each nonempty interval must have an owner.
    pub fn new(
        columns: usize,
        widths: Vec<usize>,
        intervals: Vec<GpuColumnInterval>,
    ) -> Result<Self, GpuScheduleError> {
        if widths.is_empty() {
            return Err(GpuScheduleError::EmptyFleet);
        }
        let mut owners = vec![Vec::new(); widths.len()];
        let mut local_job_counts = vec![0usize; widths.len()];
        let mut next = 0;
        for (index, interval) in intervals.iter().enumerate() {
            let width = *widths
                .get(interval.device)
                .ok_or(GpuScheduleError::InvalidOwner(interval.device))?;
            if interval.start != next || interval.start >= interval.end || interval.end > columns {
                return Err(GpuScheduleError::InvalidCoverage);
            }
            if width == 0 {
                return Err(GpuScheduleError::ZeroCapacity(interval.device));
            }
            let jobs = (interval.end - interval.start).div_ceil(width);
            local_job_counts[interval.device] = local_job_counts[interval.device]
                .checked_add(jobs)
                .ok_or(GpuScheduleError::Overflow)?;
            owners[interval.device].push(index);
            next = interval.end;
        }
        if next != columns {
            return Err(GpuScheduleError::InvalidCoverage);
        }
        let wave_count = local_job_counts.iter().copied().max().unwrap_or(0);
        Ok(Self { intervals, widths, owners, local_job_counts, wave_count })
    }

    pub fn local_job_counts(&self) -> &[usize] {
        &self.local_job_counts
    }

    /// Admitted local job width per device; idle devices may report zero.
    pub fn widths(&self) -> &[usize] {
        &self.widths
    }

    fn device_segments(&self, device: usize) -> Vec<DeviceWaveSegment> {
        let width = self.widths[device];
        let mut segments = Vec::new();
        let mut wave = 0usize;
        for &index in &self.owners[device] {
            let interval = self.intervals[index];
            let columns = interval.end - interval.start;
            let full = columns / width;
            if full > 0 {
                segments.push(DeviceWaveSegment {
                    wave_start: wave,
                    wave_end: wave + full,
                    source_interval: index,
                    column_start: interval.start,
                    width,
                });
                wave += full;
            }
            let tail = columns % width;
            if tail > 0 {
                segments.push(DeviceWaveSegment {
                    wave_start: wave,
                    wave_end: wave + 1,
                    source_interval: index,
                    column_start: interval.start + full * width,
                    width: tail,
                });
                wave += 1;
            }
        }
        segments
    }

    fn segment_job(device: usize, segment: &DeviceWaveSegment, wave: usize) -> GpuColumnJob {
        let start = segment.column_start + (wave - segment.wave_start) * segment.width;
        GpuColumnJob {
            device,
            source_interval: segment.source_interval,
            start,
            end: start + segment.width,
        }
    }

    /// The jobs of one wave without generating the preceding waves. Equal to
    /// `waves().nth(wave)`; empty when `wave >= wave_count()`.
    pub fn wave_jobs(&self, wave: usize) -> Vec<GpuColumnJob> {
        (0..self.widths.len())
            .filter_map(|device| {
                self.device_segments(device)
                    .iter()
                    .find(|segment| segment.wave_start <= wave && wave < segment.wave_end)
                    .map(|segment| Self::segment_job(device, segment, wave))
            })
            .collect()
    }

    /// Distinct wave shapes with exact multiplicities, computed from the
    /// stored ownership without enumerating every wave. Two waves share a
    /// class when the same devices are active with the same local widths.
    pub fn wave_classes(&self) -> Vec<GpuColumnWaveClass> {
        let segments =
            (0..self.widths.len()).map(|device| self.device_segments(device)).collect::<Vec<_>>();
        let mut boundaries = segments
            .iter()
            .flatten()
            .flat_map(|segment| [segment.wave_start, segment.wave_end])
            .collect::<Vec<_>>();
        boundaries.sort_unstable();
        boundaries.dedup();
        let mut classes: Vec<GpuColumnWaveClass> = Vec::new();
        for range in boundaries.windows(2) {
            let (start, end) = (range[0], range[1]);
            let jobs = segments
                .iter()
                .enumerate()
                .filter_map(|(device, segments)| {
                    segments
                        .iter()
                        .find(|segment| segment.wave_start <= start && start < segment.wave_end)
                        .map(|segment| Self::segment_job(device, segment, start))
                })
                .collect::<Vec<_>>();
            if jobs.is_empty() {
                continue;
            }
            let shape = |jobs: &[GpuColumnJob]| {
                jobs.iter().map(|job| (job.device, job.end - job.start)).collect::<Vec<_>>()
            };
            if let Some(existing) =
                classes.iter_mut().find(|class| shape(&class.jobs) == shape(&jobs))
            {
                existing.multiplicity += end - start;
            } else {
                classes.push(GpuColumnWaveClass {
                    jobs,
                    multiplicity: end - start,
                    first_wave: start,
                });
            }
        }
        classes
    }

    pub fn intervals(&self) -> &[GpuColumnInterval] {
        &self.intervals
    }

    /// Actual logical groups, not ceil(columns / sum(role capacities)).
    pub fn wave_count(&self) -> usize {
        self.wave_count
    }

    /// One job from each owner that still has work. The groups are accounting
    /// descriptors, not a requirement for a device-completion barrier.
    pub fn waves(&self) -> impl Iterator<Item = Vec<GpuColumnJob>> + '_ {
        let mut jobs = self
            .owners
            .iter()
            .enumerate()
            .map(|(device, owned)| {
                let width = self.widths[device];
                owned.iter().flat_map(move |index| {
                    let interval = self.intervals[*index];
                    (interval.start..interval.end).step_by(width).map(move |start| GpuColumnJob {
                        device,
                        source_interval: *index,
                        start,
                        end: start + width.min(interval.end - start),
                    })
                })
            })
            .collect::<Vec<_>>();
        std::iter::from_fn(move || {
            let wave = jobs.iter_mut().filter_map(Iterator::next).collect::<Vec<_>>();
            (!wave.is_empty()).then_some(wave)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preserved_ownership_determines_actual_waves() {
        let schedule = GpuColumnSchedule::new(
            100,
            vec![10, 90],
            vec![
                GpuColumnInterval { device: 0, start: 0, end: 90 },
                GpuColumnInterval { device: 1, start: 90, end: 100 },
            ],
        )
        .unwrap();
        assert_eq!(schedule.local_job_counts(), &[9, 1]);
        assert_eq!(schedule.wave_count(), 9);
        assert_eq!(schedule.waves().count(), 9);
        assert_eq!(
            schedule.waves().next().unwrap(),
            vec![
                GpuColumnJob { device: 0, source_interval: 0, start: 0, end: 10 },
                GpuColumnJob { device: 1, source_interval: 1, start: 90, end: 100 },
            ]
        );
    }

    #[test]
    fn test_wave_classes_and_wave_jobs_match_enumerated_waves() {
        let schedule = GpuColumnSchedule::new(
            100,
            vec![10, 90],
            vec![
                GpuColumnInterval { device: 0, start: 0, end: 90 },
                GpuColumnInterval { device: 1, start: 90, end: 100 },
            ],
        )
        .unwrap();
        let classes = schedule.wave_classes();
        assert_eq!(classes.iter().map(|class| class.multiplicity).sum::<usize>(), 9);
        assert_eq!(
            classes
                .iter()
                .map(|class| {
                    (
                        class.multiplicity,
                        class.first_wave,
                        class
                            .jobs
                            .iter()
                            .map(|job| (job.device, job.start, job.end))
                            .collect::<Vec<_>>(),
                    )
                })
                .collect::<Vec<_>>(),
            vec![(1, 0, vec![(0, 0, 10), (1, 90, 100)]), (8, 1, vec![(0, 10, 20)])]
        );
        for (index, wave) in schedule.waves().enumerate() {
            assert_eq!(schedule.wave_jobs(index), wave);
        }
        assert!(schedule.wave_jobs(9).is_empty());

        let uneven = GpuColumnSchedule::new(
            9,
            vec![4, 4, 0],
            vec![
                GpuColumnInterval { device: 0, start: 0, end: 2 },
                GpuColumnInterval { device: 0, start: 2, end: 5 },
                GpuColumnInterval { device: 1, start: 5, end: 7 },
                GpuColumnInterval { device: 0, start: 7, end: 9 },
            ],
        )
        .unwrap();
        let classes = uneven.wave_classes();
        assert_eq!(classes.iter().map(|class| class.multiplicity).sum::<usize>(), 3);
        // Wave 0: device 0 width 2, device 1 width 2; waves 1 and 2: device 0 only,
        // widths 3 and 2 respectively, so they are distinct classes.
        assert_eq!(classes.len(), 3);
        for (index, wave) in uneven.waves().enumerate() {
            assert_eq!(uneven.wave_jobs(index), wave);
        }
        let large = GpuColumnSchedule::new(
            usize::MAX,
            vec![1],
            vec![GpuColumnInterval { device: 0, start: 0, end: usize::MAX }],
        )
        .unwrap();
        let classes = large.wave_classes();
        assert_eq!(classes.len(), 1);
        assert_eq!(classes[0].multiplicity, usize::MAX);
        assert_eq!(large.wave_jobs(usize::MAX - 1)[0].start, usize::MAX - 1);
    }

    #[test]
    fn test_multiple_stored_intervals_on_one_owner_are_never_skipped() {
        let schedule = GpuColumnSchedule::new(
            9,
            vec![4, 4, 0],
            vec![
                GpuColumnInterval { device: 0, start: 0, end: 2 },
                GpuColumnInterval { device: 0, start: 2, end: 5 },
                GpuColumnInterval { device: 1, start: 5, end: 7 },
                GpuColumnInterval { device: 0, start: 7, end: 9 },
            ],
        )
        .unwrap();
        assert_eq!(schedule.local_job_counts(), &[3, 1, 0]);
        let mut jobs = schedule.waves().flatten().collect::<Vec<_>>();
        jobs.sort_by_key(|job| job.start);
        assert_eq!(
            jobs.iter().map(|job| job.source_interval).collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
        assert_eq!(
            jobs.iter().flat_map(|job| job.start..job.end).collect::<Vec<_>>(),
            (0..9).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_schedule_rejects_gaps_overlaps_and_invalid_owners() {
        for intervals in [
            vec![],
            vec![GpuColumnInterval { device: 0, start: 1, end: 3 }],
            vec![
                GpuColumnInterval { device: 0, start: 0, end: 2 },
                GpuColumnInterval { device: 0, start: 1, end: 3 },
            ],
            vec![GpuColumnInterval { device: 0, start: 0, end: 4 }],
        ] {
            assert_eq!(
                GpuColumnSchedule::new(3, vec![1], intervals),
                Err(GpuScheduleError::InvalidCoverage)
            );
        }
        assert_eq!(
            GpuColumnSchedule::new(
                1,
                vec![1],
                vec![GpuColumnInterval { device: 1, start: 0, end: 1 }]
            ),
            Err(GpuScheduleError::InvalidOwner(1))
        );
        assert_eq!(
            GpuColumnSchedule::new(
                1,
                vec![0],
                vec![GpuColumnInterval { device: 0, start: 0, end: 1 }]
            ),
            Err(GpuScheduleError::ZeroCapacity(0))
        );
        let empty = GpuColumnSchedule::new(0, vec![0, 0], vec![]).unwrap();
        assert_eq!(empty.wave_count(), 0);
        assert_eq!(empty.waves().next(), None);
    }

    #[test]
    fn test_large_schedules_generate_only_requested_jobs() {
        let schedule = GpuColumnSchedule::new(
            usize::MAX,
            vec![1],
            vec![GpuColumnInterval { device: 0, start: 0, end: usize::MAX }],
        )
        .unwrap();
        assert_eq!(schedule.wave_count(), usize::MAX);
        assert_eq!(
            schedule.waves().take(2).map(|wave| wave[0].start).collect::<Vec<_>>(),
            vec![0, 1]
        );
        let wide = GpuColumnSchedule::new(
            usize::MAX,
            vec![usize::MAX],
            vec![GpuColumnInterval { device: 0, start: 0, end: usize::MAX }],
        )
        .unwrap();
        assert_eq!(wide.wave_count(), 1);
        assert_eq!(wide.waves().next().unwrap()[0].end, usize::MAX);
    }
}
