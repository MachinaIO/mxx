// storage/lookup_batch.rs
use once_cell::sync::OnceCell;
use std::{collections::HashMap, fs, io, path::PathBuf, time::Instant};
use tokio::sync::{mpsc, oneshot};

pub static BATCH_CHANNEL: OnceCell<mpsc::UnboundedSender<BatchCommand>> = OnceCell::new();

#[derive(Clone)]
pub struct BatchConfig {
    /// Flush when accumulated bytes for a lookup_id reaches/exceeds this.
    pub byte_threshold: usize, // e.g., 256 * 1024 * 1024
    /// File I/O buffer size (only used if you later swap to a streaming writer).
    pub _io_buffer_bytes: usize, // kept for future PackWriter use
}

type FlushAllResult = Vec<(String, io::Result<Vec<PathBuf>>)>;

pub enum BatchCommand {
    /// Add many matrices for a single lookup.
    AddMany {
        lookup_id: String,
        dir_path: PathBuf,
        matrices: Vec<(usize, Vec<u8>)>,
        total_bytes: usize,
    },
    /// Force flush a specific lookup_id.
    Flush {
        lookup_id: String,
        resp: oneshot::Sender<io::Result<PathBuf>>,
    },
    /// Force flush everything.
    FlushAll {
        resp: oneshot::Sender<Vec<(String, io::Result<Vec<PathBuf>>)>>,
    },
    Shutdown,
}

struct Accum {
    lookup_id: String,
    dir_path: PathBuf,
    pending: Vec<(usize, Vec<u8>)>,
    pending_bytes: usize,
    seq: u32,
}

pub fn start_batcher(cfg: BatchConfig) {
    let (tx, mut rx) = mpsc::unbounded_channel::<BatchCommand>();
    // Make channel globally accessible to producers.
    let _ = BATCH_CHANNEL.set(tx);

    tokio::spawn(async move {
        let mut acc: HashMap<String, Accum> = HashMap::new();

        while let Some(cmd) = rx.recv().await {
            match cmd {
                BatchCommand::AddMany { lookup_id, dir_path, matrices, total_bytes } => {
                    let entry = acc.entry(lookup_id.clone()).or_insert_with(|| Accum {
                        lookup_id: lookup_id.clone(),
                        dir_path: dir_path.clone(),
                        pending: Vec::new(),
                        pending_bytes: 0,
                        seq: 0,
                    });
                    // If dir_path changes across calls, prefer the latest.
                    entry.dir_path = dir_path;

                    entry.pending.extend(matrices);
                    entry.pending_bytes += total_bytes;

                    // Threshold-based flush
                    if entry.pending_bytes >= cfg.byte_threshold {
                        let _ = flush_one(entry);
                    }
                }

                BatchCommand::Flush { lookup_id, resp } => {
                    let res = if let Some(entry) = acc.get_mut(&lookup_id) {
                        flush_one(entry)
                    } else {
                        Err(io::Error::new(io::ErrorKind::NotFound, "lookup_id not found"))
                    };
                    let _ = resp.send(res);
                }

                // In your match on `cmd`:
                BatchCommand::FlushAll { resp } => {
                    let mut results: FlushAllResult = Vec::with_capacity(acc.len());

                    for (k, v) in acc.iter_mut() {
                        if v.pending.is_empty() {
                            // Nothing buffered for this lookup_id.
                            results.push((k.clone(), Ok(Vec::new())));
                            continue;
                        }

                        match flush_one(v) {
                            Ok(p) => results.push((k.clone(), Ok(vec![p]))),
                            Err(e) => results.push((k.clone(), Err(e))),
                        }
                    }

                    // Send exactly once, after we've built the full result set.
                    let _ = resp.send(results);
                }

                BatchCommand::Shutdown => {
                    // Best-effort flush all on shutdown.
                    for (_, entry) in acc.iter_mut() {
                        let _ = flush_one(entry);
                    }
                    break;
                }
            }
        }
    });
}

/// Write one batch file in the SAME format your old `store_and_drop_matrices` used:
/// bincode(Vec<(usize, Vec<u8>)>) to `{lookup_id}_batch_{seq:03}.matrices`.
fn flush_one(acc: &mut Accum) -> io::Result<PathBuf> {
    if acc.pending.is_empty() {
        return Err(io::Error::new(io::ErrorKind::Other, "nothing to flush"));
    }

    acc.seq += 1;
    let filename = format!("{}_batch_{:03}.matrices", acc.lookup_id, acc.seq);
    let path = acc.dir_path.join(filename);

    let start = Instant::now();
    let encoded = bincode::encode_to_vec(&acc.pending, bincode::config::standard())
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("bincode: {e}")))?;
    fs::write(&path, encoded)?;
    let elapsed = start.elapsed();

    // Logging helpers: replace with your own logging macros if desired.
    #[allow(unused_macros)]
    macro_rules! log_mem {
        ($($t:tt)*) => { eprintln!($($t)*) }
    }

    log_mem!(
        "Flushed {} matrices for '{}' -> {} in {:?}",
        acc.pending.len(),
        acc.lookup_id,
        path.display(),
        elapsed
    );

    acc.pending.clear();
    acc.pending_bytes = 0;
    Ok(path)
}

/// Force flush all pending batches.
pub async fn flush_all_batches() -> io::Result<()> {
    let channel = BATCH_CHANNEL
        .get()
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "Batch channel not initialized"))?;

    let (tx, rx) = oneshot::channel();
    if let Err(_) = channel.send(BatchCommand::FlushAll { resp: tx }) {
        return Err(io::Error::new(io::ErrorKind::BrokenPipe, "Batch worker dropped"));
    }

    let _results = rx.await.map_err(|_| {
        io::Error::new(io::ErrorKind::BrokenPipe, "Failed to receive flush response")
    })?;

    Ok(())
}

/// Force flush all batches and combine them into a multi-lookup batch file.
/// This creates a single batch file containing all lookup tables.
pub async fn flush_all_batches_combined(base_dir: PathBuf) -> io::Result<String> {
    // First flush all individual batches
    flush_all_batches().await?;

    // Now combine them into a multi-lookup format
    let mut all_lookups = Vec::new();

    // Read all batch files in the directory
    let entries = fs::read_dir(&base_dir)?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if let Some(filename) = path.file_name() {
            if let Some(filename_str) = filename.to_str() {
                if filename_str.ends_with("_batch_001.matrices") {
                    // Extract lookup_id from filename (e.g., "L_4_batch_001.matrices" -> "L_4")
                    if let Some(lookup_id) = filename_str.strip_suffix("_batch_001.matrices") {
                        // Read the batch file
                        let data = fs::read(&path)?;
                        let matrices: Vec<(usize, Vec<u8>)> =
                            bincode::decode_from_slice(&data, bincode::config::standard())
                                .map_err(|e| {
                                    io::Error::new(
                                        io::ErrorKind::InvalidData,
                                        format!("bincode: {e}"),
                                    )
                                })?
                                .0;

                        all_lookups.push((lookup_id.to_string(), matrices));

                        // Remove the individual batch file
                        fs::remove_file(&path)?;
                    }
                }
            }
        }
    }

    if all_lookups.is_empty() {
        return Ok(String::new());
    }

    // Create combined batch filename
    let combined_filename = "combined_batch_001.matrices".to_string();
    let combined_path = base_dir.join(&combined_filename);

    // Write multi-lookup batch format
    let mut output = Vec::new();

    // Header: number of lookup tables
    let num_lookups = all_lookups.len() as u64;
    output.extend_from_slice(&num_lookups.to_le_bytes());

    // Calculate offsets for each lookup's data
    let mut data_sections = Vec::new();

    for (_, matrices) in &all_lookups {
        // Serialize the matrices data
        let mut data = Vec::new();
        for (idx, matrix_bytes) in matrices {
            data.extend_from_slice(&(*idx as u64).to_le_bytes());
            data.extend_from_slice(&(matrix_bytes.len() as u64).to_le_bytes());
            data.extend_from_slice(matrix_bytes);
        }
        data_sections.push(data);
    }

    // Write lookup table headers
    let mut current_offset = 0;
    for (i, (lookup_id, matrices)) in all_lookups.iter().enumerate() {
        // ID length and ID
        output.extend_from_slice(&(lookup_id.len() as u64).to_le_bytes());
        output.extend_from_slice(lookup_id.as_bytes());

        // Lookup info
        output.extend_from_slice(&(matrices.len() as u64).to_le_bytes());
        output.extend_from_slice(&(current_offset as u64).to_le_bytes());
        output.extend_from_slice(&(data_sections[i].len() as u64).to_le_bytes());

        current_offset += data_sections[i].len();
    }

    // Write data sections
    for data in data_sections {
        output.extend_from_slice(&data);
    }

    fs::write(&combined_path, output)?;

    Ok(combined_filename)
}
