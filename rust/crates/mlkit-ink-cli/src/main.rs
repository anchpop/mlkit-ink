//! Command line driver for `mlkit-ink`.
//!
//! Everything host-specific lives here — the filesystem, the network, mmap —
//! so the core crate stays portable to wasm and mobile untouched.

mod catalog;
mod model;
mod svg;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use clap::{Parser, Subcommand};
use mlkit_ink::Stroke;
use mlkit_ink::optimize::{self, FitOptions};
use mlkit_ink::settings::DecoderSettings;

#[derive(Parser)]
#[command(name = "mlkit-ink", about, version)]
struct Cli {
    /// Repo root holding manifest.json, packmapping.pb and models/.
    #[arg(long, global = true, env = "MLKIT_INK_ROOT")]
    root: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Recognize handwriting from an ink JSON file.
    Recognize {
        ink: PathBuf,
        #[arg(short, long, default_value = "en-US")]
        language: String,
        #[arg(short, long, default_value_t = 5)]
        nbest: usize,
        /// Best-path decode only. This is the configuration that matches the
        /// real SDK on 74 of 74 corpus inks.
        #[arg(long)]
        greedy: bool,
        /// Skip the language model entirely (implies greedy).
        #[arg(long)]
        no_lm: bool,
        /// Use the empirically tuned weights instead of the binary-faithful
        /// ones. Scores better on our corpus; its sign is provably wrong.
        #[arg(long)]
        empirical: bool,
        #[arg(long)]
        show_features: bool,
        #[arg(long)]
        json: bool,
        /// Native scales every net log-posterior by this before combining it
        /// with the LM costs. Its value was not recoverable from the binary;
        /// see SPEC.md section 20. Larger means "trust the ink over the LM".
        #[arg(long, default_value_t = 1.0)]
        acoustic_scale: f64,
    },
    /// Nudge an ink's coordinates until the recognizer reads it as TARGET.
    ///
    /// Gradient descent through the whole pipeline: CTC loss, the network, and
    /// the Bezier fitter. Writes the moved ink as JSON so it can be drawn, or
    /// fed back to the real SDK through android-oracle/.
    Fit {
        ink: PathBuf,
        /// The text to steer toward.
        target: String,
        #[arg(short, long, default_value = "en-US")]
        language: String,
        #[arg(short, long)]
        output: Option<PathBuf>,
        #[arg(long, default_value_t = 200)]
        steps: usize,
        #[arg(long, default_value_t = 0.01)]
        learning_rate: f64,
        /// Pull back toward the original ink. Zero turns this into an
        /// adversarial attack: the result will score well and look like noise.
        #[arg(long, default_value_t = 3.0)]
        anchor: f64,
        /// Penalize jaggedness, keeping the result a plausible pen trace.
        #[arg(long, default_value_t = 3.0)]
        smoothness: f64,
        #[arg(long, default_value_t = 10)]
        resegment_every: usize,
        /// Width, as a fraction of the ink's diagonal, of the blur applied to
        /// the descent direction. This is what keeps the result looking
        /// hand-drawn; 0 lets the optimizer jitter individual samples.
        #[arg(long, default_value_t = 0.4)]
        smoothing: f64,
        /// Average the gradient over this many jittered copies of the ink.
        /// Above 1, the search must find a shape that reads correctly even when
        /// nudged — which is what real handwriting does and an adversarial
        /// example does not.
        #[arg(long, default_value_t = 1)]
        robust: usize,
        /// How far to jitter, as a fraction of the ink's diagonal.
        #[arg(long, default_value_t = 0.006)]
        jitter: f64,
        /// Draw the original and the result overlaid, so the fit can be looked
        /// at and not just scored.
        #[arg(long)]
        svg: Option<PathBuf>,
        /// Print the per-step loss, for telling a plateau apart from
        /// divergence, oscillation, or a resegmentation cliff.
        #[arg(long)]
        history: bool,
    },
    /// List every language tag in the catalog.
    Languages,
    /// Download and extract the packs for a language tag.
    Fetch { language: String },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let root = match cli.root {
        Some(path) => path,
        None => default_root()?,
    };

    match cli.command {
        Command::Languages => {
            let bytes = std::fs::read(root.join("packmapping.pb"))?;
            let mapping =
                mlkit_ink::packs::PackMapping::parse(&bytes).map_err(|e| anyhow!("{e}"))?;
            let mut tags: Vec<_> = mapping.tags().collect();
            tags.sort_unstable();
            for tag in &tags {
                println!("{tag}");
            }
            eprintln!("{} language tags", tags.len());
        }
        Command::Fetch { language } => {
            let loaded = model::load(&root, &language, true, false)?;
            for path in &loaded.paths {
                println!("{}", path.display());
            }
        }
        Command::Recognize {
            ink,
            language,
            nbest,
            greedy,
            no_lm,
            empirical,
            show_features,
            json,
            acoustic_scale,
        } => {
            let strokes = read_ink(&ink)?;
            let loaded = model::load(&root, &language, !no_lm, true)?;
            let mut recognizer = loaded.recognizer()?;
            if empirical {
                recognizer.settings = DecoderSettings {
                    beam_width: recognizer.settings.beam_width,
                    ..DecoderSettings::empirical()
                };
            }
            recognizer.settings.acoustic_scale = acoustic_scale;

            let features = recognizer.features(&strokes).map_err(|e| anyhow!("{e}"))?;
            if show_features {
                for row in features.iter_rows() {
                    let cells: Vec<_> = row.iter().map(|v| format!("{v:8.3}")).collect();
                    println!("  [{}]", cells.join(" "));
                }
            }

            let candidates = if greedy || no_lm {
                alloc_one(
                    recognizer
                        .recognize_greedy(&strokes)
                        .map_err(|e| anyhow!("{e}"))?,
                )
            } else {
                recognizer
                    .recognize(&strokes, nbest)
                    .map_err(|e| anyhow!("{e}"))?
            };

            if json {
                let payload: Vec<_> = candidates
                    .iter()
                    .map(|c| serde_json::json!({ "text": c.text, "score": c.score }))
                    .collect();
                println!("{}", serde_json::to_string_pretty(&payload)?);
            } else {
                eprintln!(
                    "{} strokes -> {} curves ({} features each)",
                    strokes.len(),
                    features.rows(),
                    features.cols()
                );
                for (i, candidate) in candidates.iter().enumerate() {
                    println!(
                        "  {}. {:?}  ({:.4})",
                        i + 1,
                        candidate.text,
                        candidate.score
                    );
                }
            }
        }
        Command::Fit {
            ink,
            target,
            language,
            output,
            steps,
            learning_rate,
            anchor,
            smoothness,
            resegment_every,
            smoothing,
            robust,
            jitter,
            svg: svg_path,
            history,
        } => {
            let strokes = read_ink(&ink)?;
            let loaded = model::load(&root, &language, false, true)?;
            let recognizer = loaded.recognizer()?;

            let before = recognizer
                .recognize_greedy(&strokes)
                .map_err(|e| anyhow!("{e}"))?;
            let options = FitOptions {
                steps,
                learning_rate,
                anchor_weight: anchor,
                smoothness_weight: smoothness,
                resegment_every,
                smoothing,
                robust_samples: robust,
                jitter,
            };
            let report = optimize::fit_strokes(&recognizer, &strokes, &target, &options)
                .map_err(|e| anyhow!("{e}"))?;
            let after = recognizer
                .recognize_greedy(&report.strokes)
                .map_err(|e| anyhow!("{e}"))?;

            if history {
                println!("  step   ctc_loss   regularization  reseg");
                for record in &report.history {
                    println!(
                        "  {:>4}   {:>8.4}   {:>13.4}  {}",
                        record.step,
                        record.ctc_loss,
                        record.regularization,
                        if record.resegmented { "*" } else { "" }
                    );
                }
            }
            println!("target      {target:?}");
            println!("before      {:?}", before.text);
            println!(
                "after       {:?}{}",
                after.text,
                if report.matched {
                    ""
                } else {
                    "   (target never became the top candidate)"
                }
            );
            println!(
                "CTC loss    {:.4} -> {:.4}  (best at step {} of {steps})",
                report.initial_ctc_loss, report.best_ctc_loss, report.best_step
            );
            println!(
                "moved       {:.1}% of the ink's diagonal (rms)",
                rms_shift(&strokes, &report.strokes)
            );

            if let Some(path) = svg_path {
                let caption = format!(
                    "{:?} -> {:?}  (target {target:?}, loss {:.2} -> {:.2})",
                    before.text, after.text, report.initial_ctc_loss, report.best_ctc_loss
                );
                std::fs::write(&path, svg::overlay(&strokes, &report.strokes, &caption))?;
                eprintln!("wrote {}", path.display());
            }

            if let Some(path) = output {
                let payload = serde_json::json!({
                    "language": language,
                    "strokes": report.strokes,
                });
                std::fs::write(&path, serde_json::to_string_pretty(&payload)?)?;
                eprintln!("wrote {}", path.display());
            }
        }
    }
    Ok(())
}

fn alloc_one<T>(value: T) -> Vec<T> {
    vec![value]
}

/// How far the ink actually moved, as a percentage of its own diagonal — the
/// number that says whether a successful fit is a nudge or a rewrite.
fn rms_shift(before: &[Stroke], after: &[Stroke]) -> f64 {
    let (mut total, mut count) = (0.0, 0usize);
    let (mut lo, mut hi) = ((f64::MAX, f64::MAX), (f64::MIN, f64::MIN));
    for (a, b) in before.iter().zip(after) {
        for i in 0..a.x.len().min(b.x.len()) {
            let (dx, dy) = (b.x[i] - a.x[i], b.y[i] - a.y[i]);
            total += dx * dx + dy * dy;
            count += 1;
            lo = (lo.0.min(a.x[i]), lo.1.min(a.y[i]));
            hi = (hi.0.max(a.x[i]), hi.1.max(a.y[i]));
        }
    }
    if count == 0 {
        return 0.0;
    }
    let diagonal = ((hi.0 - lo.0).powi(2) + (hi.1 - lo.1).powi(2)).sqrt();
    if diagonal > 0.0 {
        100.0 * (total / count as f64).sqrt() / diagonal
    } else {
        0.0
    }
}

/// Walk up from the executable's manifest directory looking for the catalog.
/// Keeps `mlkit-ink recognize foo.json` working from anywhere in the repo.
fn default_root() -> Result<PathBuf> {
    let start = std::env::current_dir()?;
    for dir in start.ancestors() {
        if dir.join("manifest.json").is_file() && dir.join("packmapping.pb").is_file() {
            return Ok(dir.to_path_buf());
        }
    }
    let fallback = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    if fallback.join("manifest.json").is_file() {
        return Ok(fallback.canonicalize()?);
    }
    anyhow::bail!("could not find manifest.json; pass --root or set MLKIT_INK_ROOT")
}

/// The oracle harness's ink schema: `{"strokes": [{"x": [...], "y": [...], "t": [...]}]}`.
fn read_ink(path: &Path) -> Result<Vec<Stroke>> {
    #[derive(serde::Deserialize)]
    struct Ink {
        strokes: Vec<Stroke>,
    }
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let ink: Ink =
        serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
    Ok(ink.strokes)
}
