use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand, ValueEnum};
use genai_benchmark::{
    generate_output_filename, load_dataset, load_scenario_from_file, print_comparison,
    print_result, run_benchmark, run_scenario, save_results_to_file, BenchmarkConfig,
    DatasetConfig,
};
use tracing::info;

#[derive(Parser, Debug)]
#[command(name = "genai-benchmark")]
#[command(about = "Load testing tool for OpenAI-compatible LLM APIs")]
#[command(version)]
struct Args {
    #[command(subcommand)]
    command: Option<Commands>,

    // Direct benchmark options (when no subcommand)
    /// Base URL of the OpenAI-compatible API
    #[arg(long, env = "OPENAI_BASE_URL", global = true)]
    base_url: Option<String>,

    /// API key for authentication
    #[arg(long, env = "OPENAI_API_KEY", default_value = "", global = true)]
    api_key: String,

    /// Model name to use
    #[arg(long, short, default_value = "gpt-3.5-turbo", global = true)]
    model: String,

    /// Dataset source
    #[arg(long, value_enum, default_value = "synthetic", global = true)]
    dataset: DatasetSourceArg,

    /// Path to local dataset file (JSON/JSONL)
    #[arg(long, global = true)]
    dataset_path: Option<String>,

    /// Hugging Face dataset name
    #[arg(
        long,
        default_value = "anon8231489123/ShareGPT_Vicuna_unfiltered",
        global = true
    )]
    hf_dataset: String,

    /// Number of requests to send
    #[arg(long, short, default_value = "100", global = true)]
    num_requests: usize,

    /// Maximum concurrent requests
    #[arg(long, short, default_value = "5", global = true)]
    concurrency: usize,

    /// Target requests per second (0 = unlimited)
    #[arg(long, default_value = "100", global = true)]
    rps: f64,

    /// Maximum tokens to generate per request
    #[arg(long, default_value = "256", global = true)]
    max_tokens: u32,

    /// Request timeout in seconds
    #[arg(long, default_value = "300", global = true)]
    timeout: u64,

    /// Random seed for reproducibility
    #[arg(long, global = true)]
    seed: Option<u64>,

    /// Enable verbose logging
    #[arg(long, short, global = true)]
    verbose: bool,

    /// Skip first N conversations from dataset
    #[arg(long, default_value = "0", global = true)]
    skip: usize,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run a benchmark from a YAML scenario file
    Scenario {
        /// Path to the YAML scenario file
        #[arg(required = true)]
        file: String,

        /// Output directory for results (default: ./output)
        #[arg(long, short, default_value = "output")]
        output_dir: String,

        /// Skip saving results to file
        #[arg(long)]
        no_save: bool,
    },
    /// Run a single benchmark (default behavior)
    Run,
}

#[derive(Debug, Clone, ValueEnum)]
enum DatasetSourceArg {
    Sharegpt,
    Prompts,
    Synthetic,
}

impl DatasetSourceArg {
    fn to_config(&self, args: &Args) -> DatasetConfig {
        match self {
            DatasetSourceArg::Sharegpt => DatasetConfig::Sharegpt {
                path: args.dataset_path.clone(),
                hf_dataset: args.hf_dataset.clone(),
                skip: args.skip,
            },
            DatasetSourceArg::Prompts => DatasetConfig::Prompts {
                path: args.dataset_path.clone().unwrap_or_default(),
                skip: args.skip,
            },
            DatasetSourceArg::Synthetic => DatasetConfig::Synthetic { seed: args.seed },
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging
    // Use RUST_LOG env var if set, otherwise use --verbose flag or default to "warn"
    let filter = std::env::var("RUST_LOG").unwrap_or_else(|_| {
        if args.verbose {
            "debug".to_string()
        } else {
            "warn".to_string()
        }
    });
    tracing_subscriber::fmt().with_env_filter(&filter).init();

    match &args.command {
        Some(Commands::Scenario {
            file,
            output_dir,
            no_save,
        }) => run_scenario_command(file, output_dir, *no_save).await,
        Some(Commands::Run) | None => run_single_benchmark(&args).await,
    }
}

async fn run_scenario_command(file: &str, output_dir: &str, no_save: bool) -> Result<()> {
    info!("Loading scenario from: {}", file);
    let scenario = load_scenario_from_file(file)?;

    let results = run_scenario(&scenario).await?;

    for result in &results {
        print_result(result);
    }

    if results.len() > 1 {
        print_comparison(&results);
    }

    // Save results to file
    if !no_save {
        let output_path = generate_output_filename(&scenario.name, output_dir);
        save_results_to_file(&results, &scenario.name, &output_path)?;
    }

    Ok(())
}

async fn run_single_benchmark(args: &Args) -> Result<()> {
    let base_url = args
        .base_url
        .clone()
        .ok_or_else(|| anyhow!("--base-url is required"))?;

    info!("GenAI Benchmark - LLM Load Testing Tool");
    info!("Target: {} (model: {})", base_url, args.model);
    info!(
        "Requests: {}, Concurrency: {}, RPS: {}",
        args.num_requests, args.concurrency, args.rps
    );

    let dataset_config = args.dataset.to_config(args);
    let prompts = load_dataset(&dataset_config, args.num_requests).await?;

    if prompts.is_empty() {
        return Err(anyhow!("No prompts loaded from dataset"));
    }

    info!("Loaded {} prompts, starting benchmark...", prompts.len());

    let config = BenchmarkConfig {
        name: None,
        base_url,
        api_key: args.api_key.clone(),
        model: args.model.clone(),
        max_tokens: args.max_tokens,
        concurrency: args.concurrency,
        rps: args.rps,
        timeout_secs: args.timeout,
        disable_prewarm: false,
    };

    let result = run_benchmark(&config, prompts, args.num_requests).await?;

    print_result(&result);

    Ok(())
}
