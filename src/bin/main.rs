use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand, ValueEnum};
use genai_benchmark::{
    expand_scenario_env_vars, generate_output_filename, load_dataset, load_scenario_from_file,
    print_comparison, print_result, run_benchmark, run_scenario, save_results_to_file,
    BenchmarkConfig, DatasetConfig, Scenario,
};
use tracing::info;

// Embedded scenario files
const SCENARIO_NEAR_VS_BEDROCK: &str = include_str!("../../scenarios/near_vs_bedrock.yaml");
const SCENARIO_LOW_CONCURRENCY: &str = include_str!("../../scenarios/low_concurrency.yaml");
const SCENARIO_SINGLE_PROVIDER: &str = include_str!("../../scenarios/single_provider.yaml");

fn get_embedded_scenario(name: &str) -> Option<&'static str> {
    match name {
        "near-vs-bedrock" => Some(SCENARIO_NEAR_VS_BEDROCK),
        "low-concurrency" => Some(SCENARIO_LOW_CONCURRENCY),
        "single-provider" => Some(SCENARIO_SINGLE_PROVIDER),
        _ => None,
    }
}

fn list_embedded_scenarios() -> Vec<&'static str> {
    vec!["near-vs-bedrock", "low-concurrency", "single-provider"]
}

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
    /// Run an embedded scenario by name
    Run {
        /// Scenario name (e.g., near-vs-bedrock, low-concurrency, single-provider)
        #[arg(required = true)]
        scenario: String,

        /// Output directory for results (default: ./output)
        #[arg(long, short, default_value = "output")]
        output_dir: String,

        /// Skip saving results to file
        #[arg(long)]
        no_save: bool,
    },
    /// Export an embedded scenario to stdout
    Export {
        /// Scenario name to export (e.g., near-vs-bedrock, low-concurrency, single-provider)
        #[arg(required = true)]
        scenario: String,
    },
    /// Describe a scenario (show config and required env vars)
    Describe {
        /// Scenario name to describe (e.g., near-vs-bedrock, low-concurrency, single-provider)
        #[arg(required = true)]
        scenario: String,
    },
    /// List available embedded scenarios
    List,
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
        }) => run_scenario_file(file, output_dir, *no_save).await,
        Some(Commands::Run {
            scenario,
            output_dir,
            no_save,
        }) => run_embedded_scenario(scenario, output_dir, *no_save).await,
        Some(Commands::Export { scenario }) => export_scenario(scenario),
        Some(Commands::Describe { scenario }) => describe_scenario(scenario),
        Some(Commands::List) => list_scenarios(),
        None => run_single_benchmark(&args).await,
    }
}

async fn run_scenario_file(file: &str, output_dir: &str, no_save: bool) -> Result<()> {
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

async fn run_embedded_scenario(name: &str, output_dir: &str, no_save: bool) -> Result<()> {
    let yaml_content = get_embedded_scenario(name).ok_or_else(|| {
        anyhow!(
            "Unknown scenario: '{}'. Use 'list' to see available scenarios.",
            name
        )
    })?;

    info!("Running embedded scenario: {}", name);
    let scenario: Scenario = serde_yaml::from_str(yaml_content)?;
    let scenario = expand_scenario_env_vars(scenario)?;

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

fn export_scenario(name: &str) -> Result<()> {
    let yaml_content = get_embedded_scenario(name).ok_or_else(|| {
        anyhow!(
            "Unknown scenario: '{}'. Use 'list' to see available scenarios.",
            name
        )
    })?;

    println!("{}", yaml_content);
    Ok(())
}

fn describe_scenario(name: &str) -> Result<()> {
    let yaml_content = get_embedded_scenario(name).ok_or_else(|| {
        anyhow!(
            "Unknown scenario: '{}'. Use 'list' to see available scenarios.",
            name
        )
    })?;

    let scenario: Scenario = serde_yaml::from_str(yaml_content)?;

    println!("Scenario: {}", name);
    println!("Name: {}", scenario.name);
    if let Some(desc) = &scenario.description {
        println!("Description: {}", desc);
    }
    println!();

    println!("Configuration:");
    println!("  Model: {}", scenario.model);
    println!("  Requests: {}", scenario.num_requests);
    println!("  Concurrency: {}", scenario.concurrency);
    println!("  RPS: {}", scenario.rps);
    println!("  Max tokens: {}", scenario.max_tokens);
    println!("  Timeout: {}s", scenario.timeout_secs);
    println!();

    println!("Providers:");
    for provider in &scenario.providers {
        println!("  - {}", provider.name);
        println!("    URL: {}", provider.base_url);
        println!(
            "    Model: {}",
            provider.model.as_ref().unwrap_or(&scenario.model)
        );
    }
    println!();

    // Extract environment variables from the YAML content
    let env_vars = extract_env_vars(yaml_content);
    if !env_vars.is_empty() {
        println!("Required environment variables:");
        for var in env_vars {
            println!("  - {}", var);
        }
        println!();
    }

    println!("Usage:");
    println!("  genai-benchmark run {}  # Run this scenario", name);

    Ok(())
}

fn extract_env_vars(yaml_content: &str) -> Vec<String> {
    use std::collections::HashSet;
    let mut vars = HashSet::new();

    // Match ${VAR_NAME} pattern
    let re = regex::Regex::new(r"\$\{([A-Z_][A-Z0-9_]*)\}").unwrap();
    for caps in re.captures_iter(yaml_content) {
        if let Some(var) = caps.get(1) {
            vars.insert(var.as_str().to_string());
        }
    }

    let mut vars: Vec<String> = vars.into_iter().collect();
    vars.sort();
    vars
}

fn list_scenarios() -> Result<()> {
    println!("Available embedded scenarios:");
    for scenario in list_embedded_scenarios() {
        println!("  - {}", scenario);
    }
    println!();
    println!("Usage:");
    println!("  genai-benchmark describe <scenario-name>  # Show details and required env vars");
    println!("  genai-benchmark run <scenario-name>       # Run an embedded scenario");
    println!("  genai-benchmark export <scenario-name>    # Export scenario to file");
    println!("  genai-benchmark scenario <path>           # Run a custom YAML file");
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
        verify: false,
    };

    let result = run_benchmark(&config, prompts, args.num_requests).await?;

    print_result(&result);

    Ok(())
}
