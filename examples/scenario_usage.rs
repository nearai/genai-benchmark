use genai_benchmark::{load_multi_phase_scenario_from_file, run_multi_phase_scenario};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load a multi-phase scenario from a YAML file
    let scenario = load_multi_phase_scenario_from_file("scenarios/rag.yaml")?;

    println!("Running scenario: {}", scenario.name);
    if let Some(desc) = &scenario.description {
        println!("Description: {desc}");
    }

    // Execute the scenario against all configured providers
    let results = run_multi_phase_scenario(&scenario).await?;

    // Print results
    for result in results {
        genai_benchmark::print_result(&result);
    }

    Ok(())
}
