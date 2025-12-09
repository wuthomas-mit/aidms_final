from dotenv import load_dotenv
load_dotenv()  # Load API keys from .env file

from datasets import load_dataset
import pandas as pd
from evaluator import evaluate_dataframe

# Load EmoBench - emotional_application config (has scenario + choices + label)
print("Loading EmoBench dataset from HuggingFace...")
ds_ea = load_dataset("SahandSab/EmoBench", "emotional_application")

# Convert to pandas
df = ds_ea['train'].to_pandas()

print("Dataset loaded!")
print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}")

# Filter to English only
df = df[df['language'] == 'en']
print(f"Rows after filtering to English: {len(df)}")

print("\nFirst 3 records:")
print(df.head(3))

# Format the scenario for evaluation
def format_context(row):
    """Format the EmoBench scenario for cultural evaluation."""
    scenario = str(row['scenario']) if row.get('scenario') is not None else ""
    
    # Handle choices as a list
    choices_val = row.get('choices')
    if choices_val is not None and len(choices_val) > 0:
        choices = ", ".join(choices_val) if isinstance(choices_val, list) else str(choices_val)
    else:
        choices = ""
    
    label = str(row['label']) if row.get('label') is not None else ""
    
    return f"""SCENARIO: {scenario}

CHOICES: {choices}

ANSWER: {label}"""

df['combined_context'] = df.apply(format_context, axis=1)

# Run the cultural representation evaluation with simple random sampling
print("\n" + "=" * 50)
print("Running Cultural Representation Evaluation...")
print("=" * 50 + "\n")

SAMPLE_SIZE = 200  # Number of random samples to evaluate

results = evaluate_dataframe(
    df=df,
    text_column="combined_context",
    sample_size=SAMPLE_SIZE,
    llm_provider="openai",
    random_seed=3950,
    show_plot=True,
    save=False  # We'll save manually with a specific name
)

# Save results to file
import json
with open("emo_bench_results.json", "w") as f:
    json.dump({
        "counts": results["counts"],
        "total": results["total"],
        "labels": results["labels"],
    }, f, indent=2)

print("\nResults saved to emo_bench_results.json")
