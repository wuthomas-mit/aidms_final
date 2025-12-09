from dotenv import load_dotenv
load_dotenv()  # Load API keys from .env file

from datasets import load_dataset
import pandas as pd
from evaluator import evaluate_dataframe

# Load the MentalChat16K dataset from HuggingFace
print("Loading MentalChat16K dataset from HuggingFace...")
ds = load_dataset("ShenLab/MentalChat16K")

# Convert to pandas DataFrame
print(f"Available splits: {list(ds.keys())}")
split_name = list(ds.keys())[0]
df = ds[split_name].to_pandas()

print("Dataset loaded!")
print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
print("\nFirst 3 records:")
print(df.head(3))

# Format the conversation for evaluation
def format_context(row):
    """Format the counseling conversation for cultural evaluation."""
    patient_input = str(row['input']) if pd.notna(row.get('input')) else ""
    counselor_output = str(row['output']) if pd.notna(row.get('output')) else ""
    
    return f"""PATIENT: {patient_input}

COUNSELOR: {counselor_output}"""

df['combined_context'] = df.apply(format_context, axis=1)

# # Show a sample of the combined context
# print("\n" + "=" * 50)
# print("SAMPLE COMBINED CONTEXT:")
# print("=" * 50)
# import random
# random.seed(42)
# sample_idx = random.choice(range(len(df)))
# print(df['combined_context'].iloc[sample_idx])

# Run the cultural representation evaluation with simple random sampling
print("\n" + "=" * 50)
print("Running Cultural Representation Evaluation...")
print("=" * 50 + "\n")

SAMPLE_SIZE = 416  # Number of random samples to evaluate

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
with open("mental_chat_results.json", "w") as f:
    json.dump({
        "counts": results["counts"],
        "total": results["total"],
        "labels": results["labels"],
    }, f, indent=2)

print("\nResults saved to mental_chat_results.json")
