# Install dependencies as needed:
# pip install kagglehub[pandas-datasets] openai python-dotenv
# Or for other providers:
# pip install kagglehub[pandas-datasets] anthropic python-dotenv
# pip install kagglehub[pandas-datasets] google-generativeai python-dotenv

from dotenv import load_dotenv
load_dotenv()  # Load API keys from .env file

import kagglehub
import pandas as pd
from evaluator import evaluate_dataframe

# Download the dataset
dataset_path = kagglehub.dataset_download("atharvjairath/empathetic-dialogues-facebook-ai")
print(f"Dataset downloaded to: {dataset_path}")

# Load the CSV file
import os
csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
print(f"Available CSV files: {csv_files}")

# Load the first CSV file found (or specify the exact filename if multiple)
df = pd.read_csv(os.path.join(dataset_path, csv_files[0]))

print("Dataset loaded!")
print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
print("\nFirst 5 records:")
print(df.head())

print(f"Unique emotions: {df['emotion'].nunique()}")
print(f"Entries per emotion:\n{df['emotion'].value_counts()}")

# Filter to only include rows where labels has length > 10
df = df[df['labels'].apply(lambda x: len(str(x)) > 10 if pd.notna(x) else False)]
print(f"\nRows after filtering (labels length > 10): {len(df)}")

# # Filter to only keep the top 32 valid emotions (exclude parsing errors)
# top_32_emotions = df['emotion'].value_counts().head(32).index.tolist()
# df = df[df['emotion'].isin(top_32_emotions)]
# print(f"Rows after filtering to top 32 emotions: {len(df)}")
# print(f"Valid emotions: {sorted(top_32_emotions)}")

# Combine Situation, empathetic_dialogues, and labels into a single formatted text
# This gives the LLM full context to determine what kind of person would respond this way
def format_context(row):
    situation = str(row['Situation']) if pd.notna(row['Situation']) else ""
    dialogue = str(row['empathetic_dialogues']) if pd.notna(row['empathetic_dialogues']) else ""
    response = str(row['labels']) if pd.notna(row['labels']) else ""
    
    return f"""SITUATION: {situation}

DIALOGUE: {dialogue}

RESPONSE: {response}"""

df['combined_context'] = df.apply(format_context, axis=1)

# # Show a sample of the combined context
# print("\n" + "=" * 50)
# print("SAMPLE COMBINED CONTEXT:")
# print("=" * 50)
# import random
# random.seed(42)
# sample_idx = random.choice(range(len(df)))
# print(df['combined_context'].iloc[sample_idx])

# Stratified sampling: sample equally from each emotion category
def stratified_sample_by_emotion(df, samples_per_emotion, random_seed=42):
    """
    Sample equally from each emotion category for balanced representation.
    
    Args:
        df: DataFrame with an 'emotion' column
        samples_per_emotion: Number of samples to take from each emotion
        random_seed: For reproducibility
    """
    sampled_indices = []
    for emotion in df['emotion'].unique():
        emotion_df = df[df['emotion'] == emotion]
        n_samples = min(len(emotion_df), samples_per_emotion)
        sampled_indices.extend(emotion_df.sample(n=n_samples, random_state=random_seed).index.tolist())
    
    return df.loc[sampled_indices].reset_index(drop=True)

# number of samples per emotion (32 valid emotions)
SAMPLES_PER_EMOTION = 13
df_stratified = stratified_sample_by_emotion(df, samples_per_emotion=SAMPLES_PER_EMOTION, random_seed=42)

print(f"\nStratified sampling: {SAMPLES_PER_EMOTION} samples per emotion")
print(f"Total stratified sample size: {len(df_stratified)}")
print(f"Emotions represented: {df_stratified['emotion'].nunique()}")
print(f"\nSamples per emotion:\n{df_stratified['emotion'].value_counts().sort_index()}")

# Run the cultural representation evaluation on the stratified sample
print("\n" + "=" * 50)
print("Running Cultural Representation Evaluation...")
print("=" * 50 + "\n")

results = evaluate_dataframe(
    df=df_stratified,
    text_column="combined_context",  # Full context: Situation + Dialogue + Response
    sample_size=len(df_stratified),  # Use entire stratified sample
    llm_provider="openai",
    random_seed=3950,                  # For reproducibility
    show_plot=True
)

# print("\n" + "=" * 50)
# print("Running Cultural Representation Evaluation...")
# print("=" * 50 + "\n")

# results = evaluate_dataframe(
#     df=df,
#     text_column="combined_context",  # Full context: Situation + Dialogue + Response
#     sample_size=50,         # Number of random samples to evaluate
#     llm_provider="openai",
#     random_seed=39,         # For reproducibility
#     show_plot=True
# )

# Save results to file
import json
with open("empathetic_dialogues_results.json", "w") as f:
    json.dump({
        "counts": results["counts"],
        "total": results["total"],
        "labels": results["labels"],
    }, f, indent=2)

print("\nResults saved to empathetic_dialogues_results.json")
