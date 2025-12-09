"""
Cultural Representation Evaluator

Evaluates whether datasets are representative of different cultural backgrounds
in America, using LLM-based classification on random sampling.

Labels:
    1  = First/Second Generation American (stronger cultural ties to heritage)
    0  = Neutral/Ambiguous (cannot be clearly classified)
   -1  = Third+ Generation American (culturally assimilated)
"""

import os
import json
import random
from datetime import datetime
from typing import Literal, Optional
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# LLM API imports - install as needed
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
except ImportError:
    genai = None


# Label definitions (following recitation format)
LABELS = {
    1: "First/Second Generation American",
    0: "Neutral/Ambiguous",
    -1: "Third+ Generation American"
}

CLASSIFICATION_PROMPT = '''Given the SITUATION, DIALOGUE, and RESPONSE below, classify the norms and way of thinking behind the RESPONSE.

First/Second Generation American Norms and Way of Thinking (label: 1): defined by immigrant norms and values,

Third+ Generation American Norms and Way of Thinking (label: -1): defined by American norms and values,

Neutral Norms and Way of Thinking(label: 0): neutral way of thinking,

{text}

Based on the NORMS AND WAY OF THINKING behind the response (not just the words), answer with only the label: "1", "-1", or "0".'''


class CulturalRepresentationEvaluator:
    """
    Evaluates cultural representation in datasets using LLM classification.
    Following the AIDMS recitation format with -1, 0, 1 labels.
    """
    
    def __init__(
        self,
        llm_provider: Literal["openai", "anthropic", "gemini"] = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the evaluator.
        
        Args:
            llm_provider: Which LLM provider to use ("openai", "anthropic", or "gemini")
            model: Model name (defaults based on provider)
            api_key: API key (defaults to environment variable)
        """
        self.llm_provider = llm_provider
        
        if llm_provider == "openai":
            if OpenAI is None:
                raise ImportError("openai package not installed. Run: pip install openai")
            self.model = model or "gpt-4o"
            self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
            
        elif llm_provider == "anthropic":
            if anthropic is None:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")
            self.model = model or "claude-sonnet-4-20250514"
            self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
            
        elif llm_provider == "gemini":
            if genai is None:
                raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")
            self.model = model or "gemini-1.5-flash"
            api_key = api_key or os.getenv("GOOGLE_API_KEY")
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(self.model)
        else:
            raise ValueError(f"Unknown LLM provider: {llm_provider}")
    
    def _call_llm(self, prompt: str) -> str:
        """Make an LLM API call and return the response text."""
        import time
        if self.llm_provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            time.sleep(1)  # Rate limiting for OpenAI
            return response.choices[0].message.content.strip()
            
        elif self.llm_provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=16,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
            
        elif self.llm_provider == "gemini":
            response = self.client.generate_content(
                prompt,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
                }
            )
            time.sleep(5)  # Rate limiting for Gemini free tier
            return response.text.strip()
    
    def classify_text(self, text: str) -> int:
        """
        Classify a single text sample.
        
        Args:
            text: The text to classify
            
        Returns:
            Label: 1 (first/second gen), 0 (neutral), -1 (third+ gen)
        """
        prompt = CLASSIFICATION_PROMPT.format(text=text)
        response = self._call_llm(prompt)
        
        try:
            response_clean = response.strip().replace('"', '').replace("'", "")
            label = int(response_clean)
            if label in [-1, 0, 1]:
                return label
        except (ValueError, TypeError):
            pass
        
        print(f"Warning: Could not parse response '{response}', defaulting to 0 (neutral)")
        return 0
    
    def evaluate_dataset(
        self,
        data: pd.DataFrame | list[str],
        text_column: Optional[str] = None,
        sample_size: int = 100,
        random_seed: Optional[int] = None,
    ) -> dict:
        """
        Evaluate cultural representation in a dataset.
        
        Args:
            data: DataFrame or list of text samples
            text_column: Column name containing text (required if data is DataFrame)
            sample_size: Number of random samples to evaluate
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary with labels list and counts
        """
        if isinstance(data, pd.DataFrame):
            if text_column is None:
                raise ValueError("text_column must be specified for DataFrame input")
            texts = data[text_column].dropna().tolist()
        else:
            texts = [t for t in data if t]
        
        if random_seed is not None:
            random.seed(random_seed)
        
        sample_size = min(sample_size, len(texts))
        sampled_texts = random.sample(texts, sample_size)
        
        labels_list = []
        for i, text in enumerate(sampled_texts):
            print(f"Classifying sample {i+1}/{sample_size}...")
            label = self.classify_text(text)
            labels_list.append(label)
            print(f"  -> {LABELS[label]} ({label})")
        
        counts = {1: 0, 0: 0, -1: 0}
        for label in labels_list:
            counts[label] += 1
        
        return {
            "labels": labels_list,
            "texts": sampled_texts,
            "counts": counts,
            "total": sample_size
        }


def plot_results(counts: dict, title: str = "Cultural Representation in Dataset"):
    """
    Plot the distribution of cultural representation labels.
    
    Args:
        counts: Dictionary with label counts {1: n, 0: n, -1: n}
        title: Plot title
    """
    categories = ['First/Second Gen\n(1)', 'Neutral\n(0)', 'Third+ Gen\n(-1)']
    values = [counts[1], counts[0], counts[-1]]
    colors = ['#4CAF50', '#9E9E9E', '#2196F3']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=1.2)
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('# Samples', fontsize=12)
    ax.set_xlabel('Cultural Classification', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    total = sum(values)
    if total > 0:
        percentages = [v/total*100 for v in values]
        for i, (bar, pct) in enumerate(zip(bars, percentages)):
            ax.annotate(f'({pct:.1f}%)',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2),
                        ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return fig


def print_summary(results: dict):
    """Print a summary of the evaluation results."""
    counts = results["counts"]
    total = results["total"]
    
    print("\n" + "=" * 50)
    print("CULTURAL REPRESENTATION EVALUATION SUMMARY")
    print("=" * 50)
    print(f"\nTotal Samples Evaluated: {total}")
    print("\nDistribution:")
    print(f"  First/Second Gen (1):  {counts[1]:3d} ({counts[1]/total*100:5.1f}%)")
    print(f"  Neutral (0):           {counts[0]:3d} ({counts[0]/total*100:5.1f}%)")
    print(f"  Third+ Gen (-1):       {counts[-1]:3d} ({counts[-1]/total*100:5.1f}%)")
    
    print("\n" + "-" * 50)
    print("REPRESENTATION ANALYSIS")
    print("-" * 50)
    
    non_neutral = counts[1] + counts[-1]
    if non_neutral > 0:
        first_second_ratio = counts[1] / non_neutral
        print(f"\nAmong non-neutral samples:")
        print(f"  First/Second Gen: {first_second_ratio*100:.1f}%")
        print(f"  Third+ Gen:       {(1-first_second_ratio)*100:.1f}%")
        
        if first_second_ratio < 0.3:
            print("\nWARNING: Dataset appears UNDER-REPRESENTATIVE of First/Second Gen Americans")
        elif first_second_ratio > 0.7:
            print("\nWARNING: Dataset appears OVER-REPRESENTATIVE of First/Second Gen Americans")
        else:
            print("\nDataset appears reasonably balanced between cultural perspectives")
    else:
        print("\nAll samples classified as neutral - unable to assess representation")
    
    print("=" * 50 + "\n")


def save_results(results: dict, output_dir: str = ".") -> str:
    """
    Save evaluation results to a JSON file with a unique timestamp.
    
    Args:
        results: Dictionary with evaluation results
        output_dir: Directory to save the results file
        
    Returns:
        Path to the saved file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"evaluation_results_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "counts": results["counts"],
            "total": results["total"],
            "labels": results["labels"],
            "texts": results.get("texts", [])
        }, f, indent=2)
    
    print(f"Results saved to: {filepath}")
    return filepath


def evaluate_texts(
    texts: list[str],
    sample_size: int = 50,
    llm_provider: Literal["openai", "anthropic", "gemini"] = "openai",
    model: Optional[str] = None,
    random_seed: Optional[int] = 42,
    show_plot: bool = True,
    save: bool = True,
    output_dir: str = ".",
) -> dict:
    """
    Convenience function to evaluate a list of texts.
    
    Args:
        texts: List of text samples to evaluate
        sample_size: Number of samples to randomly select
        llm_provider: LLM provider to use
        model: Model name (optional)
        random_seed: Random seed for reproducibility
        show_plot: Whether to display the results plot
        save: Whether to save results to a timestamped JSON file
        output_dir: Directory to save results file
        
    Returns:
        Dictionary with evaluation results
    """
    evaluator = CulturalRepresentationEvaluator(
        llm_provider=llm_provider,
        model=model,
    )
    
    results = evaluator.evaluate_dataset(
        data=texts,
        sample_size=sample_size,
        random_seed=random_seed,
    )
    
    print_summary(results)
    
    if save:
        save_results(results, output_dir)
    
    if show_plot:
        plot_results(results["counts"])
    
    return results


def evaluate_dataframe(
    df: pd.DataFrame,
    text_column: str,
    sample_size: int = 50,
    llm_provider: Literal["openai", "anthropic", "gemini"] = "openai",
    model: Optional[str] = None,
    random_seed: Optional[int] = 42,
    show_plot: bool = True,
    save: bool = True,
    output_dir: str = ".",
) -> dict:
    """
    Convenience function to evaluate a DataFrame.
    
    Args:
        df: DataFrame containing text data
        text_column: Name of column containing text to analyze
        sample_size: Number of samples to randomly select
        llm_provider: LLM provider to use
        model: Model name (optional)
        random_seed: Random seed for reproducibility
        show_plot: Whether to display the results plot
        save: Whether to save results to a timestamped JSON file
        output_dir: Directory to save results file
        
    Returns:
        Dictionary with evaluation results
    """
    evaluator = CulturalRepresentationEvaluator(
        llm_provider=llm_provider,
        model=model,
    )
    
    results = evaluator.evaluate_dataset(
        data=df,
        text_column=text_column,
        sample_size=sample_size,
        random_seed=random_seed,
    )
    
    print_summary(results)
    
    if save:
        save_results(results, output_dir)
    
    if show_plot:
        plot_results(results["counts"])
    
    return results


# Example usage
if __name__ == "__main__":
    print("empty main")
    pass
    
