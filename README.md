# Cultural Representation Evaluator

LLM-powered tool to evaluate cultural bias in mental health AI datasets by classifying responses as immigrant vs. assimilated American cultural norms.

## Labels

| Label | Description |
|-------|-------------|
| `1` | First/Second Generation American (immigrant norms) |
| `0` | Neutral/Ambiguous |
| `-1` | Third+ Generation American (assimilated norms) |

## Usage

```python
from evaluator import evaluate_dataframe

results = evaluate_dataframe(
    df=your_dataframe,
    text_column="text",
    sample_size=100,
    llm_provider="openai",  # or "anthropic", "gemini"
)
```

## Setup

```bash
pip install -r requirements.txt
```

Set your API key in a `.env` file:
```
OPENAI_API_KEY=your_key_here
```

## Datasets Evaluated

- **Empathetic Dialogues** (Facebook AI)
- **MentalChat16K** (mental health counseling)
- **EmoBench** (emotional intelligence scenarios)
