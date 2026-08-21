# Phase 1: Synthetic Data Generation

Generate 2,000+ synthetic retail tool-calling samples using a local Ollama model. The resulting dataset is used for Phase 2 QLoRA fine-tuning of Phi-3-mini.

## Pipeline

```text
Ollama Teacher (Llama 3.1)
        ↓
generate_dataset.py
        ↓
Product + Query → Tool-calling decision
        ↓
Parse + Validate
        ↓
data/training_data.json
        ↓
Phase 2: QLoRA Fine-tuning
```

## Setup

### 1. Install Ollama

Install from [Ollama](https://ollama.ai?utm_source=chatgpt.com).

Pull a model:

```bash
ollama pull llama3.1
# Alternatives:
# ollama pull llama2
# ollama pull mistral
```

Start the server:

```bash
ollama serve
```

Verify:

```bash
curl http://localhost:11434/api/tags
```

### 2. Create the environment

```bash
cd Phase_1_DataGeneration

uv venv venv_phase1 --python 3.10
source venv_phase1/bin/activate

uv pip install -r requirements_phase1.txt
```

Check dependencies:

```bash
python -c "
import requests
import pandas
import pydantic
print('All packages installed')
"
```

## Generate the Dataset

```bash
python data/generate_dataset.py
```

The script:

1. Creates random product/query combinations.
2. Sends them to Ollama.
3. Parses the tool-calling response.
4. Keeps valid samples.
5. Saves them to `data/training_data.json`.
6. Prints dataset statistics and examples.

For testing, reduce:

```python
NUM_SAMPLES = 100
```

Default:

```python
NUM_SAMPLES = 2000
```

## Dataset Format

```json
{
  "input": "Product: Red mesh running shoe, sporty style\nQuery: Find me something similar but cheaper",
  "output": {
    "tool": "retrieve_similar",
    "filters": {
      "color": "white",
      "price": "lower",
      "category": "running shoe"
    },
    "reasoning": "User wants similar product at lower price"
  },
  "product_metadata": {
    "category": "running shoe",
    "color": "red",
    "material": "mesh",
    "style": "sporty",
    "description": "Red running shoe made of mesh, sporty style"
  },
  "query": "Find me something similar but cheaper"
}
```

Expected tools include:

* `retrieve_similar`
* `compare_prices`
* `answer_query`
* `extract_attributes`
* `check_availability`

## Configuration

`generate_dataset.py` contains the main settings:

```python
NUM_SAMPLES = 2000
MODEL_NAME = "llama3.1"
REQUEST_TIMEOUT = 120
```

You can also modify product categories, colors, materials, styles, and retail queries.

For slower hardware:

```python
REQUEST_TIMEOUT = 300
```

For more consistent JSON output, lower the temperature to around `0.5`.

## Troubleshooting

**Ollama connection failed**

```bash
ollama serve
curl http://localhost:11434/api/tags
```

**Generation is too slow**

Try a smaller model:

```bash
ollama pull mistral
```

Then change:

```python
MODEL_NAME = "mistral"
```

**High failure rate**

Lower the temperature and try another model.

**Missing Python packages**

```bash
source venv_phase1/bin/activate
uv pip install -r requirements_phase1.txt
```

Monitor GPU usage with:

```bash
nvidia-smi -l 1
```

## Check the Dataset

```bash
python -c "
import json
with open('data/training_data.json') as f:
    data = json.load(f)

print(f'Total samples: {len(data)}')
print(f'Tools: {set(s[\"output\"][\"tool\"] for s in data)}')
print(f'First sample: {data[0]}')
"
```

## Project Structure

```text
Phase_1_DataGeneration/
├── data/
│   ├── training_data.json
│   └── .gitkeep
├── venv_phase1/
├── generate_dataset.py
├── requirements_phase1.txt
└── README_Phase1.md
```


