# Phase 2: QLoRA Fine-tuning (Phi-3 Mini)

Fine-tune **Phi-3 Mini 3.8B** on the retail tool-calling dataset generated in Phase 1 using QLoRA.

**Input:** `training_data.json`
**Output:** `models/phi3_retail_adapter/`
**GPU:** 6–8 GB VRAM (recommended)

## Training Flow

```text
training_data.json
        ↓
qlora_finetune.py
        ↓
4-bit Phi-3 + LoRA adapters
        ↓
Trained adapter weights
```

## Setup

```bash
cd Phase_2_ModelTraining

uv venv venv_phase2 --python 3.10
source venv_phase2/bin/activate

uv pip install -r requirements_phase2.txt
```

Verify CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Train

Make sure Phase 1 has generated `training_data.json`, then run:

```bash
python training/qlora_finetune.py
```

Default settings:

```python
EPOCHS = 3
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 512
```

## Output

```text
models/
└── phi3_retail_adapter/
    ├── checkpoint-*/
    └── final/
        ├── adapter_config.json
        └── adapter_model.bin
```

The adapter is typically **40–50 MB**.

## Common Issues

**CUDA out of memory**

```python
BATCH_SIZE = 2
MAX_SEQ_LENGTH = 256
```

**Model download failed**

```bash
huggingface-cli login
```

**Check GPU**

```bash
nvidia-smi
```

## Expected Time

| Hardware       |    Time |
| -------------- | ------: |
| RTX 3060 (8GB) | ~1.5 hr |
| RTX 4080       | ~45 min |
| CPU            |  6–9 hr |

## Next

After training, use the adapter in **Phase 3** for the retail agent and UI.
