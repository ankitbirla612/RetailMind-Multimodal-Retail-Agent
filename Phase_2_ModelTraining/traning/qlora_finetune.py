"""
Phase 2: QLoRA Fine-tuning of Phi-3-mini-3.8B
Purpose: Fine-tune Phi-3-mini on retail tool-calling data generated in Phase 1
Output: adapter weights saved in models/phi3_retail_adapter
Training time: 30-120 minutes (depends on GPU)
"""

import json
import os
import sys
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# HuggingFace libraries
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Logging & tracking
import logging
from tqdm import tqdm

# Configuration
DATA_PATH = "/work/retailmind/data/training_data.json"
MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"
OUTPUT_DIR = "models/phi3_retail_adapter"
ADAPTER_OUTPUT = "models/phi3_retail_adapter/final"

# Training hyperparameters
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 2
LEARNING_RATE = 2e-4
EPOCHS = 3
MAX_SEQ_LENGTH = 512
WARMUP_STEPS = 100

# Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_MEMORY = {0: "8GB"} if torch.cuda.is_available() else {}

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{OUTPUT_DIR}/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def ensure_output_dir():
    """Create output directories"""
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    logger.info(f"✓ Output directory ready: {OUTPUT_DIR}/")


def check_requirements():
    """Check GPU availability and memory"""
    print(f"\n{'='*60}")
    print(f"🔧 Environment Check")
    print(f"{'='*60}\n")
    
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Total GPU Memory: {total_memory:.1f} GB")
        allocated = torch.cuda.memory_allocated(0) / 1e9
        print(f"Allocated: {allocated:.1f} GB")
    
    print(f"Device Type: {DEVICE.upper()}")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # Memory check
    if cuda_available and total_memory < 8:
        logger.warning(" GPU has <8GB memory. Training might be slow or fail.")
        logger.warning("   Consider reducing BATCH_SIZE or MAX_SEQ_LENGTH")
    
    print(f"\n{'='*60}\n")


def load_training_data(data_path: str) -> Dataset:
    """Load and prepare training data from Phase 1"""
    logger.info(f"Loading training data from {data_path}...")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found: {data_path}")
    
    # Load JSON
    with open(data_path, 'r') as f:
        raw_data = json.load(f)
    
    logger.info(f"Loaded {len(raw_data)} samples from Phase 1")
    
    # Prepare data for training
    processed_data = []
    for sample in raw_data:
        # Format: input + output as structured text
        input_text = sample['input']
        output_dict = sample['output']
        
        # Create structured output string
        output_text = f"""Tool: {output_dict.get('tool', 'unknown')}
Filters: {json.dumps(output_dict.get('filters', {}))}
Reasoning: {output_dict.get('reasoning', '')}"""
        
        processed_data.append({
            "text": f"Input: {input_text}\n\nOutput:\n{output_text}"
        })
    
    # Create HuggingFace Dataset
    dataset = Dataset.from_dict({
        "text": [d["text"] for d in processed_data]
    })
    
    # Split into train and validation
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    logger.info(f"✓ Training samples: {len(split_dataset['train'])}")
    logger.info(f"✓ Validation samples: {len(split_dataset['test'])}")
    
    # Show sample
    logger.info(f"\nSample training example:")
    logger.info(f"{split_dataset['train'][0]['text'][:200]}...\n")
    
    return split_dataset


def setup_model_and_tokenizer():
    """Load model and tokenizer with 4-bit quantization"""
    logger.info(f"Loading model: {MODEL_NAME}...")
    
    # 4-bit quantization config for 8GB VRAM
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="right"
    )
    
    # Add padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager"  # Faster than flash_attention on some GPUs
    )
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    
    logger.info(f"✓ Model loaded with 4-bit quantization")
    logger.info(f"  Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    return model, tokenizer


def setup_lora_config() -> LoraConfig:
    """Configure LoRA adapters"""
    lora_config = LoraConfig(
        r=16,                           # LoRA rank
        lora_alpha=32,                  # LoRA scaling
        target_modules=[
            "q_proj",                   # Query projection
            "v_proj",                   # Value projection
            "k_proj",                   # Key projection
            "o_proj",                   # Output projection
            "gate_proj",                # Gate projection
            "up_proj",                  # Up projection
            "down_proj"                 # Down projection
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    logger.info(f"✓ LoRA config ready")
    logger.info(f"  Rank: {lora_config.r}")
    logger.info(f"  Alpha: {lora_config.lora_alpha}")
    
    return lora_config


def preprocess_function(examples, tokenizer, max_length=MAX_SEQ_LENGTH):
    """Preprocess text for training"""
    # Tokenize
    result = tokenizer(
        examples["text"],
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors=None
    )
    
    result["labels"] = result["input_ids"].copy()
    return result


def train_model(
    model,
    tokenizer,
    train_dataset,
    val_dataset,
    lora_config
):
    """Train model with QLoRA"""
    logger.info(f"\n{'='*60}")
    logger.info(f"🚀 Starting QLoRA Fine-tuning")
    logger.info(f"{'='*60}\n")
    
    # Add LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        seed=42,
        dataloader_pin_memory=True
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8
    )
    
    # Preprocess datasets
    logger.info("Preprocessing datasets...")
    train_tokenized = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Processing train dataset"
    )
    
    val_tokenized = val_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Processing val dataset"
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    logger.info(f"✓ Training complete!")
    
    return model, trainer


def save_adapter(model, output_path=ADAPTER_OUTPUT):
    """Save trained adapter weights"""
    logger.info(f"\nSaving adapter to {output_path}...")
    
    Path(output_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    
    logger.info(f"✓ Adapter saved!")
    logger.info(f"  Location: {output_path}")
    
    return output_path


def evaluate_model(trainer, val_dataset):
    """Evaluate fine-tuned model"""
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 Evaluation")
    logger.info(f"{'='*60}\n")
    
    metrics = trainer.evaluate(eval_dataset=val_dataset)
    
    logger.info(f"Validation Loss: {metrics.get('eval_loss', 'N/A'):.4f}")
    logger.info(f"Perplexity: {np.exp(metrics.get('eval_loss', 0)):.4f}")
    
    return metrics


def test_inference(model, tokenizer, test_prompts=None):
    """Test the fine-tuned model"""
    logger.info(f"\n{'='*60}")
    logger.info(f"🧪 Inference Test")
    logger.info(f"{'='*60}\n")
    
    if test_prompts is None:
        test_prompts = [
            "Input: Product: Red mesh running shoe\nQuery: Find cheaper\n\nOutput:\n",
            "Input: Product: Blue leather jacket\nQuery: Is there a better option\n\nOutput:\n",
            "Input: Product: Black smartphone\nQuery: Compare prices\n\nOutput:\n"
        ]
    
    model.eval()
    with torch.no_grad():
        for i, prompt in enumerate(test_prompts):
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            outputs = model.generate(
                **inputs,
                max_length=150,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"\nTest {i+1}:")
            logger.info(f"Prompt: {prompt[:100]}...")
            logger.info(f"Response: {response[len(prompt):]}")


def main():
    """Main execution"""
    print(f"\n{'='*60}")
    print(f"Phase 2: QLoRA Fine-tuning of Phi-3-mini")
    print(f"{'='*60}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE.upper()}")
    print()
    
    # Step 1: Setup
    ensure_output_dir()
    check_requirements()
    
    # Step 2: Load data
    dataset = load_training_data(DATA_PATH)
    
    # Step 3: Setup model
    model, tokenizer = setup_model_and_tokenizer()
    
    # Step 4: Setup LoRA
    lora_config = setup_lora_config()
    
    # Step 5: Train
    model, trainer = train_model(
        model,
        tokenizer,
        dataset['train'],
        dataset['test'],
        lora_config
    )
    
    # Step 6: Evaluate
    metrics = evaluate_model(trainer, dataset['test'])
    
    # Step 7: Save
    adapter_path = save_adapter(model)
    
    # Step 8: Test inference
    test_inference(model, tokenizer)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"✅ Phase 2 Complete!")
    print(f"{'='*60}")
    print(f"Adapter saved: {adapter_path}")
    print(f"Next: Phase 3 - Agent & UI")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()