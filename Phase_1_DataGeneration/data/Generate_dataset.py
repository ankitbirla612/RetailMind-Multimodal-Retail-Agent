"""
Phase 1: Synthetic Training Data Generation using Ollama
Purpose: Generate retail tool-calling training data by using Ollama as a teacher model
Output: JSON file with (input, output) pairs for fine-tuning Phi-3-mini
"""

import json
import requests
import time
from typing import List, Dict, Any
from datetime import datetime
import os
from pathlib import Path
from tqdm import tqdm
import random

# Configuration
OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "llama2"  # Change to "llama3.1" if available
OUTPUT_DIR = "data"
OUTPUT_FILE = "training_data.json"
NUM_SAMPLES = 2000
BATCH_SIZE = 10
REQUEST_TIMEOUT = 120

# Retail product categories and attributes
PRODUCT_CATEGORIES = [
    "running shoe", "dress", "jacket", "laptop", "smartphone",
    "headphones", "backpack", "watch", "lamp", "chair",
    "coffee maker", "blender", "keyboard", "monitor", "camera"
]

COLORS = [
    "red", "blue", "green", "white", "black", "gray", "yellow",
    "purple", "pink", "orange", "brown", "silver", "gold"
]

MATERIALS = [
    "leather", "cotton", "polyester", "mesh", "plastic", "metal",
    "rubber", "wool", "silk", "denim", "canvas", "nylon"
]

STYLES = [
    "sporty", "casual", "formal", "vintage", "modern", "minimalist",
    "classic", "trendy", "professional", "comfortable"
]

RETAIL_QUERIES = [
    "Find me something similar but cheaper",
    "Do you have this in a different color?",
    "Is this good quality?",
    "Show me similar items",
    "What's the price comparison?",
    "Find cheaper alternatives",
    "Do you have it in white?",
    "Is there a better option?",
    "Compare this with similar products",
    "What's a good alternative?",
    "Find premium versions",
    "Show me budget options",
    "Is this waterproof?",
    "What material is this?",
    "Do you have darker colors?"
]

# Tool definitions
TOOLS = [
    "retrieve_similar",
    "compare_prices",
    "extract_attributes",
    "answer_query",
    "check_availability"
]


def ensure_output_dir():
    """Create output directory if it doesn't exist"""
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory ready: {OUTPUT_DIR}/")


def check_ollama_connection():
    """Check if Ollama is running"""
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={"model": MODEL_NAME, "prompt": "test", "stream": False},
            timeout=10
        )
        if response.status_code == 200:
            print(f"✓ Ollama connection successful (Model: {MODEL_NAME})")
            return True
    except Exception as e:
        print(f"✗ Ollama connection failed: {e}")
        print(f"  Make sure Ollama is running: ollama serve")
        return False


def generate_product_description() -> Dict[str, str]:
    """Generate a random product description"""
    category = random.choice(PRODUCT_CATEGORIES)
    color = random.choice(COLORS)
    material = random.choice(MATERIALS)
    style = random.choice(STYLES)
    
    return {
        "category": category,
        "color": color,
        "material": material,
        "style": style,
        "description": f"{color.capitalize()} {category} made of {material}, {style} style"
    }


def generate_training_sample_prompt(product: Dict[str, str], query: str) -> str:
    """Generate a prompt for Ollama to create training samples"""
    prompt = f"""Given the following product and user query, determine which tool should be used.

Product: {product['description']}
User Query: "{query}"

You must respond with ONLY a JSON object (no markdown, no explanation) in this exact format:
{{"tool": "<tool_name>", "filters": {{...}}, "reasoning": "<brief reason>"}}

Available tools:
- retrieve_similar: Find similar products
- compare_prices: Compare prices of products
- extract_attributes: Extract product attributes
- answer_query: Answer questions about the product
- check_availability: Check if product is available

Respond with only the JSON object:"""
    
    return prompt


def call_ollama(prompt: str) -> str:
    """Call Ollama API and get response"""
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.7,
            },
            timeout=REQUEST_TIMEOUT
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "").strip()
        else:
            return None
            
    except requests.exceptions.Timeout:
        print(f"  ⚠ Timeout - retrying...")
        return None
    except Exception as e:
        print(f"  ⚠ Error: {e}")
        return None


def parse_tool_response(response: str) -> Dict[str, Any] | None:
    """Parse Ollama response to extract tool call"""
    try:
        # Clean response - remove markdown if present
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]
        
        # Parse JSON
        data = json.loads(response.strip())
        
        # Validate tool
        if data.get("tool") not in TOOLS:
            return None
            
        return data
        
    except json.JSONDecodeError:
        return None
    except Exception as e:
        return None


def generate_dataset(num_samples: int = NUM_SAMPLES) -> List[Dict[str, Any]]:
    """Generate training dataset using Ollama"""
    
    print(f"\n{'='*60}")
    print(f"Generating {num_samples} training samples...")
    print(f"{'='*60}\n")
    
    dataset = []
    failed_count = 0
    max_retries = 3
    
    for i in tqdm(range(num_samples), desc="Generating samples"):
        product = generate_product_description()
        query = random.choice(RETAIL_QUERIES)
        
        prompt = generate_training_sample_prompt(product, query)
        
        # Retry logic
        retry_count = 0
        response = None
        
        while retry_count < max_retries and response is None:
            response = call_ollama(prompt)
            if response is None:
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(2)  # Wait before retry
        
        if response is None:
            failed_count += 1
            continue
        
        # Parse response
        parsed = parse_tool_response(response)
        
        if parsed is None:
            failed_count += 1
            continue
        
        # Create training sample
        sample = {
            "input": f"Product: {product['description']}\nQuery: {query}",
            "output": parsed,
            "product_metadata": product,
            "query": query
        }
        
        dataset.append(sample)
        
        # Small delay to avoid rate limiting
        if (i + 1) % BATCH_SIZE == 0:
            time.sleep(1)
    
    print(f"\n{'='*60}")
    print(f"✓ Generation complete!")
    print(f"  Valid samples: {len(dataset)}/{num_samples}")
    print(f"  Failed samples: {failed_count}")
    print(f"  Success rate: {(len(dataset)/num_samples)*100:.1f}%")
    print(f"{'='*60}\n")
    
    return dataset


def save_dataset(dataset: List[Dict[str, Any]], output_file: str = OUTPUT_FILE):
    """Save dataset to JSON file"""
    output_path = os.path.join(OUTPUT_DIR, output_file)
    
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ Dataset saved: {output_path}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Total samples: {len(dataset)}")


def load_dataset(output_file: str = OUTPUT_FILE) -> List[Dict[str, Any]]:
    """Load dataset from JSON file"""
    output_path = os.path.join(OUTPUT_DIR, output_file)
    
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Dataset not found: {output_path}")
    
    with open(output_path, 'r') as f:
        return json.load(f)


def analyze_dataset(dataset: List[Dict[str, Any]]):
    """Analyze generated dataset"""
    print(f"\n{'='*60}")
    print(f"Dataset Analysis")
    print(f"{'='*60}\n")
    
    # Tool distribution
    tool_counts = {}
    for sample in dataset:
        tool = sample["output"].get("tool", "unknown")
        tool_counts[tool] = tool_counts.get(tool, 0) + 1
    
    print("Tool Distribution:")
    for tool, count in sorted(tool_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(dataset)) * 100
        print(f"  {tool}: {count} ({percentage:.1f}%)")
    
    # Category distribution
    category_counts = {}
    for sample in dataset:
        category = sample["product_metadata"].get("category", "unknown")
        category_counts[category] = category_counts.get(category, 0) + 1
    
    print(f"\nTop 5 Product Categories:")
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        percentage = (count / len(dataset)) * 100
        print(f"  {category}: {count} ({percentage:.1f}%)")
    
    # Sample examples
    print(f"\nSample Examples:")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"\n  Example {i+1}:")
        print(f"    Input: {sample['input'][:80]}...")
        print(f"    Tool: {sample['output']['tool']}")
        print(f"    Filters: {sample['output'].get('filters', {})}")
    
    print(f"\n{'='*60}\n")


def main():
    """Main execution function"""
    print(f"\n{'='*60}")
    print(f"🛒 RetailMind Phase 1: Data Generation")
    print(f"{'='*60}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output file: {OUTPUT_DIR}/{OUTPUT_FILE}")
    print(f"Target samples: {NUM_SAMPLES}\n")
    
   
    ensure_output_dir()
    
   
    if not check_ollama_connection():
        print("\n❌ Cannot proceed without Ollama. Please start Ollama:")
        print("   ollama serve")
        print("   Then run this script again.")
        return
    
   
    dataset = generate_dataset(NUM_SAMPLES)
    
    if len(dataset) == 0:
        print("❌ No samples generated. Please check Ollama connection.")
        return
    
  
    save_dataset(dataset)
    
    
    analyze_dataset(dataset)
    
    print(f"✅ Phase 1 Complete!")
    print(f"Next step: Run Phase 2 to fine-tune Phi-3-mini on this data")


if __name__ == "__main__":
    main()