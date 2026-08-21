"""
Phase 3: MCP Server - Attribute Extraction using Fine-tuned Phi-3-mini
Purpose: Extract structured product attributes (color, material, category, etc.) from product descriptions
Uses: Your trained Phi-3-mini adapter from Phase 2
"""

import json
import torch
from pathlib import Path
from typing import Dict, List, Any
import logging
from mcp.server import Server
from mcp.types import Tool, TextContent, ToolResult

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Imports
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
except ImportError:
    logger.error("Missing dependencies. Install: pip install transformers peft")
    raise

# Configuration
BASE_MODEL = "microsoft/phi-3-mini-4k-instruct"
ADAPTER_PATH = "../../Phase_2_ModelTraining/models/phi3_retail_adapter/final"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 150

# Global state
tokenizer = None
model = None

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_fine_tuned_model():
    """Load base model + your trained LoRA adapter"""
    global tokenizer, model
    
    logger.info(f"Loading base model: {BASE_MODEL}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        padding_side="right"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True
    )
    
    logger.info("✓ Base model loaded")
    
    # Load your trained adapter
    if Path(ADAPTER_PATH).exists():
        logger.info(f"Loading adapter from {ADAPTER_PATH}...")
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
        logger.info("✓ Adapter loaded successfully!")
    else:
        logger.warning(f"⚠️  Adapter not found at {ADAPTER_PATH}")
        logger.warning("   Using base model without fine-tuning")
    
    model.eval()
    return tokenizer, model


# ============================================================================
# ATTRIBUTE EXTRACTION
# ============================================================================

def extract_product_attributes(product_description: str) -> Dict[str, Any]:
    """
    Extract structured attributes from a product description
    
    Input: "Red Nike running shoe made of mesh, sporty style"
    Output: {
        "category": "running shoe",
        "color": "red",
        "material": "mesh",
        "style": "sporty",
        "brand": "Nike"
    }
    """
    
    if model is None or tokenizer is None:
        raise RuntimeError("Model not loaded")
    
    # Create prompt for attribute extraction
    prompt = f"""Extract product attributes from this description and return as JSON:

Description: {product_description}

Extract these attributes (if not mentioned, return "unknown"):
- category: product type
- color: main color
- material: material type
- style: style/aesthetic
- brand: brand name
- condition: new/used/etc

Return ONLY valid JSON, no markdown:"""
    
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    ).to(DEVICE)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=MAX_LENGTH,
            temperature=0.5,
            top_p=0.9,
            do_sample=False
        )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract JSON from response
    try:
        # Find JSON in response
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            json_str = response[start_idx:end_idx]
            attributes = json.loads(json_str)
        else:
            # Fallback: basic attribute parsing
            attributes = {
                "category": "unknown",
                "color": "unknown",
                "material": "unknown",
                "style": "unknown",
                "brand": "unknown",
                "condition": "new"
            }
    
    except json.JSONDecodeError:
        attributes = {
            "category": "unknown",
            "color": "unknown",
            "material": "unknown",
            "style": "unknown",
            "brand": "unknown",
            "condition": "new"
        }
    
    return attributes


def extract_multiple_attributes(descriptions: List[str]) -> List[Dict[str, Any]]:
    """Extract attributes for multiple products"""
    results = []
    for desc in descriptions:
        try:
            attrs = extract_product_attributes(desc)
            results.append({
                "description": desc,
                "attributes": attrs
            })
        except Exception as e:
            results.append({
                "description": desc,
                "error": str(e)
            })
    
    return results


# ============================================================================
# MCP SERVER SETUP
# ============================================================================

server = Server("attribute-extraction-server")


# Define MCP tools
TOOLS = [
    Tool(
        name="extract_attributes",
        description="Extract structured attributes (color, material, category, style, brand) from a product description using your fine-tuned Phi-3 model",
        inputSchema={
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Product description (e.g., 'Red Nike running shoe made of mesh')"
                }
            },
            "required": ["description"]
        }
    ),
    Tool(
        name="batch_extract_attributes",
        description="Extract attributes from multiple product descriptions at once",
        inputSchema={
            "type": "object",
            "properties": {
                "descriptions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of product descriptions"
                }
            },
            "required": ["descriptions"]
        }
    ),
    Tool(
        name="analyze_attributes",
        description="Analyze which attributes are most common across products",
        inputSchema={
            "type": "object",
            "properties": {
                "descriptions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of product descriptions"
                }
            },
            "required": ["descriptions"]
        }
    )
]


@server.list_tools()
async def list_tools():
    """Return available MCP tools"""
    return TOOLS


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> List[ToolResult]:
    """Handle MCP tool calls"""
    
    try:
        if name == "extract_attributes":
            description = arguments.get("description")
            
            attributes = extract_product_attributes(description)
            
            return [
                ToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=json.dumps(attributes, indent=2)
                        )
                    ],
                    is_error=False
                )
            ]
        
        elif name == "batch_extract_attributes":
            descriptions = arguments.get("descriptions", [])
            
            results = extract_multiple_attributes(descriptions)
            
            return [
                ToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=json.dumps(results, indent=2)
                        )
                    ],
                    is_error=False
                )
            ]
        
        elif name == "analyze_attributes":
            descriptions = arguments.get("descriptions", [])
            
            results = extract_multiple_attributes(descriptions)
            
            # Analyze
            color_counts = {}
            material_counts = {}
            category_counts = {}
            
            for result in results:
                if "attributes" in result:
                    attrs = result["attributes"]
                    color = attrs.get("color", "unknown")
                    material = attrs.get("material", "unknown")
                    category = attrs.get("category", "unknown")
                    
                    color_counts[color] = color_counts.get(color, 0) + 1
                    material_counts[material] = material_counts.get(material, 0) + 1
                    category_counts[category] = category_counts.get(category, 0) + 1
            
            analysis = {
                "total_products": len(descriptions),
                "top_colors": sorted(color_counts.items(), key=lambda x: x[1], reverse=True)[:5],
                "top_materials": sorted(material_counts.items(), key=lambda x: x[1], reverse=True)[:5],
                "top_categories": sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            }
            
            return [
                ToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=json.dumps(analysis, indent=2)
                        )
                    ],
                    is_error=False
                )
            ]
        
        else:
            return [
                ToolResult(
                    content=[TextContent(type="text", text=f"Unknown tool: {name}")],
                    is_error=True
                )
            ]
    
    except Exception as e:
        logger.error(f"Error in {name}: {str(e)}")
        return [
            ToolResult(
                content=[TextContent(type="text", text=f"Error: {str(e)}")],
                is_error=True
            )
        ]


# ============================================================================
# INITIALIZATION & STARTUP
# ============================================================================

async def initialize():
    """Initialize server components"""
    logger.info("Initializing Attribute Extraction Server...")
    
    # Load model
    load_fine_tuned_model()
    
    logger.info("✓ Server initialized successfully!")


async def main():
    """Main server loop"""
    logger.info("🚀 Starting Attribute Extraction MCP Server...")
    
    # Initialize
    await initialize()
    
    # Start server
    async with server:
        logger.info("✓ Server running. Waiting for connections...")
        # Server runs indefinitely
        while True:
            await asyncio.sleep(1)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())