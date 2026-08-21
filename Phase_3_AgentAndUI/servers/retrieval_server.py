"""
Phase 3: MCP Server - Product Retrieval using CLIP
Purpose: Search similar products using CLIP embeddings and FAISS vector database
Type: MCP (Model Context Protocol) Server - works with any MCP client
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any
import logging
from mcp.server import Server
from mcp.types import Tool, TextContent, ToolResult
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# IMPORTS & SETUP
# ============================================================================

try:
    import open_clip
    import faiss
    from PIL import Image
except ImportError:
    logger.error("Missing dependencies. Install: pip install open-clip-torch faiss-cpu pillow")
    raise

# Configuration
MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"
CATALOG_PATH = "../../Phase_1_DataGeneration/data/training_data.json"
EMBEDDINGS_PATH = "data/product_embeddings.pkl"
FAISS_INDEX_PATH = "data/faiss_index.bin"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# GLOBAL STATE - Product Catalog & Embeddings
# ============================================================================

product_catalog = []
clip_model = None
clip_processor = None
faiss_index = None
embeddings_array = None

# ============================================================================
# CLIP MODEL SETUP
# ============================================================================

def load_clip_model():
    """Load CLIP model for image/text understanding"""
    global clip_model, clip_processor
    
    logger.info(f"Loading CLIP model: {MODEL_NAME}...")
    
    # Load model and processor
    clip_model, _, clip_processor = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED,
        device=DEVICE
    )
    
    clip_model.eval()
    logger.info(f"✓ CLIP model loaded on {DEVICE.upper()}")
    
    return clip_model, clip_processor


def get_text_embedding(text: str) -> np.ndarray:
    """Convert text to CLIP embedding"""
    if clip_model is None:
        raise RuntimeError("CLIP model not loaded")
    
    # Tokenize and embed
    text_tokens = open_clip.tokenize([text])
    
    with torch.no_grad():
        text_embedding = clip_model.encode_text(text_tokens.to(DEVICE))
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
    
    return text_embedding.cpu().numpy()[0].astype(np.float32)


# ============================================================================
# PRODUCT CATALOG & EMBEDDINGS
# ============================================================================

def load_product_catalog():
    """Load products from Phase 1 training data"""
    global product_catalog
    
    logger.info(f"Loading product catalog from {CATALOG_PATH}...")
    
    if not Path(CATALOG_PATH).exists():
        raise FileNotFoundError(f"Catalog not found: {CATALOG_PATH}")
    
    with open(CATALOG_PATH, 'r') as f:
        training_data = json.load(f)
    
    # Extract unique products
    seen_products = set()
    for sample in training_data:
        product_desc = sample['product_metadata']['description']
        
        if product_desc not in seen_products:
            product_catalog.append({
                "id": len(product_catalog),
                "description": product_desc,
                "category": sample['product_metadata'].get('category', 'unknown'),
                "color": sample['product_metadata'].get('color', 'unknown'),
                "material": sample['product_metadata'].get('material', 'unknown'),
                "style": sample['product_metadata'].get('style', 'unknown')
            })
            seen_products.add(product_desc)
    
    logger.info(f"✓ Loaded {len(product_catalog)} unique products")
    return product_catalog


def build_embeddings():
    """Create embeddings for all products"""
    global embeddings_array, faiss_index
    
    logger.info("Building CLIP embeddings for all products...")
    
    embeddings = []
    for i, product in enumerate(product_catalog):
        if i % 100 == 0:
            logger.info(f"  Processing {i}/{len(product_catalog)}...")
        
        # Create text description for embedding
        text = f"{product['description']}"
        embedding = get_text_embedding(text)
        embeddings.append(embedding)
    
    embeddings_array = np.array(embeddings).astype(np.float32)
    logger.info(f"✓ Created embeddings: shape {embeddings_array.shape}")
    
    # Build FAISS index
    logger.info("Building FAISS index...")
    dimension = embeddings_array.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings_array)
    logger.info(f"✓ FAISS index created: {faiss_index.ntotal} vectors")
    
    return embeddings_array, faiss_index


def search_similar_products(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Search for similar products using CLIP"""
    
    if faiss_index is None:
        raise RuntimeError("FAISS index not initialized")
    
    # Get query embedding
    query_embedding = get_text_embedding(query)
    query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
    
    # Search
    distances, indices = faiss_index.search(query_embedding, top_k)
    
    # Format results
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        product = product_catalog[int(idx)]
        results.append({
            "id": product["id"],
            "description": product["description"],
            "category": product["category"],
            "color": product["color"],
            "material": product["material"],
            "style": product["style"],
            "similarity_score": float(1 / (1 + distance))  # Convert distance to similarity
        })
    
    return results


# ============================================================================
# MCP SERVER SETUP
# ============================================================================

# Create MCP server
server = Server("product-retrieval-server")


# Define MCP tools
TOOLS = [
    Tool(
        name="search_products",
        description="Search for similar products using natural language query. Uses CLIP embeddings to find products matching your description.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Product description or search query (e.g., 'cheap red shoes', 'blue leather jacket')"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5)",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    ),
    Tool(
        name="get_product_details",
        description="Get detailed information about a specific product by ID",
        inputSchema={
            "type": "object",
            "properties": {
                "product_id": {
                    "type": "integer",
                    "description": "Product ID to retrieve details for"
                }
            },
            "required": ["product_id"]
        }
    ),
    Tool(
        name="list_all_products",
        description="List all available products in the catalog",
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Max products to return (default: 10)",
                    "default": 10
                }
            }
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
        if name == "search_products":
            query = arguments.get("query")
            top_k = arguments.get("top_k", 5)
            
            results = search_similar_products(query, top_k)
            
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
        
        elif name == "get_product_details":
            product_id = arguments.get("product_id")
            
            if product_id < 0 or product_id >= len(product_catalog):
                return [
                    ToolResult(
                        content=[TextContent(type="text", text=f"Product {product_id} not found")],
                        is_error=True
                    )
                ]
            
            product = product_catalog[product_id]
            return [
                ToolResult(
                    content=[TextContent(type="text", text=json.dumps(product, indent=2))],
                    is_error=False
                )
            ]
        
        elif name == "list_all_products":
            limit = arguments.get("limit", 10)
            products = product_catalog[:limit]
            
            return [
                ToolResult(
                    content=[TextContent(type="text", text=json.dumps(products, indent=2))],
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
    logger.info("Initializing Product Retrieval Server...")
    
    # Load CLIP model
    load_clip_model()
    
    # Load product catalog
    load_product_catalog()
    
    # Build embeddings and FAISS index
    build_embeddings()
    
    logger.info("✓ Server initialized successfully!")


async def main():
    """Main server loop"""
    logger.info("🚀 Starting Product Retrieval MCP Server...")
    
    # Initialize
    await initialize()
    
    # Start server
    async with server:
        logger.info("✓ Server running. Waiting for connections...")
        # Server runs indefinitely until interrupted
        while True:
            await asyncio.sleep(1)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())