"""
Phase 3: RetailMind Agent - Main Orchestration
Purpose: Orchestrate all MCP servers, fine-tuned Phi-3, and handle user queries
This is the brain of the system that decides which tools to use
"""

import json
import asyncio
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# IMPORTS
# ============================================================================

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
except ImportError:
    logger.error("Missing dependencies. Install: pip install transformers peft torch")
    raise

# Configuration
BASE_MODEL = "microsoft/phi-3-mini-4k-instruct"
ADAPTER_PATH = "../../Phase_2_ModelTraining/models/phi3_retail_adapter/final"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global state
tokenizer = None
model = None
conversation_history = []

# ============================================================================
# MODEL SETUP
# ============================================================================

def load_agent_model():
    """Load fine-tuned Phi-3 for agent decision-making"""
    global tokenizer, model
    
    logger.info(f"Loading agent model: {BASE_MODEL}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True
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
    
    # Load adapter
    if Path(ADAPTER_PATH).exists():
        logger.info(f"Loading adapter from {ADAPTER_PATH}...")
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
        logger.info("✓ Adapter loaded")
    
    model.eval()
    logger.info(f"✓ Agent model ready on {DEVICE.upper()}")
    
    return tokenizer, model


# ============================================================================
# AGENT LOGIC
# ============================================================================

def decide_tool_call(user_query: str, product_description: str) -> Dict[str, Any]:
    """
    Use fine-tuned Phi-3 to decide which tool to call
    
    User query: "Find me something similar but cheaper"
    Product: "Red Nike running shoe"
    
    Output: {
        "tool": "retrieve_similar",
        "filters": {"color": "red", "price": "lower", "category": "shoe"}
    }
    """
    
    if model is None:
        raise RuntimeError("Model not loaded")
    
    # Create prompt for tool decision
    prompt = f"""Based on the product and user query, decide which tool to use.

Product: {product_description}
User Query: {user_query}

Available tools:
- retrieve_similar: Find products with similar characteristics
- compare_prices: Compare prices (if user asks about cost)
- answer_query: Answer questions about the product
- check_availability: Check if product is available

Respond with ONLY JSON (no markdown):
{{"tool": "tool_name", "filters": {{...}}, "reasoning": "why this tool"}}"""
    
    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=200,
            temperature=0.5,
            do_sample=False
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Parse JSON
    try:
        start = response.find('{')
        end = response.rfind('}') + 1
        json_str = response[start:end]
        decision = json.loads(json_str)
    except:
        decision = {
            "tool": "answer_query",
            "filters": {},
            "reasoning": "Could not parse response"
        }
    
    return decision


def generate_response(
    product_description: str,
    user_query: str,
    tool_results: Optional[Dict[str, Any]] = None
) -> str:
    """Generate natural language response to user"""
    
    if model is None:
        raise RuntimeError("Model not loaded")
    
    # Build context
    context = f"""Product: {product_description}
User Ask: {user_query}"""
    
    if tool_results:
        context += f"\nTool Results: {json.dumps(tool_results, indent=2)}"
    
    prompt = f"""{context}

Generate a helpful, concise response to the user's question about this product.
Be friendly and provide relevant information.

Response:"""
    
    # Generate
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=200,
            temperature=0.7,
            top_p=0.9
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the response part
    if "Response:" in response:
        response = response.split("Response:")[-1].strip()
    
    return response


# ============================================================================
# AGENT INTERFACE
# ============================================================================

class RetailMindAgent:
    """Main agent for handling user queries"""
    
    def __init__(self):
        """Initialize agent"""
        self.conversation_history = []
        self.available_tools = {
            "retrieve_similar": self.tool_retrieve_similar,
            "compare_prices": self.tool_compare_prices,
            "answer_query": self.tool_answer_query,
            "check_availability": self.tool_check_availability
        }
    
    def add_to_history(self, role: str, content: str):
        """Add to conversation history"""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": str(__import__('datetime').datetime.now())
        })
    
    def process_query(
        self,
        user_query: str,
        product_description: str
    ) -> Dict[str, Any]:
        """
        Main agent logic:
        1. Understand product
        2. Decide which tool to use
        3. Execute tool
        4. Generate response
        """
        
        logger.info(f"Processing query: {user_query}")
        
        # Step 1: Decide tool
        tool_decision = decide_tool_call(user_query, product_description)
        logger.info(f"Tool decision: {tool_decision['tool']}")
        
        # Step 2: Execute tool
        tool_name = tool_decision.get("tool", "answer_query")
        tool_func = self.available_tools.get(tool_name, self.tool_answer_query)
        
        try:
            tool_results = tool_func(user_query, product_description, tool_decision)
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            tool_results = {"error": str(e)}
        
        # Step 3: Generate response
        response = generate_response(product_description, user_query, tool_results)
        
        # Add to history
        self.add_to_history("user", user_query)
        self.add_to_history("agent", response)
        
        # Return result
        return {
            "user_query": user_query,
            "product": product_description,
            "tool_used": tool_name,
            "tool_results": tool_results,
            "agent_response": response,
            "decision_trace": tool_decision
        }
    
    # ========================================================================
    # TOOLS
    # ========================================================================
    
    def tool_retrieve_similar(
        self,
        query: str,
        product: str,
        decision: Dict
    ) -> Dict:
        """Tool: Find similar products"""
        
        logger.info("Executing: retrieve_similar")
        
        # Simulated CLIP retrieval
        # In real implementation, would call CLIP server
        return {
            "similar_products": [
                {
                    "description": "White mesh running shoe, sporty style",
                    "similarity": 0.92
                },
                {
                    "description": "Black athletic shoe, casual style",
                    "similarity": 0.85
                }
            ],
            "filters_applied": decision.get("filters", {})
        }
    
    def tool_compare_prices(
        self,
        query: str,
        product: str,
        decision: Dict
    ) -> Dict:
        """Tool: Compare prices"""
        
        logger.info("Executing: compare_prices")
        
        # Simulated price comparison
        return {
            "current_product": {
                "price": 120,
                "currency": "USD"
            },
            "alternatives": [
                {
                    "description": "Similar white shoe",
                    "price": 85
                },
                {
                    "description": "Premium version",
                    "price": 180
                }
            ]
        }
    
    def tool_answer_query(
        self,
        query: str,
        product: str,
        decision: Dict
    ) -> Dict:
        """Tool: Answer questions about product"""
        
        logger.info("Executing: answer_query")
        
        # Direct answer without external tools
        return {
            "answer_source": "product_description",
            "relevant_info": {
                "category": "shoe",
                "style": "sporty"
            }
        }
    
    def tool_check_availability(
        self,
        query: str,
        product: str,
        decision: Dict
    ) -> Dict:
        """Tool: Check product availability"""
        
        logger.info("Executing: check_availability")
        
        # Simulated availability check
        return {
            "available": True,
            "stock": 15,
            "locations": ["New York", "San Francisco"]
        }
    
    def get_conversation_history(self) -> List[Dict]:
        """Return conversation history"""
        return self.conversation_history


# ============================================================================
# TESTING & DEMO
# ============================================================================

def demo():
    """Demo the agent"""
    
    logger.info("="*60)
    logger.info("🛒 RetailMind Agent Demo")
    logger.info("="*60)
    
    # Load model
    load_agent_model()
    
    # Create agent
    agent = RetailMindAgent()
    
    # Test queries
    test_cases = [
        {
            "product": "Red mesh running shoe, sporty style",
            "query": "Find me something similar but cheaper"
        },
        {
            "product": "Black leather jacket, formal style",
            "query": "Is there a better quality option?"
        },
        {
            "product": "Blue denim backpack, casual style",
            "query": "How much does this cost?"
        }
    ]
    
    # Process each query
    for i, test in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"Test {i+1}")
        print(f"{'='*60}")
        print(f"Product: {test['product']}")
        print(f"Query: {test['query']}")
        
        result = agent.process_query(test['query'], test['product'])
        
        print(f"\nTool used: {result['tool_used']}")
        print(f"Response: {result['agent_response']}")
        print(f"Decision trace: {json.dumps(result['decision_trace'], indent=2)}")
    
    print(f"\n{'='*60}")
    print("✅ Demo complete!")
    print(f"Total exchanges: {len(agent.get_conversation_history())}")
    print(f"{'='*60}\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    demo()