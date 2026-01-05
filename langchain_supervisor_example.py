"""
Sample 1: LangChain Supervisor Pattern (Multi-Agent)
=====================================================

This demonstrates LangChain's supervisor pattern where a central supervisor
coordinates specialized worker agents. This shows the MANUAL WRAPPER approach.

Key Feature: You must manually write wrapper functions to expose agents as tools.

Based on: https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant

Run: python samples/07_langchain_supervisor_example.py

Requirements:
    pip install langchain langchain-openai
    export OPENAI_API_KEY='your-key' (or use another provider)
"""

import os
import asyncio
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check for LangChain
try:
    from langchain.agents import create_agent
    from langchain.tools import tool
    from langchain.chat_models import init_chat_model
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️  LangChain not installed. Install with: pip install langchain langchain-openai")

# ------------------------------------------------------------------
# Step 1: Define Function Tools
# ------------------------------------------------------------------

@tool
def get_weather_info(city: str) -> str:
    """Get weather information for a city"""
    # Stub implementation - in production would call weather API
    return f"Weather in {city}: 72°F, sunny with light breeze. Perfect for outdoor activities!"

@tool
def create_workout_plan(fitness_level: str, goal: str) -> str:
    """Create a personalized workout plan"""
    plans = {
        "beginner": "Start with 3 days/week: 20 min cardio + 10 min strength training",
        "intermediate": "4-5 days/week: Mix of cardio, strength, and flexibility exercises",
        "advanced": "5-6 days/week: Periodized training with focused recovery days"
    }
    level = fitness_level.lower()
    plan = plans.get(level, plans["beginner"])
    return f"Workout plan for {level} level ({goal}): {plan}"

@tool
def suggest_meal(meal_type: str, dietary_preferences: str = "none") -> str:
    """Suggest a healthy meal based on type and preferences"""
    suggestions = {
        "breakfast": "Greek yogurt with berries and granola",
        "lunch": "Grilled chicken salad with mixed vegetables",
        "dinner": "Salmon with quinoa and steamed broccoli"
    }
    meal = suggestions.get(meal_type.lower(), suggestions["lunch"])
    return f"Suggested {meal_type}: {meal} (Dietary preferences: {dietary_preferences})"

# ------------------------------------------------------------------
# Step 2: Create Specialized Sub-Agents
# ------------------------------------------------------------------

if LANGCHAIN_AVAILABLE:
    # Initialize model (using OpenAI by default, but can use others)
    # You can also use: init_chat_model("anthropic:claude-3-5-sonnet-20241022")
    # or: init_chat_model("google_genai:gemini-2.5-flash-lite")
    
    # Try to get API key (prefer OpenAI, fallback to others)
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    if api_key:
        if os.getenv("OPENAI_API_KEY"):
            model = init_chat_model("openai:gpt-4o-mini")
        elif os.getenv("ANTHROPIC_API_KEY"):
            model = init_chat_model("anthropic:claude-3-5-haiku-20241022")
        elif os.getenv("GOOGLE_API_KEY"):
            model = init_chat_model("google_genai:gemini-2.5-flash-lite")
        else:
            model = None
    else:
        model = None
    
    if model:
        # Weather Agent
        WEATHER_PROMPT = (
            "You are a weather assistant. "
            "Help users with weather-related questions and recommendations. "
            "Use the get_weather_info tool to get current weather data. "
            "Be friendly and helpful."
        )
        
        weather_agent = create_agent(
            model,
            tools=[get_weather_info],
            system_prompt=WEATHER_PROMPT
        )
        
        # Fitness Agent
        FITNESS_PROMPT = (
            "You are a fitness coach. "
            "Help users create workout plans and provide fitness advice. "
            "Use the create_workout_plan tool when users ask for workout plans. "
            "Be encouraging and provide actionable advice."
        )
        
        fitness_agent = create_agent(
            model,
            tools=[create_workout_plan],
            system_prompt=FITNESS_PROMPT
        )
        
        # Nutrition Agent
        NUTRITION_PROMPT = (
            "You are a nutrition advisor. "
            "Help users with meal planning and nutritional advice. "
            "Use the suggest_meal tool when users ask for meal suggestions. "
            "Provide evidence-based advice."
        )
        
        nutrition_agent = create_agent(
            model,
            tools=[suggest_meal],
            system_prompt=NUTRITION_PROMPT
        )
        
        # ------------------------------------------------------------------
        # Step 3: ⚠️ MANUAL WRAPPING - Write wrapper functions for each agent
        # ------------------------------------------------------------------
        # This is the key difference: LangChain REQUIRES you to manually
        # write wrapper functions to expose agents as tools.
        
        @tool
        def get_weather_help(request: str) -> str:
            """Get weather information and recommendations. Use this for weather-related questions."""
            # Manual invocation of the agent
            result = weather_agent.invoke({
                "messages": [{"role": "user", "content": request}],
            })
            # Manual response extraction
            return result["messages"][-1].content
        
        @tool
        def get_fitness_advice(request: str) -> str:
            """Get fitness advice and workout plans. Use this for fitness and exercise questions."""
            result = fitness_agent.invoke({
                "messages": [{"role": "user", "content": request}],
            })
            return result["messages"][-1].content
        
        @tool
        def get_nutrition_help(request: str) -> str:
            """Get nutrition advice and meal suggestions. Use this for diet and meal planning questions."""
            result = nutrition_agent.invoke({
                "messages": [{"role": "user", "content": request}],
            })
            return result["messages"][-1].content
        
        # ------------------------------------------------------------------
        # Step 4: Create Supervisor Agent
        # ------------------------------------------------------------------
        
        SUPERVISOR_PROMPT = (
            "You are a personal assistant supervisor that routes user requests "
            "to specialized agents.\n\n"
            "You have access to three specialized agents:\n"
            "1. get_weather_help - For weather-related questions\n"
            "2. get_fitness_advice - For fitness, workouts, and exercise questions\n"
            "3. get_nutrition_help - For nutrition, diet, and meal planning questions\n\n"
            "Your job:\n"
            "- Analyze the user's request\n"
            "- Determine which specialized agent should handle it\n"
            "- Call the appropriate agent tool\n\n"
            "If the request doesn't fit any category, respond directly as a general assistant."
        )
        
        supervisor_agent = create_agent(
            model,
            tools=[get_weather_help, get_fitness_advice, get_nutrition_help],
            system_prompt=SUPERVISOR_PROMPT
        )

# ------------------------------------------------------------------
# Demo Function
# ------------------------------------------------------------------

async def demo_langchain_supervisor():
    """Run a demo of the LangChain supervisor pattern"""
    print("=" * 70)
    print("LANGCHAIN SUPERVISOR PATTERN - Multi-Agent Demo")
    print("=" * 70)
    
    if not LANGCHAIN_AVAILABLE:
        print("\n LangChain is not installed.")
        print("Install with: pip install langchain langchain-openai")
        print("Or: pip install langchain langchain-anthropic")
        print("Or: pip install langchain langchain-google-genai")
        return
    
    if not model:
        print("\n No API key found.")
        print("Please set one of:")
        print("  export OPENAI_API_KEY='your-key'")
        print("  export ANTHROPIC_API_KEY='your-key'")
        print("  export GOOGLE_API_KEY='your-key'")
        return
    
    print("\nThis demonstrates the LANGCHAIN approach:")
    print("• Supervisor routes requests to specialized agents")
    print("• ⚠️  Manual wrapper functions required for each agent")
    print("• Each wrapper manually invokes the agent and extracts response")
    print("\nNotice the boilerplate code in Step 3 (wrapper functions).")
    print("-" * 70)
    
    # Test cases
    test_cases = [
        "What's the weather like in San Francisco?",
        "I want a workout plan for beginners",
        "What should I eat for breakfast?",
    ]
    
    for i, user_message in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"TEST CASE {i}/{len(test_cases)}")
        print(f"{'='*70}")
        print(f"\n👤 User: {user_message}\n")
        print(" Supervisor → Routing to specialized agent...\n")
        
        try:
            result = supervisor_agent.invoke({
                "messages": [{"role": "user", "content": user_message}],
            })
            
            response = result["messages"][-1].content
            print(f" Response:\n{response}\n")
            
        except Exception as e:
            print(f" Error: {e}")
        
        print("-" * 70)
    
    print("\n" + "=" * 70)
    print("KEY POINTS: LangChain Supervisor Pattern")
    print("=" * 70)
    print("\n Works: Supervisor routes to specialized agents")
    print(" Manual Work: Must write 3 wrapper functions:")
    print("   - get_weather_help() - manually invokes weather_agent")
    print("   - get_fitness_advice() - manually invokes fitness_agent")
    print("   - get_nutrition_help() - manually invokes nutrition_agent")
    print("\n Code Lines:")
    print("   - Function tools: ~15 lines")
    print("   - Sub-agents: ~20 lines")
    print("   - Wrapper functions: ~20 lines (manual boilerplate)")
    print("   - Supervisor: ~5 lines")
    print("   - Total: ~60 lines")
    print("\n Compare this with Google ADK's AgentTool approach!")

if __name__ == "__main__":
    if LANGCHAIN_AVAILABLE:
        # LangChain uses synchronous invoke by default
        import asyncio
        try:
            asyncio.run(demo_langchain_supervisor())
        except RuntimeError:
            # If already in event loop, run directly
            demo_langchain_supervisor()
    else:
        print("LangChain not installed.")
        print("\nTo run this example:")
        print("  pip install langchain langchain-openai")
        print("  export OPENAI_API_KEY='your-key'")
        print("  python samples/07_langchain_supervisor_example.py")

