"""
Sample 2: Google ADK AgentTool Pattern (Comparison with LangChain)
===================================================================

This demonstrates Google ADK's AgentTool pattern for the SAME use case as
Sample 1 (LangChain). This shows the AUTOMATIC WRAPPING approach.

Key Feature: AgentTool automatically wraps agents as tools - NO manual wrappers needed!

This is a direct comparison to samples/07_langchain_supervisor_example.py

Run: python adk_agenttool_comparison.py

Requirements:
    export GOOGLE_API_KEY='your-key'
"""

import asyncio
from google.adk import Agent, Runner
from google.adk.tools import AgentTool, FunctionTool
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.genai import types
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in .env file or as an environment variable.")

# Setup services
memory_service = InMemoryMemoryService()
session_service = InMemorySessionService()
artifact_service = InMemoryArtifactService()

# ------------------------------------------------------------------
# Step 1: Define Function Tools (Same as LangChain example)
# ------------------------------------------------------------------

def get_weather_info(city: str) -> str:
    """Get weather information for a city"""
    # Stub implementation - in production would call weather API
    return f"Weather in {city}: 72°F, sunny with light breeze. Perfect for outdoor activities!"

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

def suggest_meal(meal_type: str, dietary_preferences: str = "none") -> str:
    """Suggest a healthy meal based on type and preferences"""
    suggestions = {
        "breakfast": "Greek yogurt with berries and granola",
        "lunch": "Grilled chicken salad with mixed vegetables",
        "dinner": "Salmon with quinoa and steamed broccoli"
    }
    meal = suggestions.get(meal_type.lower(), suggestions["lunch"])
    return f"Suggested {meal_type}: {meal} (Dietary preferences: {dietary_preferences})"

# Wrap functions as tools
weather_tool = FunctionTool(get_weather_info)
workout_tool = FunctionTool(create_workout_plan)
meal_tool = FunctionTool(suggest_meal)

# ------------------------------------------------------------------
# Step 2: Create Specialized Sub-Agents (Same as LangChain example)
# ------------------------------------------------------------------

weather_agent = Agent(
    name="weather_agent",
    model="gemini-2.5-flash-lite",
    instruction="""
    You are a weather assistant. 
    Help users with weather-related questions and recommendations.
    Use the get_weather_info tool to get current weather data.
    Be friendly and helpful.
    """,
    tools=[weather_tool],
)

fitness_agent = Agent(
    name="fitness_agent",
    model="gemini-2.5-flash-lite",
    instruction="""
    You are a fitness coach.
    Help users create workout plans and provide fitness advice.
    Use the create_workout_plan tool when users ask for workout plans.
    Be encouraging and provide actionable advice.
    """,
    tools=[workout_tool],
)

nutrition_agent = Agent(
    name="nutrition_agent",
    model="gemini-2.5-flash-lite",
    instruction="""
    You are a nutrition advisor.
    Help users with meal planning and nutritional advice.
    Use the suggest_meal tool when users ask for meal suggestions.
    Provide evidence-based advice.
    """,
    tools=[meal_tool],
)

# ------------------------------------------------------------------
# Step 3: AUTOMATIC WRAPPING - AgentTool handles it!
# ------------------------------------------------------------------
# This is the key difference: Google ADK's AgentTool automatically
# wraps agents as tools - NO manual wrapper functions needed!

# No wrapper functions required! Just use AgentTool:
# - AgentTool(weather_agent) automatically creates the tool wrapper
# - Handles invocation internally
# - Extracts responses automatically
# - Manages sessions automatically

# ------------------------------------------------------------------
# Step 4: Create Router/Supervisor Agent
# ------------------------------------------------------------------

router_agent = Agent(
    name="router_agent",
    model="gemini-2.5-flash-lite",
    instruction="""
    You are a personal assistant router that routes user requests to specialized agents.
    
    You have access to three specialized agents:
    1. weather_agent - For weather-related questions
    2. fitness_agent - For fitness, workouts, and exercise questions
    3. nutrition_agent - For nutrition, diet, and meal planning questions
    
    Your job:
    - Analyze the user's request
    - Determine which specialized agent should handle it
    - Call the appropriate agent using AgentTool
    
    If the request doesn't fit any category, respond directly as a general assistant.
    """,
    tools=[
        AgentTool(weather_agent),    # ← Automatic wrapping! No manual function!
        AgentTool(fitness_agent),    # ← Just wrap the agent!
        AgentTool(nutrition_agent),  # ← That's it!
    ],
)

# ------------------------------------------------------------------
# Runner Setup
# ------------------------------------------------------------------

runner = Runner(
    agent=router_agent,
    app_name="ADK AgentTool Comparison Demo",
    memory_service=memory_service,
    session_service=session_service,
    artifact_service=artifact_service,
)

# ------------------------------------------------------------------
# Demo Function
# ------------------------------------------------------------------

async def demo_adk_agenttool():
    """Run a demo of the Google ADK AgentTool pattern"""
    print("=" * 70)
    print("GOOGLE ADK AGENTTOOL PATTERN - Multi-Agent Demo")
    print("=" * 70)
    print("\nThis demonstrates the GOOGLE ADK approach:")
    print("• Router routes requests to specialized agents")
    print("• AgentTool automatically wraps agents (NO manual code needed!)")
    print("• AgentTool handles invocation, response extraction, session management")
    print("\nNotice: NO wrapper functions needed - AgentTool does it automatically!")
    print("-" * 70)
    
    # Test cases (same as LangChain example for comparison)
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
        print("🤖 Router → Routing to specialized agent...\n")
        
        # Create session
        session = await session_service.create_session(
            app_name="ADK AgentTool Comparison Demo",
            user_id=f"demo-user-{i}",
            state={},
        )
        
        # Convert to Content format
        content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)],
        )
        
        # Run the router
        event_count = 0
        full_response = ""
        agent_called = None
        
        try:
            async for event in runner.run_async(
                user_id=f"demo-user-{i}",
                session_id=session.id,
                new_message=content,
            ):
                event_count += 1
                
                # Track which agent was called
                tool_calls = getattr(event, 'tool_calls', None)
                if tool_calls:
                    for tool_call in tool_calls:
                        tool_name = getattr(tool_call, 'name', 'unknown')
                        if 'agent' in tool_name.lower():
                            agent_called = tool_name
                            print(f"[🔀 Router called: {agent_called}]\n")
                
                # Extract text content
                if hasattr(event, 'content') and event.content:
                    # Defensive check: ensure parts exists and is not None before iterating
                    if hasattr(event.content, 'parts') and event.content.parts is not None:
                        for part in event.content.parts:
                            if hasattr(part, 'text') and part.text:
                                text = part.text
                                full_response += text
                                print(text, end="", flush=True)
            
            print("\n")
            if agent_called:
                print(f" Agent called: {agent_called}")
            elif not full_response:
                print("  No response received from agent")
        except TypeError as e:
            error_msg = str(e)
            if "'NoneType' object is not iterable" in error_msg:
                print("\n  Error: Known issue in google-adk library (awaiting PR #3988 fix)")
                print("   The AgentTool encountered a response with None parts.")
                print("   This will be fixed in an upcoming google-adk release.")
                print(f"   Error details: {error_msg}\n")
            else:
                print(f"\n TypeError: {error_msg}\n")
                raise
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            
            # Handle rate limit / quota errors
            if "ResourceExhausted" in error_type or "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower():
                print("\n  Rate Limit / Quota Error:")
                print("   You've exceeded your API quota for the free tier.")
                print("   Free tier limit: 20 requests per day per model")
                print("   Solutions:")
                print("   1. Wait for the quota to reset (usually 24 hours)")
                print("   2. Use a different API key with higher limits")
                print("   3. Reduce the number of test cases")
                if full_response:
                    print(f"   Note: Partial response received: {full_response[:100]}...")
                print(f"   Error: {error_msg[:200]}...\n")
                # Break out of this test case - the outer loop will continue to next test case
                break
            else:
                print(f"\n Error: {error_type}: {error_msg}\n")
                import traceback
                traceback.print_exc()
                raise
        
        print("-" * 70)
    
    print("\n" + "=" * 70)
    print("KEY POINTS: Google ADK AgentTool Pattern")
    print("=" * 70)
    print("\n Works: Router routes to specialized agents")
    print(" Automatic: AgentTool handles everything - NO wrapper functions needed!")
    print("   - AgentTool(weather_agent) - automatically wraps")
    print("   - AgentTool(fitness_agent) - automatically wraps")
    print("   - AgentTool(nutrition_agent) - automatically wraps")
    print("\n Code Lines:")
    print("   - Function tools: ~15 lines")
    print("   - Sub-agents: ~20 lines")
    print("   -  Wrapper: 0 lines (AgentTool handles it!)")
    print("   - Router: ~5 lines")
    print("   - Total: ~40 lines")
    print("\n Compare: ~40 lines (ADK) vs ~60 lines (LangChain)")
    print("   ADK eliminates ~33% of boilerplate code!")

if __name__ == "__main__":
    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        print(" Error: GOOGLE_API_KEY environment variable not set")
        print("Please set it: export GOOGLE_API_KEY='your-key'")
        exit(1)
    
    asyncio.run(demo_adk_agenttool())

