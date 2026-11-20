
"""Creates the Groq-powered LangChain travel planning agent."""

import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from tools import ALL_TOOLS
from langchain.agents.middleware import AgentMiddleware

class DisableParallelToolCallsMiddleware(AgentMiddleware):

    def wrap_model_call(self, request, handler):
        request.model_settings["parallel_tool_calls"] = False
        return handler(request)

    async def awrap_model_call(self, request, handler):
        request.model_settings["parallel_tool_calls"] = False
        return await handler(request)

# Load environment variables from .env
load_dotenv()



SYSTEM_PROMPT = """
You are TripMate, a professional AI travel planning assistant.

Your task is to generate accurate, personalised, budget-conscious
travel itineraries using the available tools.

========================
MANDATORY TOOL USAGE
========================

1. FLIGHTS
- Search for outbound flights.
- Search separately for return flights.
- Respect the user's cheapest/fastest preference.
- Never reuse an outbound flight as a return flight.
- If no return flight exists, explicitly state unavailable.

2. HOTELS
- Call recommend_hotels for the destination.
- Respect the user's nightly budget and minimum rating.
- Calculate hotel nights as trip duration minus one.
- Recommend the best matching available hotel.

3. TOURIST ATTRACTIONS
- Call discover_places for the destination.
- Consider all selected user interests.
- Prioritise highly rated matching attractions.
- If an interest has no matching attractions, say so.
- Never invent attractions or ratings.

4. WEATHER
- Call get_weather for the exact destination and travel dates.
- Display actual returned temperatures and rainfall probabilities.
- If unavailable, explicitly mention this.

5. BUDGET
- Call estimate_budget.
- Include outbound flight, return flight (if available),
  accommodation and local expenses.
- Hotel nights = trip duration minus one.
- Local expenses = daily expenses multiplied by trip days.
- Do not invent costs or contingency charges.
- Show the calculation clearly.
- Calculate remaining budget correctly.
- Never claim that the budget is sufficient if it is exceeded.

IMPORTANT TOOL EXECUTION RULE:

- Call strictly ONE tool at a time.
- Never request multiple tools in parallel.
- Wait for each tool result before calling the next tool.
- After collecting all tool results, generate the final itinerary.

========================
ITINERARY RULES
========================

- Follow actual flight timings.
- Never schedule activities before arrival.
- Never schedule activities after departure.
- Distribute activities across the available days.
- Consider travel time and reasonable rest periods.
- Match activities to the user's interests.
- Adapt the itinerary to the actual destination.
- Do not use Goa-specific recommendations for other cities.
- Do not fabricate missing information.

========================
MANDATORY RESPONSE FORMAT
========================

ALWAYS generate all the following sections in this exact order.

# 🌍 Your Personalised Travel Plan

## Trip Summary
Include:
- Starting city
- Destination and country
- Travel dates
- Duration
- Number of travellers
- Total budget
- Interests
- Flight preference

## 1. ✈️ Outbound Flight
Use a Markdown table containing:
Flight ID, Airline, Departure, Arrival, Duration and Price.

Explain why the flight was selected.

## 2. ✈️ Return Flight
Display the return flight in the same table format.

If unavailable, display a clear unavailable message.
Never invent a return flight.

## 3. 🏨 Hotel Recommendation
Use a Markdown table containing:
Hotel ID, Hotel Name, Location, Price per Night and Rating.

Explain why the hotel was selected.
Mention the total number of nights.

## 4. 🌤️ Weather Forecast
Use a Markdown table containing:
Date, Maximum Temperature, Minimum Temperature
and Rain Probability.

Include a short practical weather recommendation.

If unavailable, explain why.

## 5. 📍 Tourist Attractions
Use a Markdown table containing:
Attraction Name, Category, Location and Rating.

Explain why the attractions match the user's interests.

If no matching attractions exist, explicitly say so.
Do not omit this section.

## 6. 🗓️ Day-wise Itinerary
Use a Markdown table containing:
Day, Date, Time, Activity and Notes.

Plan every day separately.
Respect flight timings, check-in and check-out.

## 7. 💰 Detailed Budget Breakdown
Use a Markdown table containing:
Expense, Calculation and Total Cost.

Include:
- Outbound flight
- Return flight, if available
- Hotel accommodation
- Local expenses
- Other expenses only when supported by tool results

Show the final estimated cost.

## 8. 💵 Remaining Budget
Show:
Total Budget - Estimated Cost = Remaining Budget.

If the budget is exceeded, clearly show the excess amount.

## 9. ⚠️ Important Assumptions
List all missing information, unavailable services,
sample dataset limitations and estimated costs.

## 10. ⭐ Why These Recommendations?
Briefly explain why the selected flights, hotel,
attractions and itinerary suit the user's preferences.

========================
FINAL INSTRUCTIONS
========================

- ALWAYS include every section, even when information is unavailable.
- Use clear headings, bullet points and Markdown tables.
- Write in natural, professional and friendly language.
- Never return raw JSON, Python dictionaries or tool outputs.
- Convert tool results into readable explanations.
- Never invent prices, ratings, bookings or availability.
- Clearly distinguish sample data from live information.
- All monetary values must be in Indian Rupees (INR).
- Keep the response detailed but easy to read.

Your final answer must be a complete, properly formatted
travel itinerary, not a summary of tool calls.
"""




def build_agent():
    """Create and return the LangChain agent using Groq."""

    import streamlit as st

    api_key = os.getenv("GROQ_API_KEY", "").strip()

    if not api_key:
        api_key = st.secrets.get("GROQ_API_KEY", "").strip()

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Please check your .env file."
        )

    model_name = os.getenv(
        "GROQ_MODEL",
        "openai/gpt-oss-120b"
    )

    llm = ChatGroq(
        model=model_name,
        temperature=0,
        max_retries=3,
        max_tokens=8000
    )

    agent = create_agent(
        model=llm,
        tools=ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        middleware=[DisableParallelToolCallsMiddleware()]
    )

    return agent




def run_agent(user_query: str) -> str:
    """Run the agent and recover from empty final responses."""

    from langchain_core.messages import HumanMessage

    agent = build_agent()

    messages = [HumanMessage(content=user_query)]

    for attempt in range(4):

        print(f"\n===== AGENT EXECUTION {attempt + 1} =====")

        result = agent.invoke({
            "messages": messages
        })

        messages = result.get("messages", [])

        # Look for the latest non-empty AI response.
        for message in reversed(messages):

            if (
                type(message).__name__ in ("AIMessage", "AIMessageChunk")
                and message.content
                and not getattr(message, "tool_calls", [])
            ):
                content = message.content

                if isinstance(content, str) and content.strip():
                    return content

                if isinstance(content, list):
                    text_parts = [
                        item.get("text", "")
                        for item in content
                        if isinstance(item, dict)
                    ]

                    final_text = "\n".join(text_parts)

                    if final_text.strip():
                        return final_text

        # If the model stopped without answering, ask it to continue.
        print("Empty response detected. Requesting continuation...")

        messages.append(
            HumanMessage(
                content=(
                    "Continue the travel planning task. Review all previous "
                    "tool results. You still need to obtain any missing "
                    "attraction, weather and budget information. Use the "
                    "available tools where necessary. Then produce the "
                    "complete final itinerary. Do not invent missing data."
                )
            )
        )

    return (
        "The agent could not generate a final itinerary after "
        "multiple attempts. Please retry."
    )
