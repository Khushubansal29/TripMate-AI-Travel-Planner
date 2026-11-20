
from agent import run_agent

query = """
Plan a 3-day trip from Delhi to Goa.

Budget: 15000 INR
Travellers: 1
Interests: Beaches and sightseeing.

You must:
1. Search flights from Delhi to Goa.
2. Find hotels in Goa.
3. Discover tourist attractions in Goa.
4. Get weather for Goa starting 2026-11-23 for 3 days.
5. Estimate the budget using the available budget tool.

Generate a complete day-wise itinerary.
Explain your selections and show the budget breakdown.
Do not invent unavailable information.
"""

print("\n===== TRIPMATE AI AGENT TEST =====\n")

try:
    result = run_agent(query)

    print("===== GENERATED TRAVEL PLAN =====\n")
    print(result)

    if result.strip():
        print("\nSUCCESS: Agent returned a response!")
    else:
        print("\nWARNING: Agent returned an empty response.")

except Exception as error:
    print("\nAGENT ERROR:")
    print(repr(error))
