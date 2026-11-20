
from tools import get_weather

result = get_weather.invoke({
    "city": "Goa",
    "travel_date": "2026-11-23",
    "days": 3
})

print("\n===== GOA WEATHER TEST =====\n")
print(result)
