
"""LangChain tools used by the AI travel planning agent."""
from datetime import date, timedelta
import requests
from langchain.tools import tool
from utils.helpers import load_json

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

CITY_COORDINATES = {
    "goa": (15.2993, 74.1240, "Goa"),
    "delhi": (28.6139, 77.2090, "Delhi"),
    "mumbai": (19.0760, 72.8777, "Mumbai"),
    "jaipur": (26.9124, 75.7873, "Jaipur"),
    "bengaluru": (12.9716, 77.5946, "Bengaluru"),
}


def _normalise(value):
    return str(value).strip().casefold()


@tool
def search_flights(source: str, destination: str, preference: str = "cheapest") -> str:
    """Search the local flights JSON dataset by source and destination.
    preference can be cheapest or fastest. Returns matching flight options."""
    flights = load_json("flights.json")
    matches = [
        flight for flight in flights
        if _normalise(flight["source"]) == _normalise(source)
        and _normalise(flight["destination"]) == _normalise(destination)
    ]
    if not matches:
        return f"No sample flights found from {source} to {destination}. The dataset may need more routes."
    if _normalise(preference) == "fastest":
        matches.sort(key=lambda item: item["duration_hours"])
    else:
        matches.sort(key=lambda item: item["price"])
    return str(matches[:5])


@tool
def recommend_hotels(city: str, max_price_per_night: int = 10000, minimum_rating: float = 0) -> str:
    """Recommend hotels from hotels.json for a city, filtered by maximum nightly price
    and minimum guest rating. Results are sorted by rating, then price."""
    hotels = load_json("hotels.json")
    matches = [
        hotel for hotel in hotels
        if _normalise(hotel["city"]) == _normalise(city)
        and float(hotel["price_per_night"]) <= int(max_price_per_night)
        and float(hotel["rating"]) >= float(minimum_rating)
    ]
    matches.sort(key=lambda item: (-float(item["rating"]), float(item["price_per_night"])))
    if not matches:
        return f"No hotels in the sample dataset match the filters for {city}. Try increasing the price limit or lowering the rating."
    return str(matches[:5])


@tool
def discover_places(city: str, interest: str = "all", minimum_rating: float = 4.0) -> str:
    """Search tourist attractions in a city.

    Arguments:
    city: City name, for example Goa.
    interest: Use exactly one of these values:
    all, beach, heritage, adventure, culture, sightseeing.
    minimum_rating: Minimum attraction rating from 0 to 5.

    Always provide all three arguments.
    """
    places = load_json("places.json")
    matches = [
        place for place in places
        if _normalise(place["city"]) == _normalise(city)
        and float(place["rating"]) >= float(minimum_rating)
        and (_normalise(interest) == "all" or _normalise(place["type"]) == _normalise(interest))
    ]
    matches.sort(key=lambda item: -float(item["rating"]))
    if not matches:
        return f"No attractions match {interest} in {city} in the sample dataset."
    return str(matches[:12])



@tool
def get_weather(city: str, travel_date: str, days: int = 3) -> str:
    """Fetch weather forecasts for an Indian city using Open-Meteo.
    Arguments:
    city: Destination city.
    travel_date: Start date in YYYY-MM-DD format.
    days: Number of forecast days, from 1 to 16.
    """

    try:
        # Known Indian city coordinates prevent incorrect city matching.
        city_key = city.strip().casefold()

        if city_key in CITY_COORDINATES:
            latitude, longitude, city_name = CITY_COORDINATES[city_key]

        else:
            # Restrict fallback geocoding to India.
            geo_response = requests.get(
                GEOCODING_URL,
                params={
                    "name": city,
                    "count": 10,
                    "language": "en",
                    "format": "json",
                    "countryCode": "IN"
                },
                timeout=15
            )
            geo_response.raise_for_status()

            locations = geo_response.json().get("results", [])

            if not locations:
                return f"No Indian location found for {city}."

            # Prefer an exact city-name match.
            match = next(
                (
                    loc for loc in locations
                    if loc.get("name", "").casefold() == city_key
                ),
                None
            )

            if match is None:
                return (
                    f"Could not confidently identify {city} in India. "
                    "Please provide a more specific city name."
                )

            latitude = match["latitude"]
            longitude = match["longitude"]
            city_name = match["name"]

        requested_date = date.fromisoformat(travel_date)
        days = max(1, min(int(days), 16))

        today = date.today()

        if requested_date < today or requested_date > today + timedelta(days=15):
            return (
                "Forecast unavailable. Open-Meteo supports forecasts "
                "only within its 16-day forecast window. "
                "Do not invent weather information."
            )

        response = requests.get(
            FORECAST_URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "daily": (
                    "temperature_2m_max,"
                    "temperature_2m_min,"
                    "precipitation_probability_max,"
                    "weather_code"
                ),
                "timezone": "auto",
                "forecast_days": 16
            },
            timeout=20
        )

        response.raise_for_status()

        daily = response.json().get("daily", {})

        forecast = []

        for index, day_text in enumerate(daily.get("time", [])):

            forecast_date = date.fromisoformat(day_text)

            if requested_date <= forecast_date < requested_date + timedelta(days=days):

                forecast.append({
                    "date": day_text,
                    "max_temperature_c": daily["temperature_2m_max"][index],
                    "min_temperature_c": daily["temperature_2m_min"][index],
                    "rain_probability_percent": daily["precipitation_probability_max"][index],
                    "weather_code": daily["weather_code"][index]
                })

        if not forecast:
            return "No weather forecast available for these dates."

        return str({
            "city": city_name,
            "country": "India",
            "forecast": forecast
        })

    except (requests.RequestException, ValueError, KeyError) as error:
        return f"Weather API error: {error}"




@tool
def estimate_budget(
    flight_cost: int,
    hotel_per_night: int,
    nights: int,
    daily_local_expenses: int,
    travellers: int = 1
) -> str:
    """Calculate a trip budget in Indian Rupees.

    Arguments:
    flight_cost: One-way flight fare per traveller.
    hotel_per_night: Hotel price per room per night.
    nights: Actual number of hotel nights.
    daily_local_expenses: Estimated local expenses per traveller per day.
    travellers: Number of travellers.

    IMPORTANT: Do not assume a return flight is included.
    """

    travellers = max(1, int(travellers))
    nights = max(0, int(nights))

    flight_total = int(flight_cost) * travellers
    hotel_total = int(hotel_per_night) * nights

    travel_days = nights + 1

    local_total = (
        int(daily_local_expenses)
        * travel_days
        * travellers
    )

    total = flight_total + hotel_total + local_total

    return str({
        "currency": "INR",
        "flight_type": "One-way only",
        "flight_total": flight_total,
        "hotel_nights": nights,
        "hotel_total": hotel_total,
        "travel_days": travel_days,
        "local_expenses_total": local_total,
        "estimated_total": total,
        "remaining_budget": None,
        "assumptions": [
            "Flight fare covers one-way travel only.",
            "Return flight is excluded unless separately found.",
            "Hotel cost is calculated per room per night.",
            "Local expenses are estimated per traveller per day.",
            "No additional contingency charge is included.",
            "Prices are estimates, not confirmed bookings."
        ]
    })



ALL_TOOLS = [
    search_flights,
    recommend_hotels,
    discover_places,
    get_weather,
    estimate_budget
]
