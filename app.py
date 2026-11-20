"""TripMate - Agentic AI Travel Planner"""

from datetime import date, timedelta
import ast
import json

import pandas as pd
import streamlit as st

from agent import run_agent
from tools import (
    search_flights,
    recommend_hotels,
    discover_places,
    get_weather,
)

st.set_page_config(
    page_title="TripMate | AI Travel Planner",
    page_icon="✈️",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

st.markdown("""
<style>
    .hero {
        padding: 2rem;
        border-radius: 18px;
        background: linear-gradient(115deg, #102a43, #176b87);
        color: white;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        color: white;
        font-size: 2.8rem;
        margin-bottom: 5px;
    }

    .hero p {
        color: #e6f3f7;
        font-size: 1.05rem;
    }

    .section-heading {
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .small-note {
        color: #8b9bad;
        font-size: .85rem;
    }

    div.stButton > button {
        border-radius: 10px;
        min-height: 2.8rem;
        font-weight: 600;
    }

    div[data-testid="stMetric"] {
        background: rgba(128,128,128,0.08);
        padding: 14px;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def parse_result(result):
    """Convert JSON or Python dictionary strings into Python objects."""

    if isinstance(result, (dict, list)):
        return result

    if not isinstance(result, str):
        return result

    try:
        return json.loads(result)
    except (ValueError, TypeError):
        pass

    try:
        return ast.literal_eval(result)
    except (ValueError, SyntaxError):
        return result


def display_table(data, columns=None):
    """Display records in a readable table."""

    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list) or not data:
        st.info("No matching records found.")
        return

    if not all(isinstance(item, dict) for item in data):
        st.write(data)
        return

    df = pd.DataFrame(data)

    if columns:
        available = [c for c in columns if c in df.columns]
        if available:
            df = df[available]

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


def display_weather(data):
    """Display weather information as daily forecast cards."""

    if not isinstance(data, dict):
        st.info(str(data))
        return

    city = data.get("city", "Destination")
    country = data.get("country", "")

    st.subheader(f"🌤️ Weather in {city}, {country}")

    forecasts = data.get("forecast", [])

    if not forecasts:
        st.warning("No weather forecast is available.")
        return

    for i in range(0, len(forecasts), 3):

        cols = st.columns(3)

        for col, item in zip(cols, forecasts[i:i + 3]):

            with col:
                with st.container(border=True):

                    st.markdown(f"**📅 {item.get('date', 'N/A')}**")

                    st.metric(
                        "Maximum temperature",
                        f"{item.get('max_temperature_c', 'N/A')} °C"
                    )

                    st.metric(
                        "Minimum temperature",
                        f"{item.get('min_temperature_c', 'N/A')} °C"
                    )

                    st.metric(
                        "Rain probability",
                        f"{item.get('rain_probability_percent', 'N/A')}%"
                    )


def display_records(result, kind):
    """Display tool results in a readable format."""

    data = parse_result(result)

    if isinstance(data, str):
        st.info(data)
        return

    if kind == "weather":
        display_weather(data)

    elif kind == "flights":

        st.subheader("✈️ Available flights")

        if isinstance(data, list) and data:
            for flight in data:

                with st.container(border=True):

                    c1, c2, c3 = st.columns([2, 2, 1])

                    with c1:
                        st.markdown(
                            f"**{flight.get('airline', 'Airline')}**"
                        )
                        st.caption(
                            f"Flight ID: {flight.get('flight_id', 'N/A')}"
                        )

                    with c2:
                        st.markdown(
                            f"**{flight.get('departure', 'N/A')} → "
                            f"{flight.get('arrival', 'N/A')}**"
                        )
                        st.caption(
                            f"Duration: {flight.get('duration_hours', 'N/A')} hours"
                        )

                    with c3:
                        st.metric(
                            "Ticket price",
                            f"₹{flight.get('price', 'N/A'):,}"
                        )

        else:
            st.info("No matching flights found.")

    elif kind == "hotels":

        st.subheader("🏨 Recommended hotels")

        if isinstance(data, list) and data:
            for hotel in data:

                with st.container(border=True):

                    c1, c2 = st.columns([3, 1])

                    with c1:
                        st.markdown(
                            f"### {hotel.get('name', 'Hotel')}"
                        )
                        st.caption(
                            f"📍 {hotel.get('area', 'Location unavailable')}"
                        )
                        st.write(
                            f"Category: {hotel.get('category', 'N/A')}"
                        )

                    with c2:
                        st.metric(
                            "Price per night",
                            f"₹{hotel.get('price_per_night', 0):,}"
                        )
                        st.metric(
                            "Rating",
                            f"⭐ {hotel.get('rating', 'N/A')}/5"
                        )

        else:
            st.info("No matching hotels found.")

    elif kind == "places":

        st.subheader("📍 Tourist attractions")

        if isinstance(data, list) and data:

            for place in data:

                with st.container(border=True):

                    c1, c2 = st.columns([3, 1])

                    with c1:
                        st.markdown(
                            f"**{place.get('name', 'Attraction')}**"
                        )
                        st.caption(
                            place.get(
                                'description',
                                place.get('category', 'Tourist attraction')
                            )
                        )

                    with c2:
                        st.metric(
                            "Rating",
                            f"⭐ {place.get('rating', 'N/A')}/5"
                        )

        else:
            st.info("No matching attractions found.")


# --------------------------------------------------
# RULE-BASED FALLBACK (NO GROQ CALL REQUIRED)
# --------------------------------------------------

def _as_records(value, possible_keys=()):
    """Normalize tool output into a list of dictionaries."""
    data = parse_result(value)

    if isinstance(data, dict):
        for key in possible_keys:
            if isinstance(data.get(key), list):
                return data[key]
        for key in ("results", "data", "records", "items"):
            if isinstance(data.get(key), list):
                return data[key]
        return [data]

    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]

    return []


def _money(value):
    """Return a numeric amount when a tool provides one."""
    try:
        if isinstance(value, str):
            value = value.replace(",", "").replace("₹", "").strip()
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_value(record, keys, default="N/A"):
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return default


def generate_fallback_plan(
    source, destination, start_date, duration, travellers,
    budget, interests, preference, min_rating
):
    """
    Build a useful itinerary from the existing data tools.
    This function does not call Groq or any language model.
    """
    errors = []

    def call_tool(tool, label, arguments):
        try:
            return tool.invoke(arguments)
        except Exception as exc:
            errors.append(f"{label}: {exc}")
            return None

    nights = max(duration - 1, 0)
    return_date = start_date + timedelta(days=max(duration - 1, 0))
    interest_text = ", ".join(interests) if interests else "all"

    outbound_raw = call_tool(
        search_flights, "Outbound flight search",
        {
            "source": source,
            "destination": destination,
            "preference": preference.lower()
        }
    )
    return_raw = call_tool(
        search_flights, "Return flight search",
        {
            "source": destination,
            "destination": source,
            "preference": preference.lower()
        }
    )

    nightly_limit = max(
        500, int(budget / max(nights, 1) / max(travellers, 1))
    )
    hotels_raw = (
        call_tool(
            recommend_hotels, "Hotel search",
            {
                "city": destination,
                "max_price_per_night": nightly_limit,
                "minimum_rating": min_rating
            }
        )
        if nights > 0 else None
    )

    places_raw = call_tool(
        discover_places, "Attraction search",
        {
            "city": destination,
            "interest": interest_text,
            "minimum_rating": 0
        }
    )
    weather_raw = call_tool(
        get_weather, "Weather lookup",
        {
            "city": destination,
            "travel_date": start_date.isoformat(),
            "days": duration
        }
    )

    outbound = _as_records(outbound_raw, ("flights", "outbound_flights"))
    inbound = _as_records(return_raw, ("flights", "return_flights"))
    hotels = _as_records(hotels_raw, ("hotels",))
    places = _as_records(places_raw, ("places", "attractions"))
    weather_data = parse_result(weather_raw) if weather_raw is not None else None

    lines = [
        "# ✈️ TripMate — Your Travel Plan",
        "",
        "> **Fallback mode:** Groq was unavailable or rate-limited. "
        "This plan was assembled directly from TripMate's available data tools; "
        "it was not generated by the AI model.",
        "",
        "## 1. Trip Summary",
        f"- **Route:** {source} → {destination} → {source}",
        f"- **Dates:** {start_date.strftime('%d %b %Y')} – "
        f"{return_date.strftime('%d %b %Y')}",
        f"- **Duration:** {duration} day(s) / {nights} hotel night(s)",
        f"- **Travellers:** {travellers}",
        f"- **Total budget:** ₹{budget:,.0f}",
        f"- **Interests:** {interest_text}",
        f"- **Flight preference:** {preference}",
        "",
        "## 2. Outbound Flight",
    ]

    selected_outbound = outbound[0] if outbound else None
    selected_return = inbound[0] if inbound else None

    def add_flight(title, flight):
        lines.append(f"### {title}")
        if not flight:
            lines.append("- No matching flight was returned by the dataset.")
            return None
        price = _money(_first_value(flight, ("price", "fare", "price_inr"), None))
        lines.extend([
            f"- **Airline:** {_first_value(flight, ('airline', 'carrier'))}",
            f"- **Flight ID:** {_first_value(flight, ('flight_id', 'id'))}",
            f"- **Departure:** {_first_value(flight, ('departure', 'departure_time'))}",
            f"- **Arrival:** {_first_value(flight, ('arrival', 'arrival_time'))}",
            f"- **Duration:** {_first_value(flight, ('duration_hours', 'duration'))}",
            f"- **Price:** ₹{price:,.0f}" if price is not None
              else "- **Price:** Not supplied by the dataset",
        ])
        return price

    outbound_cost = add_flight("Outbound — " + source + " to " + destination, selected_outbound)
    lines.extend(["", "## 3. Return Flight"])
    return_cost = add_flight("Return — " + destination + " to " + source, selected_return)

    lines.extend(["", "## 4. Hotel Recommendation"])
    selected_hotel = hotels[0] if hotels else None
    hotel_cost = None
    if selected_hotel and nights:
        nightly_price = _money(_first_value(
            selected_hotel, ("price_per_night", "price", "nightly_price"), None
        ))
        lines.extend([
            f"- **Hotel:** {_first_value(selected_hotel, ('name', 'hotel_name'))}",
            f"- **Area:** {_first_value(selected_hotel, ('area', 'location'))}",
            f"- **Rating:** {_first_value(selected_hotel, ('rating', 'stars'))}",
            f"- **Price per night:** ₹{nightly_price:,.0f}" if nightly_price is not None
              else "- **Price per night:** Not supplied by the dataset",
            f"- **Nights:** {nights}",
        ])
        if nightly_price is not None:
            hotel_cost = nightly_price * nights * travellers
            lines.append(f"- **Estimated hotel total:** ₹{hotel_cost:,.0f}")
    elif nights == 0:
        lines.append("- No hotel night is required for a one-day trip.")
    else:
        lines.append("- No hotel matching the supplied filters was returned.")

    lines.extend(["", "## 5. Weather Forecast"])
    if isinstance(weather_data, dict):
        forecasts = weather_data.get("forecast", [])
        if forecasts:
            for item in forecasts:
                lines.append(
                    f"- **{item.get('date', 'Date unavailable')}:** "
                    f"High {item.get('max_temperature_c', 'N/A')}°C, "
                    f"low {item.get('min_temperature_c', 'N/A')}°C, "
                    f"rain probability {item.get('rain_probability_percent', 'N/A')}%."
                )
        else:
            lines.append("- Weather service did not return forecast days for these dates.")
    else:
        lines.append("- Weather information is unavailable.")

    lines.extend(["", "## 6. Day-wise Itinerary"])
    if duration == 1:
        lines.extend([
            f"### Day 1 — {start_date.strftime('%d %b %Y')}",
            f"- Arrive in {destination}; check in or leave luggage if applicable.",
            "- Visit a nearby attraction from the list below if time permits.",
            "- Keep sufficient time for the return journey."
        ])
    else:
        for day in range(duration):
            current_day = start_date + timedelta(days=day)
            lines.append(f"### Day {day + 1} — {current_day.strftime('%d %b %Y')}")
            if day == 0:
                lines.append("- Travel to the destination. Begin sightseeing only after arrival and check-in.")
            elif day == duration - 1:
                lines.append("- Check out and keep the schedule flexible around the return flight.")
            else:
                lines.append("- Explore the destination; take breaks between activities.")

            if places:
                place = places[min(day, len(places) - 1)]
                place_name = _first_value(place, ("name", "place_name", "attraction"))
                description = _first_value(
                    place, ("description", "category", "interest"), ""
                )
                rating = _first_value(place, ("rating",), None)
                detail = f" — {description}" if description not in ("", "N/A") else ""
                score = f" (rating: {rating}/5)" if rating not in (None, "N/A") else ""
                lines.append(f"- Suggested attraction: **{place_name}**{detail}{score}")
            else:
                lines.append("- No attraction records were returned; use the Places tab to explore options.")

    lines.extend(["", "## 7. Detailed Budget Breakdown"])
    known_costs = []
    for label, amount in (
        ("Outbound airfare", outbound_cost),
        ("Return airfare", return_cost),
        ("Hotel stay", hotel_cost),
    ):
        if amount is not None:
            known_costs.append((label, amount))
            lines.append(f"- {label}: ₹{amount:,.0f}")
        else:
            lines.append(f"- {label}: Not included (price unavailable or option not found)")

    known_total = sum(amount for _, amount in known_costs)
    remaining = budget - known_total
    lines.extend([
        f"- **Known-cost subtotal:** ₹{known_total:,.0f}",
        "- Meals, local transport and attraction entry fees are not included "
        "because the available tools did not provide verified amounts.",
        "",
        "## 8. Remaining Budget",
        f"- **Budget minus known costs:** ₹{remaining:,.0f}",
        "- This is a provisional balance, not a complete trip-cost estimate.",
        "",
        "## 9. Important Assumptions",
        "- The first option returned by each flight search is shown; verify the "
        "dataset's ordering and details before making travel decisions.",
        "- Outbound and return flights were searched separately.",
        f"- Hotel nights are calculated as duration minus one ({nights} night(s)).",
        "- Hotel total is multiplied by the number of travellers.",
        "- Flight, hotel and attraction information comes from sample datasets; "
        "it is not live booking inventory.",
        "- Weather is shown only when returned by the weather service.",
        "- No reservation or payment has been made.",
    ])

    if errors:
        lines.extend(["", "### Tool availability notes"])
        for error in errors:
            lines.append(f"- {error}")

    return "\n".join(lines)


# --------------------------------------------------
# HERO SECTION
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <h1>✈️ TripMate</h1>
    <p>Your AI-powered travel companion — plan smarter, travel better.</p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SIDEBAR INPUTS
# --------------------------------------------------

def sync_flight_destination():
    st.session_state["f_dest"] = st.session_state["destination"]

# Keep the flight explorer synchronised with sidebar inputs.

st.session_state.setdefault("f_source", "Delhi")
st.session_state.setdefault("f_dest", "Goa")


def sync_flight_source():
    st.session_state["f_source"] = st.session_state["source_city"]


def sync_flight_destination():
    st.session_state["f_dest"] = st.session_state["destination"]

with st.sidebar:

    st.header("🧭 Plan your trip")

    source = st.text_input(
        "Starting city",
        value="Delhi",
        key="source_city",
        on_change=sync_flight_source
    )

    destination = st.text_input(
        "Destination",
        value="Goa",
        key="destination",
        on_change=sync_flight_destination
    )

    start_date = st.date_input(
        "Departure date",
        value=date.today() + timedelta(days=3),
        min_value=date.today()
    )

    duration = st.slider(
        "Trip duration (days)",
        1, 7, 3
    )

    travellers = st.number_input(
        "Number of travellers",
        min_value=1,
        max_value=10,
        value=1
    )

    budget = st.number_input(
        "Total budget (₹)",
        min_value=1000,
        max_value=500000,
        value=15000,
        step=500
    )

    interests = st.multiselect(
        "What do you enjoy?",
        ["Beach", "Heritage", "Adventure", "Culture", "Sightseeing"],
        default=["Beach", "Sightseeing"]
    )

    preference = st.selectbox(
        "Flight preference",
        ["Cheapest", "Fastest"]
    )

    min_rating = st.slider(
        "Minimum hotel rating",
        0.0, 5.0, 4.0, 0.1
    )

    generate = st.button(
        "✨ Generate my travel plan",
        type="primary",
        use_container_width=True
    )


# --------------------------------------------------
# MAIN AI TRAVEL PLANNER
# --------------------------------------------------

st.caption(
    "Plan using flight, hotel and attraction datasets "
    "plus the Open-Meteo weather service."
)

if generate:

    if not source.strip() or not destination.strip():

        st.error("Please enter both starting city and destination.")

    else:

        interests_text = (
            ", ".join(interests)
            if interests
            else "all types of attractions"
        )

        query = f"""
        Plan a {duration}-day trip for {travellers} traveller(s)
        from {source} to {destination}.

        Start date: {start_date.isoformat()}.
        Total budget: INR {budget}.
        Interests: {interests_text}.
        Flight preference: {preference.lower()}.
        Minimum hotel rating: {min_rating}.

        Use all five available tools:
        search_flights, recommend_hotels, discover_places,
        get_weather and estimate_budget.

        Search outbound and reverse flight routes separately.
        Never invent a return flight.
        Calculate hotel nights as duration minus one.
        Use actual weather results.
        Ensure all budget calculations are mathematically correct.

        Give a complete day-wise itinerary.
        Clearly state unavailable information.
        """

        with st.spinner(
            "✈️ TripMate is searching and preparing your itinerary..."
        ):

            try:
                answer = run_agent(query)

                if answer and answer.strip():
                    st.session_state["last_plan"] = answer
                    st.session_state["last_query"] = query
                    st.session_state["plan_mode"] = "AI"
                else:
                    raise RuntimeError("The AI agent returned an empty response.")

            except Exception as error:
                # Automatically switch to deterministic tool-based planning.
                st.warning(
                    "Groq is temporarily unavailable. TripMate is switching "
                    "to fallback mode and building your plan from the available datasets."
                )
                try:
                    fallback_answer = generate_fallback_plan(
                        source=source,
                        destination=destination,
                        start_date=start_date,
                        duration=duration,
                        travellers=int(travellers),
                        budget=float(budget),
                        interests=interests,
                        preference=preference,
                        min_rating=float(min_rating),
                    )
                    st.session_state["last_plan"] = fallback_answer
                    st.session_state["last_query"] = query
                    st.session_state["plan_mode"] = "Fallback"
                except Exception as fallback_error:
                    st.error(
                        "Both AI planning and fallback planning failed. "
                        f"Fallback error: {fallback_error}"
                    )


# --------------------------------------------------
# DISPLAY AI-GENERATED ITINERARY
# --------------------------------------------------

if "last_plan" in st.session_state:
    st.divider()

    st.subheader("🌍 Your Personalised Travel Plan")
    mode = st.session_state.get("plan_mode", "AI")
    if mode == "Fallback":
        st.info("🛟 Fallback itinerary — generated from available tools without Groq.")
    else:
        st.caption("Generated by TripMate AI.")

    st.markdown(
        st.session_state["last_plan"]
    )

    st.download_button(
        label="⬇️ Download your itinerary",
        data=st.session_state["last_plan"],
        file_name="TripMate_Travel_Itinerary.md",
        mime="text/markdown"
    )

elif generate:
    st.warning("No itinerary was generated. Please try again.")


# --------------------------------------------------
# DATA EXPLORER
# --------------------------------------------------

st.divider()

st.header("🔎 Explore the travel data")

st.caption(
    "Explore flights, hotels, attractions and weather "
    "using structured information."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "✈️ Flights",
    "🏨 Hotels",
    "📍 Places",
    "🌤️ Weather"
])


# --------------------------------------------------
# FLIGHTS TAB
# --------------------------------------------------

with tab1:
    c1, c2 = st.columns(2)

    with c1:
        f_source = st.text_input(
            "Flight source",
            key="f_source"
        )

    with c2:
        f_dest = st.text_input(
            "Flight destination",
            key="f_dest"
        )

    if st.button("🔍 Search flights"):

        with st.spinner("Searching flights..."):

            result = search_flights.invoke({
                "source": f_source,
                "destination": f_dest,
                "preference": preference.lower()
            })

        display_records(result, "flights")


# --------------------------------------------------
# HOTELS TAB
# --------------------------------------------------

with tab2:

    st.subheader("Find your ideal accommodation")

    max_hotel_price = st.number_input(
        "Maximum price per night (₹)",
        500, 50000, 5000, 500
    )

    if st.button("🏨 Find hotels"):

        with st.spinner("Finding hotels..."):

            result = recommend_hotels.invoke({
                "city": destination,
                "max_price_per_night": max_hotel_price,
                "minimum_rating": min_rating
            })

        display_records(result, "hotels")


# --------------------------------------------------
# PLACES TAB
# --------------------------------------------------

with tab3:

    st.subheader("Discover tourist attractions")

    place_interest = st.selectbox(
        "Attraction type",
        [
            "all",
            "Beach",
            "Heritage",
            "Adventure",
            "Culture",
            "Sightseeing"
        ]
    )

    if st.button("📍 Discover attractions"):

        with st.spinner("Discovering attractions..."):

            result = discover_places.invoke({
                "city": destination,
                "interest": place_interest,
                "minimum_rating": 4.0
            })

        display_records(result, "places")


# --------------------------------------------------
# WEATHER TAB
# --------------------------------------------------

with tab4:

    st.subheader("Check destination weather")

    weather_date = st.date_input(
        "Weather start date",
        value=start_date,
        key="weather_date"
    )

    if st.button("🌤️ Get weather forecast"):

        with st.spinner("Fetching weather forecast..."):

            result = get_weather.invoke({
                "city": destination,
                "travel_date": weather_date.isoformat(),
                "days": duration
            })

        display_records(result, "weather")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.markdown("""
<div class="small-note">
<strong>About TripMate:</strong>

TripMate is an Agentic AI academic prototype using LangChain,
Groq, Streamlit, JSON datasets and the Open-Meteo weather API.

Flight, hotel and attraction information comes from sample datasets.
Weather is retrieved from Open-Meteo when the requested dates fall
within its forecast window.

This application provides recommendations only. It does not
make reservations, process payments or guarantee availability.
</div>
""", unsafe_allow_html=True)
