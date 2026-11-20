
# TripMate — Agentic AI-Based Travel Planning Assistant Using LangChain

## 1. Overview
TripMate is an academic prototype that helps users plan trips through a conversational AI agent. It combines local JSON datasets, LangChain tool calling, the Open-Meteo weather API and a Streamlit interface.

## 2. Problem statement
Trip planning requires comparing transport, accommodation, attractions, weather, distance and budget. TripMate brings several of these steps into one interface and produces a personalised itinerary.

## 3. Features
- Flight search by source and destination, with cheapest/fastest sorting.
- Hotel recommendations by city, nightly price and minimum rating.
- Attraction discovery by city, interest category and rating.
- Weather forecast using Open-Meteo (no API key required).
- Budget estimation with clearly stated assumptions.
- LangChain agent that decides which tools to call.
- Streamlit interface and downloadable itinerary.

## 4. Technology stack
Python, LangChain, LangChain OpenAI integration, OpenAI API, Streamlit, JSON, Requests, Open-Meteo.

## 5. Project structure
```text
AI_Travel_Planner/
├── app.py
├── agent.py
├── tools.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── data/
│   ├── flights.json
│   ├── hotels.json
│   └── places.json
└── utils/
    ├── __init__.py
    └── helpers.py
```

## 6. Setup instructions

### Step 1: Create and activate a virtual environment
Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Configure the OpenAI API key
Copy `.env.example` to `.env` and add your own key:
```env
OPENAI_API_KEY=your_real_api_key
OPENAI_MODEL=gpt-4o-mini
```
Keep `.env` private. Never upload it to GitHub.

### Step 4: Run the app
```bash
streamlit run app.py
```
The application opens in your browser, usually at `http://localhost:8501`.

## 7. Dataset format
The application reads `flights.json`, `hotels.json` and `places.json` from the `data` folder. Replace the included demonstration records with the larger datasets supplied with the assignment, keeping the field names or adapting the loader functions in `tools.py`.

## 8. Agentic workflow
1. User submits trip details through Streamlit.
2. The application converts the preferences into a natural-language task.
3. The LangChain agent receives the task and available tool descriptions.
4. The model chooses relevant tools and supplies arguments.
5. Tools retrieve and filter flight, hotel and attraction records and request weather data.
6. Tool results are returned to the agent.
7. The agent reasons over the results and produces an itinerary and budget estimate.

## 9. Important limitations
- JSON flight, hotel and attraction records are demonstration data, not live booking inventory.
- Weather forecasts are available only within the API's forecast horizon; unavailable forecasts are not fabricated.
- Prices and budget totals are estimates. They are not confirmed quotes.
- The application does not book flights or hotels or process payments.

## 10. Future scope
- Connect real flight and hotel provider APIs.
- Add maps, route-distance optimisation and public transport data.
- Store user preferences and past itineraries in a database.
- Add multi-city planning and currency conversion.
- Add authentication and deploy the application.

## 11. References
- LangChain: https://docs.langchain.com/
- Streamlit: https://docs.streamlit.io/
- Open-Meteo Forecast API: https://open-meteo.com/en/docs
- Open-Meteo Geocoding API: https://open-meteo.com/en/docs/geocoding-api
