
"""Common helper functions for the travel planner."""
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"


def load_json(filename):
    """Load a JSON dataset from the project's data directory."""
    path = DATA_DIR / filename
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def money(amount):
    """Format a numeric amount as Indian Rupees."""
    return f"₹{amount:,.0f}"
