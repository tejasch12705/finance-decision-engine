"""
Local storage for named portfolios.

Just a JSON file on disk - no database needed for something only one
person uses on one laptop. If this ever needs to sync across devices or
be shared, that's the point to introduce a real database, not before.
"""
import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PORTFOLIOS_PATH = os.path.join(DATA_DIR, "portfolios.json")


def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def load_all_portfolios() -> dict:
    _ensure_data_dir()
    if not os.path.exists(PORTFOLIOS_PATH):
        return {}
    with open(PORTFOLIOS_PATH, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def save_portfolio(name: str, holdings_records: list[dict], base_currency: str):
    """holdings_records: list of dicts like {"ticker": ..., "quantity": ..., "currency": ..., "benchmark": ...}"""
    _ensure_data_dir()
    portfolios = load_all_portfolios()
    portfolios[name] = {"holdings": holdings_records, "base_currency": base_currency}
    with open(PORTFOLIOS_PATH, "w", encoding="utf-8") as f:
        json.dump(portfolios, f, indent=2)


def delete_portfolio(name: str):
    _ensure_data_dir()
    portfolios = load_all_portfolios()
    if name in portfolios:
        del portfolios[name]
        with open(PORTFOLIOS_PATH, "w", encoding="utf-8") as f:
            json.dump(portfolios, f, indent=2)


def list_portfolio_names() -> list[str]:
    return sorted(load_all_portfolios().keys())
