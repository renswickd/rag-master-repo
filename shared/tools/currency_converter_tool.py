import os
import requests
from dotenv import load_dotenv
from typing import List
from langchain.tools import StructuredTool
from shared.components.agentic_rag_states import CurrencyConvertInput

load_dotenv()
exchangerate_api_key = os.getenv("EXCHANGERATE_API_KEY")

def _currency_convert(amount: float, from_currency: str, to_currency: str) -> str:
    try:
        resp = requests.get(
            "https://api.exchangerate.host/convert",
            params={"from": from_currency.upper(), "to": to_currency.upper(), "amount": amount},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        result = data.get("result", None)
        if result is None:
            return f"Conversion error: unexpected response: {data}"
        rate = data.get("info", {}).get("rate")
        return f"{amount} {from_currency.upper()} = {result} {to_currency.upper()} (rate: {rate})"
    except Exception as e:
        return f"Conversion error: {e}"

exchangerate_converter = StructuredTool.from_function(
    func=_currency_convert,
    name="currency_convert",
    description=(
        "Convert currency amounts using live foreign exchange rates."
    ),
    args_schema=CurrencyConvertInput,
)