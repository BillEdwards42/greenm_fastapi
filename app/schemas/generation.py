import re
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional

# --- 1. The Normalized "Clean" Object ---
class Generation(BaseModel):
    name: str
    fuel_type: str
    capacity_mw: float
    current_generation_mw: float
    status: str

# --- 2. The Raw Row Parser ---
# This handles the ugly list ["<HTML>", "", "Name", "800", ...]
class RawGeneration(BaseModel):
    # We don't define fields 1-by-1 because the input is a list, not a dict.
    # We use a trick to parse the list into this object.

    name: str
    fuel_type: str
    capacity_mw: float
    current_generation_mw: float
    status: str

    @model_validator(mode="before")
    @classmethod
    def parse_raw_list(cls, data: List[str]) -> dict:
        """
        Input: ["<A..>Coal</A>", "", "Linkou#1", "800.0", "757.3", "94%", "Run", ""]
        Output: Dict mapping to the fields above.
        """
        if not isinstance(data, list) or len(data)<7:
            raise ValueError("Invalid row format.")

        # Helper to strip HTML tags
        def clean_str(s: str) -> str:
            return re.sub('<[^<]+?>', '', s).strip()

        # Helper to handle "-", "N/A", or valid floats
        def parse_float(s: str) -> float:
            s = s.replace(',', '').strip()
            if s in ['', '-', 'N/A']:
                return 0.0
            try:
                if '(' in s:
                    s = s.split('(')[0]
                return float(s)
            except ValueError:
                return 0.0

        # Mapping by Index (The fragile part of scraping)
        # Index 0: Type (Has HTML)
        # Index 2: Name
        # Index 3: Capacity
        # Index 4: Current Generation
        # Index 6: Status notes
        return {
            "fuel_type": clean_str(data[0]),
            "name": clean_str(data[2]),
            "capacity_mw": parse_float(data[3]),
            "current_generation_mw": parse_float(data[4]),
            "status": clean_str(data[6])
        }

# --- 3. The Root Response ---
# This matches the JSON structure exactly.
class TaipowerResponse(BaseModel):
    timestamp: str = Field(alias='')
    aaData: List[RawGeneration]
    
    #Logic to filter out "Subtotals" or "Trash" rows
    @property
    def valid_generators(self) -> List[Generation]:
        """Returns only actual power plants, filtering out 'Subtotal' rows"""
        return [
            Generation(
                name=row.name,
                fuel_type=row.fuel_type,
                capacity=row.capacity_mw,
                current_generation=row.current_generation_mw,
                status=row.status
            )
            for row in self.aaData
            if "小計" not in row.name # Filter out subtotals
        ]