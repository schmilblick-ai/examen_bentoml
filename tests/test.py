from src.service import InputModel
from pydantic import ValidationError

VALID_PAYLOAD = {
  "input_data": {
    "GRE Score": 320,
    "TOEFL Score": 110,
    "University Rating": 3,
    "SOP": 3.5,
    "LOR": 3.5,
    "CGPA": 8.5,
    "Research": 1,
    }
}

payload = {**VALID_PAYLOAD, "Research": 2}

try:
    obj = InputModel(**payload)
    print("❌ ACCEPTÉ :", obj.model_dump())
except ValidationError as e:
    print("✅ REJETÉ :")
    print(e.json(indent=2))