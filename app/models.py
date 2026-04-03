from pydantic import BaseModel
from typing import Dict, List

class StudentsInput(BaseModel):
    students: Dict[str, list[float]]
    iterations: int = 100 
    