from pydantic import BaseModel
from typing import List

class User (BaseModel):
    name: str
    disability: str
    skills: List[str]
    # history: List[str]
    