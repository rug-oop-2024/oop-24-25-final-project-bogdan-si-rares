from pydantic import BaseModel, Field
from typing import Literal


class Feature(BaseModel):
    name: str = Field(description="The name of the feature")
    type: Literal["numerical", "categorical"] = Field(
        description="Type of the feature"
    )

    def __str__(self):
        return f"Feature(name={self.name}, type={self.type})"
