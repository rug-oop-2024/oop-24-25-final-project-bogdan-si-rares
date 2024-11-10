from pydantic import BaseModel, Field
import base64
import uuid


class Artifact(BaseModel):
    name: str
    data: bytes
    type: str = Field(description="Type of artifact")
    asset_path: str = Field(
        default=None,
        description="Path where the artifact is stored"
    )
    version: str = Field(
        default="1.0.0",
        description="Version of the artifact"
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Additional metadata about the artifact"
    )
    tags: list = Field(
        default_factory=list,
        description="Tags associated with the artifact"
    )
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the artifact"
    )

    def encode_data(self) -> str:
        """Encodes artifact data into a base64 string."""
        return base64.b64encode(self.data).decode('utf-8')

    @classmethod
    def decode_data(cls, encoded_data: str) -> bytes:
        """Decodes a base64 string back into bytes."""
        return base64.b64decode(encoded_data)

    def save(self, data: bytes) -> None:
        """Updates the artifact's data."""
        self.data = data

    def read(self) -> bytes:
        """Reads the artifact's data."""
        return self.data
