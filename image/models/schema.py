from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ImageEstimationRequestBody(BaseModel):
    check_target_file_id: str = Field(
        ...,
        title="fact check ID",
        example="aedf4c2b-3e1d-4a5b-8c7f-9a2e0d3f4b5e",
    )
    file_path: str = Field(
        ...,
        title="Path of the file",
        example="xxxxx.png",
    )


class PostImageFileDetailRequestBody(BaseModel):
    class BoundingBox(BaseModel):
        x_position: int = Field(..., title="x position", examples=[50])
        y_position: int = Field(..., title="y position", examples=[50])
        width: int = Field(..., title="width", examples=[100])
        height: int = Field(..., title="height", examples=[100])
        fake_percentage: float = Field(..., title="fake percentage", examples=[0.6])

    check_target_file_id: str = Field(..., title="Fact Check ID", examples=[1])
    bounding_boxes: List[BoundingBox] = Field(..., title="bounding_boxes")
    fake_percentage: Optional[float] = Field(
        ..., title="fake percentage", examples=[0.6]
    )
    general_object_fake_percentage: Optional[float] = Field(
        ..., title="general object fake percentage", examples=[0.6]
    )
