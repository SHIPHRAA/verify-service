from __future__ import annotations

import os
from tempfile import TemporaryDirectory
from typing import List, Optional
from urllib.parse import urlparse

from uuid import UUID

import cv2
import pytorch_lightning as pl
import requests
from deepface import DeepFace

from configs.env_config import settings
from image.utils import request_backend, _request
from image.models.schema import (
    ImageEstimationRequestBody,
    PostImageFileDetailRequestBody,
)
from image.services.analysis_modules.config import (
    INFERENCE_CHECKPOINT_FACE,
    INFERENCE_CHECKPOINT_NON_FACE,
)
from image.services.analysis_modules.datamodule import BinaryClassificationDataModule
from image.services.analysis_modules.datamodule2 import BinaryClassificationDataModule2
from image.services.analysis_modules.model import BinaryClassificationEfficientNet
from shared.util import logging_utils
from shared.util.logging_utils import log_execution_time, logger

IMAGE_FILE_DETAIL_API_PATH = settings.IMAGE_FILE_DETAIL_API_PATH
IS_GPU = settings.IS_GPU

binary_classification_face = BinaryClassificationEfficientNet.load_from_checkpoint(
    INFERENCE_CHECKPOINT_FACE, map_location="cuda:0"
)
binary_classification_non_face = BinaryClassificationEfficientNet.load_from_checkpoint(
    INFERENCE_CHECKPOINT_NON_FACE, map_location="cuda:0"
)
trainer = pl.Trainer(devices=[0])


def clear_temp_folder(temp_dir: str):
    """一時フォルダ内のファイルをすべて削除します。"""
    try:
        for file in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    except Exception as e:
        logger.error(f"一時フォルダのクリア中にエラーが発生しました: {e}")


def _get_tmp_file_path_by_url(
    url: str, directory: str
) -> tuple[Optional[str], Optional[Exception]]:
    try:
        response = requests.get(url)
    except Exception as e:
        logger.error(f"外部APIリクエストエラー: {e}")
        return None, e

    if response.status_code != 200:
        logger.error(
            f"外部APIのステータスコードが200以外です。| status_code: {response.status_code} | url: {url}"
        )
        return None, Exception("ステータスコードが200以外です。")
    # 保存先ディレクトリが存在しない場合は作成
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # 保存するファイル名の決定
    parsed_url = urlparse(url)
    _, suffix = os.path.splitext(parsed_url.path)
    if not suffix:
        logger.error(f"suffixがありません。 | url: {url}")
        return None, Exception("suffixがありません。")

    tmp_file_path = os.path.join(directory, f"temp_file{suffix}")
    try:
        with open(tmp_file_path, "wb") as file:
            file.write(response.content)
    except Exception as e:
        logger.error(f"ファイルの保存中にエラーが発生しました。 | error: {e}")
        return None, e

    return tmp_file_path, None


def _detect_and_crop_face(image_path: str, detector_backend: str = "yunet"):
    print(image_path)
    image = cv2.imread(str(image_path))
    face_objs = DeepFace.extract_faces(
        img_path=image,
        detector_backend=detector_backend,
        enforce_detection=False,
        align=False,
    )

    if len(face_objs) == 0:
        return None

    if len(face_objs) == 1:
        box = face_objs[0]["facial_area"]
    else:
        largest_face = max(
            face_objs,
            key=lambda face: face["facial_area"]["w"] * face["facial_area"]["h"],
        )
        box = largest_face["facial_area"]

    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    # cropped_face = image[y:y+h, x:x+w]
    # cv2.imwrite("/work/data/img.png", cropped_face)
    return x, y, w, h


def _run_inference(image_path: str, check_target_file_id: str):
    face_coords = _detect_and_crop_face(image_path)
    if face_coords is None:
        predictions = trainer.predict(
            binary_classification_non_face,
            datamodule=BinaryClassificationDataModule2(
                file_dir=os.path.dirname(image_path)
            ),
        )
        fake_score = predictions[0][0][0]

        return PostImageFileDetailRequestBody(
            check_target_file_id=check_target_file_id,
            bounding_boxes=[],
            fake_percentage=None,
            general_object_fake_percentage=fake_score,
        )

    x, y, w, h = face_coords

    predictions = trainer.predict(
        binary_classification_non_face,
        datamodule=BinaryClassificationDataModule(file_dir=os.path.dirname(image_path)),
    )
    fake_score = predictions[0][0][0]

    bounding_boxes = [
        PostImageFileDetailRequestBody.BoundingBox(
            x_position=x, y_position=y, width=w, height=h, fake_percentage=fake_score
        )
    ]

    return PostImageFileDetailRequestBody(
        check_target_file_id=check_target_file_id,
        bounding_boxes=bounding_boxes,
        fake_percentage=fake_score,
        general_object_fake_percentage=None,
    )


@log_execution_time
def image_estimation_pipeline(
    authentication: str,
    image_estimation_request_body: ImageEstimationRequestBody,
) -> Exception | None:
    logger.info(
        f"check_target_file_id: {image_estimation_request_body.check_target_file_id}の処理をスタートします。"
    )

    with TemporaryDirectory() as tmp_dir:
        tmp_file_path, err = _get_tmp_file_path_by_url(
            image_estimation_request_body.file_path, tmp_dir
        )
        if err is not None:
            return None, err
        _, err = request_backend(
            path=IMAGE_FILE_DETAIL_API_PATH,
            method="POST",
            authentication=authentication,
            data=_run_inference(
                image_path=tmp_file_path,
                check_target_file_id=image_estimation_request_body.check_target_file_id,
            ).model_dump(),
        )
        if err is not None:
            logger.error(err)

    logger.info(
        f"check_target_file_id: {image_estimation_request_body.check_target_file_id}の処理を終了しました。"
    )
    return None
