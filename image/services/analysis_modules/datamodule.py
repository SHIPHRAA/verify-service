from typing import Optional
import pickle

import albumentations as A
import cv2
import pytorch_lightning as pl
from albumentations.pytorch.transforms import ToTensorV2
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader, Dataset
import random
from deepface import DeepFace
import time

import numpy as np
from sklearn import metrics

from .config import (
    BATCH_SIZE,
    IMG_SIZE,
    # NG_TEST_DIR,
    # NG_TRAIN_DIR,
    # NG_VALID_DIR,
    # OK_TEST_DIR,
    # OK_TRAIN_DIR,
    # OK_VALID_DIR,
    FFHQ_REAL_DIR,
    AiBOS_REAL_DIR_TR,
    AiBOS_REAL_DIR_VAL,
    SD_FACESWAP_DIR_TR,
    SD_FACESWAP_DIR_VAL,
    SD_IMG2IMG_DIR_TR,
    SD_IMG2IMG_DIR_VAL,
    GHOST_FACESWAP_DIR_TR,
    GHOST_FACESWAP_DIR_VAL,
    GHOST_FACESWAP_DIR2_TR,
    GHOST_FACESWAP_DIR2_VAL,
    COMMERCIAL_FACESWAP_DIR_TR,
    COMMERCIAL_FACESWAP_DIR_VAL,
    VIDRO_FACESWAPPED_DIR_TR,
    REAL_VIDRO_DIR_TR,
)
from .utils import list_file_paths


def validate_images(image_paths):
    valid_paths = []
    for path in image_paths:
        image = cv2.imread(path)
        if image is not None:
            valid_paths.append(path)
        else:
            print(f"Invalid image file: {path}")
    return valid_paths


def save_files_to_pickle(files, pickle_path):
    """
    ファイルリストをpickle形式で保存する関数。
    """
    try:
        with open(pickle_path, "wb") as f:
            pickle.dump(files, f)
        print(f"ファイルリストを保存しました: {pickle_path}")
    except Exception as e:
        print(f"エラー: ファイルリストを保存できませんでした -> {e}")


def load_files_from_pickle(pickle_path):
    """
    pickle形式で保存されたファイルリストをロードする関数。
    """
    try:
        with open(pickle_path, "rb") as f:
            files = pickle.load(f)
        print(f"ファイルリストをロードしました: {pickle_path}")
        return files
    except Exception as e:
        print(f"エラー: ファイルリストをロードできませんでした -> {e}")
        return []


class BinaryClassificationDataset(Dataset):
    """
    Albumentations変換を適用した画像データのカスタムデータセット。
    DeepFaceのextract_facesを使用して顔を検出し、最大のバウンディングボックスを使用します。
    パラメータ
    ----------
    images_list : List[Tuple[str, int]]
        ファイルパスとラベルを含むタプルのリスト。
    transform : A.Compose
        各画像に適用されるAlbumentations変換。
    detector_backend : str
        DeepFaceで使用する顔検出バックエンド（例: 'mtcnn', 'opencv', 'dlib'など）。
    メソッド
    -------
    __len__()
        データセット内のアイテム数を返します。
    __getitem__(item)
        指定されたインデックスの変換済み画像とそのラベルを返します。
    """

    def __init__(
        self,
        images_list: list[tuple[str, int]],
        transform: A.Compose,
        detector_backend: str = "yunet",
    ):
        super().__init__()
        self.images_list = images_list
        self.transform = transform
        self.detector_backend = detector_backend

    def detect_and_crop_face(self, image):
        """
        DeepFaceのextract_facesを使用して顔を検出し、最大のバウンディングボックスでクロップします。
        顔が検出されない場合はNoneを返します。
        """
        face_objs = DeepFace.extract_faces(
            img_path=image,
            detector_backend=self.detector_backend,
            enforce_detection=False,
            align=False,
        )

        if len(face_objs) == 0:
            return None

        if len(face_objs) == 1:
            # 1つの顔が検出された場合、その顔を使用
            box = face_objs[0]["facial_area"]
        else:
            # 複数の顔が検出された場合、最大のバウンディングボックスを選択
            largest_face = max(
                face_objs,
                key=lambda face: face["facial_area"]["w"] * face["facial_area"]["h"],
            )
            box = largest_face["facial_area"]

        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        cropped_face = image[y : y + h, x : x + w]
        cv2.imwrite("/app/data/img.png", cropped_face)
        return cropped_face

    def __len__(self) -> int:
        return len(self.images_list)

    def __getitem__(self, item: int) -> dict:
        path, label = self.images_list[item]
        image = cv2.imread(str(path))
        if "trump" in str(path):
            print(path)
        if image is None:
            print(f"Skipping invalid image: {path}")
            return self.__getitem__((item + 1) % len(self.images_list))  # 次の画像を試す

        # 顔検出とクロップ処理
        cropped_face = self.detect_and_crop_face(image)
        if cropped_face is None:
            print(f"No face detected, skipping: {path}")
            return self.__getitem__((item + 1) % len(self.images_list))  # 次の画像を試す
        # RGBに変換してトランスフォームを適用
        cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
        transformed_image = self.transform(image=cropped_face)["image"]
        return {"image": transformed_image, "label": label, "path": str(path)}


class BinaryClassificationDataModule(pl.LightningDataModule):
    """
    PyTorch Lightningを使用して画像データを読み込み、変換するためのデータモジュール。
    メソッド
    -------
    setup(stage: Optional[str] = None)
        トレーニング、検証、テストフェーズ用のデータセットを準備します。
    train_dataloader()
        トレーニングデータのDataLoaderを返します。
    val_dataloader()
        検証データのDataLoaderを返します。
    predict_dataloader()
        テストデータのDataLoaderを返します。
    """

    def __init__(self, file_dir="", num_of_train_ng_images=10):
        super().__init__()
        self.file_dir = file_dir
        self.num_of_train_ng_images = num_of_train_ng_images
        self.train_transform = A.Compose(
            [
                A.Resize(IMG_SIZE, IMG_SIZE),
                # A.OneOf(
                #     [
                #         A.Resize(IMG_SIZE, IMG_SIZE),# 元の画像サイズを保持
                #         A.Compose(  # パターン1: 縮小＋パディング
                #             [
                #                 A.Resize(IMG_SIZE // 2, IMG_SIZE // 2),  # サイズを1/2に縮小
                #                 A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=0, value=0),
                #             ],
                #             p=1.0,
                #         ),
                #         A.Compose(  # パターン2: サイズを2/3に縮小＋パディング
                #             [
                #                 A.Resize(IMG_SIZE * 2 // 3, IMG_SIZE * 2 // 3),  # サイズを2/3に縮小
                #                 A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=0, value=0),
                #             ],
                #             p=1.0,
                #         ),
                #         A.Compose(  # パターン3: サイズを3/4に縮小＋パディング
                #             [
                #                 A.Resize(IMG_SIZE * 3 // 4, IMG_SIZE * 3 // 4),  # サイズを3/4に縮小
                #                 A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=0, value=0),
                #             ],
                #             p=1.0,
                #         ),
                #     ],
                #     p=1.0,  # 全体として確率0.3で適用
                # ),
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.ShiftScaleRotate(),
                A.OneOf(
                    [
                        A.OpticalDistortion(p=0.3),
                        A.GridDistortion(p=0.1),
                    ],
                    p=0.2,
                ),
                A.Transpose(),
                A.GaussNoise(p=0.4),
                A.PiecewiseAffine(scale=(0.01, 0.03), p=0.3),
                A.CoarseDropout(num_holes=4, max_h_size=8, max_w_size=8, p=0.3),
                A.ImageCompression(quality_lower=20, quality_upper=100, p=0.4),
                A.OneOf(
                    [
                        A.MotionBlur(p=0.6),
                        A.MedianBlur(blur_limit=3, p=0.6),
                        A.Blur(blur_limit=3, p=0.6),
                    ],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.CLAHE(clip_limit=2),
                        A.RandomBrightnessContrast(),
                    ],
                    p=0.3,
                ),
                A.HueSaturationValue(p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(p=1.0),
            ]
        )
        self.valid_transform = A.Compose(
            [
                A.Resize(IMG_SIZE, IMG_SIZE),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(p=1.0),
            ]
        )

    def setup(self, stage: Optional[str] = None):
        """
        トレーニング、検証、テストフェーズ用のデータセットを準備します。
        パラメータ
        ----------
        stage : Optional[str], optional
            設定するステージ（'train', 'val', 'test'など）、デフォルトはNone
        """

        ok_train_path_list = []
        ng_train_path_list = []
        ok_valid_path_list = []
        ng_valid_path_list = []
        ok_test_path_list = list_file_paths(self.file_dir)
        ng_test_path_list = []

        print(ok_test_path_list)

        ok_valid_path_list = ok_test_path_list
        ng_valid_path_list = ng_test_path_list
        print(set(ng_train_path_list) & set(ng_valid_path_list))
        print(set(ok_train_path_list) & set(ok_valid_path_list))

        train_images_list = [(path, 0) for path in ok_train_path_list] + [
            (path, 1) for path in ng_train_path_list
        ]
        random.shuffle(train_images_list)
        valid_images_list = [(path, 0) for path in ok_valid_path_list] + [
            (path, 1) for path in ng_valid_path_list
        ]
        random.shuffle(valid_images_list)
        test_images_list = [(path, 0) for path in ok_test_path_list] + [
            (path, 1) for path in ng_test_path_list
        ]

        self.train_dataset = BinaryClassificationDataset(
            images_list=train_images_list, transform=self.train_transform
        )
        self.valid_dataset = BinaryClassificationDataset(
            images_list=valid_images_list, transform=self.valid_transform
        )
        self.test_dataset = BinaryClassificationDataset(
            images_list=test_images_list, transform=self.valid_transform
        )

    def train_dataloader(self) -> DataLoader:
        """
        トレーニングデータのDataLoaderを返します。
        Returns
        -------
        DataLoader
            トレーニングデータのDataLoaderオブジェクト
        """
        return DataLoader(
            self.train_dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        """
        検証データのDataLoaderを返します。
        Returns
        -------
        DataLoader
            検証データのDataLoaderオブジェクト
        """
        return DataLoader(
            self.valid_dataset, batch_size=BATCH_SIZE * 2, num_workers=4, shuffle=True
        )

    def predict_dataloader(self) -> DataLoader:
        """
        テストデータのDataLoaderを返します。
        Returns
        -------
        DataLoader
            テストデータのDataLoaderオブジェクト
        """
        return DataLoader(self.test_dataset, batch_size=BATCH_SIZE, num_workers=4)
