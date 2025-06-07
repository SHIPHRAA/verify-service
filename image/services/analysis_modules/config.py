INFERENCE_CHECKPOINT_FACE = (
    "ckpt/face-deepfake-detection-epoch=07-val_metric=1.000.ckpt"
)
INFERENCE_CHECKPOINT_NON_FACE = (
    "ckpt/face-deepfake-detection-epoch=07-val_metric=1.000.ckpt"
)
BACKBONE = "efficientnet_b0"
IMG_SIZE = 224
BATCH_SIZE = 128
FFHQ_REAL_DIR = "/dataset/from_cropped_1p3b"
AiBOS_REAL_DIR_TR = "/dataset/PREPROCESSED_FACE_DATASET/train_real"
AiBOS_REAL_DIR_VAL = "/dataset/PREPROCESSED_FACE_DATASET/test_real"
SD_FACESWAP_DIR_TR = "/dataset/FACE_SWAP_DATASET/train_fake_SD_faceswap"
SD_FACESWAP_DIR_VAL = "/dataset/FACE_SWAP_DATASET/test_fake_SD_faceswap"
GHOST_FACESWAP_DIR_TR = "/dataset/GHOST_DATASET/train"
GHOST_FACESWAP_DIR_VAL = "/dataset/GHOST_DATASET/test"
GHOST_FACESWAP_DIR2_TR = "/dataset/GHOST_DATASET2/train"
GHOST_FACESWAP_DIR2_VAL = "/dataset/GHOST_DATASET2/test"
COMMERCIAL_FACESWAP_DIR_TR = "/dataset/FACESWAP_COMMERCIAL_DATASET_20241201_CP/train"
COMMERCIAL_FACESWAP_DIR_VAL = "/dataset/FACESWAP_COMMERCIAL_DATASET_20241201_CP/test"
SD_IMG2IMG_DIR_TR = "/dataset/IMG_2_IMG_FACE_DATASET/train_real"
SD_IMG2IMG_DIR_VAL = "/dataset/IMG_2_IMG_FACE_DATASET/test_real"
VIDRO_FACESWAPPED_DIR_TR = "/dataset/FRAMES_FROM_VIDEO_FACESWAPPED_DATASET"
REAL_VIDRO_DIR_TR = "/dataset/FRAMES_FROM_REAL_HUMAN_VIDEOS_video-processor-dg_CP"
