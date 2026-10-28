import sys
sys.path.insert(0, "/workspace/LightM-UNet/enet")
from nnunetv2.training.dataloading.utils import unpack_dataset

if __name__ == "__main__":
    unpack_dataset(
        "/workspace/LightM-UNet/data/nnUNet_preprocessed/Dataset510_ARCADE_256_4c/nnUNetPlans_2d",
        unpack_segmentation=True, overwrite_existing=False, num_processes=8,
    )
    print("UNPACK_DONE")
