import sys
sys.path.insert(0, "/workspace/LightM-UNet/enet")
sys.argv = ["nnUNetv2_preprocess", "-d", "510", "-c", "2d", "-np", "8"]
from nnunetv2.experiment_planning.plan_and_preprocess_entrypoints import preprocess_entry

if __name__ == "__main__":
    preprocess_entry()
