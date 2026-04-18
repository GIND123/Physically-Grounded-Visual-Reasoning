"""
InpaintingTripletDataset — PyTorch Dataset for LoRA training.

Each sample is a (normal, mask, defect) triplet from MVTec AD.
The model learns: given normal image + defect mask → produce defect in masked region.
"""

import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class InpaintingTripletDataset(Dataset):
    """
    Loads (normal, mask, defect) triplets from a manifest JSON file.

    Manifest format (built by pipeline/stage1.py::build_triplets):
        {
            "category": "bottle",
            "triplets": [
                {
                    "normal_image": "/path/to/normal.png",
                    "mask_image":   "/path/to/mask.png",
                    "defect_image": "/path/to/defect.png",
                    "defect_type":  "broken_large",
                    ...
                },
                ...
            ]
        }
    """

    def __init__(self, manifest_path: str, resolution: int = 512):
        with open(manifest_path) as f:
            data = json.load(f)
        self.triplets = data["triplets"]
        self.category = data["category"]
        self.resolution = resolution

        self.img_transform = transforms.Compose([
            transforms.Resize(
                (resolution, resolution),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),   # SD expects [-1, 1]
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(
                (resolution, resolution),
                interpolation=transforms.InterpolationMode.NEAREST,
            ),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> dict:
        t = self.triplets[idx]

        normal = Image.open(t["normal_image"]).convert("RGB")
        defect = Image.open(t["defect_image"]).convert("RGB")
        mask   = Image.open(t["mask_image"]).convert("L")

        normal_tensor = self.img_transform(normal)     # [-1, 1], 3×H×W
        defect_tensor = self.img_transform(defect)     # [-1, 1], 3×H×W  (TARGET)
        mask_tensor   = self.mask_transform(mask)      # [0, 1], 1×H×W

        # Binarize mask: >0.5 → defect region
        mask_tensor = (mask_tensor > 0.5).float()

        # SD inpainting expects the normal image with defect region zeroed out
        masked_normal = normal_tensor * (1 - mask_tensor)

        defect_type = t["defect_type"].replace("_", " ")
        category    = t["category"].replace("_", " ")
        prompt = (
            f"a {defect_type} defect on a {category}, "
            f"industrial inspection photograph, macro, high resolution"
        )

        return {
            "normal":        normal_tensor,
            "defect":        defect_tensor,
            "mask":          mask_tensor,
            "masked_normal": masked_normal,
            "prompt":        prompt,
            "defect_type":   t["defect_type"],
        }
