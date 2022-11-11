import cv2
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def img_to_tensor(img_file, image_shape=(299, 299)):
    if isinstance(img_file, str):
        image = cv2.imread(img_file)
    else:
        image = img_file

    if (image_shape is not None) and (image.shape[:2] != image_shape[:2]):
        transform = A.Compose([
            A.Resize(image_shape[0], image_shape[1]),  # (height,weight),
            ToTensorV2(),
        ])
    else:
        transform = A.Compose([ToTensorV2()])  #ToTensorV2 HWC -> CHW, return torch.uint8 do not normalized to 0-1

    image = transform(image=image)['image'].float()
    tensor = torch.unsqueeze(image, dim=0)  # (C,H,W) -> (B,C,H,W) B=1

    return tensor

