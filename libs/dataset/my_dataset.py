
import os
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2



class Dataset_multiclass(Dataset):
    def __init__(self, csv_file, transform=None, image_shape=None):
        if isinstance(csv_file, pd.DataFrame):
            df = csv_file
        else:
            assert os.path.exists(csv_file), f'csv file {csv_file} does not exists'
            df = pd.read_csv(csv_file)
        assert len(df) > 0, 'csv file is empty!'

        self.img_files = df['images'].tolist()  # or df.values[:, 0].tolist()
        self.labels = df['labels'].tolist()

        self.image_shape = image_shape
        self.transform = transform

    def __getitem__(self, index):
        file_img = self.img_files[index]
        assert os.path.exists(file_img), f'image file {file_img} does not exists'
        image = cv2.imread(file_img)  # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        assert image is not None, f'{file_img} error.'
        label = int(self.labels[index])  # if using df self.df.iloc[index][1]

        if self.transform is None:
            if (self.image_shape is not None) and (image.shape[:2] != self.image_shape[:2]):
                self.transform = A.Compose([
                    A.Resize(self.image_shape[0], self.image_shape[1]),  # (height,weight),
                    ToTensorV2(),
                ])
            else:
                self.transform = A.Compose([ToTensorV2()])  # ToTensorV2 HWC -> CHW  do not normalized to 0-1

        transformed = self.transform(image=image)
        image = transformed['image'].float()  # torch.uint8 -> torch.float32
        # adding  .float() here or during training or inference using inputs.float()
        # otherwise,  "Input type (torch.cuda.ByteTensor) and weight type (torch.cuda.HalfTensor) should be the same."

        return image, label

    def __len__(self):
        return len(self.img_files)



