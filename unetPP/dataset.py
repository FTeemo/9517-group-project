import os

import cv2
import numpy as np
import torch
import torch.utils.data


# class Dataset(torch.utils.data.Dataset):
#     def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
#         """
#         Args:
#             img_ids (list): Image ids.
#             img_dir: Image file directory.
#             mask_dir: Mask file directory.
#             img_ext (str): Image file extension.
#             mask_ext (str): Mask file extension.
#             num_classes (int): Number of classes.
#             transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        
#         Note:
#             Make sure to put the files as the following structure:
#             <dataset name>
#             ├── images
#             |   ├── 0a7e06.jpg
#             │   ├── 0aab0a.jpg
#             │   ├── 0b1761.jpg
#             │   ├── ...
#             |
#             └── masks
#                 ├── 0
#                 |   ├── 0a7e06.png
#                 |   ├── 0aab0a.png
#                 |   ├── 0b1761.png
#                 |   ├── ...
#                 |
#                 ├── 1
#                 |   ├── 0a7e06.png
#                 |   ├── 0aab0a.png
#                 |   ├── 0b1761.png
#                 |   ├── ...
#                 ...
#         """
#         self.img_ids = img_ids
#         self.img_dir = img_dir
#         self.mask_dir = mask_dir
#         self.img_ext = img_ext
#         self.mask_ext = mask_ext
#         self.num_classes = num_classes
#         self.transform = transform

#     def __len__(self):
#         return len(self.img_ids)

#     def __getitem__(self, idx):
#         img_id = self.img_ids[idx]
#         # print('111111111111111',self.img_dir[0:])
#         img = cv2.imread(os.path.join(self.img_dir[0:], img_id + self.img_ext))
       
#         mask = []
#         for i in range(self.num_classes):
#             mask.append(cv2.imread(os.path.join(self.mask_dir[0:],
#                         img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
#         #数组沿深度方向进行拼接。
#         mask = np.dstack(mask)

#         if self.transform is not None:
#             augmented = self.transform(image=img, mask=mask)#这个包比较方便，能把mask也一并做掉
#             img = augmented['image']#参考https://github.com/albumentations-team/albumentations
#             mask = augmented['mask']
        
#         img = img.astype('float32') / 255
#         img = img.transpose(2, 0, 1)
#         # mask = mask.astype('float32') / 255
#         # mask = mask.transpose(2, 0, 1)
#         mask = torch.from_numpy(mask).long()
        
#         return img, mask, {'img_id': img_id}




class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, transform=None):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.img_dir, img_id + self.img_ext)
        mask_path = os.path.join(self.mask_dir, img_id + self.mask_ext)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 读取 mask，确保它是单通道
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 将 mask 中的灰度值映射到类别索引（例如：0, 1, 2, ...）
        mask = np.where(mask == 85, 1, mask)
        mask = np.where(mask == 170, 2, mask)
        mask = np.where(mask == 255, 3, mask)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        img = img.astype('float32') / 255.0
        img = img.transpose(2, 0, 1)  # (C, H, W)

        # 确保 mask 为 [height, width] 并转换为 Long 类型
        mask = torch.from_numpy(mask).long()

        return img, mask, {'img_id': img_id}
