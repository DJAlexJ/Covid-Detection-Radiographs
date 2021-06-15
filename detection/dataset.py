import numpy as np
import torch
from torch.utils.data import Dataset

import albumentations as A
import cv2


class DetectionDataset(Dataset):

    def __init__(self, image_ids, df, transforms=None, test=False):
        super().__init__()

        self.image_ids = image_ids
        self.df = df
        self.file_names = df['jpg_path'].values
        self.transforms = transforms
        self.test = test

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]

        image, boxes, labels = self.load_image_and_boxes(index)

        target = dict()
        target['boxes'] = boxes
        target['labels'] = torch.tensor(labels)
        target['image_id'] = torch.tensor([index])

        if self.transforms:
            sample = self.transforms(**{
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            })
            image = sample['image']
            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)

        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def load_image_and_boxes(self, index):
        image_id = self.image_ids[index]
        image = cv2.imread(self.file_names[index], cv2.IMREAD_COLOR).copy().astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        records = self.df[self.df['id'] == image_id]
        boxes = []
        for bbox in records[['frac_xmin', 'frac_ymin', 'frac_xmax', 'frac_ymax']].values:
            bbox = np.clip(bbox, 0, 1.0)
            temp = A.convert_bbox_from_albumentations(bbox, 'pascal_voc', image.shape[0], image.shape[0])
            boxes.append(temp)
        '''
        [0: 'negative', 1: 'not negative']
        '''
        # '''
        # [0: 'atypical', 1: 'indeterminate', 2: 'negative', 3: 'typical']
        # '''
        labels = records['integer_label'].values
        return image, boxes, labels
