from torchvision import datasets, transforms
import torch
import torch.nn.functional as F
import numpy as np

class CocoDatasets:
    def __init__(self, batch_size=32):
        self.train_coco = datasets.CocoDetection("train2017", "annotations/instances_train2017.json", transform=transforms.ToTensor())
        self.valid_coco = datasets.CocoDetection("val2017", "annotations/instances_val2017.json", transform=transforms.ToTensor())
        self.batch_size = batch_size
        self.train_end = False
        self.valid_end = False
        self.valid_num = 0
        self.train_num = 0

    def bbox_to_norm(self, bbox, size):
        x1 = bbox[0] / size[0] * 256
        y1 = bbox[1] / size[1] * 256
        w = bbox[2] / size[0] * 256
        h = bbox[3] / size[1] * 256
        x2 = x1 + w
        y2 = y1 + h
        return x1, y1, x2, y2

    def core_area(self, x1, y1, x2, y2):
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        left = center_x - 0.1 * (x2 - x1)
        right = center_x + 0.1 * (x2 - x1)
        top = center_y - 0.1 * (y2 - y1)
        bottom = center_y + 0.1 * (y2 - y1)
        return left, top, right, bottom

    def init_one_input(self, inputs, targets):
        origin_w = inputs.shape[2]
        origin_h = inputs.shape[1]
        inputs = inputs.permute(0,2,1)
        bbox_list = [(anno['bbox'], anno['category_id']) for anno in targets]
        ground = [[False for _ in range(256)] for _ in range(256)]
        category = torch.zeros((256, 256, 91))
        boxes = torch.zeros((256, 256, 4))
        inputs = torch.unsqueeze(inputs, 0)
        inputs = F.interpolate(inputs, size=(256, 256))
        inputs = torch.squeeze(inputs)
        for bbox in bbox_list:
            # print(origin_w, origin_h)
            # print(bbox[0])
            norm_bbox = self.bbox_to_norm(bbox[0], (origin_w, origin_h))
            ca = self.core_area(*norm_bbox)
            # print(ca)
            for i in range(int(ca[0]), int(ca[2]) + 1):
                for j in range(int(ca[1]), int(ca[3]) + 1):
                    ground[i][j] = True
                    category[i][j][bbox[1]] = 1
                    for k in range(4):
                        boxes[i][j][k] = norm_bbox[k]
        ground = np.asarray(ground)
        ground = torch.from_numpy(ground)
        return inputs, ground, category, boxes

    def train_get_batch(self):
        if self.train_end:
            self.train_end = False
            self.train_num = 0
        batch_inputs = []
        batch_front_ground = []
        batch_category = []
        batch_boxes = []
        if len(self.train_coco) <= self.train_num + self.batch_size:
            self.train_end = True
        for i in range(self.train_num, self.train_num + self.batch_size):
            inputs, target = self.train_coco[i]
            inputs, ground, category, boxes = self.init_one_input(inputs, target)
            batch_inputs.append(inputs)
            batch_front_ground.append(ground)
            batch_category.append(category)
            batch_boxes.append(boxes)
        self.train_num += self.batch_size
        batch_inputs = torch.stack(batch_inputs, axis=0)
        batch_front_ground = torch.stack(batch_front_ground, axis=0)
        batch_category  = torch.stack(batch_category, axis=0)
        batch_boxes = torch.stack(batch_boxes, axis=0)
        return batch_inputs, batch_front_ground, batch_category, batch_boxes
    
    def valid_get_batch(self):
        if self.valid_end:
            self.valid_end = False
            self.valid_num = 0
        batch_inputs = []
        batch_front_ground = []
        batch_category = []
        batch_boxes = []
        if len(self.valid_coco) <= self.valid_num + self.batch_size:
            self.valid_end = True
        for i in range(self.valid_num, self.valid_num + self.batch_size):
            inputs, target = valid_coco[i]
            inputs, ground, category, boxes = self.init_one_input(inputs, target)
            batch_inputs.append(inputs)
            batch_front_ground.append(ground)
            batch_category.append(category)
            batch_boxes.append(boxes)
        batch_inputs = torch.stack(batch_inputs, axis=0)
        batch_front_ground = torch.stack(batch_front_ground, axis=0)
        batch_category  = torch.stack(batch_category, axis=0)
        batch_boxes = torch.stack(batch_boxes, axis=0)
        return batch_inputs, batch_front_ground, batch_category, batch_boxes




