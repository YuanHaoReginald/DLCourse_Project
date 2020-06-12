import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from utils import *
from torchvision.ops import nms
from anchors import Anchors
import os
import json
import cv2
from dataloader import *
from PIL import Image


os.environ["CUDA_VISIBLE_DEVICES"] = "7"

dataset = CocoDataset('../dataset', set_name='val2017', transform=transforms.Compose([Normalizer(), Resizer()]))

device = torch.device('cuda:0')

anchors = Anchors()
fpn = torch.load("fpn.pt")
net = torch.load("fsaf.pt")

fpn = fpn.to(device)
net = net.to(device)

fpn.eval()
net.eval()

def normalizer(image):
    mean = np.array([[[0.485, 0.456, 0.406]]])
    std = np.array([[[0.229, 0.224, 0.225]]])
    return (image - mean) / std

def pad(image):
    rows, cols, cns = image.shape
    pad_w = 32 - rows%32
    pad_h = 32 - cols%32
    new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
    new_image[:rows, :cols, :] = image.astype(np.float32)
    return new_image

def run(image_path):
    img = Image.open(image_path)
    img = np.asarray(img)
    img = normalizer(img)
    img = pad(img)
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    img = img.permute(2, 0, 1)
    images = img.unsqueeze(0)
    img = cv2.imread(image_path)
    threshold = 0.5
    p_feature_maps = fpn(images)
    cb, rb, cf, rf = net(p_feature_maps)
    cb = torch.cat(cb, dim=1)
    rb = torch.cat(rb, dim=1)
    all_anchors = anchors(images)
    all_anchors = torch.cat(all_anchors, dim=1)
    transformed_anchors = generate_predict_boxes(all_anchors, rb)
    transformed_anchors = clip_boxes(transformed_anchors, images)
    image_shape = torch.zeros(5, 2)
    level_id = [3, 4, 5, 6, 7]
    strides = []
    width = images.shape[3]
    height = images.shape[2]
    for i, l in enumerate(level_id):
        strides.append(2 ** l)
        image_shape[i][0] = height / (1.0 * (2 ** l))
        image_shape[i][1] = width / (1.0 * (2 ** l))
    transformed_anchors2 = anchor_free_predict_boxes(rf, strides, image_shape).unsqueeze(dim=0)
    transformed_anchors = torch.cat((transformed_anchors, transformed_anchors2), dim=1)
    classifications = torch.cat((cb, cf), dim=1)
    scores = torch.max(classifications, dim=2, keepdim=True)[0]
    print(torch.max(scores))
    scores_over_thresh = (scores > threshold)[0, :, 0]
    if scores_over_thresh.sum() == 0:
        print("scores is too low")
        return
    classifications = classifications[:, scores_over_thresh, :]
    transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
    scores = scores[:, scores_over_thresh, :]

    anchors_nms_idx = nms(transformed_anchors[0, :, :], scores[0, :, 0], 0.5)
    nms_scores, nms_class = classifications[0, anchors_nms_idx, :].max(dim=1)
    transformed_anchors = transformed_anchors[0, anchors_nms_idx, :]

    if transformed_anchors.shape[0] == 0:
        print("no anchors")
        return
    for box_id in range(transformed_anchors.shape[0]):
        score = float(nms_scores[box_id])
        label = int(nms_class[box_id])
        box = transformed_anchors[box_id, :]
        box = box.tolist()
        if score < threshold:
            break
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        label_name = dataset.labels[label]
        caption = "%s:%.3f" % (label_name, score)
        cv2.putText(img, caption, (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(img, caption, (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
        cv2.rectangle(img, (x1, y1, x2, y2), color=(0, 0, 255), thickness=2)
    filename = image_path.split('/')[-1]
    print(transformed_anchors.shape)
    cv2.imwrite("ret/%s" % (filename), img)

if __name__ == "__main__":
    for path, dir_list, file_list in os.walk("test"):
        for filename in file_list:
            print(filename)
            run(os.path.join(path, filename))
