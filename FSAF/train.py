import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from fpn import FPN
from fsaf import *
from utils import *
from torchvision.ops import nms
from dataloader import *
from anchors import Anchors
from loss import *
import os
from pycocotools.cocoeval import COCOeval
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

device = torch.device('cuda:0')

dataset_train = CocoDataset('.', set_name='train2017',
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
dataset_val = CocoDataset('.', set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))

sampler = AspectRatioBasedSampler(dataset_train, batch_size=2, drop_last=False)
dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

fpn = FPN()
net = Net()
anchors = Anchors()

fpn = fpn.to(device)
net = net.to(device)

criterion = AnchorBasedLoss()

optimizer1 = optim.Adam(fpn.parameters(), lr=1e-4)
optimizer2 = optim.Adam(net.parameters(), lr=1e-4)



def train():
    num = len(dataloader_train)
    done_num = 0
    fpn.train()
    net.train()
    for iter_num, data in enumerate(dataloader_train):
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        images = data['img'].cuda()
        annots = data['annot'].cuda()
        p_feature_maps = fpn(images)
        cb, rb, cf, rf = net(p_feature_maps)
        levels = []
        loss1 = 0
        level_id = [3, 4, 5, 6, 7]
        image_shape = torch.zeros(5, 2)
        strides = []
        width = images.shape[3]
        height = images.shape[2]
        batch_size = images.shape[0]
        for i, l in enumerate(level_id):
            strides.append(2 ** l)
            image_shape[i][0] = height / (2 ** l)
            image_shape[i][1] = width / (2 ** l)
        for i in range(batch_size):
            level, ls = level_select(cf[i], rf[i], annots[i], image_shape, strides)
            levels.append(level)
            loss1 += ls
        loss1 /= batch_size
        loss1.backward()
        optimizer1.step()
        optimizer2.step()
        p_feature_maps = fpn(images)
        cb, rb, cf, rf = net(p_feature_maps)
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        levels = torch.stack(levels, dim = 0)
        all_anchors = anchors(images)
        boxes = annots[:, :, :4]
        clss = annots[:, :, 4]
        loss2 = 0
        for i, l in enumerate(level_id):
            clss_level = torch.ones(clss.shape, dtype=torch.float).cuda() * -1
            clss_level_mask = levels.eq(i).cuda()
            clss_level = clss_level.masked_scatter(clss_level_mask, clss)
            level_annot = torch.cat((boxes, clss_level.unsqueeze(dim=2)), dim=2)
            anchor = all_anchors[i]
            cls_pred = cb[i]
            regr_pred = rb[i]
            l1, l2 = criterion(cls_pred, regr_pred, anchor, level_annot)
            l = l1 + l2
            loss2 = loss2 + l
        loss2 /= len(level_id)
        if loss2 != 0:
            loss2.backward()
            optimizer1.step()
            optimizer2.step()
        done_num += batch_size
        if done_num % 100 == 0:
            print("%d/%d................Loss:%.6f" %(done_num, num, loss2.item()))

def valid():
    fpn.eval()
    net.eval()
    results = []
    image_ids = []
    threshold = 0.05
    for index in range(len(dataset_val)):
        data = dataset_val[index]
        scale = data['scale']
        images = data['img'].permute(2, 0, 1).cuda()
        p_feature_maps = fpn(images.unsequeeze(dim=0))
        cb, rb, cf, rf = net(p_feature_maps)
        cb = torch.cat(cb, dim=1)
        rb = torch.cat(rb, dim=1)
        all_anchors = anchors(images)
        transformed_anchors = generate_predict_boxes(all_anchors, rb)
        transformed_anchors = clip_boxes(transformed_anchors, images)
        image_shape = torch.zeros(5, 2)
        level_id = [3, 4, 5, 6, 7]
        strides = []
        width = images.shape(3)
        height = images.shape(2)
        for i, l in enumerate(level_id):
            strides.append(2 ** l)
            image_shape[i][0] = height / (2 ** l)
            image_shape[i][1] = width / (2 ** l)
        transformed_anchors2 = anchor_free_predict_boxes(rf, strides, image_shape)
        transformed_anchors = torch.cat((transformed_anchors, transformed_anchors2), dim=1)
        transformed_anchors /= scale
        classifications = torch.cat((cb, cf), dim=1)
        scores = torch.max(classifications, dim=2, keepdim=True)[0]
        scores_over_thresh = (scores > 0.05)[0, :, 0]
        if scores_over_thresh.sum() == 0:
            continue
        classifications = classifications[:, scores_over_thresh, :]
        transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
        scores = scores[:, scores_over_thresh, :]

        anchors_nms_idx = nms(transformed_anchors[0, :, :], scores[0, :, 0], 0.5)
        nms_scores, nms_class = classifications[0, anchors_nms_idx, :].max(dim=1)
        transformed_anchors = transformed_anchors[0, anchors_nms_idx, :]

        if transformed_anchors.shape[0] == 0:
            continue
        transformed_anchors[:, 2] -= transformed_anchors[:, 0]
        transformed_anchors[:, 3] -= transformed_anchors[:, 1]

        # compute predicted labels and scores
        # for box, score, label in zip(boxes[0], scores[0], labels[0]):
        for box_id in range(transformed_anchors.shape[0]):
            score = float(nms_scores[box_id])
            label = int(nms_class[box_id])
            box = transformed_anchors[box_id, :]

            # scores are sorted, so we can break
            if score < threshold:
                break

            # append detection for each positively labeled class
            image_result = {
                'image_id': dataset_val.image_ids[index],
                'category_id': dataset_val.label_to_coco_label(label),
                'score': float(score),
                'bbox': box.tolist(),
            }

            # append detection to results
            results.append(image_result)

        # append image to list of processed images
        image_ids.append(dataset_val.image_ids[index])

        # print progress
        print('{}/{}'.format(index, len(dataset_val)))
    if not len(results):
        return

        # write output
    json.dump(results, open('{}_bbox_results.json'.format(dataset.set_name), 'w'), indent=4)

    # load results in COCO evaluation tool
    coco_true = dataset_val.coco
    coco_pred = coco_true.loadRes('{}_bbox_results.json'.format(dataset.set_name))

    # run COCO evaluation
    coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == "__main__":
    for epoch in range(20):
        print(">>>>>>>>>>>>>>>>")
        print("epochs %d/20" % epoch)
        train()
        torch.save(fpn, "fpn.pt")
        torch.save(net, "fsaf.pt")
        valid()







