import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from fpn import FPN
from sapd import *
from utils import *
from torchvision.ops import nms
from dataloader import *
from loss import *
import os
from pycocotools.cocoeval import COCOeval
import json
from sapd_layers import *

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

device = torch.device('cuda:0')

dataset_train = CocoDataset('../dataset', set_name='val2017',
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
dataset_val = CocoDataset('../dataset', set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))

sampler = AspectRatioBasedSampler(dataset_train, batch_size=2, drop_last=False)
dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

fpn = FPN()
net = Net()
MetaSelectionNet = MetaSelectionNet()

fpn = fpn.to(device)
net = net.to(device)
MetaSelectionNet = MetaSelectionNet.to(device)

wiou = IoULossWithWeightAndMask()
wfocal = FocalLossWithWeightAndMask()

optimizer1 = optim.Adam(fpn.parameters(), lr=1e-4)
optimizer2 = optim.Adam(net.parameters(), lr=1e-4)
optimizer3 = optim.Adam(MetaSelectionNet.parameters(), lr=1e-4)


def train():
    num = len(dataloader_train) * 2
    done_num = 0
    fpn.train()
    net.train()
    MetaSelectionNet.train()
    for _, data in enumerate(dataloader_train):
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        images = data['img'].cuda()
        annots = data['annot'].cuda()
        num_gt_boxes_input = []
        for annot in annots:
            num_gt_boxes_input.append(annot.shape[0])
        p_feature_maps = fpn(images)
        cls_pred, regr_pred = net(p_feature_maps)
        # MetaSelectionLoss
        level_id = [3, 4, 5, 6, 7]
        image_shape = torch.zeros(5, 2)
        strides = []
        width = images.shape[3]
        height = images.shape[2]
        batch_size = images.shape[0]
        for i, l in enumerate(level_id):
            strides.append(2 ** l)
            image_shape[i][0] = height / (1.0 * (2 ** l))
            image_shape[i][1] = width / (1.0 * (2 ** l))

        # meta_select
        meta_select_input, gt_boxes_batch_ids = MetaSelectInput()(annots, p_feature_maps)
        meta_select_pred = MetaSelectionNet(meta_select_input)
        meta_select_target = MetaSelectTarget(strides=strides)(cls_pred, regr_pred, image_shape, annots)
        meta_select_loss = nn.CrossEntropyLoss(meta_select_pred, meta_select_target)

        meta_select_weight = MetaSelectWeight(soft_select=True
                                                )(meta_select_pred, gt_boxes_batch_ids, num_gt_boxes_input) #TODO!!!!

        cls_target, regr_target = SAPDTarget()(annots, meta_select_weight, image_shape)
        loss_with_weight_and_mask = 0
        l1 = FocalLossWithWeightAndMask(cls_target, cls_pred)
        l2 = IoULossWithWeightAndMask(regr_target, regr_pred)
        loss_with_weight_and_mask = l1 + l2
        #lambda == 0.1 in paper
        total_loss = 0.1 * meta_select_loss + loss_with_weight_and_mask
        if total_loss != 0:
            total_loss.backward()
            optimizer1.step()
            optimizer2.step()
            optimizer3.step()
        done_num += batch_size
        if done_num % 100 == 0:
            print("%d/%d................Loss:%.6f" %(done_num, num, total_loss.item()))

def valid():
    fpn.eval()
    net.eval()
    MetaSelectionNet.eval()
    results = []
    image_ids = []
    threshold = 0.05
    for index in range(len(dataset_val)):
        data = dataset_val[index]
        scale = data['scale']
        images = data['img'].permute(2, 0, 1).cuda()
        images = images.unsqueeze(dim=0)
        p_feature_maps = fpn(images)
        cls_pred, regr_pred = net(p_feature_maps)
        cls_pred = torch.cat(cls_pred, dim=1)
        regr_pred = torch.cat(cls_pred, dim=1)
        location, strides = Location(p_feature_maps)
        boxes = regress_boxes(location, strides, regr_pred)
        transformed_anchors = clip_boxes(images, boxes)
        transformed_anchors /= scale
        classifications = cls_pred
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
        print(transformed_anchors.shape)

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
        print("no results!")
        return

        # write output
    json.dump(results, open('{}_bbox_results.json'.format(dataset_val.set_name), 'w'), indent=4)

    # load results in COCO evaluation tool
    coco_true = dataset_val.coco
    coco_pred = coco_true.loadRes('{}_bbox_results.json'.format(dataset_val.set_name))

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
        torch.save(MetaSelectionNet, "metaselection.pt")
        valid()







