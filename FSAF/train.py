from model import Net
from data import CocoDatasets
from loss import FocalLoss, IoULoss
import torch.optim as optim
import torch
import os


def train(model, datasets, optimizer, criterion1, criterion2, device):
    model.train()
    total_loss = 0
    count = 0
    total = len(datasets.train_coco)
    curr = 0
    while not datasets.train_end:
        inputs, ground, category, boxes = datasets.train_get_batch()
        curr += inputs.shape[0]
        inputs = inputs.to(device)
        ground = ground.to(device)
        category = category.to(device)
        boxes = boxes.to(device)
        optimizer.zero_grad()
        c1, c2, c3, c4, r1, r2, r3, r4 = model(inputs)
        category = category[ground,:]
        boxes = boxes[ground,:]
        c1 = c1[ground,:]
        c2 = c2[ground,:]
        c3 = c3[ground,:]
        c4 = c4[ground,:]
        r1 = r1[ground,:]
        r2 = r2[ground,:]
        r3 = r3[ground,:]
        r4 = r4[ground,:]
        loss1 = criterion1(c1, category) + criterion2(r1, boxes)
        loss2 = criterion1(c2, category) + criterion2(r2, boxes)
        loss3 = criterion1(c3, category) + criterion2(r3, boxes)
        loss4 = criterion1(c4, category) + criterion2(r4, boxes)
        #print(loss1.mean(), loss2.mean(), loss3.mean(), loss4.mean())
        loss = torch.min(loss1, loss2)
        loss = torch.min(loss, loss3)
        loss = torch.min(loss, loss4)
        loss = loss.mean()
        loss.backward()
        total_loss += loss.item()
        count += 1
        if curr % 100 == 0:
            print("%d/%d..................Loss:%f" % (curr, total, loss.item()))
    return total_loss / count

def valid(model, datasets, criterion1, criterion2, device):
    model.eval()
    total_loss = 0
    count = 0
    while not datasets.valid_end:
        inputs, ground, category, boxes = datasets.valid_get_batch()
        category = category[ground,:]
        boxes = boxes[ground,:]
        inputs = inputs.to(device)
        ground = ground.to(device)
        category = category.to(device)
        boxes = boxes.to(device)
        c1, c2, c3, c4, r1, r2, r3, r4 = model(inputs)
        c1 = c1[ground,:]
        c2 = c2[ground,:]
        c3 = c3[ground,:]
        c4 = c4[ground,:]
        r1 = r1[ground,:]
        r2 = r2[ground,:]
        r3 = r3[ground,:]
        r4 = r4[ground,:]
        loss1 = criterion1(c1, category) + criterion2(r1, boxes)
        loss2 = criterion1(c2, category) + criterion2(r2, boxes)
        loss3 = criterion1(c3, category) + criterion2(r3, boxes)
        loss4 = criterion1(c4, category) + criterion2(r4, boxes)
        loss = torch.min(loss1, loss2)
        loss = torch.min(loss, loss3)
        loss = torch.min(loss, loss4)
        loss = loss.mean()
        total_loss += loss.item()
    return total_loss / count


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "2"
    num_epochs = 20
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net()
    model = model.to(device)
    lr = 0.001
    batch_size = 2
    datasets = CocoDatasets(batch_size=batch_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    criterion1 = FocalLoss(alpha=0.25)
    criterion2 = IoULoss()
    lowest_loss = 1e10

    for i in range(num_epochs):
        print(">>>>>>>>>>>>>>>>>>>>>>>")
        print("epochs:%d/%d" % (i, num_epochs))
        train_loss = train(model, datasets, optimizer, criterion1, criterion2, device)
        valid_loss = valid(model, datasets, criterion1, criterion2, device)
        print("train loss:%f", train_loss)
        print("valid loss:%f", valid_loss)
        if valid_loss < lowest_loss:
            lowest_loss = valid_loss
            torch.save(model, "best_model.pt")

