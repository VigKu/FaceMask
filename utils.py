### IMPT PIP INSTALLS
# !pip install https://github.com/ufoym/imbalanced-dataset-sampler/archive/master.zip
# !pip install bs4
# !pip install lxml
# conda install -> albumentations, pytorch-lightning
import cv2
import torch
# import numpy as np
# import pandas as pd
# import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from bs4 import BeautifulSoup


def generate_box(obj):
    xmin = int(obj.find('xmin').text)
    ymin = int(obj.find('ymin').text)
    xmax = int(obj.find('xmax').text)
    ymax = int(obj.find('ymax').text)

    return [xmin, ymin, xmax, ymax]


def generate_label(obj):
    map_dict = {"background": 0, "without_mask": 1, "with_mask": 2, "mask_weared_incorrect": 3}
    return map_dict[obj.find('name').text]


def reverse_generate_label(label):
    map_dict = {0: "background", 1: "without_mask", 2: "with_mask", 3: "mask_weared_incorrect"}
    return map_dict[label]


def generate_target(image_id, file):
    with open(file) as f:
        data = f.read()
        soup = BeautifulSoup(data, 'xml')
        objects = soup.find_all('object')

        # num_objs = len(objects)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        labels = []
        for i in objects:
            boxes.append(generate_box(i))
            labels.append(generate_label(i))
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Labels (In my case, I only one class: target class or background)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # Tensorise img_id
        img_id = torch.tensor([image_id])
        # Annotation is in dictionary format
        target = {"boxes": boxes, "labels": labels, "image_id": img_id}

        return target


def train(num_epochs, model, data_loader, lr, momentum, weight_decay, path, device='cpu'):
    model.to(device)

    # parameters
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    min_epoch_loss = 10 ** 5
    model_state_dict = None
    for epoch in range(num_epochs):
        model.train()
        i = 0
        epoch_loss = 0
        epoch_loss_list = []

        for imgs, annotations in data_loader:
            i += 1
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            loss_dict = model([imgs[0]], [annotations[0]])
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss_list.append(losses)
            epoch_loss += losses

        if epoch_loss < min_epoch_loss:
            min_epoch_loss = epoch_loss
            model_state_dict = model.state_dict()
        print(f'Epoch: {epoch + 1}/{num_epochs}, Loss: {epoch_loss}')
    torch.save(model_state_dict, path)


def predict(imgs, model):
    with torch.no_grad():
        model.eval()
        return model(imgs)


def plot_image(img_tensor, annotation):
    fig, ax = plt.subplots(1)
    img = img_tensor.detach().cpu().numpy()
    coords = annotation["boxes"].long().detach().cpu().numpy()

    # Display the image
    ax.imshow(img.transpose(1, 2, 0))

    for box in coords:
        xmin, ymin, xmax, ymax = box

        # Create a Rectangle patch
        rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor='r',
                                 facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()