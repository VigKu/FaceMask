import torch
import cv2
import numpy as np
from model_class import ObjectDetectionModel
import streamlit as st


@st.cache
def load_model(model_path, num_classes, device, backbone='resnet50'):
    model = ObjectDetectionModel(num_classes=num_classes, backbone_type=backbone)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


@st.cache(allow_output_mutation=True)
def predict(imgs, model):
    with torch.no_grad():
        return model(imgs)


def plot_image2(img, annotation, output_path):

    coords = annotation["boxes"].long().detach().cpu().numpy()
    # labels = [reverse_generate_label(i.item()) for i in annotation['labels']]
    labels = annotation['labels']

    for i,box in enumerate(coords):
        xmin, ymin, xmax, ymax = box

        # Add diff colour according to label
        # 'without_mask' = 1 = red, 'with_mask' = 2 = green, 'mask_weared_incorrect' = 3 = blue
        if labels[i] == 1:
            edgecolour = (255, 0, 0)
        elif labels[i] == 2:
            edgecolour = (0, 255, 0)
        else:
            edgecolour = (0, 0, 255)
        # Create a Rectangle patch
        img = cv2.rectangle(img,(xmin.item(),ymax.item()),(xmax.item(),ymin.item()),edgecolour,6)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, img)
