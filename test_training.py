# Test with training images

import torch
from utils import predict, reverse_generate_label, plot_image
from model_class import ObjectDetectionModel
from dataset_class import MaskDataset
from torchvision import transforms
from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches


def collate_fn(batch):
    return tuple(zip(*batch))


img_path = "C:/Users/vigne/PycharmProjects/FaceMask/images/"
label_path = "C:/Users/vigne/PycharmProjects/FaceMask/annotations/"
model_path = "C:/Users/vigne/PycharmProjects/FaceMask/saved_models/model.pt"

# Hyperparameters
BATCH_SIZE = 4
NUM_CLASSES = 3
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

# input data
data_transform = transforms.Compose([transforms.ToTensor()])
# collate_fn required to stitch lists or tuples tgt. Else, it uses torch.stack which only works for tensors.
# our dataset produces a list of tuples and thus this function is needed to stitch  a few tuples to from one batch.
dataset = MaskDataset(data_transform)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

for imgs, annotations in data_loader:
    imgs = list(img.to(device) for img in imgs)
    annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
    break
# print(imgs[0].shape)
# model
model = ObjectDetectionModel(num_classes=NUM_CLASSES, backbone_type='resnet50')
# model.load_state_dict(torch.load('model.pt'))
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()
model.to(device)

preds = model(imgs)
# print(preds)
# print(annotations)

print("Prediction")
plot_image(imgs[2], preds[2])
print("Target")
plot_image(imgs[2], annotations[2])
