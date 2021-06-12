# Prediction

import torch
from utils import predict, reverse_generate_label
from model_class import ObjectDetectionModel
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_image(img_tensor, annotation):
    fig, ax = plt.subplots(1)
    img = img_tensor.detach().cpu().numpy()
    coords = annotation["boxes"].long().detach().cpu().numpy()
    # labels = [reverse_generate_label(i.item()) for i in annotation['labels']]
    labels = annotation['labels']

    # Display the image
    ax.imshow(img.transpose(1, 2, 0))

    for i,box in enumerate(coords):
        xmin, ymin, xmax, ymax = box

        # Add diff colour according to label
        # 'without_mask' = 1 = red, 'with_mask' = 2 = green, 'mask_weared_incorrect' = 3 = blue
        if labels[i] == 1:
            edgecolour = 'r'
        elif labels[i] == 2:
            edgecolour = 'g'
        else:
            edgecolour = 'b'
        # Create a Rectangle patch
        rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor=edgecolour,
                                 facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()


test_img_path = "C:/Users/vigne/PycharmProjects/FaceMask/test_images/test_img4.jpg"
model_path = "C:/Users/vigne/PycharmProjects/FaceMask/saved_models/model.pt"

# Hyperparameters
BATCH_SIZE = 1
NUM_CLASSES = 4
device = 'cpu'

# input data
data_transform = transforms.Compose([transforms.ToTensor()])
img = plt.imread(test_img_path)
img = data_transform(img)
print(img.shape)

# model
model = ObjectDetectionModel(num_classes=NUM_CLASSES, backbone_type='resnet50')
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
model.to(device)
preds = model([img.to(device)])

print("Prediction")
print([reverse_generate_label(i.item()) for i in preds[0]['labels']])
print(preds)
plot_image(img, preds[0])
