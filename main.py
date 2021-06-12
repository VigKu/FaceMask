# Main function for training model

import torch
from utils import train, predict
from model_class import ObjectDetectionModel
from dataset_class import MaskDataset
from torchvision import transforms
from torch.utils.data import DataLoader


def collate_fn(batch):
    return tuple(zip(*batch))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    img_path = "C:/Users/vigne/PycharmProjects/FaceMask/images/"
    label_path = "C:/Users/vigne/PycharmProjects/FaceMask/annotations/"
    model_path = "C:/Users/vigne/PycharmProjects/FaceMask/saved_models/model.pt"

    # Hyperparameters
    BATCH_SIZE = 8
    NUM_EPOCHS = 30
    LR = 0.005
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005
    NUM_CLASSES = 4  # include background as label 0 for FasterRCNN
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # input data
    data_transform = transforms.Compose([transforms.ToTensor()])
    # collate_fn required to stitch lists or tuples tgt. Else, it uses torch.stack which only works for tensors.
    # our dataset produces a list of tuples and thus this function is needed to stitch  a few tuples to from one batch.
    dataset = MaskDataset(data_transform)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    # model
    model = ObjectDetectionModel(num_classes=NUM_CLASSES, backbone_type='resnet50')

    # train model
    train(num_epochs=NUM_EPOCHS,
          model=model,
          data_loader=data_loader,
          lr=LR,
          momentum=MOMENTUM,
          weight_decay=WEIGHT_DECAY,
          path=model_path,
          device=device)
