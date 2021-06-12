import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class ObjectDetectionModel(nn.Module):
    def __init__(self, num_classes, backbone_type='resnet50'):
        super(ObjectDetectionModel, self).__init__()
        if backbone_type == 'resnet50':
            # load an instance segmentation model pre-trained pre-trained on COCO
            mdl = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            # get number of input features for the classifier
            in_features = mdl.roi_heads.box_predictor.cls_score.in_features
            # replace the pre-trained head with a new one
            mdl.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            self.model = mdl

    def forward(self, x, y=None):
        return self.model(x, y)


# model = ObjectDetectionModel(4)
# print(model)
