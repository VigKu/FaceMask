from torch.utils.data import Dataset
import os
from PIL import Image
from utils import generate_target
# import torch
# from torchvision import transforms


class MaskDataset(Dataset):
    def __init__(self, transforms,
                 img_path="C:/Users/vigne/PycharmProjects/FaceMask/images/",
                 label_path="C:/Users/vigne/PycharmProjects/FaceMask/annotations/"):
        super(MaskDataset, self).__init__()
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.img_path = img_path
        self.label_path = label_path
        self.imgs = list(sorted(os.listdir(self.img_path)))
        # self.labels = list(sorted(os.listdir("/kaggle/input/face-mask-detection/annotations/")))

    def __getitem__(self, idx):
        # load images ad masks
        file_image = 'maksssksksss' + str(idx) + '.png'
        file_label = 'maksssksksss' + str(idx) + '.xml'
        img_path = os.path.join(self.img_path, file_image)
        label_path = os.path.join(self.label_path, file_label)
        img = Image.open(img_path).convert("RGB")
        # Generate Label
        target = generate_target(idx, label_path)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)


# data_transform = transforms.Compose([
#         transforms.ToTensor(),
#     ])
# collate_fn required to stitch lists or tuples tgt. Else, it uses torch.stack which only works for tensors.
# our dataset produces a list of tuples and thus this function is needed to stitch  a few tuples to from one batch.
# def collate_fn(batch):
#    return tuple(zip(*batch))
# dataset = MaskDataset(data_transform)
# data_loader = torch.utils.data.DataLoader(dataset, batch_size=8, collate_fn=collate_fn)
# print(dataset[0])
