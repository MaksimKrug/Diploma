import os
from glob import glob
from xml.dom import minidom

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm.notebook import tqdm

CLASSES = {
    "background": [0, 0, 0],
    "lips": [255, 0, 0],
    "eye": [0, 255, 0],
    "nose": [0, 0, 255],
    "hair": [255, 255, 0],
    "eyebrows": [255, 0, 255],
    "teeth": [255, 255, 255],
    "face": [128, 128, 128],
    "ears": [0, 255, 255],
    "glasses": [0, 128, 128],
    "beard": [255, 192, 192],
}
label_list = [
    "skin",
    "nose",
    "eye_g",
    "l_eye",
    "r_eye",
    "l_brow",
    "r_brow",
    "l_ear",
    "r_ear",
    "mouth",
    "u_lip",
    "l_lip",
    "hair",
    "hat",
    "ear_r",
    "neck_l",
    "neck",
    "cloth",
]
map_classes = {
    "background": ["_cloth", "_hat"],
    "lips": ["_l_lip", "_u_lip"],
    "eye": ["_eye_g", "_l_eye", "_r_eye"],
    "nose": ["_nose"],
    "hair": ["_hair"],
    "eyebrows": ["_l_brow", "_r_brow"],
    "teeth": ["_mouth"],
    "face": ["_neck", "_neck_l", "_skin"],
    "ears": ["_ear_r", "_r_ear", "_l_ear"],
    "glasses": [],
    "beard": [],
}


def celeb2npy():
    """
    Read Celeb images and annotations and save them to .npy
    """
    # img paths
    imgs_path = glob("../data/dataset_celebs/CelebA-HQ-img/*.jpg")

    # iterate over imgs
    for img_path in tqdm(imgs_path):
        # get list of img labels
        img_name = os.path.basename(img_path).replace(".jpg", "")
        # get img labels
        img_labels = []
        for label_name in label_list:
            temp_label = os.path.join(
                "../data/dataset_celebs/CelebA-HQ-masks/",
                img_name + "_" + label_name + ".png",
            )
            if os.path.exists(temp_label):
                img_labels.append(temp_label)

        # collect all labels
        labels_list = {}
        for class_name, mapped_classes in map_classes.items():
            # get the list of all class labels
            class_labels = []
            for mapped_class in mapped_classes:
                temp_labels = [
                    img_label for img_label in img_labels if mapped_class in img_label
                ]
                if temp_labels != []:
                    class_labels.append(temp_labels[0])
            labels_list[class_name] = class_labels

        # save as .npy
        label_img = []
        for class_name, _ in CLASSES.items():
            temp = np.zeros((512, 512, 3))
            for label_path in labels_list[class_name]:
                label_temp = Image.open(label_path)
                label_temp = np.array(label_temp)
                temp[
                    (label_temp[..., 0] != 0)
                    & (label_temp[..., 1] != 0)
                    & (label_temp[..., 2] != 0)
                ] = 1
            label_img.append(temp[..., 0])
        mask = np.stack(label_img, axis=-1).astype(np.float32)
        # correct background
        new_back = np.sum(mask[..., 1:10], axis=-1) == 0
        mask[..., 0] = new_back

        if mask[..., 0].sum() != mask[..., 0].size:
            np.save(
                os.path.join(
                    "../data/dataset_celebs/CelebA-HQ-masks-array/", img_name + ".npy"
                ),
                mask,
            )


def get_data_paths():
    # PARSE COMMUNITY DATASET
    # utils
    xml_path = "../data/dataset_community/training.xml"
    input_files = []
    labels = []
    # parse xml file
    xmldoc = minidom.parse(xml_path)
    # get features and labels
    inputs_features = xmldoc.getElementsByTagName("srcimg")
    labels_features = xmldoc.getElementsByTagName("labelimg")
    for idx, _ in enumerate(inputs_features):
        # get paths
        input_path = inputs_features[idx].attributes["name"].value
        label_path = labels_features[idx].attributes["name"].value
        # check names
        assert os.path.basename(input_path.replace("\\", "/")), os.path.basename(
            label_path.replace("\\", "/")
        )
        # update storages
        input_files.append(
            os.path.join(os.path.dirname(xml_path), input_path.replace("\\", "/"))
        )
        labels.append(
            os.path.join(os.path.dirname(xml_path), label_path.replace("\\", "/"))
        )
    # zip data
    community_dataset = list(
        zip(["community_dataset"] * len(input_files), input_files, labels)
    )

    # PARSE CELEB DATASET
    # utils
    input_files = []
    labels = []
    # img paths
    imgs_path = glob("../data/dataset_celebs/CelebA-HQ-img/*.jpg")
    labels_paths = glob("../data/dataset_celebs/CelebA-HQ-masks-arrays/*.npy")

    # iterate over imgs
    for img_path in imgs_path:
        # get list of img labels
        img_name = os.path.basename(img_path).replace(".jpg", "")
        # get img labels
        label_path = os.path.join(
            "../data/dataset_celebs/CelebA-HQ-masks-array/", img_name + ".npy"
        )

        # update inputs
        if os.path.exists(label_path):
            input_files.append(img_path)
            labels.append(label_path)

    # zip data
    celeb_dataset = list(zip(["celeb_dataset"] * len(input_files), input_files, labels))

    # full data
    data_paths = community_dataset + celeb_dataset

    return data_paths


class CustomDataset(Dataset):
    def __init__(self, data_paths, img_size=(256, 256)):
        # utils
        self.data_paths = data_paths
        self.img_size = img_size
        self.augs = self.get_augs()

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        # get data_batch
        data_batch = self.data_paths[idx]
        # get data item
        if data_batch[0] == "community_dataset":
            data = self.get_community_data(data_batch)
        elif data_batch[0] == "celeb_dataset":
            data = self.get_celeb_data(data_batch)

        return data

    def get_augs(self):
        augs = A.Compose(
            [
                A.Resize(256, 256, 1),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
            ]
        )
        return augs

    def get_community_data(self, data_batch):
        # get_community_data
        # paths
        img_path, label_path = data_batch[1], data_batch[2]
        # read data
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label_temp = cv2.imread(label_path)
        # collect labels
        masks = []
        for _, vals in CLASSES.items():
            temp = np.zeros_like(label_temp)
            temp[
                (label_temp[..., 0] == vals[0])
                & (label_temp[..., 1] == vals[1])
                & (label_temp[..., 2] == vals[2])
            ] = 1
            masks.append(temp[..., 0])
        #
        # Augs
        transformed = self.augs(image=image, masks=masks)
        image = transformed["image"] / 255
        masks = transformed["masks"]
        masks = np.stack(masks, axis=-1).astype(np.float32)
        image = torch.permute(torch.FloatTensor(image), (2, 0, 1))
        masks = torch.permute(torch.FloatTensor(masks), (2, 0, 1))

        return (image, masks)

    def get_celeb_data(self, data_batch):
        # get_celeb_data
        # paths
        img_path, label_path = data_batch[1], data_batch[2]
        # read data
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = np.load(label_path)
        masks = [label[..., i] for i in range(label.shape[-1])]

        # Augs
        transformed = self.augs(image=image, masks=masks)
        image = transformed["image"] / 255
        masks = transformed["masks"] 
        masks = np.stack(masks, axis=-1).astype(np.float32)
        image = torch.permute(torch.FloatTensor(image), (2, 0, 1))
        masks = torch.permute(torch.FloatTensor(masks), (2, 0, 1))

        return (image, masks)
