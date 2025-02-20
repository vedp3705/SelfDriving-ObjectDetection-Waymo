import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

GCS_TFRECORD_PATH = "gs://waymo_open_dataset_v_1_4_3/individual_files/training/segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord"
# gs://waymo_open_dataset_v_1_4_3/individual_files/training/segment-10023947602400723454_1120_000_1140_000_with_camera_labels.tfrecord
# gs://waymo_open_dataset_v_1_4_3/individual_files/training/segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord
# gs://waymo_open_dataset_v_1_4_3/individual_files/training/segment-10061305430875486848_1080_000_1100_000_with_camera_labels.tfrecord

raw_dataset = tf.data.TFRecordDataset(GCS_TFRECORD_PATH)

def parse_tfrecord_fn(raw_record):
    frame = open_dataset.Frame()
    frame.ParseFromString(raw_record.numpy())

    images, labels = {}, []

    for img in frame.images:
        img_array = np.frombuffer(img.image, dtype=np.uint8)
        img_decoded = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  

        images[img.name] = img_decoded

    for camera_labels in frame.camera_labels:
        for label in camera_labels.labels:
            labels.append({"type": label.type, "box": (label.box.center_x, label.box.center_y, label.box.width, label.box.height)})
    
    return images, labels

class WaymoDataset(Dataset):
    def __init__(self, raw_dataset):
        self.raw_dataset = list(raw_dataset.take(100))  

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        raw_record = self.raw_dataset[idx]
        images, labels = parse_tfrecord_fn(raw_record)

        image_name = list(images.keys())[0]
        image = images[image_name]

        img_h, img_w = image.shape[:2]
        yolo_labels = []
        for label in labels:
            class_id = label["type"] - 1
            x, y, w, h = label["box"]
            yolo_labels.append([class_id, x / img_w, y / img_h, w / img_w, h / img_h])

        return torch.tensor(image).permute(2, 0, 1), torch.tensor(yolo_labels)