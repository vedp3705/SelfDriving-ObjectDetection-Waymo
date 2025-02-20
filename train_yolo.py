from ultralytics import YOLO
from torch.utils.data import DataLoader
from dataset_loader import WaymoDataset, raw_dataset

model = YOLO("yolov8n.pt") 

dataset = WaymoDataset(raw_dataset)s
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

model.train(data="config.yaml", epochs=50, dataloader=dataloader)
