import os
import cv2
import numpy as np
import albumentations as A

import torch
from torch import nn
from torch.utils import data
from skimage.io import imread

SIZE = 100

class FacePointsDataset(data.Dataset):
    def __init__(
            self,
            mode: str,
            gt: dict,
            img_dir: str,
            train_fraction: float = 0.8,
            transform=None,
    ):
        self._items = []
        self._transform = transform
        self.mode = mode

        images = sorted([f for f in os.listdir(img_dir)]) 
        split = int(train_fraction * len(images))

        if mode == "train":
            img_names = images[:split]
        elif mode == "valid":
            img_names = images[split:]
        else:
            raise RuntimeError(f"Invalid mode {mode!r}") 

        for img in img_names:
            if img in gt:
                self._items.append((os.path.join(img_dir, img), gt[img]))
            else:
                print(f"Warning: No ground truth for {img}")  

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        img_path, facepoints = self._items[index]
        
        facepoints = np.array(facepoints).astype(np.float32)
        img = imread(img_path)
        img = np.array(img).astype(np.float32)

        if len(img.shape) == 2:
            h, w = img.shape
            img = np.stack([img]*3, axis=2)
        elif len(img.shape) == 3:
            h, w = img.shape[:2]
            if img.shape[2] == 4:
                img = img[:, :, :3]
            elif img.shape[2] == 1:
                img = np.stack([img[:,:,0]]*3, axis=2)
        else:
            raise ValueError(f"ValueError: {img.shape}")
        
        img = cv2.resize(img, (SIZE, SIZE))
        img = img / 255.0
        facepoints[::2] *= (SIZE / w)
        facepoints[1::2] *= (SIZE / h)

        if self._transform and self.mode == "train":
            keypoints = list(zip(facepoints[::2], facepoints[1::2]))
            transformed = self._transform(image=img, keypoints=keypoints)
            img = transformed['image']
            facepoints = np.ravel(transformed['keypoints']).astype(np.float32)

        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        mean = img.mean([1, 2]).reshape(3, 1, 1)
        std = img.std([1, 2]).reshape(3, 1, 1)
        img = (img - mean) / std

        return img, torch.from_numpy(facepoints).float()
    


# ========== MODEL =========
class FacePointsModel(nn.Module):
    def __init__(self, num_points):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((6, 6)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_points * 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_detector(train_gt: dict,
                   train_img_dir: str,
                   fast_train=True):

    if fast_train:
        device = torch.device("cpu")
        num_epochs = 2
        batch_size = 8
        num_workers = 0
        lr = 1e-4
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_epochs = 50
        batch_size = 32
        num_workers = 4
        lr = 1e-4

    # Train augmentations
    train_transform = A.Compose([
        A.Rotate(limit=70, p=0.3),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    

    # Create datasets
    ds_train = FacePointsDataset(
        mode="train",
        gt=train_gt,
        img_dir=train_img_dir,
        transform=train_transform,
        train_fraction=0.8
    )

    ds_valid = FacePointsDataset(
        mode="valid",
        gt=train_gt,
        img_dir=train_img_dir,
        transform=None,
        train_fraction=0.8
    )

    dl_train = data.DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
    )

    dl_valid = data.DataLoader(
        ds_valid,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    model = FacePointsModel(num_points=14).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    best_val_loss = float('inf')
    best_model_path = "facepoints_model.pt"

    for epoch in range(num_epochs):
        # ===== TRAINING =====
        model.train()
        train_losses = 0.0

        for x_batch, y_batch in dl_train:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            p_batch = model(x_batch)
            loss = loss_fn(p_batch, y_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses += loss.item()

        avg_train_loss = train_losses / len(dl_train)

        # ===== VALIDATION =====
        model.eval()
        val_losses = 0.0

        with torch.no_grad():
            for x_batch, y_batch in dl_valid:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                p_batch = model(x_batch)
                loss = loss_fn(p_batch, y_batch)
                val_losses += loss.item()

        avg_val_loss = val_losses / len(dl_valid)
        scheduler.step(avg_val_loss)

        if not fast_train and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
        elif avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

        if fast_train and epoch > 0:
            break

    if not fast_train:
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    return model


def detect(model_filename: str, test_img_dir: str):
    device = torch.device("cpu")
    model = FacePointsModel(num_points=14)
    model.load_state_dict(torch.load(model_filename, map_location=device))
    model.eval()
    
    img_names = sorted([f for f in os.listdir(test_img_dir)]) 
    predictions = {}
    
    for img_name in img_names:
        img_path = os.path.join(test_img_dir, img_name)
        image = imread(img_path)
        image = np.array(image).astype(np.float32)
        
        if len(image.shape) == 2:
            original_h, original_w = image.shape
            image_resized = cv2.resize(image, (SIZE, SIZE))
            if len(image_resized.shape) == 2:
                image_resized = np.stack([image_resized]*3, axis=2)
        else:
            original_h, original_w = image.shape[:2]
            if image.shape[2] == 4:
                image = image[:, :, :3]
            image_resized = cv2.resize(image, (SIZE, SIZE))
        
        image_resized = image_resized / 255.0
        img_tensor = torch.from_numpy(image_resized.transpose(2, 0, 1)).float()
        img_tensor = img_tensor.unsqueeze(0) 
        
        with torch.no_grad():
            pred = model(img_tensor).numpy()[0]
        
        pred[::2] *= (original_w / SIZE)
        pred[1::2] *= (original_h / SIZE)
        
        predictions[img_name] = pred.tolist()
    
    return predictions
