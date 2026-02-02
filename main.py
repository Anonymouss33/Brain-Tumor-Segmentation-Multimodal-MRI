import torch
from torch.utils.data import DataLoader, random_split
from dataset import BraTSMiddleSliceDataset
from model import UNetBESNet
from train import train_model
import os

def main():
    # Settings
    DATA_DIR = r"C:\Users\praha\project\data\BraTS2021_NPY"  # CHANGE this to your real npy path
    SAVE_DIR = "outputs"
    BATCH_SIZE = 2
    NUM_EPOCHS = 50
    VAL_SPLIT = 0.2
    PATIENCE = 5
    NUM_CLASSES = 4  # background, edema, tumor core, enhancing tumor
    IN_CHANNELS = 4  # flair, t1, t1ce, t2
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataset
    full_dataset = BraTSMiddleSliceDataset(DATA_DIR, modalities=["flair", "t1", "t1ce", "t2"])
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Model
    model = UNetBESNet(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    model = model.to(device)
    
    # Train
    history = train_model(
        model,
        train_loader,
        val_loader,
        device,
        num_epochs=NUM_EPOCHS,
        patience=PATIENCE,
        save_dir=SAVE_DIR
    )
    
    print("Training finished!")

if __name__ == "__main__":
    main()
