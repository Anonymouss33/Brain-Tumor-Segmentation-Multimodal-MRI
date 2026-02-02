import torch
import torch.nn as nn
import torch.optim as optim
from utils import dice_score, hausdorff_distance, plot_training, visualize_prediction
import os
from tqdm import tqdm

def train_model(model, train_loader, val_loader, device, num_epochs=50, patience=5, save_dir="outputs"):
    os.makedirs(save_dir, exist_ok=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_dice": [],
        "val_dice": []
    }
    
    best_val_dice = 0
    early_stop_counter = 0
    
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        
        # Training
        model.train()
        running_loss = 0
        running_dice = 0
        for images, masks in tqdm(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            running_dice += dice_score(outputs, masks).item()
        
        train_loss = running_loss / len(train_loader)
        train_dice = running_dice / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_dice = 0
        with torch.no_grad():
            for images, masks in tqdm(val_loader):
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                val_dice += dice_score(outputs, masks).item()
        
        val_loss = val_loss / len(val_loader)
        val_dice = val_dice / len(val_loader)
        
        print(f"Train Loss: {train_loss:.4f}  |  Train Dice: {train_dice:.4f}")
        print(f"Val Loss: {val_loss:.4f}  |  Val Dice: {val_dice:.4f}")
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_dice"].append(train_dice)
        history["val_dice"].append(val_dice)
        
        # Early Stopping
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            early_stop_counter = 0
            # Save best model
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print("🔥 Best model saved!")
        else:
            early_stop_counter += 1
            print(f"Early stopping patience: {early_stop_counter}/{patience}")
        
        if early_stop_counter >= patience:
            print("Early stopping triggered!")
            break
    
    # Plot training history
    plot_training(history, save_path=os.path.join(save_dir, "training_plot.png"))
    
    return history
