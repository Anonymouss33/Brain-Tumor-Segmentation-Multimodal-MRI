import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff

# Dice Coefficient for Multi-class
def dice_score(preds, targets, num_classes=4):
    preds = torch.argmax(preds, dim=1)
    dice = []
    smooth = 1e-6
    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        target_cls = (targets == cls).float()
        
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        dice.append((2. * intersection + smooth) / (union + smooth))
    return sum(dice) / len(dice)

# Hausdorff Distance for binary masks (simple version)
def hausdorff_distance(preds, targets):
    preds = torch.argmax(preds, dim=1).cpu().numpy()
    targets = targets.cpu().numpy()
    
    hausdorff_scores = []
    for pred, target in zip(preds, targets):
        pred_points = np.argwhere(pred)
        target_points = np.argwhere(target)
        if pred_points.size == 0 or target_points.size == 0:
            hausdorff_scores.append(0)
        else:
            forward_hd = directed_hausdorff(pred_points, target_points)[0]
            backward_hd = directed_hausdorff(target_points, pred_points)[0]
            hausdorff_scores.append(max(forward_hd, backward_hd))
    return np.mean(hausdorff_scores)

# Plot Training vs Validation Loss and Accuracy
def plot_training(history, save_path=None):
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    train_dice = history['train_dice']
    val_dice = history['val_dice']
    
    epochs = range(1, len(train_loss) + 1)
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss Plot
    axs[0].plot(epochs, train_loss, label="Train Loss")
    axs[0].plot(epochs, val_loss, label="Val Loss")
    axs[0].set_title("Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    
    # Dice Plot
    axs[1].plot(epochs, train_dice, label="Train Dice")
    axs[1].plot(epochs, val_dice, label="Val Dice")
    axs[1].set_title("Dice Score")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Dice Score")
    axs[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

# Visualize Predictions
def visualize_prediction(images, masks, preds, num_classes=4):
    preds = torch.argmax(preds, dim=1)
    
    images = images.cpu().numpy()
    masks = masks.cpu().numpy()
    preds = preds.cpu().numpy()
    
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    
    for i in range(3):
        img = images[i][0]  # just show flair channel
        mask = masks[i]
        pred = preds[i]
        
        axs[i][0].imshow(img, cmap='gray')
        axs[i][0].set_title("Flair Image")
        axs[i][0].axis('off')
        
        axs[i][1].imshow(mask, cmap='jet', vmin=0, vmax=num_classes-1)
        axs[i][1].set_title("Ground Truth Mask")
        axs[i][1].axis('off')
        
        axs[i][2].imshow(pred, cmap='jet', vmin=0, vmax=num_classes-1)
        axs[i][2].set_title("Predicted Mask")
        axs[i][2].axis('off')
    
    plt.tight_layout()
    plt.show()
