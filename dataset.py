import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random

class BraTSMiddleSliceDataset(Dataset):
    def __init__(self, npy_dir, modalities=["flair", "t1", "t1ce", "t2"]):
        self.npy_dir = npy_dir
        self.modalities = modalities
        self.patient_dirs = [os.path.join(npy_dir, p) for p in os.listdir(npy_dir)]
        self.slice_list = []
        self._prepare_slice_list()
    
    def _prepare_slice_list(self):
        for patient in self.patient_dirs:
            try:
                flair_path = os.path.join(patient, "flair.npy")
                if not os.path.exists(flair_path):
                    print(f"⚠️ Missing flair.npy: {flair_path}")
                    continue
                flair = np.load(flair_path)
                if flair.size == 0:
                    print(f"⚠️ Skipping {patient} due to empty flair.npy")
                    continue
                slices = flair.shape[2]
                for idx in range(60, min(100, slices)):
                    self.slice_list.append((patient, idx))
            except Exception as e:
                print(f"⚠️ Skipping {patient} due to error: {e}")
    
    def __len__(self):
        return len(self.slice_list)
    
    def __getitem__(self, idx):
        patient_path, slice_idx = self.slice_list[idx]
        
        try:
            images = []
            for mod in self.modalities:
                img_path = os.path.join(patient_path, f"{mod}.npy")
                if not os.path.exists(img_path):
                    raise ValueError(f"Missing {mod}.npy at {patient_path}")
                img = np.load(img_path)
                if img.size == 0:
                    raise ValueError(f"Empty {mod}.npy at {patient_path}")
                img_slice = img[:, :, slice_idx]
                img_slice = (img_slice - np.min(img_slice)) / (np.max(img_slice) - np.min(img_slice) + 1e-8)
                images.append(img_slice)
            
            image = np.stack(images)  # (C, H, W)
            image = torch.tensor(image, dtype=torch.float32)
            
            mask_path = os.path.join(patient_path, "seg.npy")
            if not os.path.exists(mask_path):
                raise ValueError(f"Missing seg.npy at {patient_path}")
            mask = np.load(mask_path)
            if mask.size == 0:
                raise ValueError(f"Empty seg.npy at {patient_path}")
            mask_slice = mask[:, :, slice_idx]
            mask_slice[mask_slice == 4] = 3  # Replace label 4 with 3
            mask = torch.tensor(mask_slice, dtype=torch.long)
            
            return image, mask
        
        except Exception as e:
            print(f"⚠️ Error at {patient_path}: {e} -- resampling another")
            new_idx = random.randint(0, len(self.slice_list) - 1)
            return self.__getitem__(new_idx)
