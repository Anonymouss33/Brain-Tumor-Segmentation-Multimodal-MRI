import os
import numpy as np
import nibabel as nib

# 1. Original data (your nii.gz files)
ORIGINAL_DATA_DIR = r"C:/Users/praha/project/data/BraTS2021_Training_Data"

# 2. New directory for .npy files
SAVE_DIR = r"C:/Users/praha/project/data/BraTS2021_NPY"

# 3. Modalities we want
modalities = ["flair", "t1", "t1ce", "t2"]

# 4. Create save directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

# 5. Loop over each patient
patients = [p for p in os.listdir(ORIGINAL_DATA_DIR) if os.path.isdir(os.path.join(ORIGINAL_DATA_DIR, p))]

for patient in patients:
    patient_path = os.path.join(ORIGINAL_DATA_DIR, patient)
    save_patient_dir = os.path.join(SAVE_DIR, patient)
    os.makedirs(save_patient_dir, exist_ok=True)
    
    for mod in modalities + ["seg"]:
        try:
            nii_path = os.path.join(patient_path, f"{patient}_{mod}.nii.gz")
            img = nib.load(nii_path).get_fdata()
            np.save(os.path.join(save_patient_dir, f"{mod}.npy"), img)
            print(f"Saved: {patient}/{mod}.npy")
        except Exception as e:
            print(f"Error with {patient} {mod}: {e}")
