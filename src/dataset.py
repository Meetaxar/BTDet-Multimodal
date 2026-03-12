"""
BraTS2020 Multi-Modal Dataset Pipeline
Loads 4-channel MRI slices and links clinical survival data.
"""
import os, glob, re
import numpy as np
import pandas as pd
import nibabel as nib
from PIL import Image
from torch.utils.data import Dataset
import torch

MODALITIES = ["flair", "t1", "t1ce", "t2"]

def find_modality_file(patient_path, modality):
    for pat in [f"*_{modality}.nii.gz", f"*_{modality}.nii",
                f"*{modality}*.nii.gz", f"*{modality}*.nii"]:
        matches = glob.glob(os.path.join(patient_path, pat))
        if matches:
            return matches[0]
    return None

def find_seg_file(patient_path):
    for pat in ["*_seg.nii.gz", "*_seg.nii",
                "*seg*.nii.gz", "*seg*.nii"]:
        matches = glob.glob(os.path.join(patient_path, pat))
        if matches:
            return matches[0]
    return None

def normalize_slice(s):
    mn, mx = s.min(), s.max()
    if mx - mn < 1e-8:
        return np.zeros_like(s, dtype=np.float32)
    return ((s - mn) / (mx - mn)).astype(np.float32)

def seg_to_bbox(seg_slice):
    mask = (seg_slice > 0).astype(np.uint8)
    if mask.sum() == 0:
        return None
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    H, W = seg_slice.shape
    return [((cmin+cmax)/2)/W, ((rmin+rmax)/2)/H,
            (cmax-cmin)/W,     (rmax-rmin)/H]

def parse_survival(val):
    if pd.isna(val):
        return None
    s = str(val).strip().upper()
    try:
        return float(s)
    except:
        nums = re.findall(r"\d+\.?\d*", s)
        return float(nums[0]) if nums else None

class BraTSMultiModalDataset(Dataset):
    """
    PyTorch Dataset for BraTS2020 multi-modal MRI + clinical data.
    Returns 4-channel grid image, clinical vector, bbox, survival.
    """
    def __init__(self, records, age_mean, age_std,
                 surv_mean, surv_std, img_size=640):
        self.records   = records
        self.age_mean  = age_mean
        self.age_std   = age_std
        self.surv_mean = surv_mean
        self.surv_std  = surv_std
        self.img_size  = img_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]

        # 4-channel 2x2 grid
        flair = (r["slice_array"][0] * 255).astype(np.uint8)
        t1    = (r["slice_array"][1] * 255).astype(np.uint8)
        t1ce  = (r["slice_array"][2] * 255).astype(np.uint8)
        t2    = (r["slice_array"][3] * 255).astype(np.uint8)
        grid  = np.concatenate([
            np.concatenate([flair, t1],   axis=1),
            np.concatenate([t1ce,  t2],   axis=1)
        ], axis=0)
        img = torch.tensor(
            np.array(Image.fromarray(grid).convert("RGB")
                     .resize((self.img_size, self.img_size))),
            dtype=torch.float32).permute(2,0,1) / 255.0

        # Clinical encoding
        age_norm = ((float(r["age"]) - self.age_mean) / 
                    (self.age_std + 1e-8)
                    if pd.notna(r["age"]) else 0.0)
        res = str(r["resection"]).upper()
        res_enc = 1.0 if "GTR" in res else (0.0 if "STR" in res else 0.5)
        clinical = torch.tensor([age_norm, res_enc], dtype=torch.float32)

        # Survival
        sv = parse_survival(r["survival_days"])
        surv_norm  = ((sv - self.surv_mean) / (self.surv_std + 1e-8)
                      if sv is not None else 0.0)
        surv_valid = sv is not None

        # BBox
        bbox = torch.zeros(4, dtype=torch.float32)
        if r["has_tumor"] and r["bbox"] is not None:
            xc, yc, w, h = r["bbox"]
            bbox = torch.tensor(
                [xc*0.5, 0.5+yc*0.5, w*0.5, h*0.5],
                dtype=torch.float32)

        return (img, clinical, bbox,
                torch.tensor(r["has_tumor"], dtype=torch.bool),
                torch.tensor(surv_norm,  dtype=torch.float32),
                torch.tensor(surv_valid, dtype=torch.bool))
