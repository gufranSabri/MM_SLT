import os
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

class PrecomputedFeatureDataset(Dataset):
    def __init__(
        self, 
        feature_dir,
        sp_dir,
        phm_dir,
        dataset="phoenix2014-T",
        mode="train",
        include_sp=True,
        include_pose=True,
        p2hm=True
    ):
        self.feature_dir = feature_dir
        self.sp_dir = sp_dir
        self.phm_dir = phm_dir

        self.mode = mode
        self.include_sp = include_sp
        self.include_pose = include_pose
        self.p2hm = p2hm

        self.sp_suffix = "_s2wrapping"
        self.phm_suffix = "_overlap-8"

        self.files = [
            fname
            for fname in os.listdir(feature_dir)
            if fname.endswith(".pkl")
        ]
        if not self.files:
            raise ValueError(f"No .pkl files found in directory: {feature_dir}")

        self.anno_path = f"./preprocess/{dataset}/{mode}_info_ml.npy"
        self.prep_annots(np.load(self.anno_path, allow_pickle=True).item())

    def prep_annots(self, annots):
        del annots["prefix"]

        langs = ["en", "fr", "es"]
        self.translations = {}
        for i in range(len(annots)):
            self.translations[annots[i]["fileid"]] = {lang: annots[i][f"{lang}_text"] for lang in langs if f"{lang}_text" in annots[i]}

        to_remove = []
        for key in self.translations.keys():
            if f"{key}.pkl" not in self.files:
                to_remove.append(key)

        for key in to_remove:
            del self.translations[key]

    def filter_missing(self):
        len_before = len(self.files)
        self.files = [
            fname for fname in self.files
            if os.path.exists(os.path.join(self.feature_dir, fname)) and
               os.path.exists(os.path.join(self.sp_dir, fname.replace(".pkl", f"{self.sp_suffix}.npy"))) and
               os.path.exists(os.path.join(self.phm_dir, fname.replace(".pkl", f"{self.phm_suffix}.npy")))
        ]

        removed = len_before - len(self.files)
        if removed > 0:
            print(f"Removed {removed} missing files.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sign_ft_path = os.path.join(self.feature_dir, self.files[idx])
        sp_ft_path = os.path.join(self.sp_dir, self.files[idx].replace(".pkl", f"{self.sp_suffix}.npy"))
        phm_ft_path = os.path.join(self.phm_dir, self.files[idx].replace(".pkl", f"{self.phm_suffix}.npy"))

        sign_ft, sp_ft, phm_ft, label = None, None, None, None
        with open(sign_ft_path, "rb") as f:
            entry = pickle.load(f)
            sign_ft = torch.tensor(entry["tensor"], dtype=torch.float32)
            label = entry["label"]

        sp_ft = torch.tensor(np.load(sp_ft_path), dtype=torch.float32) if self.include_sp else None
        phm_ft = torch.tensor(np.load(phm_ft_path), dtype=torch.float32) if self.include_pose else None

        icl_text = "\n".join([
            f'{self.translations.get(self.files[idx].replace(".pkl", ""), {}).get("en", "")}={label}',
            f'{self.translations.get(self.files[idx].replace(".pkl", ""), {}).get("fr", "")}={label}',
            f'{self.translations.get(self.files[idx].replace(".pkl", ""), {}).get("es", "")}={label}',
        ])

        # pad sp_ft to make sure length is divisble by 4
        if self.include_sp:
            pad_length = (4 - sp_ft.shape[0] % 4) % 4
            sp_ft = F.pad(sp_ft, (0, 0, 0, pad_length), "constant", 0)

        return sign_ft, sp_ft, phm_ft, label, icl_text
    
    @staticmethod
    def collate_fn(batch):
        sign_ft, sp_ft, phm_ft, labels, icl_text = zip(*batch)  # unpack the batch

        sign_lengths = torch.tensor([t.shape[0] for t in sign_ft], dtype=torch.long)  # time dims
        sp_lengths = torch.tensor([t.shape[0] for t in sp_ft], dtype=torch.long) if sp_ft[0] is not None else None
        phm_lengths = torch.tensor([t.shape[0] for t in phm_ft], dtype=torch.long) if phm_ft[0] is not None else None

        padded_sign_ft = pad_sequence(sign_ft, batch_first=True)  # [B, T_max, D]
        padded_sp_ft = pad_sequence(sp_ft, batch_first=True) if sp_ft[0] is not None else None
        padded_phm_ft = pad_sequence(phm_ft, batch_first=True) if phm_ft[0] is not None else None

        return padded_sign_ft, padded_sp_ft, padded_phm_ft, labels, icl_text, sign_lengths, sp_lengths, phm_lengths


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    mode="dev"
    dataset = PrecomputedFeatureDataset(
        f"/home/ahmedubc/projects/aip-lsigal/ahmedubc/MM_SLT/features/phoenix14-T/{mode}",
        f"/home/ahmedubc/projects/aip-lsigal/ahmedubc/datasets/spamo/sp/clip-vit-large-patch14_feat_Phoenix14T/{mode}",
        f"/home/ahmedubc/projects/aip-lsigal/ahmedubc/datasets/spamo/mo/mae_feat_Phoenix14T/{mode}",
        mode=mode
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=dataset.collate_fn)

    for sign_features, sp_features, phm_features, labels, icl_text, sign_lengths, sp_lengths, phm_lengths in dataloader:
        print(sign_features.shape)
        print(sp_features.shape)
        print(phm_features.shape)
        print(labels)
        print(icl_text[0])
        print(sign_lengths)
        print(sp_lengths)
        print(phm_lengths)

        break