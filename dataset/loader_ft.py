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
        pose_dir,
        mo_dir,
        phm_dir,
        dataset="phoenix2014-T",
        mode="train",
        include_sp=True,
        include_pose=True,
        include_mo=True,
        logger=None
    ):
        self.feature_dir = feature_dir
        self.sp_dir = sp_dir
        self.phm_dir = phm_dir
        self.pose_dir = pose_dir
        self.mo_dir = mo_dir

        self.mode = mode
        self.include_sp = include_sp
        self.include_pose = include_pose
        self.include_mo = include_mo

        assert self.include_sp or self.include_pose, "At least one feature must be included."
        if logger:
            logger(f"FEATURES: include_sp: {include_sp}, include_pose: {include_pose}, include_mo: {include_mo}")

        self.sp_suffix = "_s2wrapping"
        self.mo_suffix = "_overlap-8"

        self.files = [
            fname
            for fname in os.listdir(feature_dir)
        ]
        self.filter_missing()
        if not self.files:
            raise ValueError(f"No .pkl files found in directory: {feature_dir}")

        self.anno_path = f"./preprocess/{dataset}/{mode}_info_ml.npy"
        self.prep_annots(np.load(self.anno_path, allow_pickle=True).item())

        with open(os.path.join(self.pose_dir, f"{dataset}.{mode}"), "rb") as f:
            self.pose_dict = pickle.load(f)

    def filter_missing(self):
        to_remove = []
        for f in self.files:
            if not os.path.exists(os.path.join(self.sp_dir, f + ".npy")):
                to_remove.append(f)

        for f in to_remove:
            self.files.remove(f)

        

                

    def prep_annots(self, annots):
        del annots["prefix"]

        langs = ["en", "fr", "es"]
        self.translations, self.text, self.gloss = {}, {}, {}
        for i in range(len(annots)):
            self.translations[annots[i]["fileid"]] = {lang: annots[i][f"{lang}_text"] for lang in langs if f"{lang}_text" in annots[i]}
            self.text[annots[i]["fileid"]] = annots[i]["text"] if "text" in annots[i] else ""
            self.gloss[annots[i]["fileid"]] = annots[i]["gloss"] if "gloss" in annots[i] else ""

        to_remove = []
        for key in self.translations.keys():
            if key not in self.files:
                to_remove.append(key)

        for key in to_remove:
            del self.translations[key]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # sp_ft_path = os.path.join(self.sp_dir, self.files[idx] + f"{self.sp_suffix}.npy")
        sp_ft_path = os.path.join(self.sp_dir, self.files[idx] + f".npy")
        mo_ft_path = os.path.join(self.mo_dir, self.files[idx] + f"{self.mo_suffix}.npy")

        sp_ft, pose_ft, mo_ft = None, None, None

        sp_ft = torch.tensor(np.load(sp_ft_path), dtype=torch.float32) if self.include_sp else None
        pose_ft = self.pose_dict[f"{self.mode}/{self.files[idx].replace('.pkl', '')}"]["keypoint"] if self.include_pose else None
        mo_ft = torch.tensor(np.load(mo_ft_path), dtype=torch.float32) if self.include_mo else None

        # pad sp_ft to make sure length is divisble by 4
        if self.include_sp:
            pad_length = (4 - sp_ft.shape[0] % 4) % 4
            sp_ft = F.pad(sp_ft, (0, 0, 0, pad_length), "constant", 0)

        # pad pose_ft to make sure length is divisble by 4
        if self.include_pose:
            A,B,C = pose_ft.shape
            if A > 400: pose_ft = pose_ft[np.linspace(0, A - 1, 400, dtype=int)]
            pose_ft = pose_ft.reshape(pose_ft.shape[0], B * C)

            pad_length = (4 - pose_ft.shape[0] % 4) % 4
            pose_ft = F.pad(pose_ft, (0, 0, 0, pad_length), "constant", 0)

            pose_ft = pose_ft.reshape(-1, B, C)

        if self.include_mo:
            pad_length = (4 - mo_ft.shape[0] % 4) % 4
            mo_ft = F.pad(mo_ft, (0, 0, 0, pad_length), "constant", 0)


        gloss = self.gloss[self.files[idx].replace(".pkl", "")]
        text = self.text[self.files[idx].replace(".pkl", "")]

        icl_text = "\n".join([
            f'{self.translations[self.files[idx].replace(".pkl", "")]["en"]}={text}',
            f'{self.translations[self.files[idx].replace(".pkl", "")]["fr"]}={text}',
            f'{self.translations[self.files[idx].replace(".pkl", "")]["es"]}={text}',
        ])

        return sp_ft, pose_ft, mo_ft, gloss, text, icl_text
    
    @staticmethod
    def collate_fn(batch):
        sp_ft, pose_ft, mo_ft, glosses, texts, icl_text = zip(*batch)  # unpack the batch

        sp_lengths = torch.tensor([t.shape[0] for t in sp_ft], dtype=torch.long) if sp_ft[0] is not None else None
        pose_lengths = torch.tensor([t.shape[0] for t in pose_ft], dtype=torch.long) if pose_ft[0] is not None else None
        mo_lengths = torch.tensor([t.shape[0] for t in mo_ft], dtype=torch.long) if mo_ft[0] is not None else None

        padded_sp_ft = pad_sequence(sp_ft, batch_first=True) if sp_ft[0] is not None else None

        padded_pose_ft = None
        if pose_ft[0] is not None:
            pose_ft_r = [pft.reshape(pft.shape[0], -1) for pft in pose_ft]
            padded_pose_ft = pad_sequence(pose_ft_r, batch_first=True)
            B, T, KC = padded_pose_ft.shape
            padded_pose_ft = padded_pose_ft.reshape(B, T, KC//3, 3).permute(0, 3, 1, 2)
        
        padded_mo_ft = pad_sequence(mo_ft, batch_first=True) if mo_ft[0] is not None else None

        return padded_sp_ft, padded_pose_ft, padded_mo_ft, glosses, texts, icl_text, sp_lengths, pose_lengths, mo_lengths


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    mode="dev"
    dataset = PrecomputedFeatureDataset(
        f"/home/ahmedubc/projects/aip-lsigal/ahmedubc/datasets/phoenix2014T/fullFrame-256x256px/{mode}",
        f"/home/ahmedubc/projects/aip-lsigal/ahmedubc/datasets/spamo/sp/clip-vit-large-patch14_feat_Phoenix14T/{mode}",
        f"/home/ahmedubc/projects/aip-lsigal/ahmedubc/datasets/pose/",
        f"/home/ahmedubc/projects/aip-lsigal/ahmedubc/datasets/spamo/pomo/mae_feat_Phoenix14T/{mode}",
        f"/home/ahmedubc/projects/aip-lsigal/ahmedubc/datasets/spamo/posp/clip-vit-large-patch14_feat_Phoenix14T/{mode}",
        mode=mode
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=dataset.collate_fn)

    for sp_features, pose_features, mo_features, glosses, texts, icl_text, sp_lengths, pose_lengths, mo_lengths in dataloader:
        print(sp_features.shape)
        # print(pose_features.shape)
        # print(mo_features.shape)
        # print(glosses)
        # print(texts)
        # print(icl_text[0])
        # print(sp_lengths)
        # print(pose_lengths)
        # print(mo_lengths)

        # break