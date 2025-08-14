import os
import cv2
import sys
import glob
import time
import torch
import warnings
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import torch.utils.data as data
from utils import video_augmentation

sys.path.append("..")
global kernel_sizes 

class BaseFeeder(data.Dataset):
    def __init__(self, prefix, dataset='phoenix2014', drop_ratio=1, num_gloss=-1, mode="train", transform_mode=True,
                 datatype="lmdb", frame_interval=1, image_scale=1.0, kernel_size=1, input_size=224):
        self.mode = mode
        self.ng = num_gloss
        self.prefix = prefix
        self.data_type = datatype
        self.dataset = dataset
        self.input_size = input_size
        global kernel_sizes 
        kernel_sizes = kernel_size
        self.frame_interval = frame_interval # not implemented for read_features()
        self.image_scale = image_scale # not implemented for read_features()
        self.feat_prefix = f"{prefix}/fullFrame-256x256px/{mode}"
        self.transform_mode = "train" if transform_mode else "test"
        self.inputs_list = np.load(f"./preprocess/{dataset}/{mode}_info.npy", allow_pickle=True).item()

        del self.inputs_list["prefix"]
        self.labels = [self.inputs_list[i]["original_info"].split("|")[-1] for i in range(len(self.inputs_list))]

        # python main.py --device 2 --dataset phoenix2014 --loss-weights Slow=0.25 Fast=0.25 --work-dir ./work_dir/phoenix2014/
        
        if "phoenix" in self.dataset:
            indices_to_remove = []
            for key in self.inputs_list:
                if key == 'prefix': continue
                item = self.inputs_list[key]["original_info"].split("|")[1]
                item = item[:item.index("*")]
                path = os.path.join(self.feat_prefix, item)

                if not os.path.exists(path):
                    print(path)
                    indices_to_remove.append(key)

            for key in indices_to_remove:
                print(f"Removing {key} from {self.mode}", "=============================")
                del self.inputs_list[key]

        inp = []
        for key in self.inputs_list:
            if key == 'prefix': continue
            inp.append(self.inputs_list[key])

        self.inputs_list = inp
        self.data_aug = self.transform()


    def __getitem__(self, idx):
        input_data, label, fi = self.read_video(idx)
        input_data = self.normalize(input_data)
        return input_data, label, self.inputs_list[idx]['original_info']

    def read_video(self, index):
        fi = self.inputs_list[index]
        if 'phoenix' in self.dataset:
            img_folder = os.path.join(self.prefix, "fullFrame-256x256px/" , fi['folder'])  
        elif self.dataset == 'CSL':
            img_folder = os.path.join(self.prefix, "features/fullFrame-256x256px/" + fi['folder'] + "/*.jpg")
        elif self.dataset == 'CSL-Daily':
            img_folder = os.path.join(self.prefix, fi['folder'])

        img_list = sorted(glob.glob(img_folder))
        img_list = img_list[int(torch.randint(0, self.frame_interval, [1]))::self.frame_interval]
        label = self.labels[index]

        if self.dataset != 'CSL-Daily':
            return [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in img_list], label, fi
        else:
            return [cv2.cvtColor(cv2.resize(cv2.imread(img_path)[40:, ...], (256, 256)), cv2.COLOR_BGR2RGB) for img_path in img_list], label, fi

    def read_features(self, index):
        fi = self.inputs_list[index]
        data = np.load(f"./features/{self.mode}/{fi['fileid']}_features.npy", allow_pickle=True).item()
        return data['features'], data['label']
    
    def normalize(self, video, file_id=None):
        video = self.data_aug(video)
        if isinstance(video, list):
            video = torch.stack(video, dim=0)  # Shape: (T, C, H, W)
        video = ((video.float() / 255.) - 0.45) / 0.225
        
        return video

    def transform(self):
        if self.transform_mode == "train":
            return video_augmentation.Compose([
                video_augmentation.RandomCrop(self.input_size),
                video_augmentation.RandomHorizontalFlip(0.5),
                video_augmentation.Resize(self.image_scale),
                video_augmentation.ToTensor(),
                video_augmentation.TemporalRescale(0.2, self.frame_interval),
            ])
        else:
            return video_augmentation.Compose([
                video_augmentation.CenterCrop(self.input_size),
                video_augmentation.Resize(self.image_scale),
                video_augmentation.ToTensor(),
            ])

    @staticmethod
    def collate_fn(batch):
        batch = [item for item in sorted(batch, key=lambda x: len(x[0]), reverse=True)]
        video, label, info = list(zip(*batch))
        
        left_pad = 0
        last_stride = 1
        total_stride = 1
        global kernel_sizes 
        for layer_idx, ks in enumerate(kernel_sizes):
            if ks[0] == 'K':
                left_pad = left_pad * last_stride 
                left_pad += int((int(ks[1])-1)/2)
            elif ks[0] == 'P':
                last_stride = int(ks[1])
                total_stride = total_stride * last_stride

        if len(video[0].shape) > 3:
            max_len = len(video[0])
            video_length = torch.LongTensor([np.ceil(len(vid) / total_stride) * total_stride + 2*left_pad for vid in video])
            right_pad = int(np.ceil(max_len / total_stride)) * total_stride - max_len + left_pad
            max_len = max_len + left_pad + right_pad
            padded_video = [torch.cat(
                (
                    vid[0][None].expand(left_pad, -1, -1, -1),
                    vid,
                    vid[-1][None].expand(max_len - len(vid) - left_pad, -1, -1, -1),
                )
                , dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video)
        else:
            max_len = len(video[0])
            video_length = torch.LongTensor([len(vid) for vid in video])
            padded_video = [torch.cat(
                (
                    vid,
                    vid[-1][None].expand(max_len - len(vid), -1),
                )
                , dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video).permute(0, 2, 1)

        return padded_video, video_length, label, info

    def __len__(self):
        return len(self.inputs_list) - 1

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time