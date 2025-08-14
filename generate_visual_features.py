import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import yaml
import torch
import importlib
import faulthandler
import numpy as np
import shutil
from distutils.dir_util import copy_tree
import inspect
from collections import OrderedDict
from tqdm import tqdm
import pickle

faulthandler.enable()
import utils
from modules.sync_batchnorm import convert_model
from seq_scripts import seq_train, seq_eval, seq_feature_generation


class Processor():
    def __init__(self, arg):
        self.arg = arg
        
        self.recoder = utils.Recorder(self.arg.work_dir, self.arg.print_log, self.arg.log_interval)
        self.save_arg()

        # if self.arg.random_fix: 
        self.rng = utils.RandomState(seed=self.arg.random_seed)
        self.device = utils.GpuDataParallel()
        self.recoder = utils.Recorder(self.arg.work_dir, self.arg.print_log, self.arg.log_interval)
        self.dataset, self.data_loader = {}, {}

        self.arg.optimizer_args['num_epoch'] = self.arg.num_epoch

        self.model, self.optimizer = self.loading()

    def start(self):
        # splits = ["dev", "test", "train"]
        splits = ["train"]
        os.makedirs("features", exist_ok=True)
        for split in splits:
            os.makedirs(f"features/{split}", exist_ok=True)

        self.model.eval()
        for split in splits:
            loader = self.data_loader.get(split, None)

            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(loader, desc=f"Processing {split} data")):
                    vid = batch[0].to(self.arg.device)
                    vid_lgt = batch[1].to(self.arg.device)
                    info = batch[3][0]
                    label_batch = batch[2]
                    

                    path = f"split/{info.split('|')[0]}"
                    name = info.split('|')[0]
                    label = label_batch[0]

                    if os.path.exists(f"./features/{split}/{name}.pkl"):
                        continue

                    tqdm.write(f"./features/{split}/{name}.pkl")

                    features = self.model(vid, vid_lgt)['sequence_features'].mean(dim=1)  # [B, D]
                    tensor = features.cpu().numpy()


                    entry = {
                        "path": path,
                        "label": label,
                        "tensor": tensor
                    }

                    # Save each entry individually
                    save_path = f"./features/{split}/{name}.pkl"
                    with open(save_path, "wb") as f:
                        pickle.dump(entry, f)


    def loading(self):
        self.arg.llm_args["llm"] = self.arg.llm

        # self.device.set_device(self.arg.device)
        slr_class = import_class(self.arg.slr)
        
        slr = slr_class( **self.arg.slr_args, loss_weights=self.arg.loss_weights).to(self.arg.device)
        shutil.copy2(inspect.getfile(slr_class), self.arg.work_dir)

        optimizer = utils.Optimizer(slr, self.arg.optimizer_args)

        if self.arg.load_weights: self.load_model_weights(slr, self.arg.slr_weights)

        # model = self.model_to_device(model)
        self.kernel_sizes = slr.conv1d.kernel_size
        self.load_data()

        return slr, optimizer
    
    def load_data(self):
        self.feeder = import_class(self.arg.feeder)
        shutil.copy2(inspect.getfile(self.feeder), self.arg.work_dir)
        
        if self.arg.dataset == 'CSL':
            dataset_list = zip(["train", "dev"], [True, False])
        elif 'phoenix' in self.arg.dataset:
            dataset_list = zip(["train", "train_eval", "dev", "test"], [True, False, False, False]) 
        elif self.arg.dataset == 'CSL-Daily':
            dataset_list = zip(["train", "train_eval", "dev", "test"], [True, False, False, False])
        for idx, (mode, train_flag) in enumerate(dataset_list):
            arg = self.arg.feeder_args
            arg["prefix"] = self.arg.dataset_info['dataset_root']
            arg["mode"] = mode.split("_")[0]
            arg["transform_mode"] = train_flag
            self.dataset[mode] = self.feeder(kernel_size= self.kernel_sizes, dataset=self.arg.dataset, **arg)
            self.data_loader[mode] = self.build_dataloader(self.dataset[mode], mode, train_flag)
    
    @staticmethod
    def modified_weights(state_dict, modified=False):
        state_dict = {k: v for k, v in state_dict.items() if 'prems' not in k}
        state_dict = {k: v for k, v in state_dict.items() if 'conv1d.fc.weight' not in k}
        state_dict = {k: v for k, v in state_dict.items() if 'classifier.weight' not in k}
        state_dict = OrderedDict([(k.replace('.module', ''), v) for k, v in state_dict.items()])

        if not modified: return state_dict
        modified_dict = dict()
        return modified_dict
    
    def model_to_device(self, model):
        model = model.to(self.device.output_device)
        if len(self.device.gpu_list) > 1:
            model.conv2d = torch.nn.DataParallel(model.conv2d, device_ids=self.device.gpu_list, output_device=self.device.output_device)
            model = convert_model(model)
        model.cuda()
        return model
    
    def load_model_weights(self, model, weight_path):
        state_dict = torch.load(weight_path, map_location=torch.device('cpu'), weights_only=False)

        if len(self.arg.ignore_weights):
            for w in self.arg.ignore_weights:
                if state_dict.pop(w, None) is not None:
                    print('Successfully Remove Weights: {}.'.format(w))
                else:
                    print('Can Not Remove Weights: {}.'.format(w))
        weights = self.modified_weights(state_dict['model_state_dict'], False)
        msg = model.load_state_dict(weights, strict=True)
        print("Model Weights Loaded: {}".format(msg))
    
    def save_arg(self):
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir): os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f: yaml.dump(arg_dict, f)

    def init_fn(self, worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    def build_dataloader(self, dataset, mode, train_flag):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=train_flag,
            drop_last=train_flag,
            num_workers=0,
            collate_fn=self.feeder.collate_fn,
            pin_memory=True,
            worker_init_fn=self.init_fn,
        )



def import_class(name):
    components = name.rsplit('.', 1)
    mod = importlib.import_module(components[0])
    mod = getattr(mod, components[1])
    return mod

if __name__ == '__main__':
    sparser = utils.get_parser()
    p = sparser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            try:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)
            except AttributeError:
                default_arg = yaml.load(f)

        key = vars(p).keys()
        sparser.set_defaults(**default_arg)

    args = sparser.parse_args()
    with open(f"./configs/{args.dataset}.yaml", 'r') as f:
        args.dataset_info = yaml.load(f, Loader=yaml.FullLoader)

    processor = Processor(args)
    processor.start()

