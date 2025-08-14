import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import time
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
from utils.logger import Logger
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import get_cosine_schedule_with_warmup
from rouge_score import rouge_scorer

faulthandler.enable()
from datetime import datetime
import utils
from modules.sync_batchnorm import convert_model
from utils.evaluate import evaluate_results


class Processor():
    def __init__(self, arg):
        self.arg = arg

        self.arg.work_dir = self.arg.work_dir + '_' + datetime.now().strftime("%d_%H_%M_%S")
        if os.path.exists(self.arg.work_dir):
            answer = input('Remove work dir [yes, y, ok, 1]? ')
            if answer in ['yes','y','ok','1']:
                print('Dir removed!')
                shutil.rmtree(self.arg.work_dir, ignore_errors=True)
                os.makedirs(self.arg.work_dir)
            else:
                print('Dir Not removed!')
                self.arg.load_checkpoints = self.arg.work_dir + '/_best_model.pt'
        else:
            os.makedirs(self.arg.work_dir)

        if not self.arg.work_dir.endswith('/'):
            self.arg.work_dir = self.arg.work_dir + '/'

        self.rng = utils.RandomState(seed=self.arg.random_seed)
        self.dataset, self.data_loader = {}, {}

        self.arg.optimizer_args['num_epoch'] = self.arg.num_epoch

        self.main_logger = Logger(os.path.join(self.arg.work_dir, 'log.txt'))
        self.pred_logger = Logger(os.path.join(self.arg.work_dir, 'preds.txt'))
        for key, value in vars(self.arg).items():
            self.main_logger(f"{key}: {value}")

        self.model, self.optimizer, self.scheduler = self.loading()

    def calculate_bleu_and_rouge(self, gen_str, ref_str):
        smoothing = SmoothingFunction().method1
        b1, b2, b3, b4 = [], [], [], []
        rouge_l = []

        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

        for gen, ref in zip(gen_str, ref_str):
            ref_tokens = [ref.split()]
            gen_tokens = gen.split()

            b1.append(sentence_bleu(ref_tokens, gen_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing))
            b2.append(sentence_bleu(ref_tokens, gen_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing))
            b3.append(sentence_bleu(ref_tokens, gen_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing))
            b4.append(sentence_bleu(ref_tokens, gen_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing))

            rouge_l.append(scorer.score(ref, gen)['rougeL'].fmeasure)

        return {
            'BLEU-1': sum(b1) / len(b1),
            'BLEU-2': sum(b2) / len(b2),
            'BLEU-3': sum(b3) / len(b3),
            'BLEU-4': sum(b4) / len(b4),
            'ROUGE-L': sum(rouge_l) / len(rouge_l),
        }

    def start(self):
        self.main_logger("\n")
        self.main_logger("=" * 20 + " STARTING " + "=" * 20)

        self.main_logger(self.model)

        if self.arg.phase == 'train':
            train_loader = self.data_loader['train']
            val_loader = self.data_loader['dev']

            warmup = False
            patience = self.arg.patience
            least_loss = 1000
            best_b4 = 0
            for epoch in range(self.arg.num_epoch + self.arg.num_warmpup_epochs):
                total_loss = 0.0
                start_time = time.time()
                if epoch < self.arg.num_warmpup_epochs and not warmup:
                    self.main_logger("FROZE ALL")
                    self.model.freeze_all_parameters()
                    warmup = True
                elif epoch == self.arg.num_warmpup_epochs:
                    self.main_logger("UNFROZE LoRA")
                    self.model.unfreeze_lora_parameters()
                    warmup = False

                self.model.train()
                # for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.arg.num_epoch + self.arg.num_warmpup_epochs}")):
                for i, batch in enumerate(train_loader):
                    sign_features, sp_features, phm_features, labels, icl_text, sign_lengths, sp_lengths, phm_lengths = batch

                    sign_features = sign_features.to(self.arg.device)
                    sp_features = sp_features.to(self.arg.device) if sp_features is not None else None
                    phm_features = phm_features.to(self.arg.device) if phm_features is not None else None
                    sign_lengths = sign_lengths.to(self.arg.device)
                    sp_lengths = sp_lengths.to(self.arg.device) if sp_lengths is not None else None
                    phm_lengths = phm_lengths.to(self.arg.device) if phm_lengths is not None else None

                    loss = self.model(sign_features, sp_features, phm_features, sign_lengths, sp_lengths, phm_lengths, labels, icl_text, warmup=warmup)
                    total_loss += loss.item()

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    if not self.arg.score_dependent_scheduler:
                        self.scheduler.step()

                self.main_logger(f"Epoch {(epoch + 1)-self.arg.num_warmpup_epochs} - Loss: {total_loss / len(train_loader):.4f}")

                scores = None
                if not warmup:
                    gen_strings, ref_strings = [], []
                    self.model.eval()
                    # for i, batch in enumerate(tqdm(val_loader, desc="Validation")):
                    for i, batch in enumerate(val_loader):
                        sign_features, sp_features, phm_features, labels, icl_text, sign_lengths, sp_lengths, phm_lengths = batch

                        sign_features = sign_features.to(self.arg.device)
                        sp_features = sp_features.to(self.arg.device) if sp_features is not None else None
                        phm_features = phm_features.to(self.arg.device) if phm_features is not None else None
                        sign_lengths = sign_lengths.to(self.arg.device)
                        sp_lengths = sp_lengths.to(self.arg.device) if sp_lengths is not None else None
                        phm_lengths = phm_lengths.to(self.arg.device) if phm_lengths is not None else None

                        with torch.no_grad():
                            gen_str, ref_str = self.model(sign_features, sp_features, phm_features, sign_lengths, sp_lengths, phm_lengths, labels, icl_text, warmup=warmup)
                        gen_strings.extend(gen_str)
                        ref_strings.extend(ref_str)

                    self.pred_logger("=" * 10 + f"EPOCH {(epoch + 1)-self.arg.num_warmpup_epochs} - Generated and Reference Strings" + "=" * 10)
                    for i in range(min(5, len(gen_strings))):
                        self.pred_logger(f"Generated: {gen_strings[i]}")
                        self.pred_logger(f"Reference: {ref_strings[i]}")
                        self.pred_logger("-" * 50)
                    self.pred_logger("\n")

                    # scores = self.calculate_bleu_and_rouge(gen_strings, ref_strings)
                    scores = evaluate_results(gen_strings, ref_strings)
                    self.main_logger(f"Validation Scores > B1: {scores['BLEU-1']:.4f}, B2: {scores['BLEU-2']:.4f}, B3: {scores['BLEU-3']:.4f}, B4: {scores['BLEU-4']:.4f}, RG-L: {scores['ROUGE-L_F1']:.4f}")

                    if self.arg.monitor_bleu:
                        if scores['BLEU-4'] > best_b4:
                            best_b4 = scores['BLEU-4']
                            patience = self.arg.patience
                            self.main_logger(f"New best model found at epoch {epoch + 1} with BLEU-4 {best_b4:.4f}. Saving model...")
                            torch.save(self.model.state_dict(), self.arg.work_dir + '/best_model.pt')
                        elif epoch > self.arg.num_warmpup_epochs + 10:
                            patience -= 1
                            if patience <= 0:
                                self.main_logger("Early stopping triggered. No improvement in validation BLEU-4.")
                                break

                        if self.arg.score_dependent_scheduler:
                            self.scheduler.step(scores['BLEU-4'])

                    else:
                        if least_loss > total_loss/len(train_loader):
                            least_loss = total_loss/len(train_loader)
                            patience = self.arg.patience
                            self.main_logger(f"New best model found at epoch {epoch + 1} with loss {least_loss:.4f}. Saving model...")
                            torch.save(self.model.state_dict(), self.arg.work_dir + '/best_model.pt')
                        elif epoch > self.arg.num_warmpup_epochs + 10:
                            patience -= 1
                            if patience <= 0:
                                self.main_logger("Early stopping triggered. No improvement in validation loss.")
                                break

                        if self.arg.score_dependent_scheduler:
                            self.scheduler.step(total_loss / len(train_loader))
                

                current_lr = self.optimizer.param_groups[0]['lr']
                self.main_logger(f"Learning Rate: {current_lr} - Patience: [{patience}/{self.arg.patience}]")
                elapsed_time_mins = (time.time() - start_time) / 60
                self.main_logger(f"Elapsed Time: {elapsed_time_mins:.2f} mins")
                self.main_logger("\n")
                    

        assert os.path.exists(self.work_dir + '/best_model.pt'), "No best model found. Please train the model first."
        self.model.load_state_dict(torch.load(self.arg.work_dir + '/best_model.pt'))

        test_loader = self.data_loader['test']
        self.model.eval()

        all_gen_str = []
        all_ref_str = []

        with torch.no_grad():
            # for i, batch in enumerate(tqdm(test_loader)):
            for i, batch in enumerate(test_loader):
                sign_features, sp_features, phm_features, labels, icl_text, sign_lengths, sp_lengths, phm_lengths = batch

                sign_features = sign_features.to(self.arg.device)
                sp_features = sp_features.to(self.arg.device) if sp_features is not None else None
                phm_features = phm_features.to(self.arg.device) if phm_features is not None else None
                sign_lengths = sign_lengths.to(self.arg.device)
                sp_lengths = sp_lengths.to(self.arg.device) if sp_lengths is not None else None
                phm_lengths = phm_lengths.to(self.arg.device) if phm_lengths is not None else None

                gen_str, ref_str = self.model(sign_features, sp_features, phm_features, sign_lengths, sp_lengths, phm_lengths, labels, icl_text, warmup=warmup)
                all_gen_str.extend(gen_str)
                all_ref_str.extend(ref_str)

        # log all generated and reference strings
        self.pred_logger("\n\n")
        self.pred_logger("=" * 10 + "TEST - Generated and Reference Strings" + "=" * 10)
        for i in range(min(5, len(all_gen_str))):
            self.pred_logger(f"Generated: {all_gen_str[i]}")
            self.pred_logger(f"Reference: {all_ref_str[i]}")
            self.pred_logger("-" * 50)

        scores = self.calculate_bleu_and_rouge(all_gen_str, all_ref_str)
        self.main_logger(f"Test Scores > B1: {scores['BLEU-1']:.4f}, B2: {scores['BLEU-2']:.4f}, B3: {scores['BLEU-3']:.4f}, B4: {scores['BLEU-4']:.4f}, RG-L: {scores['ROUGE-L']:.4f}")
        self.pred_logger(f"Test Scores > B1: {scores['BLEU-1']:.4f}, B2: {scores['BLEU-2']:.4f}, B3: {scores['BLEU-3']:.4f}, B4: {scores['BLEU-4']:.4f}, RG-L: {scores['ROUGE-L']:.4f}")


    def loading(self):
        self.arg.llm_args["llm"] = self.arg.llm
        self.arg.llm_args["lang"] = self.arg.dataset_info["language"]

        self.arg.llm_args["include_sp"] = self.arg.include_sp
        self.arg.llm_args["include_pose"] = self.arg.include_pose
        self.arg.llm_args["p2hm"] = self.arg.p2hm
        self.arg.llm_args["sign_hidden_size"] = self.arg.sign_hidden_size
        self.arg.llm_args["sp_hidden_size"] = self.arg.sp_hidden_size
        self.arg.llm_args["pose_hidden_size"] = self.arg.pose_hidden_size

        self.load_data()

        llm_class = import_class(self.arg.model)
        llm = llm_class(self.arg.llm_args).to(self.arg.device)
        shutil.copy2(inspect.getfile(llm_class), self.arg.work_dir)

        optimizer = torch.optim.AdamW(
            llm.parameters(),
            lr=self.arg.optimizer_args['base_lr'],
            eps=1e-8, 
            weight_decay=0.01, 
            betas=(0.9, 0.98)
        )

        scheduler=None
        if self.arg.score_dependent_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max' if self.arg.monitor_bleu else 'min',
                factor=self.arg.optimizer_args['gamma'],
                patience=2,
            )
            self.main_logger("Using score-dependent learning rate scheduler.")

        else:
            scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=len(self.data_loader["train"])*5,
                num_training_steps=self.steps_info["train"],
            )
            self.main_logger("Using cosine learning rate scheduler.")
        return llm, optimizer, scheduler

    def load_data(self):
        self.ft_dataset = import_class(self.arg.feeder_ft)
        shutil.copy2(inspect.getfile(self.ft_dataset), self.arg.work_dir)
        
        self.steps_info = {}
        if self.arg.dataset == 'CSL':
            dataset_list = zip(["train", "dev"], [True, False])
        elif 'phoenix' in self.arg.dataset:
            dataset_list = zip(["train", "dev", "test"], [True, False, False]) 
        elif self.arg.dataset == 'CSL-Daily':
            dataset_list = zip(["train", "train_eval", "dev", "test"], [True, False, False, False])
        for idx, (mode, train_flag) in enumerate(dataset_list):
            self.dataset[mode] = self.ft_dataset(
                os.path.join(self.arg.dataset_info["precomputed_ft_dir"], mode), 
                os.path.join(self.arg.dataset_info["sp_path"], mode), 
                os.path.join(self.arg.dataset_info["phm_path"], mode), 
                mode=mode,
                include_sp=self.arg.include_sp,
                include_pose=self.arg.include_pose,
                p2hm=self.arg.p2hm
            )
            self.data_loader[mode] = torch.utils.data.DataLoader(
                self.dataset[mode],
                batch_size=self.arg.batch_size if mode == "train" else self.arg.test_batch_size,
                shuffle=train_flag,
                drop_last=train_flag,
                num_workers=0,
                collate_fn=self.ft_dataset.collate_fn,
                worker_init_fn=self.init_fn,
            )

            self.steps_info[mode] = (len(self.data_loader[mode]) * self.arg.num_epoch) 

    def init_fn(self, worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)


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

