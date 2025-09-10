
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.tconv import TemporalConv
from modules.pose_encoder_old import PoseEncoder

from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM, Mxfp4Config
from peft import LoraConfig, get_peft_model
from torch.distributed.nn.functional import all_gather

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0
    

def safe_derangement(lst):
    lst = list(lst)
    n = len(lst)
    for _ in range(1000):  # max attempts
        shuffled = lst[:]
        random.shuffle(shuffled)
        if all(original != shuffled[i] for i, original in enumerate(lst)):
            return shuffled
    # fallback: just return original if derangement fails
    return lst

class DecoderOnlyModel(nn.Module):
    def __init__(self, arg):
        super(DecoderOnlyModel, self).__init__()
        self.arg = arg

        self.lora_r = arg["lora_r"]
        self.lora_alpha = arg["lora_alpha"]
        self.lora_dropout = arg["lora_dropout"]
        self.max_txt_len = arg["max_txt_len"]
        self.prompt = f'{arg["prompt"]} {arg["lang"]}'
        self.tuning_mode = arg["tuning_mode"]
        self.include_sp = arg["include_sp"]
        self.include_pose = arg["include_pose"]
        self.include_mo = arg["include_mo"]
        self.include_sign = arg["include_sign"]
        self.p2hm = arg["p2hm"]
        self.sp_hidden_size = arg["sp_hidden_size"]
        self.pose_hidden_size = arg["pose_hidden_size"]
        self.mo_hidden_size = arg["mo_hidden_size"]
        self.sign_hidden_size = arg["sign_hidden_size"]
        self.gloss_dict = arg["gloss_dict"]
        self.contrastive = arg["contrastive"]
        self.num_classes = len(self.gloss_dict) + 1
        self.pose_encoder_cfg = arg["pose_encoder_cfg"]
        self.hidden_size = arg["llm_hidden_size"]
        self.include_ctc = arg["include_ctc"]

        self._prepare_llm(arg["llm"])
        self._apply_lora()

        print("PROMPT:", self.prompt)

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params / 1_000_000:.2f}M, Trainable parameters: {trainable_params / 1_000_000:.2f}M")

    def _prepare_llm(self, model: str) -> None:
        quantization_config = Mxfp4Config(dequantize=True)
        model_kwargs = dict(
            attn_implementation="eager",
            quantization_config=quantization_config,
            use_cache=False,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model, 
            cache_dir="./data/models",
            **model_kwargs
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model, 
            cache_dir="./data/models",
            max_length=self.max_txt_len,
            use_fast=False  # 👈 important: force SentencePiece
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


        print(self.tokenizer.special_tokens_map)
        try:
            self.bos_token = self.tokenizer.special_tokens_map['bos_token']
            self.eos_token = self.tokenizer.special_tokens_map['eos_token']
            self.vision_start_token = "<|vision_start|>"
            self.vision_end_token = "<|vision_end|>"
        except:
            if "qwen" in model.lower():
                self.bos_token = "<|startoftext|>"
                self.eos_token = self.tokenizer.special_tokens_map['eos_token']

                print("Using Qwen special tokens.")

        self.sp_proj = nn.Linear(self.sp_hidden_size, self.hidden_size) if self.include_sp else None
        self.pose_proj = nn.Linear(self.pose_hidden_size, self.hidden_size) if self.include_pose else None
        self.mo_proj = nn.Linear(self.mo_hidden_size, self.hidden_size) if self.include_mo else None
        self.sign_proj = nn.Linear(self.sign_hidden_size, self.hidden_size) if self.include_sign else None
        self.fusion_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.model.config.hidden_size),
            nn.GELU(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
            nn.GELU(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
        )
        if not self.p2hm and self.include_pose:
            cfg = {
                'DSTA-Net': {
                    'net': [
                        (64, 64, 16, 7, 2), (64, 64, 16, 3, 1),
                        (64, 128, 32, 3, 1), (128, 128, 32, 3, 1),
                        (128, 256, 64, 3, 2), (256, 256, 64, 3, 1),
                        (256, 256, 64, 3, 1), (256, 256, 64, 3, 1)
                    ],
                    'body': [0, 1, 3, 5, 7, 9, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 2, 4, 6, 8, 10, 112, 113,
                            114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 23, 26, 29, 33, 36, 39, 41, 43, 46, 48,
                            53, 56, 59, 62, 65, 68, 71, 72, 73, 74, 75, 76, 77, 79, 80, 81],
                    'left': [0, 1, 3, 5, 7, 9, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
                    'right': [0, 2, 4, 6, 8, 10, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132],
                    'face': [23, 26, 29, 33, 36, 39, 41, 43, 46, 48, 53, 56, 59, 62, 65, 68, 71, 72, 73, 74, 75, 76, 77, 79, 80, 81],
                    'mouth': [71, 72, 73, 74, 75, 76, 77, 79, 80, 81]
                }
            }
            self.pose_encoder = PoseEncoder(cfg=cfg)

        self.temporal_encoder = TemporalConv(self.hidden_size, self.hidden_size, conv_type=2)
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))

    def freeze_all_parameters(self):
        for param in self.model.parameters():
            param.requires_grad = False

        print("All parameters are frozen for warmup.")

    def unfreeze_lora_parameters(self):
        for name, param in self.model.named_parameters():
            if "lora" in name:
                param.requires_grad = True

        print("LoRA parameters are unfrozen for training.")
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params / 1_000_000:.2f}M, Trainable parameters: {trainable_params / 1_000_000:.2f}M")
        print("\n\n")

    def _apply_lora(self) -> None:
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules="all-linear",
            # target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=self.lora_dropout,
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_config)
        print("LoRA adapter applied to Decoder-only model model.")

    def create_mask(self, seq_lengths: list[int], device="cpu"):
        lengths = torch.tensor(seq_lengths, dtype=torch.int32, device=device)
        max_len = lengths.max().item()
        range_row = torch.arange(max_len, dtype=torch.int32, device=device).expand(len(lengths), -1)
        lengths = lengths.unsqueeze(1)
        mask = range_row < lengths  # shape: (batch_size, max_len)
        return mask
    
    def prep_cont_tensors(self, visual_ft, labels):
        # Tokenize target texts
        output_tokens = self.tokenizer(
            labels,
            padding="longest",
            return_tensors="pt",
        ).to(visual_ft.device)
        
        # Get text embeddings
        with torch.no_grad():
            text_embeds = self.model.get_input_embeddings()(output_tokens.input_ids)
        
        # Mean pooling for visual and text embeddings
        image_embeds = visual_ft.mean(1)  # global pooling
        text_embeds = text_embeds.mean(1)  # global pooling
        
        # Normalize features
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        return image_embeds, text_embeds

    def calc_ctc_loss(self, logits, gloss_labels, ft_lgt, label_lgt):
        return self.ctc_loss(
            logits.permute(1,0,2).contiguous().log_softmax(-1),
            gloss_labels,
            ft_lgt,
            label_lgt
        ).mean()

    def _prepare_visual_input(self, sp_features, pose_features, mo_features, sign_features, sp_lengths, pose_lengths, mo_lengths, sign_lengths):
        """
        Project and combine sign, spatial, and pose features.
        Returns: visual_embeds (B, T, D), visual_lengths (List[int])
        """
        bs = sp_features.shape[0] if self.include_sp else pose_features.shape[0]

        if self.include_sp:
            sp_features = self.sp_proj(sp_features)
        if self.include_mo:
            mo_features = self.mo_proj(mo_features)
        if self.include_pose:
            if not self.p2hm:
                pose_features = self.pose_encoder({'keypoint': pose_features.float()})
                pose_lengths = torch.tensor([pose_features.shape[1]]*pose_features.shape[0], dtype=torch.long, device=pose_features.device)
            pose_features = self.pose_proj(pose_features)
        if self.include_sign:
            sign_features = self.sign_proj(sign_features)


        # Build joint visual token sequences
        joint_visual = []
        visual_lengths = []
        for i in range(bs):
            parts = []
            if self.include_sp:
                parts.append(sp_features[i, :sp_lengths[i]])
            if self.include_pose:
                parts.append(pose_features[i, :pose_lengths[i]])
            if self.include_mo:
                parts.append(mo_features[i, :mo_lengths[i]])
            if self.include_sign:
                parts.append(sign_features[i, :sign_lengths[i]])

            vis_tokens = torch.cat(parts, dim=0)
            joint_visual.append(vis_tokens)
            visual_lengths.append(vis_tokens.size(0))  # store actual length

        # Pad & encode
        joint_visual = pad_sequence(joint_visual, batch_first=True)  # (B, T, D)
        visual_conv_outputs = self.temporal_encoder(
            joint_visual.permute(0, 2, 1), 
            torch.tensor(visual_lengths, device=joint_visual.device)
        )

        visual_outputs = visual_conv_outputs['visual_feat'].permute(1, 0, 2)
        visual_lengths = visual_conv_outputs['feat_len'].to(torch.int).tolist()  # final lengths after encoder

        # Project fused visual tokens
        visual_tokens = self.fusion_proj(visual_outputs)

        return visual_tokens, visual_lengths


    def _prepare_joint_input(self, visual_tokens, visual_lengths, prompts, texts):
        """
        Combine visual embeddings with text prompt embeddings.
        Returns: joint_embeds, visual_embeds, prompt_lengths, joint_mask
        """
        bs = visual_tokens.size(0)

        prompts = [f"{self.bos_token}{prompt}" for prompt in prompts]
        texts = [f"{text}{self.eos_token}" for text in texts]

        # Tokenize prompts
        input_tokens = self.tokenizer(
            prompts, 
            padding="longest", 
            truncation=True, 
            return_tensors="pt"
        ).to(visual_tokens.device)
        text_tokens = self.tokenizer(
            texts, 
            padding="longest", 
            truncation=True, 
            return_tensors="pt"
        ).to(visual_tokens.device)

        prompt_embeds = self.model.get_input_embeddings()(input_tokens.input_ids)
        prompt_lengths = input_tokens.attention_mask.sum(dim=1).tolist()
        text_embeds = self.model.get_input_embeddings()(text_tokens.input_ids)
        text_lengths = text_tokens.attention_mask.sum(dim=1).tolist()

        # Combine per sample
        joint_embeds = []
        visual_embeds = []
        new_lengths = []
        for i in range(bs):
            vis = visual_tokens[i, :visual_lengths[i], :]
            prm = prompt_embeds[i, :prompt_lengths[i], :]
            txt = text_embeds[i, :text_lengths[i], :]

            if self.training:
                combined = torch.cat([prm, vis, txt], dim=0)
            else: 
                combined = torch.cat([prm, vis], dim=0)

            joint_embeds.append(combined)
            visual_embeds.append(vis)

            if self.training:
                new_lengths.append(visual_lengths[i] + prompt_lengths[i] + text_lengths[i])
            else:
                new_lengths.append(visual_lengths[i] + prompt_lengths[i])

        # Pad sequences
        joint_embeds = pad_sequence(joint_embeds, batch_first=True)
        visual_embeds = pad_sequence(visual_embeds, batch_first=True)

        # Attention mask for Decoder-only model
        joint_mask = self.create_mask(new_lengths, device=visual_tokens.device)

        labels = []
        if self.training:
            for i in range(bs):
                seq_len = joint_embeds.size(1)
                lbl = torch.full((seq_len,), -100, dtype=torch.long, device=visual_tokens.device)
                start = prompt_lengths[i] + visual_lengths[i]  # text starts after prompt + visual
                end = start + text_lengths[i]
                lbl[start:end] = text_tokens.input_ids[i, :text_lengths[i]]
                labels.append(lbl)

            labels = torch.stack(labels, dim=0)  # shape: [batch_size, seq_len]

        joint_embeds = joint_embeds.to(torch.bfloat16)
        return joint_embeds, visual_embeds, prompt_lengths, joint_mask, labels


    def forward(
        self, 
        sp_features, 
        pose_features,
        mo_features,
        sign_features,
        sp_lengths, 
        pose_lengths, 
        mo_lengths,
        sign_lengths,
        glosses, 
        texts, 
        icl_text,
        warmup=False
    ):
        bs = sp_features.shape[0] if self.include_sp else 0
        bs = pose_features.shape[0] if self.include_pose else bs
        bs = mo_features.shape[0] if self.include_mo else bs
        bs = sign_features.shape[0] if self.include_sign else bs

        english_texts = [icl_text[i].split('=')[0] for i in range(bs)]

        # Derange ICL text
        icl_text = safe_derangement(icl_text)  # ensure this returns same length list

        # Prepare prompts
        prompts = [
            f"{self.prompt}\n\nExamples:\n{icl_text[i]}" 
            for i in range(bs)
        ]

        # Prepare visual input
        visual_tokens, visual_lengths = self._prepare_visual_input(
            sp_features, pose_features, mo_features, sign_features,
            sp_lengths, pose_lengths, mo_lengths, sign_lengths
        )

        # Prepare joint input (visual + prompt embeddings)
        joint_embeds, visual_embeds, prompt_lengths, joint_mask, labels = self._prepare_joint_input(
            visual_tokens, visual_lengths, prompts, texts
        )

        # Tokenize texts
        output_tokens = self.tokenizer(
            texts, 
            padding="longest", 
            truncation=True, 
            return_tensors="pt"
        ).to(joint_embeds.device)

        if self.training:
            loss = 0
            if self.contrastive:
                local_img, local_txt = self.prep_cont_tensors(visual_embeds, english_texts)

                # Differentiable gather -> returns [B*world, D]
                global_img = all_gather(local_img)
                global_txt = all_gather(local_txt)

                if isinstance(global_img, (tuple, list)):
                    global_img = torch.cat(global_img, dim=0)
                if isinstance(global_txt, (tuple, list)):
                    global_txt = torch.cat(global_txt, dim=0)

                logit_scale = self.logit_scale.exp()
                logits_per_text = torch.matmul(global_txt.to(global_img.dtype), global_img.t()) * logit_scale

                # Now your clip_loss works as-is
                loss = clip_loss(logits_per_text)

            if not warmup or not self.contrastive:
                outputs = self.model(
                    inputs_embeds=joint_embeds,
                    attention_mask=joint_mask,
                    labels=labels,
                    output_hidden_states=True,
                    return_dict=True
                )
                loss += outputs.loss

            return loss

        else:
            generated = self.model.generate(
                inputs_embeds=joint_embeds,
                attention_mask=joint_mask,
                num_beams=5,
                max_new_tokens=self.max_txt_len,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=False,
                temperature=None,
                top_p=None
            )

            generated_strings = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            generated_strings = [gen.lower() for gen in generated_strings]

            reference_strings = self.tokenizer.batch_decode(output_tokens.input_ids, skip_special_tokens=True)
            reference_strings = [ref.lower() for ref in reference_strings]

            return generated_strings, reference_strings
