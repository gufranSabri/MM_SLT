
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.tconv import TemporalConv

from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, T5ForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0
    
def derangement(lst):
    assert len(lst) > 1, "List must have at least two elements."
    lst = list(lst)
    
    while True:
        shuffled = lst[:]
        random.shuffle(shuffled)
        if all(original != shuffled[i] for i, original in enumerate(lst)):
            return shuffled

class FlanT5(nn.Module):
    def __init__(self, arg):
        super(FlanT5, self).__init__()
        self.arg = arg

        self.lora_r = arg["lora_r"]
        self.lora_alpha = arg["lora_alpha"]
        self.lora_dropout = arg["lora_dropout"]
        self.max_txt_len = arg["max_txt_len"]
        self.prompt = f'{arg["prompt"]} {arg["lang"]}'
        self.tuning_mode = arg["tuning_mode"]
        self.include_sp = arg["include_sp"]
        self.include_pose = arg["include_pose"]
        self.p2hm = arg["p2hm"]
        self.sign_hidden_size = arg["sign_hidden_size"]
        self.sp_hidden_size = arg["sp_hidden_size"]
        self.pose_hidden_size = arg["pose_hidden_size"]

        self._prepare_llm(arg["llm"])
        self._apply_lora()

        print("PROMPT:", self.prompt)

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params / 1_000_000:.2f}M, Trainable parameters: {trainable_params / 1_000_000:.2f}M")

    def _prepare_llm(self, t5_model: str) -> None:
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            t5_model, 
            cache_dir="./data/models",
        )
        
        self.t5_tokenizer = AutoTokenizer.from_pretrained(
            t5_model, 
            cache_dir="./data/models",
            max_length=self.max_txt_len,
        )

        self.sign_proj = nn.Linear(self.sign_hidden_size, 768)
        self.sp_proj = nn.Linear(self.sp_hidden_size, 768) if self.include_sp else None
        self.fusion_proj = nn.Sequential(
            nn.Linear(768, self.t5_model.config.d_model),
            nn.GELU(),
            nn.Linear(self.t5_model.config.d_model, self.t5_model.config.d_model),
            nn.GELU(),
            nn.Linear(self.t5_model.config.d_model, self.t5_model.config.d_model),
        )

        conv_type = 0 if not (self.include_sp or self.include_pose) else 2
        self.temporal_encoder = TemporalConv(768, 768, conv_type=conv_type)
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))

    def freeze_all_parameters(self):
        for param in self.t5_model.parameters():
            param.requires_grad = False

        print("All parameters are frozen for warmup.")

    def unfreeze_lora_parameters(self):
        for name, param in self.t5_model.named_parameters():
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
            target_modules=["q", "v"],
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        self.t5_model = get_peft_model(self.t5_model, lora_config)
        print("LoRA adapter applied to T5 model.")

    def create_mask(self, seq_lengths: list[int], device="cpu"):
        lengths = torch.tensor(seq_lengths, dtype=torch.int32, device=device)
        max_len = lengths.max().item()
        range_row = torch.arange(max_len, dtype=torch.int32, device=device).expand(len(lengths), -1)
        lengths = lengths.unsqueeze(1)
        mask = range_row < lengths  # shape: (batch_size, max_len)
        return mask
    
    def visual_textual_align(self, sign_features, labels):
        # Tokenize target texts
        output_tokens = self.t5_tokenizer(
            labels,
            padding="longest",
            return_tensors="pt",
        ).to(sign_features.device)
        
        # Get text embeddings
        with torch.no_grad():
            text_embeds = self.t5_model.encoder.embed_tokens(output_tokens.input_ids)
        
        # Mean pooling for visual and text embeddings
        image_embeds = sign_features.mean(1)  # global pooling
        text_embeds = text_embeds.mean(1)  # global pooling
        
        # Normalize features
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        # Calculate cosine similarities with temperature scaling
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale

        # Calculate contrastive loss
        loss = clip_loss(logits_per_text)
        
        return loss

    def _prepare_visual_input(self, sign_features, sp_features, phm_features, sign_lengths, sp_lengths, phm_lengths):
        """
        Project and combine sign, spatial, and pose features.
        Returns: visual_embeds (B, T, D), visual_lengths (List[int])
        """
        bs = sign_features.shape[0]

        # Project main features
        sign_features = self.sign_proj(sign_features)

        # Optional projections
        if self.include_sp:
            sp_features = self.sp_proj(sp_features)
        if self.include_pose:
            phm_features = self.phm_v2lm_proj(phm_features)

        # Build joint visual token sequences
        joint_visual = []
        visual_lengths = []
        for i in range(bs):
            parts = [sign_features[i, :sign_lengths[i]]]
            if self.include_sp:
                parts.append(sp_features[i, :sp_lengths[i]])
            if self.include_pose:
                parts.append(phm_features[i, :phm_lengths[i]])

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


    def _prepare_joint_input(self, visual_tokens, visual_lengths, prompts):
        """
        Combine visual embeddings with text prompt embeddings.
        Returns: joint_embeds, visual_embeds, prompt_lengths, joint_mask
        """
        bs = visual_tokens.size(0)

        # Tokenize prompts
        input_tokens = self.t5_tokenizer(
            prompts, 
            padding="longest", 
            truncation=True, 
            return_tensors="pt"
        ).to(visual_tokens.device)

        prompt_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
        prompt_lengths = input_tokens.attention_mask.sum(dim=1).tolist()

        # Combine per sample
        joint_embeds = []
        visual_embeds = []
        new_lengths = []
        for i in range(bs):
            vis = visual_tokens[i, :visual_lengths[i], :]
            prm = prompt_embeds[i, :prompt_lengths[i], :]
            combined = torch.cat([vis, prm], dim=0)
            joint_embeds.append(combined)
            visual_embeds.append(vis)
            new_lengths.append(visual_lengths[i] + prompt_lengths[i])

        # Pad sequences
        joint_embeds = pad_sequence(joint_embeds, batch_first=True)
        visual_embeds = pad_sequence(visual_embeds, batch_first=True)

        # Attention mask for T5
        joint_mask = self.create_mask(new_lengths, device=visual_tokens.device)

        return joint_embeds, visual_embeds, prompt_lengths, joint_mask


    def forward(
        self, 
        sign_features, 
        sp_features, 
        phm_features,
        sign_lengths, 
        sp_lengths, 
        phm_lengths, 
        labels, 
        icl_text,
        warmup=False
    ):
        bs = sign_features.shape[0]

        # Derange ICL text
        icl_text = derangement(icl_text)  # ensure this returns same length list

        # Prepare prompts
        prompts = [
            f"{self.prompt}\n\nExamples:\n{icl_text[i]}" 
            for i in range(bs)
        ]

        # Prepare visual input
        visual_tokens, visual_lengths = self._prepare_visual_input(
            sign_features, sp_features, phm_features, 
            sign_lengths, sp_lengths, phm_lengths
        )

        # Prepare joint input (visual + prompt embeddings)
        joint_embeds, visual_embeds, prompt_lengths, joint_mask = self._prepare_joint_input(
            visual_tokens, visual_lengths, prompts
        )

        # Tokenize labels
        output_tokens = self.t5_tokenizer(
            labels, 
            padding="longest", 
            truncation=True, 
            return_tensors="pt"
        ).to(joint_embeds.device)

        targets = output_tokens.input_ids.masked_fill(
            output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
        )

        if self.training:
            # Visual-textual alignment loss
            loss = self.visual_textual_align(visual_embeds, labels)

            if not warmup:
                outputs = self.t5_model(
                    inputs_embeds=joint_embeds,
                    attention_mask=joint_mask,
                    decoder_attention_mask=output_tokens.attention_mask,
                    labels=targets,
                    output_hidden_states=True,
                    return_dict=True
                )
                loss += outputs.loss

            return loss

        else:
            # Generation (avoid beam search + sampling conflict)
            generated = self.t5_model.generate(
                inputs_embeds=joint_embeds,
                attention_mask=joint_mask,
                num_beams=5,
                max_length=self.max_txt_len,
                do_sample=False  # beam search only
            )

            generated_strings = self.t5_tokenizer.batch_decode(generated, skip_special_tokens=True)
            generated_strings = [gen.lower() for gen in generated_strings]

            reference_strings = self.t5_tokenizer.batch_decode(output_tokens.input_ids, skip_special_tokens=True)
            reference_strings = [ref.lower() for ref in reference_strings]

            return generated_strings, reference_strings

