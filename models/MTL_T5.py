import torch
import torch.nn as nn

class MTL_T5(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()
        self.model = model
        
        # Keep the original lm_head for SLT
        # Add an extra head for SLR
        self.slr_clf = nn.Linear(model.config.d_model, num_classes)
        self.ctc_loss = nn.CTCLoss(reduction='none', zero_infinity=False)

    def calc_ctc_loss(self, logits, gloss_labels, ft_lgt, label_lgt):
        return self.ctc_loss(
            logits.permute(1,0,2).contiguous().log_softmax(-1),
            gloss_labels,
            ft_lgt,
            label_lgt
        ).mean()

    def forward(self, joint_embeds, joint_mask, decoder_attention_mask, labels, gloss_labels, label_lengths):
        # Normal forward pass for SLT (uses model.lm_head internally)
        slt_outputs = self.model(
            inputs_embeds=joint_embeds,
            attention_mask=joint_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,              # T5 will compute loss internally
            return_dict=True,
            output_hidden_states=True,  # <-- ensures decoder hidden states are returned
        )

        # SLR: use decoder hidden states + custom head
        hidden_states = slt_outputs.decoder_hidden_states[-1]
        slr_logits = self.slr_clf(hidden_states)

        logits_lengths = [slr_logits.shape[1]]*slr_logits.shape[0]
        logits_lengths = torch.tensor(logits_lengths, device=slr_logits.device)

        slr_loss = self.calc_ctc_loss(slr_logits, gloss_labels, logits_lengths, label_lengths) if gloss_labels is not None else 0

        return {
            "loss": slt_outputs.loss + slr_loss,
        }

    def generate(self, **kwargs):
        return self.model.generate(**kwargs)
