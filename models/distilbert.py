import torch
import torch.nn as nn
import transformers

class DistilBERTClass(torch.nn.Module):
    def __init__(self, n_class, model_path):
        super(DistilBERTClass, self).__init__()
        self.distill_bert = transformers.DistilBertModel.from_pretrained(model_path)
        self.drop = nn.Dropout(0.3)
        self.l0 = nn.Linear(768, n_class)
        torch.nn.init.normal_(self.l0.weight, std=0.02)
    
    def forward(self, ids, mask):
        distilbert_output = self.distill_bert(ids, mask)
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        output_1 = self.drop(pooled_output)
        output = self.l0(output_1)
        return output