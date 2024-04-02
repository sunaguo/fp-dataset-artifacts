import torch.nn as nn
import torch.nn.functional as F

## ENCODER CLASS
class BOWEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, class_in):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.linear = nn.Linear(emb_dim,class_in)
        self.logits = None
        self.sm = nn.LogSoftmax(dim=-1)
    
    def forward(self, data, length):
        """
        @param data: matrix of size (batch_size, max_sentence_length). Each row in data represents a 
            review that is represented using n-gram index. Note that they are padded to have same length.
        @param length: an int tensor of size (batch_size), which represents the non-trivial (excludes padding)
            length of each sentences in the data.
        """
        out = self.embed(data)  ## batch x maxlen x emb
        # print("emb", out.shape)
        out = out.sum(dim=1)  ## batch x emb
        # print("sum", out.shape)
        out /= length.view(length.size()[0],1).expand_as(out).float()
        # print("avg", out.shape)
     
        out = self.linear(out.float())
        # print("out", out.shape)
        self.logits = out.detach()
    
        out = self.sm(out)
        return out