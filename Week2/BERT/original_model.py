import torch

import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class JointEmbedding(nn.Module):

    def __init__(self, vocab_size, size) -> None:
        # why?
        super(JointEmbedding, self).__init__()

        self.size = size

        self.token_emb = nn.Embedding(vocab_size, size)
        self.segment_emb = nn.Embedding(vocab_size, size)

        self.norm = nn.LayerNorm(size)

    def forward(self, input_tensor):
        sentence_size = input_tensor.size(-1)
        pos_tensor = self.attention_position(self.size, input_tensor)

        segment_tensor = torch.zeros_like(input_tensor).to(device)
        segment_tensor[:, sentence_size//2 + 1:] = 1

        output = self.token_emb(input_tensor) + self.segment_emb(segment_tensor) + pos_tensor
        return self.norm(output)
    
    def attention_position(self, dim, input_tensor):
        batch_size = input_tensor.size(0)
        sentence_size = input_tensor.size(-1)

        pos = torch.arange(sentence_size, dtype=torch.long).to(device)
        d = torch.arange(dim, dtype=torch.long).to(device)
        d = (2 * d / dim)

        pos = pos.unsqueeze(1) / (1e4 ** d)
        pos[:, ::2], pos[:, 1::2] = torch.sin(pos[:, ::2]), torch.cos(pos[:, 1::2])

        return pos.expand(batch_size, *pos.size())
    
class AttentionHead(nn.Module):
    
    def __init__(self, dim_inp, dim_out) -> None:
        super(AttentionHead, self).__init__()
        
        self.dim_inp = dim_inp

        self.q = nn.Linear(dim_inp, dim_out)
        self.k = nn.Linear(dim_inp, dim_out)
        self.v = nn.Linear(dim_inp, dim_out)
    
    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.tensor = None):
        query, key, value = self.q(input_tensor), self.k(input_tensor), self.v(input_tensor)

        scale = query.size(1) ** 0.5
        scores = query @ key.transpose(1, 2) / scale

        scores.masked_fill_(attention_mask, -1e9)
        attn = F.softmax(scores, dim=-1)
        context = attn @ value

        return context

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, dim_inp, dim_out) -> None:
        super(MultiHeadAttention, self).__init__()

        self.heads = nn.ModuleList([
            AttentionHead(dim_inp, dim_out) for _ in range(num_heads)
        ])
        self.linear = nn.Linear(dim_out*num_heads, dim_inp)
        self.norm = nn.LayerNorm(dim_inp)

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor):
        s = [head(input_tensor, attention_mask) for head in self.heads]
        scores = torch.cat(s, dim=-1)
        scores = self.norm(self.linear(scores))

        return scores

class Encoder(nn.Module):

    def __init__(self, dim_inp, dim_out, attention_heads=4, dropout=0.1) -> None:
        super(Encoder, self).__init__()

        self.attention = MultiHeadAttention(attention_heads, dim_inp, dim_out)
        self.feedforward = nn.Sequential(
            nn.Linear(dim_inp, dim_out),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(dim_out, dim_inp),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim_inp)
    
    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor):
        context = self.attention(input_tensor, attention_mask)
        out = self.norm(self.feedforward(context))
        return out

class BERT(nn.Module):

    def __init__(self, vocab_size, dim_inp, dim_out, attention_heads) -> None:
        super(BERT, self).__init__()

        self.embedding = JointEmbedding(vocab_size, dim_inp)
        self.encoder = Encoder(dim_inp, dim_out, attention_heads)

        self.token_prediction_layer = nn.Linear(dim_inp, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.classification_layer = nn.Linear(dim_inp, 2)
    
    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor):
        embedded = self.embedding(input_tensor)
        encoded = self.encoder(embedded, attention_mask)

        token_predictions = self.token_prediction_layer(encoded)

        first_word = encoded[:, 0, :]
        return self.softmax(token_predictions), self.classification_layer(first_word)


if __name__=="__main__":
    seq = torch.zeros(32, 64, dtype=torch.long).to(device)
    seq_mask = torch.ones(64, dtype=torch.long).to(device)
    
    model = BERT(256, 128, 128*4, 4)
    out = model(seq, seq_mask)
    print(out[1].shape)