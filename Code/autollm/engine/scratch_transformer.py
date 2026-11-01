import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional

class CharTokenizer:
    """
    Lightweight, self-contained subword/char tokenizer for guaranteed offline CPU training.
    """
    def __init__(self):
        self.special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>", "<sep>"]
        self.char2idx: Dict[str, int] = {}
        self.idx2char: Dict[int, str] = {}
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3
        self.sep_id = 4
        self._init_vocab()

    def _init_vocab(self):
        for i, token in enumerate(self.special_tokens):
            self.char2idx[token] = i
            self.idx2char[i] = token

    def build_vocab(self, texts: List[str]):
        chars = sorted(list(set("".join(texts))))
        curr_idx = len(self.special_tokens)
        for c in chars:
            if c not in self.char2idx:
                self.char2idx[c] = curr_idx
                self.idx2char[curr_idx] = c
                curr_idx += 1

    @property
    def vocab_size(self) -> int:
        return len(self.char2idx)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = []
        if add_special_tokens:
            tokens.append(self.bos_id)
        for c in text:
            tokens.append(self.char2idx.get(c, self.unk_id))
        if add_special_tokens:
            tokens.append(self.eos_id)
        return tokens

    def decode(self, tokens: List[int]) -> str:
        chars = []
        for t in tokens:
            if t in [self.pad_id, self.bos_id, self.eos_id, self.sep_id]:
                continue
            chars.append(self.idx2char.get(t, ""))
        return "".join(chars)

    def save(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "char2idx": self.char2idx,
                "special_tokens": self.special_tokens
            }, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "CharTokenizer":
        tok = cls()
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok.char2idx = data["char2idx"]
        tok.idx2char = {int(v): k for k, v in tok.char2idx.items()}
        return tok


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Causal mask buffer
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        y = att @ v # (B, num_heads, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """
    Compact, highly performant Causal Small Language Model for CPU.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        max_seq_len: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.tok_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, max_seq_len, dropout)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Tie weights
        self.tok_embed.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)

        tok_emb = self.tok_embed(idx)
        pos_emb = self.pos_embed(pos)
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1), ignore_index=0)

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 150,
        temperature: float = 0.7,
        top_k: int = 20
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            # If EOS token generated (index 3)
            if idx_next.item() == 3:
                break
        return idx
