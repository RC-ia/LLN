import math
import torch
from torch import nn


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.out(y)


class MLP(nn.Module):
    def __init__(self, dim: int, multiplier: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * multiplier)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, 4.0, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class LLN(nn.Module):
    """Numeric-token language model.

    The external dictionary owns the mapping word <-> integer ID. The network
    only receives integer IDs and predicts integer IDs. Embeddings are learned
    internally, while no textual vocabulary logic lives in the model.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 512,
        layers: int = 8,
        heads: int = 8,
        max_seq_len: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.layers = layers
        self.heads = heads
        self.max_seq_len = max_seq_len

        self.token = nn.Embedding(vocab_size, dim)
        self.position = nn.Embedding(max_seq_len, dim)
        self.blocks = nn.ModuleList([
            Block(dim, heads, dropout) for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None):
        _, t = input_ids.shape
        if t > self.max_seq_len:
            raise ValueError(f"sequence length {t} > max_seq_len {self.max_seq_len}")
        pos = torch.arange(t, device=input_ids.device)
        x = self.token(input_ids) + self.position(pos)[None, :, :]
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm(x))

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, self.vocab_size), targets.reshape(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
    ):
        self.eval()
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")

        for _ in range(max_new_tokens):
            x = input_ids[:, -self.max_seq_len:]
            logits, _ = self(x)
            next_logits = logits[:, -1, :]

            if temperature == 1.0:
                next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                probabilities = torch.softmax(next_logits / temperature, dim=-1)
                next_id = torch.multinomial(probabilities, num_samples=1)

            input_ids = torch.cat([input_ids, next_id], dim=1)
            if (next_id == 2).all():
                break
        return input_ids


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def parameter_size_mb(model: nn.Module, bytes_per_param: int = 4) -> float:
    return parameter_count(model) * bytes_per_param / (1024 ** 2)
