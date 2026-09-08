import torch
from torch import nn


TYPE_SPECIAL = 0
TYPE_WORD = 1
TYPE_NUMBER = 2
TYPE_PUNCT = 3
TYPE_OPERATOR = 4
TYPE_CODE = 5


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
            q, k, v, attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        return self.out(y.transpose(1, 2).contiguous().view(b, t, c))


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


class LowRankSpecialist(nn.Module):
    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        self.act = nn.GELU()
        self.scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(x))) * self.scale


class SpecialistBank(nn.Module):
    def __init__(self, dim: int, type_count: int, specialists: int = 4):
        super().__init__()
        rank = max(8, dim // 4)
        self.type_bias = nn.Embedding(type_count, specialists)
        self.gate = nn.Linear(dim, specialists, bias=False)
        self.experts = nn.ModuleList([LowRankSpecialist(dim, rank) for _ in range(specialists)])
        self.specialists = specialists

    def forward(self, x: torch.Tensor, token_types: torch.Tensor) -> torch.Tensor:
        gate_logits = self.gate(x) + self.type_bias(token_types)
        k = min(2, self.specialists)
        topv, topi = torch.topk(gate_logits, k=k, dim=-1)
        topw = torch.softmax(topv.float(), dim=-1).to(x.dtype)
        result = torch.zeros_like(x)
        for expert_idx, expert in enumerate(self.experts):
            active = topi.eq(expert_idx)
            if not active.any():
                continue
            value = expert(x)
            weight = torch.where(active, topw, torch.zeros_like(topw)).sum(dim=-1, keepdim=True)
            result = result + value * weight
        return result


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, type_count: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, 4.0, dropout)
        self.specialists = SpecialistBank(dim, type_count)

    def forward(self, x: torch.Tensor, token_types: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        hidden = self.norm2(x)
        x = x + self.mlp(hidden) + self.specialists(hidden, token_types)
        return x


class LatentReasoningMemory(nn.Module):
    def __init__(self, dim: int, slots: int = 4):
        super().__init__()
        state_dim = max(16, dim // 8)
        self.state_in = nn.Linear(dim, state_dim, bias=False)
        self.state_out = nn.Linear(state_dim, dim, bias=False)
        self.state_gate = nn.Linear(dim, 1, bias=False)
        self.write_gate = nn.Linear(dim, slots, bias=False)
        self.write_value = nn.Linear(dim, dim, bias=False)
        self.query = nn.Linear(dim, dim, bias=False)
        self.slot_keys = nn.Parameter(torch.randn(slots, dim) * 0.02)
        self.memory_gate = nn.Linear(dim, 1, bias=False)
        self.slots = slots

    @staticmethod
    def _active_think(input_ids: torch.Tensor, think_id: int, think_end_id: int) -> torch.Tensor:
        opened = input_ids.eq(think_id).to(torch.int64)
        closed = input_ids.eq(think_end_id).to(torch.int64)
        depth = (opened.cumsum(dim=1) - closed.cumsum(dim=1)).clamp(min=0)
        return depth.gt(0)

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor, think_id: int, think_end_id: int) -> torch.Tensor:
        active = self._active_think(input_ids, think_id, think_end_id)
        mask = active.to(x.dtype).unsqueeze(-1)
        counts = mask.cumsum(dim=1).clamp_min(1.0)
        cumulative = (x * mask).cumsum(dim=1) / counts
        state_latent = torch.tanh(self.state_in(cumulative))
        state = self.state_out(state_latent)
        state = state * torch.sigmoid(self.state_gate(cumulative))
        write_logits = self.write_gate(cumulative)
        writes = torch.softmax(write_logits.float(), dim=-1).to(x.dtype)
        values = torch.tanh(self.write_value(cumulative))
        memory = (writes.unsqueeze(-1) * values.unsqueeze(2)).cumsum(dim=1)
        write_norm = writes.cumsum(dim=1).unsqueeze(-1).clamp_min(1e-4)
        memory = memory / write_norm
        query = self.query(cumulative)
        read_weights = torch.softmax(
            torch.einsum("btd,sd->bts", query.float(), self.slot_keys.float()), dim=-1
        ).to(x.dtype)
        read = torch.einsum("bts,btsd->btd", read_weights, memory)
        latent = state + read
        gated = latent * torch.sigmoid(self.memory_gate(cumulative))
        return x + mask * gated


class TiedLMHead(nn.Module):
    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.embedding = embedding

    @property
    def vocab_size(self) -> int:
        return self.embedding.num_embeddings

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.embedding.weight.transpose(0, 1))

    def loss(self, x: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.logits(x)
        flat_logits = logits.reshape(-1, self.vocab_size)
        flat_targets = targets.reshape(-1)
        token_loss = nn.functional.cross_entropy(flat_logits.float(), flat_targets, reduction="none")
        if weights is None:
            return token_loss.mean()
        flat_weights = weights.reshape(-1).to(token_loss.dtype)
        weight_sum = flat_weights.sum()
        if weight_sum.item() <= 0.0:
            raise ValueError("loss_weights must contain at least one positive weight")
        return (token_loss * flat_weights).sum() / weight_sum


class LLN(nn.Module):
    ARCHITECTURE_VERSION = 4
    THINK_TOKEN_ID = 6
    THINK_END_TOKEN_ID = 7

    def __init__(self, vocab_size: int, dim: int = 512, layers: int = 8, heads: int = 8,
                 max_seq_len: int = 256, dropout: float = 0.0, recurrent_steps: int = 2,
                 output_clusters: int = 120, memory_slots: int = 4, type_count: int = 6):
        super().__init__()
        if recurrent_steps < 1:
            raise ValueError("recurrent_steps must be >= 1")
        self.vocab_size = vocab_size
        self.dim = dim
        self.layers = layers
        self.heads = heads
        self.max_seq_len = max_seq_len
        self.recurrent_steps = recurrent_steps
        self.output_clusters = output_clusters
        self.memory_slots = memory_slots
        self.type_count = type_count
        self.token = nn.Embedding(vocab_size, dim)
        self.token_type = nn.Embedding(type_count, dim)
        self.token_cluster = nn.Embedding(max(1, min(output_clusters, vocab_size)), dim)
        self.position = nn.Embedding(max_seq_len, dim)
        self.blocks = nn.ModuleList([Block(dim, heads, type_count, dropout) for _ in range(layers)])
        self.latent_memory = LatentReasoningMemory(dim, memory_slots)
        self.norm = nn.LayerNorm(dim)
        self.lm_head = TiedLMHead(self.token)
        self.register_buffer("token_type_map", torch.zeros(vocab_size, dtype=torch.long), persistent=True)
        self.apply(self._init_weights)

    @property
    def cluster_size(self) -> int:
        return self.lm_head.vocab_size

    def set_token_types(self, token_types: list[int] | torch.Tensor) -> None:
        values = torch.as_tensor(token_types, dtype=torch.long, device=self.token_type_map.device)
        if values.numel() != self.vocab_size:
            raise ValueError(f"token_types must contain {self.vocab_size} entries")
        self.token_type_map.copy_(values.clamp(min=0, max=self.type_count - 1))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _embed(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, t = input_ids.shape
        token_types = self.token_type_map[input_ids]
        clusters = torch.div(input_ids, max(1, self.output_clusters), rounding_mode="floor").clamp(
            max=self.token_cluster.num_embeddings - 1
        )
        pos = torch.arange(t, device=input_ids.device)
        x = (self.token(input_ids) + self.token_type(token_types) + self.token_cluster(clusters)
             + self.position(pos)[None, :, :])
        return x, token_types

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None,
                loss_weights: torch.Tensor | None = None):
        _, t = input_ids.shape
        if t > self.max_seq_len:
            raise ValueError(f"sequence length {t} > max_seq_len {self.max_seq_len}")
        x, token_types = self._embed(input_ids)
        for _ in range(self.recurrent_steps):
            for block in self.blocks:
                x = block(x, token_types)
            x = self.latent_memory(x, input_ids, self.THINK_TOKEN_ID, self.THINK_END_TOKEN_ID)
        x = self.norm(x)
        logits = self.lm_head.logits(x)
        loss = None if targets is None else self.lm_head.loss(x, targets, weights=loss_weights)
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 128, temperature: float = 1.0,
                 repetition_penalty: float = 1.15, no_repeat_ngram_size: int = 3,
                 repetition_window: int = 64, frequency_penalty: float = 0.08,
                 presence_penalty: float = 0.20, hard_repeat_threshold: int = 6,
                 stop_ids=None):
        self.eval()
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        if repetition_penalty < 1.0:
            raise ValueError("repetition_penalty must be >= 1.0")
        if no_repeat_ngram_size < 0:
            raise ValueError("no_repeat_ngram_size must be >= 0")
        if repetition_window < 0:
            raise ValueError("repetition_window must be >= 0")
        if frequency_penalty < 0.0 or presence_penalty < 0.0:
            raise ValueError("frequency_penalty and presence_penalty must be >= 0")
        if hard_repeat_threshold < 2:
            raise ValueError("hard_repeat_threshold must be >= 2")
        stop_ids = set(stop_ids or {2})

        for _ in range(max_new_tokens):
            x = input_ids[:, -self.max_seq_len:]
            logits, _ = self(x)
            next_logits = logits[:, -1, :].float().clone()

            for batch_idx in range(input_ids.size(0)):
                recent = input_ids[batch_idx, -repetition_window:] if repetition_window > 0 else input_ids[batch_idx]
                recent_list = [int(token) for token in recent.tolist()]
                counts = {}
                for token_id in recent_list:
                    counts[token_id] = counts.get(token_id, 0) + 1

                if repetition_penalty > 1.0:
                    for token_id in counts:
                        value = next_logits[batch_idx, token_id]
                        next_logits[batch_idx, token_id] = (
                            value * repetition_penalty if value < 0 else value / repetition_penalty
                        )

                if presence_penalty > 0.0:
                    for token_id in counts:
                        next_logits[batch_idx, token_id] -= presence_penalty

                if frequency_penalty > 0.0:
                    for token_id, count in counts.items():
                        next_logits[batch_idx, token_id] -= frequency_penalty * count

                # Hard block tokens that have become pathological repetitions.
                if hard_repeat_threshold > 0:
                    blocked = [token_id for token_id, count in counts.items() if count >= hard_repeat_threshold]
                    if blocked:
                        next_logits[batch_idx, blocked] = float("-inf")

                if no_repeat_ngram_size >= 2 and input_ids.size(1) >= no_repeat_ngram_size - 1:
                    tokens = input_ids[batch_idx].tolist()
                    prefix = tuple(tokens[-(no_repeat_ngram_size - 1):])
                    banned = set()
                    for i in range(len(tokens) - no_repeat_ngram_size + 1):
                        ngram = tuple(tokens[i:i + no_repeat_ngram_size])
                        if ngram[:-1] == prefix:
                            banned.add(ngram[-1])
                    if banned:
                        next_logits[batch_idx, list(banned)] = float("-inf")

            if temperature == 1.0:
                next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                probabilities = torch.softmax(next_logits / temperature, dim=-1)
                next_id = torch.multinomial(probabilities, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
            if all(int(next_id[i, 0]) in stop_ids for i in range(next_id.size(0))):
                break
        return input_ids


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def parameter_size_mb(model: nn.Module, bytes_per_param: int = 4) -> float:
    return parameter_count(model) * bytes_per_param / (1024 ** 2)
