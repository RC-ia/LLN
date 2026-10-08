import math

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    """Llama-style RMS normalization, computed in float32 for stability."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        normalized = x_float * torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normalized.to(dtype=x.dtype) * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.size(-1) // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings (RoPE), applied to Q and K."""

    def __init__(self, head_dim: int, theta: float = 10_000.0):
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        if theta <= 0:
            raise ValueError("rope_theta must be positive")
        self.head_dim = head_dim
        self.theta = float(theta)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, position_offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.size(-2)
        positions = torch.arange(
            position_offset,
            position_offset + seq_len,
            device=q.device,
            dtype=torch.float32,
        )
        inv_freq = 1.0 / (
            self.theta ** (
                torch.arange(0, self.head_dim, 2, device=q.device, dtype=torch.float32)
                / self.head_dim
            )
        )
        freqs = torch.outer(positions, inv_freq)
        angles = torch.cat((freqs, freqs), dim=-1)[None, None, :, :]
        cos = angles.cos().to(dtype=q.dtype)
        sin = angles.sin().to(dtype=q.dtype)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        return q, k


class CausalSelfAttention(nn.Module):
    """Causal attention with grouped-query attention (GQA) and a compact KV cache."""

    def __init__(
        self,
        dim: int,
        heads: int,
        kv_heads: int,
        dropout: float = 0.0,
        rope_theta: float = 10_000.0,
    ):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        if heads % kv_heads:
            raise ValueError("heads must be divisible by kv_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = dim // heads
        self.groups = heads // kv_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(dim, heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim, theta=rope_theta)

    def _project(
        self, x: torch.Tensor, position_offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.rotary(q, k, position_offset=position_offset)
        return q, k, v

    def _expand_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.groups == 1:
            return x
        return x.repeat_interleave(self.groups, dim=1)

    def _attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_causal: bool,
    ) -> torch.Tensor:
        k = self._expand_kv(k)
        v = self._expand_kv(v)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        batch, _, seq_len, _ = y.shape
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, self.heads * self.head_dim)
        return self.o_proj(y)

    def forward(
        self, x: torch.Tensor, return_kv: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = self._project(x)
        output = self._attend(q, k, v, is_causal=True)
        if return_kv:
            return output, k, v
        return output

    def allocate_cache(
        self, batch_size: int, max_seq_len: int, device, dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (batch_size, self.kv_heads, max_seq_len, self.head_dim)
        return (
            torch.empty(shape, device=device, dtype=dtype),
            torch.empty(shape, device=device, dtype=dtype),
        )

    def forward_step(
        self,
        x: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        cache_pos: int,
    ) -> torch.Tensor:
        q, k, v = self._project(x, position_offset=cache_pos)
        cache_k[:, :, cache_pos:cache_pos + 1, :].copy_(k)
        cache_v[:, :, cache_pos:cache_pos + 1, :].copy_(v)
        k_all = cache_k[:, :, :cache_pos + 1, :]
        v_all = cache_v[:, :, :cache_pos + 1, :]
        return self._attend(q, k_all, v_all, is_causal=False)


class SwiGLU(nn.Module):
    """Llama-style gated feed-forward network."""

    def __init__(self, dim: int, dropout: float = 0.0, multiple_of: int = 64):
        super().__init__()
        hidden_dim = math.ceil((8 * dim / 3) / multiple_of) * multiple_of
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        kv_heads: int,
        dropout: float = 0.0,
        rope_theta: float = 10_000.0,
    ):
        super().__init__()
        self.input_layernorm = RMSNorm(dim)
        self.self_attn = CausalSelfAttention(
            dim=dim,
            heads=heads,
            kv_heads=kv_heads,
            dropout=dropout,
            rope_theta=rope_theta,
        )
        self.post_attention_layernorm = RMSNorm(dim)
        self.mlp = SwiGLU(dim, dropout=dropout)

    def forward(
        self, x: torch.Tensor, return_kv: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if return_kv:
            attention, k, v = self.self_attn(self.input_layernorm(x), return_kv=True)
        else:
            attention = self.self_attn(self.input_layernorm(x))
            k = v = None
        x = x + attention
        x = x + self.mlp(self.post_attention_layernorm(x))
        if return_kv:
            return x, k, v
        return x

    def forward_step(
        self,
        x: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        cache_pos: int,
    ) -> torch.Tensor:
        x = x + self.self_attn.forward_step(
            self.input_layernorm(x), cache_k, cache_v, cache_pos
        )
        return x + self.mlp(self.post_attention_layernorm(x))


class LLN(nn.Module):
    """Dense decoder-only Transformer inspired by modern Llama-family models."""

    ARCHITECTURE_VERSION = 6
    THINK_TOKEN_ID = 6
    THINK_END_TOKEN_ID = 7

    def __init__(
        self,
        vocab_size: int,
        dim: int = 512,
        layers: int = 8,
        heads: int = 8,
        kv_heads: int = 4,
        max_seq_len: int = 256,
        dropout: float = 0.0,
        rope_theta: float = 10_000.0,
    ):
        super().__init__()
        if vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if dim < 1 or layers < 1:
            raise ValueError("dim and layers must be positive")
        if max_seq_len < 1:
            raise ValueError("max_seq_len must be positive")
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        if (dim // heads) % 2:
            raise ValueError("dim // heads must be even for RoPE")
        if kv_heads < 1 or heads % kv_heads:
            raise ValueError("kv_heads must be positive and divide heads")

        self.vocab_size = vocab_size
        self.dim = dim
        self.layers = layers
        self.heads = heads
        self.kv_heads = kv_heads
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.rope_theta = rope_theta

        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    heads=heads,
                    kv_heads=kv_heads,
                    dropout=dropout,
                    rope_theta=rope_theta,
                )
                for _ in range(layers)
            ]
        )
        self.norm = RMSNorm(dim)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        # The output projection is weight-tied to the input token embeddings.
        return F.linear(self.norm(x), self.token_embedding.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        loss_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.size(1) > self.max_seq_len:
            raise ValueError(
                f"sequence length {input_ids.size(1)} > max_seq_len {self.max_seq_len}"
            )
        x = self.embedding_dropout(self.token_embedding(input_ids))
        for block in self.blocks:
            x = block(x)
        logits = self._logits(x)

        loss = None
        if targets is not None:
            flat_logits = logits.reshape(-1, self.vocab_size).float()
            flat_targets = targets.reshape(-1)
            token_loss = F.cross_entropy(flat_logits, flat_targets, reduction="none")
            if loss_weights is None:
                loss = token_loss.mean()
            else:
                weights = loss_weights.reshape(-1).to(token_loss.dtype)
                loss = (token_loss * weights).sum() / weights.sum().clamp_min(1e-12)
        return logits, loss

    @staticmethod
    def _adjust_generation_logits(
        next_logits: torch.Tensor,
        history: torch.Tensor,
        repetition_penalty: float,
        no_repeat_ngram_size: int,
        repetition_window: int,
        frequency_penalty: float,
        presence_penalty: float,
        hard_repeat_threshold: int,
    ) -> torch.Tensor:
        adjusted = next_logits.float().clone()
        raw = adjusted.clone()

        for batch_idx in range(history.size(0)):
            recent = history[batch_idx, -repetition_window:] if repetition_window > 0 else history[batch_idx]
            recent_list = [int(token) for token in recent.tolist()]
            counts: dict[int, int] = {}
            for token_id in recent_list:
                counts[token_id] = counts.get(token_id, 0) + 1

            if repetition_penalty > 1.0:
                for token_id in counts:
                    value = adjusted[batch_idx, token_id]
                    adjusted[batch_idx, token_id] = (
                        value * repetition_penalty if value < 0 else value / repetition_penalty
                    )

            if presence_penalty > 0.0:
                for token_id in counts:
                    adjusted[batch_idx, token_id] -= presence_penalty

            if frequency_penalty > 0.0:
                for token_id, count in counts.items():
                    adjusted[batch_idx, token_id] -= frequency_penalty * count

            if hard_repeat_threshold > 0:
                blocked = [token_id for token_id, count in counts.items() if count >= hard_repeat_threshold]
                if blocked:
                    adjusted[batch_idx, blocked] = float("-inf")

            if no_repeat_ngram_size >= 2 and history.size(1) >= no_repeat_ngram_size - 1:
                tokens = history[batch_idx].tolist()
                prefix = tuple(tokens[-(no_repeat_ngram_size - 1):])
                banned = set()
                for i in range(len(tokens) - no_repeat_ngram_size + 1):
                    ngram = tuple(tokens[i:i + no_repeat_ngram_size])
                    if ngram[:-1] == prefix:
                        banned.add(ngram[-1])
                if banned:
                    adjusted[batch_idx, list(banned)] = float("-inf")

            if not torch.isfinite(adjusted[batch_idx]).any():
                adjusted[batch_idx] = raw[batch_idx]
        return adjusted

    @staticmethod
    def _choose_next_token(next_logits: torch.Tensor, temperature: float) -> torch.Tensor:
        if temperature == 1.0:
            return torch.argmax(next_logits, dim=-1, keepdim=True)
        probabilities = torch.softmax(next_logits / temperature, dim=-1)
        return torch.multinomial(probabilities, num_samples=1)

    @torch.inference_mode()
    def _generate_full(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        repetition_penalty: float,
        no_repeat_ngram_size: int,
        repetition_window: int,
        frequency_penalty: float,
        presence_penalty: float,
        hard_repeat_threshold: int,
        stop_ids: set[int],
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.max_seq_len:]
            logits, _ = self(context)
            next_logits = self._adjust_generation_logits(
                logits[:, -1, :],
                input_ids,
                repetition_penalty,
                no_repeat_ngram_size,
                repetition_window,
                frequency_penalty,
                presence_penalty,
                hard_repeat_threshold,
            )
            next_id = self._choose_next_token(next_logits, temperature)
            input_ids = torch.cat((input_ids, next_id), dim=1)
            if all(int(next_id[i, 0]) in stop_ids for i in range(next_id.size(0))):
                break
        return input_ids

    @torch.inference_mode()
    def _prefill_cache(self, input_ids: torch.Tensor):
        batch_size, seq_len = input_ids.shape
        x = self.embedding_dropout(self.token_embedding(input_ids))
        caches = []
        for block in self.blocks:
            x, k, v = block(x, return_kv=True)
            cache_k, cache_v = block.self_attn.allocate_cache(
                batch_size, self.max_seq_len, input_ids.device, x.dtype
            )
            cache_k[:, :, :seq_len, :].copy_(k)
            cache_v[:, :, :seq_len, :].copy_(v)
            caches.append((cache_k, cache_v))
        return self._logits(x), caches

    @torch.inference_mode()
    def _generate_cached(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        repetition_penalty: float,
        no_repeat_ngram_size: int,
        repetition_window: int,
        frequency_penalty: float,
        presence_penalty: float,
        hard_repeat_threshold: int,
        stop_ids: set[int],
    ) -> torch.Tensor:
        batch_size, prompt_len = input_ids.shape
        output = torch.empty(
            batch_size,
            prompt_len + max_new_tokens,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        output[:, :prompt_len] = input_ids
        logits, caches = self._prefill_cache(input_ids)
        current_len = prompt_len

        for _ in range(max_new_tokens):
            history = output[:, :current_len]
            next_logits = self._adjust_generation_logits(
                logits[:, -1, :],
                history,
                repetition_penalty,
                no_repeat_ngram_size,
                repetition_window,
                frequency_penalty,
                presence_penalty,
                hard_repeat_threshold,
            )
            next_id = self._choose_next_token(next_logits, temperature)
            output[:, current_len:current_len + 1] = next_id
            current_len += 1
            if all(int(next_id[i, 0]) in stop_ids for i in range(batch_size)):
                break

            position = current_len - 1
            x = self.embedding_dropout(self.token_embedding(next_id))
            for block, (cache_k, cache_v) in zip(self.blocks, caches):
                x = block.forward_step(x, cache_k, cache_v, position)
            logits = self._logits(x)

        return output[:, :current_len]

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        repetition_penalty: float = 1.15,
        no_repeat_ngram_size: int = 3,
        repetition_window: int = 64,
        frequency_penalty: float = 0.08,
        presence_penalty: float = 0.20,
        hard_repeat_threshold: int = 6,
        stop_ids: set[int] | None = None,
    ) -> torch.Tensor:
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
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.size(1) == 0:
            raise ValueError("input_ids must contain at least one token")
        if max_new_tokens <= 0:
            return input_ids
        stop_ids = set(stop_ids or {2})
        if input_ids.size(1) + max_new_tokens <= self.max_seq_len:
            return self._generate_cached(
                input_ids,
                max_new_tokens,
                temperature,
                repetition_penalty,
                no_repeat_ngram_size,
                repetition_window,
                frequency_penalty,
                presence_penalty,
                hard_repeat_threshold,
                stop_ids,
            )
        return self._generate_full(
            input_ids,
            max_new_tokens,
            temperature,
            repetition_penalty,
            no_repeat_ngram_size,
            repetition_window,
            frequency_penalty,
            presence_penalty,
            hard_repeat_threshold,
            stop_ids,
        )


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def parameter_size_mb(model: nn.Module, bytes_per_param: int = 4) -> float:
    return parameter_count(model) * bytes_per_param / (1024 ** 2)
