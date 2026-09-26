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

    def _project(self, x: torch.Tensor):
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        return q, k, v

    def _merge(self, y: torch.Tensor) -> torch.Tensor:
        b, _, t, _ = y.shape
        c = self.heads * self.head_dim
        return self.out(y.transpose(1, 2).contiguous().view(b, t, c))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self._project(x)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        return self._merge(y)

    def forward_with_kv(self, x: torch.Tensor):
        q, k, v = self._project(x)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        return self._merge(y), k, v

    def allocate_cache(self, batch_size: int, max_seq_len: int, device, dtype):
        shape = (batch_size, self.heads, max_seq_len, self.head_dim)
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
        q, k, v = self._project(x)
        cache_k[:, :, cache_pos:cache_pos + 1, :].copy_(k)
        cache_v[:, :, cache_pos:cache_pos + 1, :].copy_(v)
        k_all = cache_k[:, :, :cache_pos + 1, :]
        v_all = cache_v[:, :, :cache_pos + 1, :]
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k_all, v_all,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        return self._merge(y)

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
        self.scale = nn.Parameter(torch.tensor(0.01))

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

        flat_x = x.reshape(-1, x.size(-1))
        flat_topi = topi.reshape(-1, k)
        flat_topw = topw.reshape(-1, k)
        flat_result = torch.zeros_like(flat_x)
        token_idx = torch.arange(flat_x.size(0), device=x.device)

        for expert_idx, expert in enumerate(self.experts):
            active = flat_topi.eq(expert_idx)
            selected = active.any(dim=-1)
            if not selected.any():
                continue
            selected_idx = token_idx[selected]
            values = expert(flat_x[selected])
            weights = torch.where(
                active[selected],
                flat_topw[selected],
                torch.zeros_like(flat_topw[selected]),
            ).sum(dim=-1, keepdim=True)
            flat_result = flat_result.index_add(
                0, selected_idx, values * weights
            )

        return flat_result.reshape_as(x)

class Block(nn.Module):
    def __init__(self, dim: int, heads: int, type_count: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, 4.0, dropout)
        self.specialists = SpecialistBank(dim, type_count)

    def forward(self, x: torch.Tensor, token_types: torch.Tensor, return_kv: bool = False):
        hidden = self.norm1(x)
        if return_kv:
            attn_out, k, v = self.attn.forward_with_kv(hidden)
        else:
            attn_out = self.attn(hidden)
            k = v = None
        x = x + attn_out
        hidden = self.norm2(x)
        x = x + self.mlp(hidden) + self.specialists(hidden, token_types)
        if return_kv:
            return x, k, v
        return x

    def forward_step(
        self,
        x: torch.Tensor,
        token_types: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        cache_pos: int,
    ) -> torch.Tensor:
        hidden = self.norm1(x)
        x = x + self.attn.forward_step(hidden, cache_k, cache_v, cache_pos)
        hidden = self.norm2(x)
        return x + self.mlp(hidden) + self.specialists(hidden, token_types)

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
        return self.forward_with_state(x, input_ids, think_id, think_end_id)[0]

    def forward_with_state(self, x: torch.Tensor, input_ids: torch.Tensor, think_id: int, think_end_id: int):
        active = self._active_think(input_ids, think_id, think_end_id)
        mask = active.to(x.dtype).unsqueeze(-1)
        counts = mask.cumsum(dim=1)
        cumulative = (x * mask).cumsum(dim=1) / counts.clamp_min(1.0)

        state_latent = torch.tanh(self.state_in(cumulative))
        state = self.state_out(state_latent)
        state = state * torch.sigmoid(self.state_gate(cumulative))

        write_logits = self.write_gate(cumulative)
        writes = torch.softmax(write_logits.float(), dim=-1).to(x.dtype) * mask
        values = torch.tanh(self.write_value(cumulative))
        write_accum = (writes.unsqueeze(-1) * values.unsqueeze(2)).cumsum(dim=1)
        write_norm = writes.cumsum(dim=1)
        memory = write_accum / write_norm.unsqueeze(-1).clamp_min(1e-4)

        query = self.query(cumulative)
        read_weights = torch.softmax(
            torch.einsum("btd,sd->bts", query.float(), self.slot_keys.float()), dim=-1
        ).to(x.dtype)
        read = torch.einsum("bts,btsd->btd", read_weights, memory)
        latent = state + read
        gated = latent * torch.sigmoid(self.memory_gate(cumulative))
        output = x + mask * gated

        state_out = {
            # Keep cached state rank-stable: [B], [B,D], [B,S,D], [B,S].
            "active": active[:, -1].detach(),
            "count": counts[:, -1, 0].detach(),
            "cumulative": cumulative[:, -1].detach(),
            "write_accum": write_accum[:, -1].detach(),
            "write_norm": write_norm[:, -1].detach(),
        }
        return output, state_out

    def step(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        state: dict,
        think_id: int,
        think_end_id: int,
    ):
        if x.ndim != 3 or x.size(1) != 1:
            raise ValueError(f"LatentReasoningMemory.step expects x [batch, 1, dim], got {tuple(x.shape)}")
        if input_ids.ndim != 2 or input_ids.size(1) != 1:
            raise ValueError(
                f"LatentReasoningMemory.step expects input_ids [batch, 1], got {tuple(input_ids.shape)}"
            )

        batch_size = x.size(0)
        opened = input_ids.eq(think_id)[:, 0]
        closed = input_ids.eq(think_end_id)[:, 0]

        active = state["active"]
        if active.ndim != 1:
            active = active.reshape(batch_size)
        token_active = (active | opened) & ~closed

        # Normalize cached state ranks so old/inadvertently unsqueezed states
        # cannot turn the memory tensor into [B, 1, S, D].
        previous_count = state["count"]
        if previous_count.ndim == 2:
            previous_count = previous_count[:, 0]
        previous_count = previous_count.reshape(batch_size)

        cumulative = state["cumulative"]
        if cumulative.ndim == 3:
            cumulative = cumulative[:, -1]
        if cumulative.ndim != 2:
            raise ValueError(
                f"LatentReasoningMemory.step expected cumulative [batch, dim], got {tuple(cumulative.shape)}"
            )

        write_accum_state = state["write_accum"]
        if write_accum_state.ndim == 4:
            write_accum_state = write_accum_state[:, -1]
        if write_accum_state.ndim != 3:
            raise ValueError(
                "LatentReasoningMemory.step expected write_accum [batch, slots, dim], "
                f"got {tuple(write_accum_state.shape)}"
            )

        write_norm_state = state["write_norm"]
        if write_norm_state.ndim == 3:
            write_norm_state = write_norm_state[:, -1]
        if write_norm_state.ndim != 2:
            raise ValueError(
                f"LatentReasoningMemory.step expected write_norm [batch, slots], got {tuple(write_norm_state.shape)}"
            )

        new_count = previous_count + token_active.to(x.dtype)
        updated_cumulative = (
            previous_count.unsqueeze(-1) * cumulative + x[:, 0, :]
        ) / new_count.clamp_min(1.0).unsqueeze(-1)
        cumulative = torch.where(
            token_active.unsqueeze(-1),
            updated_cumulative,
            cumulative,
        )

        state_latent = torch.tanh(self.state_in(cumulative))
        latent_state = self.state_out(state_latent)
        latent_state = latent_state * torch.sigmoid(self.state_gate(cumulative))

        write_logits = self.write_gate(cumulative)
        writes = torch.softmax(write_logits.float(), dim=-1).to(x.dtype)
        writes = writes * token_active.to(x.dtype).unsqueeze(-1)
        values = torch.tanh(self.write_value(cumulative))
        write_accum = write_accum_state + writes.unsqueeze(-1) * values.unsqueeze(1)
        write_norm = write_norm_state + writes
        memory = write_accum / write_norm.unsqueeze(-1).clamp_min(1e-4)

        query = self.query(cumulative)
        read_weights = torch.softmax(
            torch.einsum("bd,sd->bs", query.float(), self.slot_keys.float()), dim=-1
        ).to(x.dtype)
        read = torch.einsum("bs,bsd->bd", read_weights, memory).unsqueeze(1)

        gated = (latent_state.unsqueeze(1) + read) * torch.sigmoid(self.memory_gate(cumulative)).unsqueeze(1)
        output = x + token_active.to(x.dtype).view(-1, 1, 1) * gated

        return output, {
            "active": token_active,
            "count": new_count,
            "cumulative": cumulative,
            "write_accum": write_accum,
            "write_norm": write_norm,
        }

class TiedLMHead(nn.Module):
    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.embedding = embedding

    @property
    def vocab_size(self) -> int:
        return self.embedding.num_embeddings

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.embedding.weight.transpose(0, 1))

    def loss(self, logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        flat_logits = logits.reshape(-1, self.vocab_size)
        flat_targets = targets.reshape(-1)
        token_loss = nn.functional.cross_entropy(flat_logits.float(), flat_targets, reduction="none")
        if weights is None:
            return token_loss.mean()
        flat_weights = weights.reshape(-1).to(token_loss.dtype)
        weight_sum = flat_weights.sum().clamp_min(1e-12)
        return (token_loss * flat_weights).sum() / weight_sum


class LLN(nn.Module):
    ARCHITECTURE_VERSION = 5
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
        return self.token_cluster.num_embeddings

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
        loss = None if targets is None else self.lm_head.loss(logits, targets, weights=loss_weights)
        return logits, loss

    def _adjust_generation_logits(
        self,
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
            counts = {}
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
        stop_ids,
    ):
        for _ in range(max_new_tokens):
            x = input_ids[:, -self.max_seq_len:]
            logits, _ = self(x)
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
            if temperature == 1.0:
                next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                probabilities = torch.softmax(next_logits / temperature, dim=-1)
                next_id = torch.multinomial(probabilities, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
            if all(int(next_id[i, 0]) in stop_ids for i in range(next_id.size(0))):
                break
        return input_ids

    @torch.inference_mode()
    def _prefill_cache(self, input_ids: torch.Tensor):
        batch_size, seq_len = input_ids.shape
        x, token_types = self._embed(input_ids)
        caches = []
        memory_states = []

        for _ in range(self.recurrent_steps):
            layer_caches = []
            for block in self.blocks:
                x, k, v = block(x, token_types, return_kv=True)
                cache_k, cache_v = block.attn.allocate_cache(
                    batch_size, self.max_seq_len, input_ids.device, x.dtype
                )
                cache_k[:, :, :seq_len, :].copy_(k)
                cache_v[:, :, :seq_len, :].copy_(v)
                layer_caches.append((cache_k, cache_v))
            x, state = self.latent_memory.forward_with_state(
                x, input_ids, self.THINK_TOKEN_ID, self.THINK_END_TOKEN_ID
            )
            caches.append(layer_caches)
            memory_states.append(state)

        x = self.norm(x)
        return self.lm_head.logits(x), caches, memory_states

    def _embed_step(self, input_ids: torch.Tensor, position: int):
        token_types = self.token_type_map[input_ids]
        clusters = torch.div(
            input_ids, max(1, self.output_clusters), rounding_mode="floor"
        ).clamp(max=self.token_cluster.num_embeddings - 1)
        pos = torch.tensor([position], device=input_ids.device, dtype=torch.long)
        x = (
            self.token(input_ids)
            + self.token_type(token_types)
            + self.token_cluster(clusters)
            + self.position(pos)[None, :, :]
        )
        return x, token_types

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
        stop_ids,
    ):
        batch_size, prompt_len = input_ids.shape
        output = torch.empty(
            batch_size,
            prompt_len + max_new_tokens,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        output[:, :prompt_len] = input_ids

        logits, caches, memory_states = self._prefill_cache(input_ids)
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
            if temperature == 1.0:
                next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                probabilities = torch.softmax(next_logits / temperature, dim=-1)
                next_id = torch.multinomial(probabilities, num_samples=1)

            output[:, current_len:current_len + 1] = next_id
            current_len += 1
            if all(int(next_id[i, 0]) in stop_ids for i in range(next_id.size(0))):
                break

            x, token_types = self._embed_step(next_id, current_len - 1)
            cache_pos = current_len - 1
            for recurrent_idx in range(self.recurrent_steps):
                for block_idx, block in enumerate(self.blocks):
                    cache_k, cache_v = caches[recurrent_idx][block_idx]
                    x = block.forward_step(
                        x, token_types, cache_k, cache_v, cache_pos
                    )
                x, memory_states[recurrent_idx] = self.latent_memory.step(
                    x,
                    next_id,
                    memory_states[recurrent_idx],
                    self.THINK_TOKEN_ID,
                    self.THINK_END_TOKEN_ID,
                )

            logits = self.lm_head.logits(self.norm(x))

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
        stop_ids=None,
    ):
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
        if max_new_tokens <= 0:
            return input_ids
        stop_ids = set(stop_ids or {2})
        if input_ids.size(1) + max_new_tokens <= self.max_seq_len:
            return self._generate_cached(
                input_ids, max_new_tokens, temperature, repetition_penalty,
                no_repeat_ngram_size, repetition_window, frequency_penalty,
                presence_penalty, hard_repeat_threshold, stop_ids
            )
        return self._generate_full(
            input_ids, max_new_tokens, temperature, repetition_penalty,
            no_repeat_ngram_size, repetition_window, frequency_penalty,
            presence_penalty, hard_repeat_threshold, stop_ids
        )


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def parameter_size_mb(model: nn.Module, bytes_per_param: int = 4) -> float:
    return parameter_count(model) * bytes_per_param / (1024 ** 2)
