#!/usr/bin/env python
"""GPT-style Transformer training with flexium.

A decoder-only transformer language model (character-level).
Demonstrates handling of large sequence models,
attention mechanisms, and causal masking.

Usage:
    # With flexium (default)
    python gpt_train.py

    # With orchestrator
    python gpt_train.py --orchestrator localhost:50051

    # Without flexium (baseline)
    python gpt_train.py --disabled

    # With custom data
    python gpt_train.py --data-file path/to/text.txt
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import flexium.auto
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with causal masking."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Linear projections
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Causal mask
        if mask is None:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device), diagonal=1
            ).bool()
        scores = scores.masked_fill(mask, float("-inf"))

        # Softmax and output
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.w_o(out)


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Transformer decoder block (pre-norm)."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.ln1(x), mask))
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x


class GPT(nn.Module):
    """GPT-style decoder-only transformer."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape

        # Embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.token_emb(x) + self.pos_emb(positions)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Output
        x = self.ln_f(x)
        logits = self.head(x)

        return logits

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate new tokens autoregressively."""
        for _ in range(max_new_tokens):
            # Crop to max sequence length
            idx_cond = idx[:, -self.pos_emb.num_embeddings:]

            # Forward pass
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)

        return idx


class TextDataset(Dataset):
    """Simple character-level text dataset."""

    def __init__(self, text: str, seq_len: int):
        self.seq_len = seq_len
        self.chars = sorted(set(text))
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        self.data = torch.tensor(
            [self.char_to_idx[c] for c in text], dtype=torch.long
        )

    def __len__(self) -> int:
        return len(self.data) - self.seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + self.seq_len + 1]
        return x, y

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def decode(self, indices: list[int]) -> str:
        return "".join([self.idx_to_char[i] for i in indices])


def main() -> None:
    parser = argparse.ArgumentParser(description="GPT training with flexium")
    parser.add_argument(
        "--orchestrator",
        default="localhost:50051",
        help="Orchestrator address (e.g., localhost:50051). Use empty string for local mode.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Initial device (or set GPU_DEVICE env var)",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--data-file", type=str, default=None, help="Path to text file")
    parser.add_argument("--disabled", action="store_true", help="Run without flexium")
    args = parser.parse_args()

    with flexium.auto.run(
        orchestrator=args.orchestrator,
        device=args.device,
        disabled=args.disabled,
    ):
        # Load text data
        if args.data_file and Path(args.data_file).exists():
            with open(args.data_file, "r") as f:
                text = f.read()
            print(f"Loaded {len(text):,} characters from {args.data_file}")
        else:
            # Generate sample text if no file provided
            print("No data file provided, generating sample text...")
            sample_text = """
The quick brown fox jumps over the lazy dog.
Machine learning is transforming the world.
Deep neural networks can learn complex patterns.
Transformers revolutionized natural language processing.
Attention is all you need for sequence modeling.
GPU acceleration makes training faster.
"""
            text = (sample_text * 1000).strip()

        dataset = TextDataset(text, args.seq_len)
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
        )

        # Model
        model = GPT(
            vocab_size=dataset.vocab_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_model * 4,
            max_seq_len=args.seq_len,
        ).cuda()

        print(f"Vocabulary size: {dataset.vocab_size}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print("-" * 50)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs * len(dataloader)
        )

        for epoch in range(args.epochs):
            total_loss = 0

            for i, (x, y) in enumerate(dataloader):
                x, y = x.cuda(), y.cuda()

                optimizer.zero_grad()
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, dataset.vocab_size), y.view(-1))
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

                if i % 100 == 0:
                    ppl = math.exp(min(loss.item(), 10))  # Clamp for stability
                    print(
                        f"Epoch [{epoch}/{args.epochs}] "
                        f"Batch [{i}/{len(dataloader)}] "
                        f"Loss: {loss.item():.4f} "
                        f"PPL: {ppl:.2f}"
                    )

            avg_loss = total_loss / len(dataloader)
            avg_ppl = math.exp(min(avg_loss, 10))
            print(f">>> Epoch {epoch} | Avg Loss: {avg_loss:.4f} | PPL: {avg_ppl:.2f}")

            # Generate sample text every 5 epochs
            if epoch % 5 == 0:
                model.eval()
                start_char = text[0]
                start_idx = torch.tensor([[dataset.char_to_idx[start_char]]]).cuda()

                generated = model.generate(
                    start_idx,
                    max_new_tokens=200,
                    temperature=0.8,
                )
                generated_text = dataset.decode(generated[0].tolist())
                print(f"\n--- Generated Sample ---\n{generated_text[:300]}\n")
                model.train()

        print("Training complete!")


if __name__ == "__main__":
    main()
