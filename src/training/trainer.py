"""Multi-task training loop with task-alternating batches and masked (per-task) losses.

Each step processes ONE task's batch and only that head's loss is computed, so a sentiment
example never contributes a toxicity loss (the datasets come from different sources).
Across an epoch the objective is  L = L_sentiment + lambda * L_toxicity.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import get_linear_schedule_with_warmup

from src.nlp.model import MultiTaskModel
from src.training.dataset import make_loader
from src.training.metrics import sentiment_metrics, toxicity_metrics

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    eval_batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    toxicity_loss_weight: float
    max_length: int
    fp16: bool
    seed: int


def class_weights(labels: pd.Series, n: int) -> torch.Tensor:
    """Inverse-frequency weights, normalised to mean 1 (counters Jigsaw's ~10% toxic rate)."""
    counts = np.bincount(labels.astype(int), minlength=n).astype(float)
    w = counts.sum() / (n * np.maximum(counts, 1))
    return torch.tensor(w / w.mean(), dtype=torch.float)


@torch.inference_mode()
def evaluate(model: MultiTaskModel, tokenizer, data: dict[str, pd.DataFrame], cfg: TrainConfig,
             device, threshold: float = 0.5) -> dict:
    model.eval()
    out: dict = {}
    for task, df in data.items():
        loader = make_loader(df, tokenizer, cfg.max_length, cfg.eval_batch_size, False, cfg.seed)
        scores = []
        for batch in loader:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                           task=task)[f"{task}_logits"]
            scores.append(torch.softmax(logits.float(), -1).cpu().numpy())
        probs = np.concatenate(scores)
        y = df["label"].to_numpy()
        if task == "sentiment":
            out.update(sentiment_metrics(y, probs.argmax(1)))
        else:
            out.update(toxicity_metrics(y, probs[:, 1], threshold))
    return out


def train(model: MultiTaskModel, tokenizer, train_data: dict[str, pd.DataFrame],
          val_data: dict[str, pd.DataFrame], cfg: TrainConfig, device,
          on_epoch_end=None) -> dict:
    model.to(device)
    loaders = {t: make_loader(df, tokenizer, cfg.max_length, cfg.batch_size, True, cfg.seed)
               for t, df in train_data.items()}
    loss_fns = {
        "sentiment": nn.CrossEntropyLoss(),
        "toxicity": nn.CrossEntropyLoss(
            weight=class_weights(train_data["toxicity"]["label"], 2).to(device)
        ) if "toxicity" in train_data else None,
    }
    task_weight = {"sentiment": 1.0, "toxicity": cfg.toxicity_loss_weight}

    steps_per_epoch = sum(len(dl) for dl in loaders.values())
    total = steps_per_epoch * cfg.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                            weight_decay=cfg.weight_decay)
    sched = get_linear_schedule_with_warmup(opt, int(total * cfg.warmup_ratio), total)
    use_amp = cfg.fp16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = random.Random(cfg.seed)

    history = []
    t0 = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        # deterministic interleaving of task batches, proportional to dataset size
        schedule = [t for t, dl in loaders.items() for _ in range(len(dl))]
        rng.shuffle(schedule)
        iters = {t: iter(dl) for t, dl in loaders.items()}
        running = {t: [] for t in loaders}
        for step, task in enumerate(schedule, 1):
            batch = next(iters[task])
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                               task=task)[f"{task}_logits"]
            loss = task_weight[task] * loss_fns[task](logits.float(), batch["labels"].to(device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running[task].append(loss.item())
            if step % 200 == 0:
                log.info("epoch %d step %d/%d %s", epoch + 1, step, len(schedule),
                         {t: round(float(np.mean(v[-200:])), 4) for t, v in running.items() if v})
        val = evaluate(model, tokenizer, val_data, cfg, device)
        rec = {"epoch": epoch + 1,
               **{f"train_loss_{t}": float(np.mean(v)) for t, v in running.items() if v},
               **{f"val_{k}": v for k, v in val.items() if isinstance(v, float)}}
        history.append(rec)
        log.info("epoch %d done: %s", epoch + 1,
                 {k: round(v, 4) for k, v in rec.items() if isinstance(v, float)})
        if on_epoch_end:
            on_epoch_end(rec)
    return {"history": history, "training_seconds": time.time() - t0}
