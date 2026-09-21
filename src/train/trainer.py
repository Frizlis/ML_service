"""
Цикл дообучения HF-модели классификации: train/val эпохи, ранняя остановка
по val loss, градиентная аккумуляция (эффективный batch = batch_size * accum_steps
— нужно, когда GPU ограничивает физический размер батча).

Mixed precision (bf16): без него roberta-base на батче 32 при max_len=500
почти упирается в лимит VRAM и считает в fp32 заметно медленнее, чем могла
бы современная GPU. bf16 выбран вместо fp16 намеренно — у него тот же
диапазон экспоненты, что у fp32 (в отличие от fp16), поэтому не нужен
GradScaler и не возникает риска переполнения/исчезновения градиентов;
current-gen GPU (Ampere+) считают bf16 через тензорные ядра так же быстро,
как fp16. На GPU без поддержки bf16 автоматически используется обычный fp32
(autocast(enabled=False)) — код работает одинаково в обоих случаях.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.common.logging_config import get_logger

logger = get_logger(__name__)

# Как часто логировать промежуточный прогресс внутри train-эпохи — на
# больших датасетах (100k+ строк) эпоха может идти часами, и без этого лог
# молчит от "epoch_completed" до следующего "epoch_completed", создавая
# ложное впечатление, что процесс завис.
_LOG_EVERY_N_STEPS = 100


def _autocast(device: torch.device):
    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    return torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp)


def freeze_base_model(model) -> None:
    """
    Замораживает энкодер (тело трансформера) и оставляет обучаемой только
    классификационную голову. `model.base_model` — стандартное свойство
    HF `PreTrainedModel`, отдающее тело модели без головы независимо от
    конкретной архитектуры (RoBERTa/BERT/DistilBERT/...), поэтому решение
    не привязано жёстко к TRAIN_BASE_MODEL=roberta-base.

    Полезно для быстрого дообучения на малых датасетах или при нехватке
    GPU-памяти: градиенты и состояние оптимизатора считаются только для
    головы, а не для всей сети.
    """
    for param in model.base_model.parameters():
        param.requires_grad = False

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "base_model_frozen",
        extra={"trainable_params": trainable, "total_params": total},
    )


def _run_epoch(
    model,
    dataloader: DataLoader,
    device: torch.device,
    criterion,
    optimizer=None,
    scheduler=None,
    accum_steps: int = 1,
    epoch: int | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Один проход по dataloader. Если передан optimizer — это train-режим
    (считаются градиенты и делается шаг оптимизатора), иначе — eval
    (torch.no_grad, веса не меняются). Общий код для train/val эпох, чтобы
    не дублировать forward-pass дважды с риском их незаметно рассинхронить.
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    all_preds, all_labels = [], []
    total_steps = len(dataloader)

    with torch.set_grad_enabled(is_train):
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with _autocast(device):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)

            if is_train:
                (loss / accum_steps).backward()
                if (step + 1) % accum_steps == 0 or (step + 1) == total_steps:
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()

            total_loss += loss.item()
            # bf16-логиты приводим к float32 перед argmax/numpy — argmax
            # корректен и в bf16, но numpy с bf16 не работает напрямую.
            all_preds.append(outputs.logits.detach().float().argmax(dim=-1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            if is_train and (step + 1) % _LOG_EVERY_N_STEPS == 0:
                logger.info(
                    "train_step_progress",
                    extra={
                        "epoch": epoch,
                        "step": step + 1,
                        "total_steps": total_steps,
                        "running_loss": total_loss / (step + 1),
                    },
                )

    avg_loss = total_loss / max(total_steps, 1)
    return avg_loss, np.concatenate(all_preds), np.concatenate(all_labels)


def train_model(
    model,
    train_dataset,
    val_dataset,
    class_weights: torch.Tensor,
    device: torch.device,
    *,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    warmup_frac: float,
    accum_steps: int,
    early_stop_patience: int,
    freeze_base: bool = False,
    min_delta: float = 1e-3,
) -> dict:
    """
    Обучает model in-place. Ранняя остановка следит за val loss: если он не
    улучшается `early_stop_patience` эпох подряд — обучение прерывается, а в
    модель в конце загружаются веса ЛУЧШЕЙ (не последней) эпохи по val loss,
    чтобы на диск в итоге не сохранилась переобученная версия.

    `min_delta` — минимальное снижение val loss, которое считается реальным
    улучшением. Без него на плато loss иногда еле заметно колеблется (шум
    bf16/порядка обхода батчей), и формальное "val_loss < best_val_loss" на
    долю процента бесконечно сбрасывает счётчик патиенса — early stopping
    в итоге не срабатывает вообще, даже если модель фактически перестала
    учиться десятки эпох назад.

    `freeze_base=True` замораживает энкодер и обучает только классификационную
    голову (см. `freeze_base_model`) — быстрее и требует меньше памяти, но
    обычно даёт более слабое качество, чем full fine-tuning.
    """
    if freeze_base:
        freeze_base_model(model)

    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    # filter(requires_grad) — при freeze_base=True в оптимизатор не попадают
    # замороженные параметры энкодера: иначе AdamW всё равно завёл бы для них
    # состояние (moment-буферы), впустую расходуя память.
    trainable_params = (p for p in model.parameters() if p.requires_grad)
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(len(train_dl) // accum_steps, 1)
    total_steps = steps_per_epoch * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * warmup_frac),
        num_training_steps=total_steps,
    )
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        train_loss, _, _ = _run_epoch(
            model, train_dl, device, criterion, optimizer, scheduler, accum_steps, epoch=epoch
        )
        val_loss, _, _ = _run_epoch(model, val_dl, device, criterion)

        logger.info(
            "epoch_completed",
            extra={"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss},
        )

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stop_patience:
                logger.info("early_stopping_triggered", extra={"epoch": epoch})
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"best_val_loss": best_val_loss, "epochs_ran": epoch + 1}
