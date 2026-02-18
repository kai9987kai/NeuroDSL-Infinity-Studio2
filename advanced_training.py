"""
Advanced training techniques for NeuroDSL Infinity Studio
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Optional, Dict, List, Callable, Any
import copy
from collections import defaultdict
from contextlib import nullcontext


def _split_model_output(output: Any) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if isinstance(output, tuple):
        if len(output) == 0:
            raise ValueError("Model returned an empty tuple.")
        if len(output) == 1:
            return output[0], None
        return output[0], output[1]
    if isinstance(output, list):
        if not output:
            raise ValueError("Model returned an empty list.")
        if len(output) == 1:
            return output[0], None
        return output[0], output[1]
    return output, None


def _iter_batches(batch_source):
    if isinstance(batch_source, DataLoader):
        for batch in batch_source:
            yield batch
        return
    if isinstance(batch_source, list) or isinstance(batch_source, tuple):
        for item in batch_source:
            if isinstance(item, tuple) and len(item) == 2:
                yield item
        return
    raise TypeError("Batch source must be a DataLoader or iterable of (data, target) tuples.")


class AdvancedTrainingEngine:
    """
    Advanced training engine with sophisticated techniques
    """
    def __init__(
        self, 
        model: nn.Module, 
        device: str = 'cpu',
        accumulation_steps: int = 1,
        label_smoothing: float = 0.0,
        gradient_clipping: float = 1.0,
        use_amp: bool = False,
        ema_decay: float = 0.0,
        mixup_alpha: float = 0.0,
    ):
        self.model = model.to(device)
        self.device = device
        self.accumulation_steps = max(1, int(accumulation_steps))
        self.label_smoothing = label_smoothing
        self.gradient_clipping = gradient_clipping
        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler() if use_amp and device.startswith('cuda') else None
        self.ema_decay = max(0.0, min(0.9999, float(ema_decay)))
        self.mixup_alpha = max(0.0, float(mixup_alpha))
        self._ema_state = {}
        self._ema_applied_backup = None
        if self.ema_decay > 0.0:
            for name, p in self.model.named_parameters():
                self._ema_state[name] = p.detach().clone()
        
        # Metrics tracking
        self.metrics_history = defaultdict(list)
        
        # For early stopping
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
        
    def _forward_model(self, X: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        outputs, aux_loss = _split_model_output(self.model(X))
        if aux_loss is None and hasattr(self.model, "get_aux_loss"):
            try:
                aux_loss = self.model.get_aux_loss()
            except Exception:
                aux_loss = None
        return outputs, aux_loss

    def _mixup(self, X: torch.Tensor, y: torch.Tensor):
        if self.mixup_alpha <= 0:
            return X, y, y, 1.0
        lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
        index = torch.randperm(X.size(0), device=X.device)
        mixed_x = (lam * X) + ((1.0 - lam) * X[index])
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def _update_ema(self):
        if not self._ema_state:
            return
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if name not in self._ema_state:
                    self._ema_state[name] = p.detach().clone()
                else:
                    self._ema_state[name].mul_(self.ema_decay).add_(p.detach(), alpha=1.0 - self.ema_decay)

    def apply_ema_weights(self):
        if not self._ema_state:
            return
        self._ema_applied_backup = {}
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if name in self._ema_state:
                    self._ema_applied_backup[name] = p.detach().clone()
                    p.copy_(self._ema_state[name])

    def restore_training_weights(self):
        if not self._ema_applied_backup:
            return
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if name in self._ema_applied_backup:
                    p.copy_(self._ema_applied_backup[name])
        self._ema_applied_backup = None

    def train_step(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        noise_std: float = 0.0,
        zero_grad: bool = True,
        step_optimizer: bool = True,
    ):
        """Perform a single advanced training step.

        Returns: loss, lr, grad_norm, acc, f1
        """
        self.model.train()
        if zero_grad:
            optimizer.zero_grad(set_to_none=True)

        X = X.to(self.device)
        y = y.to(self.device)
        if noise_std > 0:
            X = X + (torch.randn_like(X) * noise_std)

        X_in, y_a, y_b, lam = self._mixup(X, y)
        use_amp = self.use_amp and self.scaler is not None
        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()

        with amp_ctx:
            outputs, aux_loss = self._forward_model(X_in)

            if isinstance(criterion, nn.CrossEntropyLoss):
                target_a = y_a.long().view(-1)
                target_b = y_b.long().view(-1)
                raw_loss = (lam * criterion(outputs, target_a)) + ((1.0 - lam) * criterion(outputs, target_b))
            else:
                if torch.is_floating_point(y_a):
                    mixed_target = (lam * y_a) + ((1.0 - lam) * y_b)
                else:
                    mixed_target = y_a
                raw_loss = criterion(outputs, mixed_target)

            if aux_loss is not None:
                raw_loss = raw_loss + aux_loss
            if self.label_smoothing > 0 and isinstance(criterion, nn.CrossEntropyLoss):
                raw_loss = self._apply_label_smoothing(raw_loss, outputs, y_a)

            loss = raw_loss / self.accumulation_steps

        if use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        total_norm = 0.0
        with torch.no_grad():
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        if step_optimizer:
            if self.gradient_clipping > 0:
                if use_amp:
                    self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)

            if use_amp:
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()
            self._update_ema()

        with torch.no_grad():
            if outputs.dim() >= 2 and outputs.shape[-1] > 1:
                pred = outputs.argmax(dim=1)
                if y_a.dim() > 1 and y_a.shape[-1] > 1:
                    target = y_a.argmax(dim=1)
                else:
                    target = y_a.long().view(-1)
                correct = (pred == target).sum().item()
                acc = correct / max(1, len(target))
                f1 = self._calculate_f1_score(pred, target)
            else:
                acc = 0.0
                f1 = 0.0

        lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else 0.0
        return raw_loss.item(), lr, total_norm, acc, f1

    def _apply_label_smoothing(self, loss: torch.Tensor, outputs: torch.Tensor, targets: torch.Tensor):
        """Apply lightweight label smoothing adjustment for CE-style targets."""
        if self.label_smoothing <= 0 or outputs.dim() < 2:
            return loss
        num_classes = outputs.shape[-1]
        target = targets.long().view(-1)
        with torch.no_grad():
            smooth = torch.full_like(outputs, self.label_smoothing / max(1, num_classes - 1))
            smooth.scatter_(1, target.unsqueeze(1), 1.0 - self.label_smoothing)
        log_probs = torch.log_softmax(outputs, dim=-1)
        smoothed_loss = -(smooth * log_probs).sum(dim=-1).mean()
        return 0.5 * loss + 0.5 * smoothed_loss

    def _calculate_f1_score(self, pred: torch.Tensor, target: torch.Tensor):
        """Compute macro-F1 for integer class labels."""
        pred = pred.view(-1).long()
        target = target.view(-1).long()
        classes = torch.unique(target)
        eps = 1e-7
        f1_sum = 0.0
        for cls in classes:
            cls = int(cls.item())
            p = pred == cls
            t = target == cls
            tp = (p & t).sum().float()
            fp = (p & (~t)).sum().float()
            fn = ((~p) & t).sum().float()
            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            f1_sum += float((2 * precision * recall) / (precision + recall + eps))
        return f1_sum / max(1, int(classes.numel()))

    def evaluate(self, X_val: torch.Tensor, y_val: torch.Tensor, criterion: nn.Module):
        """
        Evaluate the model on validation data
        """
        self.model.eval()
        X_val = X_val.to(self.device)
        y_val = y_val.to(self.device)
        with torch.no_grad():
            if self.use_amp and self.scaler is not None:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs, aux_loss = self._forward_model(X_val)
            else:
                outputs, aux_loss = self._forward_model(X_val)

            if isinstance(criterion, nn.CrossEntropyLoss):
                target = y_val.long().view(-1)
                val_loss = criterion(outputs, target)
            else:
                val_loss = criterion(outputs, y_val)
                
            if aux_loss is not None:
                val_loss += aux_loss
                
            if outputs.dim() >= 2 and outputs.shape[-1] > 1:  # Classification
                pred = outputs.argmax(dim=1)
                target = y_val.argmax(dim=1) if (y_val.dim() > 1 and y_val.shape[1] > 1) else y_val.long().view(-1)
                correct = (pred == target).sum().item()
                val_acc = correct / max(1, len(target))
                
                # Calculate F1 score
                val_f1 = self._calculate_f1_score(pred, target)
            else:  # Regression
                val_acc = 0.0
                val_f1 = 0.0
                
        return val_loss.item(), val_acc, val_f1

    def train_with_advanced_techniques(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        early_stopping_patience: int = 10,
        gradient_clip_val: float = 1.0
    ):
        """
        Train the model with advanced techniques
        """
        for epoch in range(epochs):
            # Training phase
            train_loss = 0.0
            train_acc = 0.0
            num_batches = 0
            optimizer.zero_grad(set_to_none=True)

            train_batches = list(_iter_batches(train_loader))
            if not train_batches:
                raise ValueError("train_loader is empty.")

            for batch_idx, (data, target) in enumerate(train_batches):
                data, target = data.to(self.device), target.to(self.device)

                do_step = ((batch_idx + 1) % self.accumulation_steps == 0) or ((batch_idx + 1) == len(train_loader))

                # Perform training step
                loss, lr, grad_norm, acc, f1 = self.train_step(
                    data,
                    target,
                    optimizer,
                    criterion,
                    zero_grad=False,
                    step_optimizer=do_step,
                )

                if do_step:
                    optimizer.zero_grad(set_to_none=True)

                train_loss += loss
                train_acc += acc
                num_batches += 1

            # Calculate average metrics for the epoch
            avg_train_loss = train_loss / max(1, num_batches)
            avg_train_acc = train_acc / max(1, num_batches)

            # Validation phase
            val_loss = 0.0
            val_acc = 0.0
            val_f1 = 0.0
            num_val_batches = 0

            val_batches = list(_iter_batches(val_loader))
            for data, target in val_batches:
                data, target = data.to(self.device), target.to(self.device)
                v_loss, v_acc, v_f1 = self.evaluate(data, target, criterion)
                val_loss += v_loss
                val_acc += v_acc
                val_f1 += v_f1
                num_val_batches += 1

            avg_val_loss = val_loss / max(1, num_val_batches)
            avg_val_acc = val_acc / max(1, num_val_batches)
            avg_val_f1 = val_f1 / max(1, num_val_batches)

            # Update metrics history
            self.metrics_history['train_loss'].append(avg_train_loss)
            self.metrics_history['train_acc'].append(avg_train_acc)
            self.metrics_history['val_loss'].append(avg_val_loss)
            self.metrics_history['val_acc'].append(avg_val_acc)
            self.metrics_history['val_f1'].append(avg_val_f1)
            self.metrics_history['lr'].append(lr)
            self.metrics_history['grad_norm'].append(grad_norm)
            
            # Update scheduler if provided
            if scheduler:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(avg_val_loss)
                else:
                    scheduler.step()
            
            # Early stopping
            if avg_val_loss < self.best_loss:
                self.best_loss = avg_val_loss
                self.patience_counter = 0
                self.best_model_state = copy.deepcopy(self.model.state_dict())
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
            
            # Print progress
            print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, "
                  f"Train Acc: {avg_train_acc:.2%}, "
                  f"Val Acc: {avg_val_acc:.2%}, "
                  f"Val F1: {avg_val_f1:.3f}")

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        self.metrics_history["report"] = self.generate_training_report()
        return self.metrics_history

    def predict_with_uncertainty(
        self,
        X: torch.Tensor,
        mc_samples: int = 16,
        as_probs: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Monte-Carlo dropout uncertainty estimation."""
        X = X.to(self.device)
        prev_mode = self.model.training
        samples = max(1, int(mc_samples))
        preds = []

        self.model.train(True if samples > 1 else False)
        with torch.no_grad():
            for _ in range(samples):
                out, _ = self._forward_model(X)
                if as_probs and out.dim() >= 2:
                    out = torch.softmax(out, dim=-1)
                preds.append(out.unsqueeze(0))
        stack = torch.cat(preds, dim=0)
        mean = stack.mean(dim=0)
        std = stack.std(dim=0, unbiased=False)
        self.model.train(prev_mode)
        return mean, std

    def generate_training_report(self) -> Dict[str, Any]:
        """Generate compact diagnostics from tracked metrics."""
        report: Dict[str, Any] = {}
        train_loss = self.metrics_history.get("train_loss", [])
        val_loss = self.metrics_history.get("val_loss", [])
        val_acc = self.metrics_history.get("val_acc", [])
        grad_norm = self.metrics_history.get("grad_norm", [])

        if train_loss:
            report["best_train_loss"] = float(min(train_loss))
            report["final_train_loss"] = float(train_loss[-1])
        if val_loss:
            report["best_val_loss"] = float(min(val_loss))
            report["final_val_loss"] = float(val_loss[-1])
        if val_acc:
            report["best_val_acc"] = float(max(val_acc))
            report["final_val_acc"] = float(val_acc[-1])
        if train_loss and val_loss:
            report["generalization_gap"] = float(val_loss[-1] - train_loss[-1])
        if grad_norm:
            report["avg_grad_norm"] = float(sum(grad_norm) / max(1, len(grad_norm)))
            report["max_grad_norm"] = float(max(grad_norm))
        report["ema_enabled"] = bool(self._ema_state)
        report["mixup_alpha"] = float(self.mixup_alpha)
        return report


class CurriculumLearning:
    """
    Implements curriculum learning techniques
    """
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
    
    def sort_by_difficulty(self, dataset: TensorDataset) -> DataLoader:
        """
        Sort the dataset by difficulty based on model predictions
        """
        self.model.eval()
        difficulties = []
        
        with torch.no_grad():
            for data, target in dataset:
                data, target = data.to(self.device), target.to(self.device)
                if data.dim() == 1:
                    data = data.unsqueeze(0)
                output, _ = _split_model_output(self.model(data))
                
                # Calculate difficulty as negative confidence
                probs = torch.softmax(output, dim=1)
                confidence = torch.max(probs, dim=1)[0]
                difficulty = 1 - confidence.item()
                difficulties.append(difficulty)
        
        # Sort dataset by difficulty
        sorted_indices = sorted(range(len(difficulties)), key=lambda i: difficulties[i])
        
        sorted_data = [dataset[i][0] for i in sorted_indices]
        sorted_targets = [dataset[i][1] for i in sorted_indices]
        
        sorted_dataset = TensorDataset(torch.stack(sorted_data), torch.stack(sorted_targets))
        return DataLoader(sorted_dataset, batch_size=32, shuffle=False)


class KnowledgeDistillation:
    """
    Implements knowledge distillation between teacher and student models
    """
    def __init__(self, teacher_model: nn.Module, student_model: nn.Module, device: str = 'cpu'):
        self.teacher = teacher_model.to(device)
        self.student = student_model.to(device)
        self.device = device
    
    def distill(
        self,
        train_loader: DataLoader,
        epochs: int,
        temperature: float = 4.0,
        alpha: float = 0.7,
        optimizer: Optional[optim.Optimizer] = None
    ):
        """
        Perform knowledge distillation
        """
        if optimizer is None:
            optimizer = optim.Adam(self.student.parameters())
        
        criterion = nn.CrossEntropyLoss()
        kl_div = nn.KLDivLoss(reduction='batchmean')
        
        self.teacher.eval()
        self.student.train()
        
        for epoch in range(epochs):
            total_loss = 0.0
            for data, targets in train_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                
                # Get predictions from both models
                with torch.no_grad():
                    teacher_outputs, _ = _split_model_output(self.teacher(data))
                
                student_outputs, _ = _split_model_output(self.student(data))
                
                # Calculate hard loss (regular classification loss)
                hard_loss = criterion(student_outputs, targets)
                
                # Calculate soft loss (distillation loss)
                soft_teacher = torch.softmax(teacher_outputs / temperature, dim=1)
                soft_student = torch.log_softmax(student_outputs / temperature, dim=1)
                soft_loss = kl_div(soft_student, soft_teacher) * (temperature ** 2)
                
                # Combined loss
                loss = alpha * hard_loss + (1 - alpha) * soft_loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Distillation Epoch {epoch}: Average Loss: {total_loss / len(train_loader):.4f}")


class MetaLearningTrainer:
    """
    Implements MAML (Model-Agnostic Meta-Learning) for few-shot learning
    """
    def __init__(self, model: nn.Module, device: str = 'cpu', inner_lr: float = 0.01):
        self.model = model
        self.device = device
        self.inner_lr = inner_lr
    
    def adapt_to_task(self, support_data: tuple, num_steps: int = 5):
        """
        Adapt the model to a new task using support data
        """
        model_adapted = copy.deepcopy(self.model)
        optimizer = torch.optim.SGD(model_adapted.parameters(), lr=self.inner_lr)
        
        X_support, y_support = support_data
        X_support, y_support = X_support.to(self.device), y_support.to(self.device)
        
        for _ in range(num_steps):
            optimizer.zero_grad()
            outputs, _ = _split_model_output(model_adapted(X_support))
            loss = nn.CrossEntropyLoss()(outputs, y_support)
            loss.backward()
            optimizer.step()
        
        return model_adapted
    
    def meta_update(self, task_batch: List[tuple], meta_optimizer: optim.Optimizer):
        """
        Perform a meta-update step across multiple tasks
        """
        meta_loss = 0.0
        
        for support_data, query_data in task_batch:
            # Adapt to the task
            adapted_model = self.adapt_to_task(support_data, num_steps=5)
            
            # Evaluate on query data
            X_query, y_query = query_data
            X_query, y_query = X_query.to(self.device), y_query.to(self.device)
            
            outputs, _ = _split_model_output(adapted_model(X_query))
            task_loss = nn.CrossEntropyLoss()(outputs, y_query)
            meta_loss += task_loss
        
        meta_loss /= len(task_batch)
        
        meta_optimizer.zero_grad()
        meta_loss.backward()
        meta_optimizer.step()
        
        return meta_loss.item()


def create_cyclic_scheduler(optimizer: optim.Optimizer, base_lr: float, max_lr: float, step_size: int):
    """
    Create a cyclic learning rate scheduler
    """
    return optim.lr_scheduler.CyclicLR(
        optimizer, 
        base_lr=base_lr, 
        max_lr=max_lr, 
        step_size_up=step_size,
        cycle_momentum=False
    )


def create_cosine_annealing_warmup(optimizer: optim.Optimizer, warmup_epochs: int, total_epochs: int, min_lr: float = 1e-6):
    """
    Create a cosine annealing scheduler with warmup
    """
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,
        total_iters=warmup_epochs
    )
    
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=min_lr
    )
    
    return optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )
