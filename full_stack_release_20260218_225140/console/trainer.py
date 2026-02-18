import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import csv
import os


class FocalLoss(nn.Module):
    """Focal Loss for class-imbalanced classification.
    
    Down-weights well-classified examples and focuses on hard negatives.
    Paper: Lin et al. 2017 — "Focal Loss for Dense Object Detection"
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # inputs: (B, C) logits, targets: (B,) class indices
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # Probability of correct class
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingLoss(nn.Module):
    """Cross-entropy with label smoothing to reduce overconfidence.
    
    Distributes a fraction of the label mass uniformly across all classes,
    improving calibration and generalization.
    """
    def __init__(self, num_classes: int = 10, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.confidence = 1.0 - smoothing
    
    def forward(self, inputs, targets):
        # inputs: (B, C), targets: (B,) long
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # Create smooth targets
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / (self.num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        loss = -(smooth_targets * log_probs).sum(dim=-1)
        return loss.mean()

class EMA:
    """Exponential Moving Average of model weights.
    
    Maintains a shadow copy of the model weights that decays exponentially.
    Using EMA weights often leads to better generalization and stability.
    """
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        """Initialize shadow weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update shadow weights: shadow = decay * shadow + (1 - decay) * param."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Replace model weights with shadow weights (for eval/inference)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        """Restore original model weights (after eval/inference)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class ContrastiveLoss(nn.Module):
    """NT-Xent / InfoNCE Contrastive Loss.

    Computes temperature-scaled cross-entropy over cosine-similarity
    matrix of paired embeddings. Each sample in a batch is treated
    as a positive pair with itself from another augmentation view,
    and all other pairs are negatives.
    Reference: Chen et al. 2020 — SimCLR.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, inputs, targets):
        # inputs: (B, D) normalized embeddings
        # In pure contrastive mode, we compare first half vs second half
        # For fallback compatibility, we compute self-similarity
        z = F.normalize(inputs, p=2, dim=-1)
        B = z.shape[0]
        sim = (z @ z.t()) / self.temperature

        # Mask out self-similarity on diagonal
        mask = torch.eye(B, device=sim.device).bool()
        sim.masked_fill_(mask, -1e9)

        # Labels: each sample is most similar to itself (diagonal)
        # Use cross-entropy where positive is the shifted diagonal
        labels = torch.arange(B, device=sim.device)
        # Shift labels for positive pairs: (0->1, 1->0, 2->3, 3->2, ...)
        if B >= 2:
            labels_pos = labels ^ 1  # XOR to pair adjacent samples
        else:
            labels_pos = labels

        loss = F.cross_entropy(sim, labels_pos)
        return loss


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization (SAM) Wrapper."""
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]), p=2
        )
        return norm


class Lion(torch.optim.Optimizer):
    """Lion (EvoLved Sign Momentum) Optimizer.

    A memory-efficient, sign-based optimizer that often outperforms AdamW.
    Reference: Chen et al. 2023 — Symbolic Discovery of Optimization Algorithms.
    """
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if not 0.0 <= lr: raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0: raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0: raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                
                # Perform stepweight decay
                p.data.mul_(1 - group["lr"] * group["weight_decay"])

                grad, lr, (beta1, beta2) = p.grad, group["lr"], group["betas"]
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                
                # Weight update
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-lr)

                # Decay the momentum running average term
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss


class TrainingEngine:
    """v4.0 Training Engine with multi-loss, gradient clipping, LR warmup, and CSV loading."""
    
    LOSS_FUNCTIONS = {
        'MSE': nn.MSELoss,
        'CrossEntropy': nn.CrossEntropyLoss,
        'Huber': nn.HuberLoss,
        'MAE': nn.L1Loss,
        'Focal': lambda: FocalLoss(alpha=0.25, gamma=2.0),
        'LabelSmooth': lambda: LabelSmoothingLoss(num_classes=10, smoothing=0.1),
        'CosineSim': lambda: nn.CosineEmbeddingLoss(margin=0.0), # Expects inputs x1, x2, target
        'Contrastive': lambda: ContrastiveLoss(temperature=0.07),
    }
    
    def __init__(
        self,
        model,
        loss_fn='MSE',
        max_epochs=250,
        grad_clip=1.0,
        warmup_steps=10,
        base_lr=0.01,
        aux_loss_coef=0.02,
        use_ema=False,
        accumulate_grad_batches=1,
        use_sam=False,
        use_lion=False,
        sam_rho=0.05,
    ):
        self.model = model
        self.max_epochs = max_epochs
        self.grad_clip = grad_clip
        self.warmup_steps = warmup_steps
        self.aux_loss_coef = max(0.0, float(aux_loss_coef))
        self.base_lr = float(base_lr)
        self.step_count = 0
        self.last_base_loss = 0.0
        self.last_aux_loss = 0.0
        self.accumulation_steps = max(1, int(accumulate_grad_batches))
        self.use_sam = use_sam
        self.use_lion = use_lion
        self.sam_rho = sam_rho
        
        # EMA initialization
        self.use_ema = use_ema
        self.ema = EMA(model, decay=0.999) if use_ema else None
        
        # Initialize training history
        self.training_history = {
            'loss': [],
            'val_loss': [],
            'lr': [],
            'grad_norm': [],
            'epoch': [],
            'accuracy': [],  # New metric
            'f1': []         # New metric
        }
        
        # Loss function
        self.loss_name = loss_fn
        if loss_fn in self.LOSS_FUNCTIONS:
            self.criterion = self.LOSS_FUNCTIONS[loss_fn]()
        else:
            self.criterion = nn.MSELoss()
        
        # Optimizer Selection
        if self.use_sam:
            base_opt = Lion if self.use_lion else optim.AdamW
            self.optimizer = SAM(model.parameters(), base_opt, rho=self.sam_rho, lr=self.base_lr, weight_decay=0.01)
        elif self.use_lion:
            self.optimizer = Lion(model.parameters(), lr=self.base_lr, weight_decay=0.01)
        else:
            self.optimizer = optim.AdamW(model.parameters(), lr=self.base_lr, weight_decay=0.1)
        
        # Scheduler: supports standard cosine, warm restarts, and OneCycle
        self.scheduler_type = 'cosine'
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, max_epochs)
        )
        
        # Automatic Mixed Precision
        self.scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
        
        # Curriculum learning state
        self.curriculum_enabled = False
        self.curriculum_progress = 0.0  # 0.0 = easy only, 1.0 = full dataset
        
        # LAWA (Learning-rate Aware Weight Averaging)
        self.lawa_enabled = False
        self.lawa_interval = 50  # Merge every N steps
        self.lawa_window_size = 5  # Keep last N snapshots
        self.lawa_snapshots = []  # List of (state_dict, lr) tuples
        
    def enable_lawa(self, interval=50, window_size=5):
        """Enable LAWA (Learning-rate Aware Weight Averaging).
        
        Periodically averages recent checkpoint weight snapshots with
        learning-rate-aware weighting for faster convergence.
        Reference: Kaddour et al. 2022 — LAWA.
        """
        self.lawa_enabled = True
        self.lawa_interval = max(1, interval)
        self.lawa_window_size = max(2, window_size)
        self.lawa_snapshots = []
        print(f"LAWA enabled: interval={self.lawa_interval}, window={self.lawa_window_size}")
    
    def _lawa_step(self):
        """Run LAWA checkpoint merging if enabled."""
        if not self.lawa_enabled:
            return
        
        # Take snapshot at interval
        if self.step_count % self.lawa_interval == 0:
            import copy
            current_lr = self.optimizer.param_groups[0]['lr']
            snapshot = copy.deepcopy(self.model.state_dict())
            self.lawa_snapshots.append((snapshot, current_lr))
            
            # Keep only last N snapshots
            if len(self.lawa_snapshots) > self.lawa_window_size:
                self.lawa_snapshots = self.lawa_snapshots[-self.lawa_window_size:]
            
            # Merge if we have enough snapshots
            if len(self.lawa_snapshots) >= 2:
                # Weight by learning rate (lower LR = more converged = higher weight)
                lrs = torch.tensor([lr for _, lr in self.lawa_snapshots])
                # Inverse LR weighting: lower lr gets higher weight
                weights = 1.0 / (lrs + 1e-8)
                weights = weights / weights.sum()
                
                # Weighted average of state dicts
                merged = {}
                for key in self.lawa_snapshots[0][0]:
                    merged[key] = sum(
                        w * snap[key].float()
                        for (snap, _), w in zip(self.lawa_snapshots, weights)
                    ).to(self.lawa_snapshots[0][0][key].dtype)
                
                self.model.load_state_dict(merged)
        
    def update_epochs(self, epochs):
        """Dynamically update epoch count and recreate scheduler."""
        self.max_epochs = epochs
        # Re-initialize scheduler with new epoch count
        self.set_scheduler(self.scheduler_type)
    
    def set_scheduler(self, scheduler_type='cosine', **kwargs):
        """Set the LR scheduler type. Options: 'cosine', 'warm_restarts', 'onecycle'."""
        self.scheduler_type = scheduler_type
        if scheduler_type == 'warm_restarts':
            t0 = kwargs.get('T_0', max(1, self.max_epochs // 4))
            t_mult = kwargs.get('T_mult', 2)
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=t0, T_mult=t_mult
            )
        elif scheduler_type == 'onecycle':
            # OneCycle requires total steps, so we estimate based on epochs
            # Assuming ~100 steps per epoch as a default if not provided
            steps_per_epoch = kwargs.get('steps_per_epoch', 100)
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=self.base_lr, 
                epochs=self.max_epochs, steps_per_epoch=steps_per_epoch,
                pct_start=0.3
            )
        else:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(1, self.max_epochs)
            )
    
    def enable_curriculum(self, enabled=True):
        """Enable/disable curriculum learning (easy-to-hard sample ordering)."""
        self.curriculum_enabled = enabled
        self.curriculum_progress = 0.0
    
    def _apply_curriculum(self, X, y):
        """Sort samples by difficulty and return a subset based on curriculum progress."""
        if not self.curriculum_enabled:
            return X, y
        
        self.model.eval()
        with torch.no_grad():
            device = next(self.model.parameters()).device
            preds = self.model(X.to(device))
            if self.loss_name in ('CrossEntropy', 'Focal', 'LabelSmooth'):
                targets_for_loss = y.squeeze().long().to(device)
                per_sample_loss = F.cross_entropy(preds, targets_for_loss, reduction='none')
            else:
                per_sample_loss = F.mse_loss(preds, y.to(device), reduction='none').mean(dim=-1)
        
        self.model.train()
        
        # Sort by loss (ascending = easiest first)
        sorted_idx = per_sample_loss.cpu().argsort()
        
        # Gradually include harder samples
        n_samples = max(int(len(sorted_idx) * max(0.3, self.curriculum_progress)), 4)
        selected = sorted_idx[:n_samples]
        
        # Advance curriculum
        self.curriculum_progress = min(1.0, self.curriculum_progress + 1.0 / max(1, self.max_epochs))
        
        return X[selected], y[selected]
        
    def _warmup_lr(self):
        """Linear warmup for the first N steps."""
        if self.step_count < self.warmup_steps:
            warmup_factor = (self.step_count + 1) / self.warmup_steps
            for pg in self.optimizer.param_groups:
                pg['lr'] = self.base_lr * warmup_factor
            return True
        return False
        
    def generate_dummy_data(self, input_dim, output_dim, n_samples=500):
        """Generate synthetic training data for quick experiments."""
        X = torch.randn(n_samples, input_dim)
        y = torch.sin(X[:, [0]]) * 0.5 + 0.5
        if output_dim > 1:
            y = torch.cat([y, torch.cos(X[:, [0]])], dim=1)
            if y.shape[1] < output_dim:
                # Fill remaining dims with linear combinations
                for j in range(y.shape[1], output_dim):
                    col_idx = j % input_dim
                    y = torch.cat([y, torch.tanh(X[:, [col_idx]])], dim=1)
                y = y[:, :output_dim]
            elif y.shape[1] > output_dim:
                y = y[:, :output_dim]
        return X, y
    
    def load_csv_data(self, csv_path, target_col=-1):
        """Load data from a CSV file. Last column is target by default.
        Returns: (X_tensor, y_tensor) or raises exception."""
        data = []
        with open(csv_path, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)  # skip header
            for row in reader:
                try:
                    vals = [float(v.strip()) for v in row if v.strip()]
                    if vals:
                        data.append(vals)
                except ValueError:
                    continue  # skip non-numeric rows
        
        if not data:
            raise ValueError(f"No valid numeric data found in {csv_path}")
        
        arr = np.array(data, dtype=np.float32)
        if target_col == -1:
            X = torch.from_numpy(arr[:, :-1])
            y = torch.from_numpy(arr[:, -1:])
        else:
            X = torch.from_numpy(np.delete(arr, target_col, axis=1))
            y = torch.from_numpy(arr[:, target_col:target_col+1])
        
        return X, y

        self.step_count += 1
        return loss.item(), self.optimizer.param_groups[0]['lr'], grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)

    def preview_scheduler(self, epochs):
        """Returns a list of LR values for the configured schedule."""
        # Create a dummy scheduler to avoid messing up the real one
        dummy_optimizer = optim.AdamW([torch.zeros(1)], lr=self.base_lr)
        dummy_scheduler = optim.lr_scheduler.CosineAnnealingLR(dummy_optimizer, T_max=max(1, epochs))
        
        lrs = []
        # Warmup simulation
        steps_per_epoch = 1 # Simplified: 1 step per epoch for visualization
        for epoch in range(epochs):
            # Warmup logic
            if epoch < self.warmup_steps:
                warmup_factor = (epoch + 1) / self.warmup_steps
                lr = self.base_lr * warmup_factor
            else:
                dummy_scheduler.step()
                lr = dummy_scheduler.get_last_lr()[0]
            lrs.append(lr)
        return lrs

    def train_step(self, X, y, noise_std=0.0):
        """Execute one training step. Returns (loss, lr, grad_norm, acc, f1)."""
        self.model.train()
        self.optimizer.zero_grad()
        
        device = next(self.model.parameters()).device
        X, y = X.to(device), y.to(device)
        
        # Data Augmentation (Noise)
        if noise_std > 0:
            noise = torch.randn_like(X) * noise_std
            X = X + noise
        
        # LR Warmup
        is_warmup = self._warmup_lr()
        
        # Curriculum learning: select subset of samples
        X, y = self._apply_curriculum(X, y)
        
        # AMP Training
        use_amp = self.scaler is not None
        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = self.model(X)
            
            # Metric Calculation
            with torch.no_grad():
                if self.loss_name in ('CrossEntropy', 'Focal', 'LabelSmooth'):
                    preds = torch.argmax(outputs, dim=1)
                    if y.dim() > 1 and y.shape[1] == 1:
                        t = y.squeeze(1)
                    elif y.dim() > 1:
                        t = y.argmax(dim=1)
                    else:
                        t = y
                    correct = (preds == t).float().sum()
                    accuracy = correct / t.shape[0]
                    
                    # Simple Macro F1 approximation
                    tp = ((preds == t) & (t == 1)).sum().float()
                    fp = ((preds == 1) & (t == 0)).sum().float()
                    fn = ((preds == 0) & (t == 1)).sum().float()
                    precision = tp / (tp + fp + 1e-8)
                    recall = tp / (tp + fn + 1e-8)
                    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
                else:
                    accuracy = torch.tensor(0.0)
                    f1 = torch.tensor(0.0)

            # Handle CrossEntropy (needs long targets)
            if self.loss_name == 'CrossEntropy':
                if y.dim() > 1 and y.shape[1] == 1:
                    y = y.squeeze(1).long()
                else:
                    y = y.argmax(dim=1) if y.dim() > 1 else y.long()
            
            base_loss = self.criterion(outputs, y)
            aux_loss = outputs.new_zeros(())
            if self.aux_loss_coef > 0 and hasattr(self.model, "get_aux_loss"):
                aux_loss = self.model.get_aux_loss()
            loss = base_loss + self.aux_loss_coef * aux_loss

        self.last_base_loss = float(base_loss.detach().item())
        self.last_aux_loss = float(aux_loss.detach().item())
        
        # Backward pass
        if use_amp:
            self.scaler.scale(loss / self.accumulation_steps).backward()
            if (self.step_count + 1) % self.accumulation_steps == 0:
                if self.use_sam:
                    self.scaler.unscale_(self.optimizer)
                    self.optimizer.first_step(zero_grad=True)
                    with torch.amp.autocast('cuda', enabled=True):
                        outputs_sam = self.model(X)
                        loss_sam = self.criterion(outputs_sam, y)
                        if self.aux_loss_coef > 0 and hasattr(self.model, "get_aux_loss"):
                            loss_sam += self.aux_loss_coef * self.model.get_aux_loss()
                    self.scaler.scale(loss_sam).backward()
                    self.scaler.unscale_(self.optimizer)
                    self.optimizer.second_step(zero_grad=True)
                else:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                
                self.scaler.update()
                self.optimizer.zero_grad()
                if self.use_ema: self.ema.update()
                grad_norm = torch.tensor(0.0) # placeholder for sam
            else:
                grad_norm = torch.tensor(0.0) # No update this step
        else:
            (loss / self.accumulation_steps).backward()
            if (self.step_count + 1) % self.accumulation_steps == 0:
                if self.use_sam:
                    self.optimizer.first_step(zero_grad=True)
                    outputs_sam = self.model(X)
                    loss_sam = self.criterion(outputs_sam, y)
                    if self.aux_loss_coef > 0 and hasattr(self.model, "get_aux_loss"):
                        loss_sam += self.aux_loss_coef * self.model.get_aux_loss()
                    loss_sam.backward()
                    self.optimizer.second_step(zero_grad=True)
                    grad_norm = torch.tensor(0.0)
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
                self.optimizer.zero_grad()
                if self.use_ema: self.ema.update()
            else:
                grad_norm = torch.tensor(0.0)
        
        # Scheduler step (skip during warmup)
        if (self.step_count + 1) % self.accumulation_steps == 0 and not is_warmup:
            self.scheduler.step()
        
        self.step_count += 1
        self._lawa_step()  # LAWA weight averaging (if enabled)
        # Record metrics in training history
        current_lr = self.optimizer.param_groups[0]['lr']
        self.training_history['loss'].append(loss.item())
        self.training_history['lr'].append(current_lr)
        self.training_history['grad_norm'].append(grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm))
        self.training_history['epoch'].append(self.step_count)
        self.training_history['accuracy'].append(accuracy.item())
        self.training_history['f1'].append(f1.item())
        
        return loss.item(), current_lr, grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm), accuracy.item(), f1.item()

    def export_onnx(self, path, input_dim):
        """Export model to ONNX format with graceful error handling."""
        if self.use_ema: self.ema.apply_shadow()  # Use EMA weights for export
        self.model.eval()
        device = next(self.model.parameters()).device
        dummy_input = torch.randn(1, input_dim).to(device)
        
        try:
            torch.onnx.export(
                self.model, dummy_input, path, 
                export_params=True, 
                opset_version=11, 
                do_constant_folding=True,
                input_names=['input'], 
                output_names=['output']
            )
            if self.use_ema: self.ema.restore()
            return path
        except Exception as e:
            if self.use_ema: self.ema.restore()
            # Provide helpful fallback message
            error_msg = str(e)
            if 'onnxscript' in error_msg.lower() or 'onnx' in error_msg.lower():
                raise RuntimeError(
                    f"ONNX export failed. You may need to install onnx: "
                    f"pip install onnx onnxscript\n\nOriginal error: {e}"
                )
            raise
    
    def export_torchscript(self, path, input_dim):
        """Export model as TorchScript (always available, no extra deps)."""
        if self.use_ema: self.ema.apply_shadow()
        self.model.eval()
        device = next(self.model.parameters()).device
        dummy_input = torch.randn(1, input_dim).to(device)
        traced = torch.jit.trace(self.model, dummy_input)
        traced.save(path)
        if self.use_ema: self.ema.restore()
        return path

    def compute_confusion_matrix(self, X, y):
        """
        Computes the confusion matrix for classification tasks.
        Returns: Tuple (matrix_string, raw_matrix_list)
        """
        if self.use_ema: self.ema.apply_shadow()
        self.model.eval()
        device = next(self.model.parameters()).device
        
        # Ensure data is on device
        X = X.to(device)
        y = y.to(device)
        
        with torch.no_grad():
            outputs = self.model(X)
            preds = torch.argmax(outputs, dim=1)
            
            # Handle targets
            if y.dim() > 1 and y.shape[1] > 1: # One-hot
                targets = torch.argmax(y, dim=1)
            else:
                targets = y.flatten().long()
                
        # Move to CPU
        preds = preds.cpu().numpy()
        targets = targets.cpu().numpy()
        
        if self.use_ema: self.ema.restore()
        
        # Get unique classes
        classes = sorted(list(set(targets) | set(preds)))
        n_classes = len(classes)
        
        # Build matrix
        matrix = [[0] * n_classes for _ in range(n_classes)]
        for p, t in zip(preds, targets):
            if t in classes and p in classes:
                t_idx = classes.index(t)
                p_idx = classes.index(p)
                matrix[t_idx][p_idx] += 1
                
        # Format as string
        header = "      " + " ".join([f"P{c:<4}" for c in classes])
        rows = [header]
        for i, row in enumerate(matrix):
            row_str = f"T{classes[i]:<4} " + " ".join([f"{val:<5}" for val in row])
            rows.append(row_str)
            
        return "\n".join(rows), matrix

    def evaluate(self, X_val, y_val):
        """Evaluate the model on validation data. Returns (loss, acc, f1)."""
        if self.use_ema: self.ema.apply_shadow()
        self.model.eval()
        with torch.no_grad():
            device = next(self.model.parameters()).device
            X_val, y_val = X_val.to(device), y_val.to(device)
            
            with torch.amp.autocast('cuda', enabled=(self.scaler is not None)):
                outputs = self.model(X_val)
                # Handle CrossEntropy (needs long targets)
                if self.loss_name == 'CrossEntropy':
                    if y_val.dim() > 1 and y_val.shape[1] == 1:
                        y_val = y_val.squeeze(1).long()
                    else:
                        y_val = y_val.argmax(dim=1) if y_val.dim() > 1 else y_val.long()
                
                val_loss = self.criterion(outputs, y_val)
                
                # Metrics
                if self.loss_name in ('CrossEntropy', 'Focal', 'LabelSmooth'):
                    preds = torch.argmax(outputs, dim=1)
                    if y_val.dim() > 1:
                        t = y_val if y_val.dtype == torch.long else y_val.argmax(dim=1)
                    else:
                        t = y_val
                    correct = (preds == t).float().sum()
                    accuracy = correct / t.shape[0]
                    
                    tp = ((preds == t) & (t == 1)).sum().float()
                    fp = ((preds == 1) & (t == 0)).sum().float()
                    fn = ((preds == 0) & (t == 1)).sum().float()
                    precision = tp / (tp + fp + 1e-8)
                    recall = tp / (tp + fn + 1e-8)
                    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
                else:
                    accuracy = torch.tensor(0.0)
                    f1 = torch.tensor(0.0)
        
        if self.use_ema: self.ema.restore()
        
        # Record validation loss in history
        self.training_history['val_loss'].append(val_loss.item())
        return val_loss.item(), accuracy.item(), f1.item()

    def get_training_history(self):
        """Get a copy of the training history."""
        return {key: value.copy() for key, value in self.training_history.items()}
    
    def reset_training_history(self):
        """Reset the training history."""
        self.training_history = {
            'loss': [],
            'val_loss': [],
            'lr': [],
            'grad_norm': [],
            'epoch': []
        }

# ============================================================
# PHASE 25: SWARM OPTIMIZATION
# ============================================================

class SwarmOptimizer:
    """
    Particle Swarm Optimization (PSO) for neural network weights.
    Each 'particle' is a full set of model weights.
    """
    def __init__(self, model, n_particles=8, inertia=0.5, cognitive=1.5, social=2.0):
        self.model = model
        self.n_particles = n_particles
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        
        self.device = next(model.parameters()).device
        
        # Flatten all parameters into a single vector
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.param_shapes = [p.shape for p in self.params]
        self.n_params = sum(p.numel() for p in self.params)
        
        # Particles: [n_particles, n_params]
        self.positions = torch.randn(n_particles, self.n_params, device=self.device) * 0.1
        self.velocities = torch.zeros(n_particles, self.n_params, device=self.device)
        
        # Best positions
        self.p_best_pos = self.positions.clone()
        self.p_best_val = torch.full((n_particles,), float('inf'), device=self.device)
        
        self.g_best_pos = self.positions[0].clone()
        self.g_best_val = float('inf')

    def _set_model_params(self, flat_params):
        offset = 0
        for p, shape in zip(self.params, self.param_shapes):
            numel = p.numel()
            p.data.copy_(flat_params[offset:offset+numel].view(shape))
            offset += numel

    def step(self, loss_eval_fn):
        """
        One PSO iteration. loss_eval_fn takes no args and returns a scalar loss 
        based on current model parameters.
        """
        for i in range(self.n_particles):
            # Evaluate particle
            self._set_model_params(self.positions[i])
            current_loss = loss_eval_fn()
            
            # Update personal best
            if current_loss < self.p_best_val[i]:
                self.p_best_val[i] = current_loss
                self.p_best_pos[i] = self.positions[i].clone()
                
            # Update global best
            if current_loss < self.g_best_val:
                self.g_best_val = current_loss
                self.g_best_pos = self.positions[i].clone()
        
        # Update velocities and positions
        r1 = torch.rand(self.n_particles, self.n_params, device=self.device)
        r2 = torch.rand(self.n_particles, self.n_params, device=self.device)
        
        self.velocities = (self.inertia * self.velocities + 
                           self.cognitive * r1 * (self.p_best_pos - self.positions) + 
                           self.social * r2 * (self.g_best_pos.unsqueeze(0) - self.positions))
        
        self.positions = self.positions + self.velocities
        
        # Final set: leave model at best found position
        self._set_model_params(self.g_best_pos)
        return self.g_best_val

class SwarmTrainingEngine:
    """Alternative training engine using Swarm Optimization."""
    def __init__(self, model, n_particles=5, loss_fn='MSE'):
        self.model = model
        self.swarm = SwarmOptimizer(model, n_particles=n_particles)
        
        if loss_fn in TrainingEngine.LOSS_FUNCTIONS:
            self.criterion = TrainingEngine.LOSS_FUNCTIONS[loss_fn]()
        else:
            self.criterion = nn.MSELoss()
            
    def train_step(self, X, y):
        device = next(self.model.parameters()).device
        X, y = X.to(device), y.to(device)
        
        def loss_fn():
            with torch.no_grad():
                out = self.model(X)
                return self.criterion(out, y).item()
        
        best_loss = self.swarm.step(loss_fn)
        return best_loss, 0.0, 0.0, 0.0, 0.0 # Standardize return for GUI compatibility
