"""Training wrapper for CPM.

The wrapper deliberately owns optimisation only.  Uplift metrics are computed
in :meth:`evaluate`, after training, from the two potential-outcome estimates.
This keeps an observed factual outcome from being confused with an uplift
supervision signal.
"""

import logging
import os
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from metrics.uplift_metrics import (
        qini_auc_score1,
        uplift_at_k1,
        uplift_auc_score1,
    )

    HAS_METRICS = True
except ImportError:
    HAS_METRICS = False


logger = logging.getLogger(__name__)


class CPMTrainer:
    """Train a CPM model using factual-outcome Huber loss.

    ``model`` must return a mapping containing ``"y_factual"`` when called
    as ``model(x_ids, treatment)``.  The trainer intentionally never derives
    its optimisation loss from an uplift output.

    The expected dataloader batch is ``(x_ids, treatment, outcome)``, which is
    the same shape convention used by the project's other trainers.  ``x_ids``
    is passed through without a dtype conversion because CPM inputs are already
    bucketed integer feature ids by the preprocessing step.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        lr_factor: float = 0.6,
        lr_patience: int = 2,
        min_lr: float = 1e-6,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=lr_factor,
            patience=lr_patience,
            min_lr=min_lr,
        )
        self.criterion = nn.HuberLoss(delta=1.0)

    def _forward(self, x_ids: torch.Tensor, treatment: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.model(x_ids, treatment)
        if not isinstance(outputs, dict):
            raise TypeError("CPM model.forward must return a dictionary of outputs")
        if "y_factual" not in outputs:
            raise KeyError("CPM model output must contain 'y_factual' for factual-outcome training")
        return outputs

    def compute_loss(
        self, x_ids: torch.Tensor, treatment: torch.Tensor, outcome: torch.Tensor
    ) -> torch.Tensor:
        """Return Huber loss between observed outcomes and ``outputs['y_factual']``."""
        outputs = self._forward(x_ids, treatment)
        y_factual = outputs["y_factual"]
        target = outcome.to(dtype=y_factual.dtype).reshape_as(y_factual)
        return self.criterion(y_factual, target)

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Optimise factual Huber loss for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for x_ids, treatment, outcome in dataloader:
            x_ids = x_ids.to(self.device)
            treatment = treatment.to(self.device)
            outcome = outcome.to(self.device)

            self.optimizer.zero_grad()
            loss = self.compute_loss(x_ids, treatment, outcome)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Compute factual Huber loss and AUUC on val_loader."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        uplift_scores, treatments, outcomes = [], [], []

        with torch.no_grad():
            for x_ids, treatment, outcome in val_loader:
                x_ids_dev = x_ids.to(self.device)
                treatment_dev = treatment.to(self.device)
                outcome_dev = outcome.to(self.device)

                outputs = self._forward(x_ids_dev, treatment_dev)
                y_factual = outputs["y_factual"]
                target = outcome_dev.to(dtype=y_factual.dtype).reshape_as(y_factual)
                total_loss += self.criterion(y_factual, target).item()
                n_batches += 1

                y0_hat, y1_hat = self._potential_outcomes(outputs)
                uplift_scores.append((y1_hat - y0_hat).reshape(-1).cpu().numpy())
                treatments.append(treatment.reshape(-1).cpu().numpy())
                outcomes.append(outcome.reshape(-1).cpu().numpy())

        val_loss = total_loss / max(n_batches, 1)
        val_auuc = float("nan")
        if HAS_METRICS and len(uplift_scores) > 0:
            try:
                uplift_arr = np.concatenate(uplift_scores)
                treatment_arr = np.concatenate(treatments)
                outcome_arr = np.concatenate(outcomes)
                val_auuc = float(uplift_auc_score1(outcome_arr, uplift_arr, treatment_arr))
            except Exception as exc:
                logger.warning("Failed to compute validation AUUC: %s", exc)

        return val_loss, val_auuc

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        early_stopping_patience: int = 3,
        checkpoint_dir: Optional[str] = None,
        model_name: str = "cpm",
        writer: Optional[Any] = None,
        verbose: int = 1,
        monitor: str = "val_auuc",
    ) -> Dict[str, list]:
        """Train CPM and save ``*_best.pth`` and ``*_final.pth`` checkpoints.

        The monitor metric ('val_auuc' or 'val_loss') is used for early stopping
        and best-model selection. Learning rate scheduling (ReduceLROnPlateau) continues
        to step on val_loss for smooth gradient descent.
        """
        if epochs < 1:
            raise ValueError("epochs must be at least 1")

        history: Dict[str, list] = {"train_loss": [], "val_loss": [], "val_auuc": []}
        best_val_loss = float("inf")
        best_val_auuc = -float("inf")
        best_loss_epoch = 0
        best_auuc_epoch = 0

        monitor_mode = "max" if monitor == "val_auuc" else "min"
        best_monitor_score = -float("inf") if monitor_mode == "max" else float("inf")
        patience_counter = 0

        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)

        if verbose:
            logger.info("Starting CPM training | Device: %s | Epochs: %s | Monitor: %s", self.device, epochs, monitor)

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            history["train_loss"].append(train_loss)

            if val_loader is not None:
                val_loss, val_auuc = self.validate(val_loader)
                history["val_loss"].append(val_loss)
                history["val_auuc"].append(val_auuc)
                self.scheduler.step(val_loss)

                # 1. Lưu checkpoint tốt nhất theo val_loss (để đối chiếu)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_loss_epoch = epoch
                    if checkpoint_dir is not None:
                        ckpt_loss = os.path.join(checkpoint_dir, f"{model_name}_best_loss.pth")
                        self.save(ckpt_loss)
                        if verbose >= 2:
                            logger.info("  -> Saved best val_loss checkpoint: %s", ckpt_loss)

                # 2. Lưu checkpoint tốt nhất theo val_auuc (để đối chiếu)
                if not np.isnan(val_auuc) and val_auuc > best_val_auuc:
                    best_val_auuc = val_auuc
                    best_auuc_epoch = epoch
                    if checkpoint_dir is not None:
                        ckpt_auuc = os.path.join(checkpoint_dir, f"{model_name}_best_auuc.pth")
                        self.save(ckpt_auuc)
                        if verbose >= 2:
                            logger.info("  -> Saved best val_auuc checkpoint: %s", ckpt_auuc)

                # 3. Model selection & early stopping theo monitor_metric đã chọn
                if monitor == "val_auuc" and not np.isnan(val_auuc):
                    current_score = val_auuc
                    improved = current_score > best_monitor_score
                else:
                    current_score = val_loss
                    improved = current_score < best_monitor_score
            else:
                # Keep the history aligned by epoch while making the absence of
                # validation explicit. This is not used as a validation score.
                val_loss = float("nan")
                val_auuc = float("nan")
                history["val_loss"].append(val_loss)
                history["val_auuc"].append(val_auuc)
                self.scheduler.step(train_loss)
                current_score = train_loss
                improved = current_score < best_monitor_score

            if writer is not None:
                writer.add_scalar("Loss/train", train_loss, epoch)
                if val_loader is not None:
                    writer.add_scalar("Loss/val", val_loss, epoch)
                    if not np.isnan(val_auuc):
                        writer.add_scalar("AUUC/val", val_auuc, epoch)
                writer.add_scalar("LearningRate", self.optimizer.param_groups[0]["lr"], epoch)

            if improved:
                best_monitor_score = current_score
                patience_counter = 0
                if checkpoint_dir is not None:
                    best_path = os.path.join(checkpoint_dir, f"{model_name}_best.pth")
                    self.save(best_path)
                    if verbose >= 2:
                        logger.info("  -> Saved best checkpoint (%s=%.5f): %s", monitor, best_monitor_score, best_path)
            else:
                patience_counter += 1

            if verbose:
                val_text = f" | Val Loss: {val_loss:.5f} | Val AUUC: {val_auuc:.5f}" if val_loader is not None else ""
                logger.info(
                    "Epoch [%02d/%02d]  Train Loss: %.5f%s | Best (%s): %.5f | LR: %.6f | EarlyStop: %d/%d",
                    epoch,
                    epochs,
                    train_loss,
                    val_text,
                    monitor,
                    best_monitor_score,
                    self.optimizer.param_groups[0]["lr"],
                    patience_counter,
                    early_stopping_patience,
                )

            if patience_counter >= early_stopping_patience:
                if verbose:
                    logger.info("Early stopping triggered at epoch %d", epoch)
                break

        if checkpoint_dir is not None:
            final_path = os.path.join(checkpoint_dir, f"{model_name}_final.pth")
            self.save(final_path)
            if verbose >= 2:
                logger.info("  -> Saved final checkpoint: %s", final_path)

            best_path = os.path.join(checkpoint_dir, f"{model_name}_best.pth")
            if os.path.exists(best_path):
                self.load(best_path)

        history["best_loss_epoch"] = best_loss_epoch
        history["best_val_loss"] = best_val_loss
        history["best_auuc_epoch"] = best_auuc_epoch
        history["best_val_auuc"] = best_val_auuc
        history["best_epoch"] = best_loss_epoch if monitor == "val_loss" else best_auuc_epoch
        if best_loss_epoch > 0 and len(history.get("val_auuc", [])) >= best_loss_epoch:
            history["val_auuc_at_best_loss"] = history["val_auuc"][best_loss_epoch - 1]
        else:
            history["val_auuc_at_best_loss"] = float("nan")

        return history

    def _potential_outcomes(
        self, outputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read potential outcomes for metric calculation, never for training."""
        if "y0_hat" in outputs and "y1_hat" in outputs:
            return outputs["y0_hat"], outputs["y1_hat"]
        if "y0" in outputs and "y1" in outputs:
            return outputs["y0"], outputs["y1"]
        raise KeyError("CPM model output must contain y0_hat/y1_hat (or y0/y1) for uplift metrics")

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader, k: float = 0.3, print_diagnostics: bool = True) -> Dict[str, float]:
        """Evaluate factual loss and uplift metrics, separately from optimisation."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        uplift_scores, treatments, outcomes = [], [], []
        y0_all, y1_all = [], []
        g0_all, g1_all = [], []
        ind0_all, ind1_all = [], []

        for x_ids, treatment, outcome in test_loader:
            x_ids_dev = x_ids.to(self.device)
            treatment_dev = treatment.to(self.device)
            outcome_dev = outcome.to(self.device)

            outputs = self._forward(x_ids_dev, treatment_dev)
            y_factual = outputs["y_factual"]
            target = outcome_dev.to(dtype=y_factual.dtype).reshape_as(y_factual)
            total_loss += self.criterion(y_factual, target).item()
            n_batches += 1

            y0_hat, y1_hat = self._potential_outcomes(outputs)
            uplift_scores.append((y1_hat - y0_hat).reshape(-1).cpu().numpy())
            treatments.append(treatment.reshape(-1).cpu().numpy())
            outcomes.append(outcome.reshape(-1).cpu().numpy())

            if print_diagnostics:
                y0_all.append(y0_hat.detach().cpu())
                y1_all.append(y1_hat.detach().cpu())
                if "g0" in outputs and outputs["g0"] is not None:
                    g0_all.append(outputs["g0"].detach().cpu())
                if "g1" in outputs and outputs["g1"] is not None:
                    g1_all.append(outputs["g1"].detach().cpu())
                if "ind0" in outputs and outputs["ind0"] is not None:
                    ind0_all.append(outputs["ind0"].detach().cpu())
                if "ind1" in outputs and outputs["ind1"] is not None:
                    ind1_all.append(outputs["ind1"].detach().cpu())

        if print_diagnostics and y0_all and y1_all:
            y0 = torch.cat(y0_all, dim=0)
            y1 = torch.cat(y1_all, dim=0)

            logger.info(
                "Diagnostics | y0: mean=%.6f, std=%.6f, min=%.6f, max=%.6f",
                y0.mean().item(),
                y0.std().item(),
                y0.min().item(),
                y0.max().item(),
            )
            logger.info(
                "Diagnostics | y1: mean=%.6f, std=%.6f, min=%.6f, max=%.6f",
                y1.mean().item(),
                y1.std().item(),
                y1.min().item(),
                y1.max().item(),
            )

            uplift = y1 - y0
            logger.info(
                "Diagnostics | uplift: mean=%.6f, std=%.6f, min=%.6f, max=%.6f",
                uplift.mean().item(),
                uplift.std().item(),
                uplift.min().item(),
                uplift.max().item(),
            )

            if g0_all and g1_all:
                g0 = torch.cat(g0_all, dim=0)
                g1 = torch.cat(g1_all, dim=0)
                logger.info("Diagnostics | control gate mean: %s", g0.mean(dim=0).tolist())
                logger.info("Diagnostics | treat gate mean: %s", g1.mean(dim=0).tolist())

            if ind0_all and ind1_all:
                ind0 = torch.cat(ind0_all, dim=0)
                ind1 = torch.cat(ind1_all, dim=0)
                logger.info(
                    "Diagnostics | indicator0: mean=%.6f, std=%.6f, min=%.6f, max=%.6f",
                    ind0.mean().item(),
                    ind0.std().item(),
                    ind0.min().item(),
                    ind0.max().item(),
                )
                logger.info(
                    "Diagnostics | indicator1: mean=%.6f, std=%.6f, min=%.6f, max=%.6f",
                    ind1.mean().item(),
                    ind1.std().item(),
                    ind1.min().item(),
                    ind1.max().item(),
                )

            t0 = torch.tensor([0], dtype=torch.long, device=self.device)
            t1 = torch.tensor([1], dtype=torch.long, device=self.device)
            t_emb0 = self.model.encoder.encode_treatment(t0)
            t_emb1 = self.model.encoder.encode_treatment(t1)
            e_gui0, e_ind0 = self.model.treatment_refine(t_emb0)
            e_gui1, e_ind1 = self.model.treatment_refine(t_emb1)
            g0_vec = self.model.control_gate(e_gui0)
            g1_vec = self.model.treatment_gate(e_gui1)

            logger.info("Diagnostics | treatment emb diff: %.6f", (t_emb0 - t_emb1).abs().mean().item())
            logger.info("Diagnostics | guidance diff:      %.6f", (e_gui0 - e_gui1).abs().mean().item())
            logger.info("Diagnostics | indicator diff:     %.6f", (e_ind0 - e_ind1).abs().mean().item())
            logger.info("Diagnostics | gate diff:          %.6f", (g0_vec - g1_vec).abs().mean().item())

        results: Dict[str, float] = {"loss": total_loss / max(n_batches, 1)}
        if not HAS_METRICS or not uplift_scores:
            return results

        uplift_arr = np.concatenate(uplift_scores)
        treatment_arr = np.concatenate(treatments)
        outcome_arr = np.concatenate(outcomes)
        try:
            results["auuc"] = float(uplift_auc_score1(outcome_arr, uplift_arr, treatment_arr))
        except Exception as exc:
            logger.warning("Failed to compute AUUC: %s", exc)
        try:
            results["qini"] = float(qini_auc_score1(outcome_arr, uplift_arr, treatment_arr))
        except Exception as exc:
            logger.warning("Failed to compute Qini: %s", exc)
        try:
            results[f"lift@{int(k * 100)}%"] = float(
                uplift_at_k1(outcome_arr, uplift_arr, treatment_arr, strategy="overall", k=k)
            )
        except Exception as exc:
            logger.warning("Failed to compute Uplift@%s: %s", k, exc)
        return results

    @torch.no_grad()
    def print_diagnostics(self, dataloader: DataLoader) -> None:
        """Evaluate model and print distribution statistics for y0, y1, uplift, gates, and indicators."""
        self.evaluate(dataloader, print_diagnostics=True)

    def save(self, path: str) -> None:
        """Save model weights using the checkpoint convention of other trainers."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        """Load model weights saved by :meth:`save`."""
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)


# Short alias retained for compatibility with existing CDUM callers.
CPM = CPMTrainer
