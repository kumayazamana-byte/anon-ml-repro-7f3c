"""Training wrapper for new, isolated legacy-clock UEA baselines only."""

from __future__ import annotations

import copy
import datetime
import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from utilsd import use_cuda
from utilsd.earlystop import EarlyStopStatus

from .masktimeseries import MaskTS
from ..common.utils import AverageMeter, GlobalTracker, printt, to_torch


class LegacyBaselineMaskTS(MaskTS):
    """MaskTS with traceable epochs and resumable scheduler/RNG for new baselines.

    Its forward path is inherited unchanged from ``MaskTS``.  This class is not
    registered and is never used by historical experiment configurations.
    """

    def __init__(
        self,
        *args: Any,
        s5_ssm_lr: Optional[float] = None,
        s5_weight_decay: float = 0.04,
        s5_warmup_epochs: int = 1,
        persistent_workers: bool = False,
        prefetch_factor: int = 2,
        non_blocking_transfer: bool = False,
        epoch_metrics_path: Optional[Path] = None,
        interrupt_after_epoch: Optional[int] = None,
        **kwargs: Any,
    ):
        self.s5_ssm_lr = s5_ssm_lr
        self.s5_weight_decay = s5_weight_decay
        self.s5_warmup_epochs = s5_warmup_epochs
        self.persistent_workers = bool(persistent_workers)
        self.prefetch_factor = int(prefetch_factor)
        self.non_blocking_transfer = bool(non_blocking_transfer)
        if self.prefetch_factor < 1:
            raise ValueError("prefetch_factor must be positive.")
        self.epoch_metrics_path = epoch_metrics_path
        self.interrupt_after_epoch = interrupt_after_epoch
        self.scheduler: Optional[LambdaLR] = None
        self._data_loader_generator: Optional[torch.Generator] = None
        super().__init__(*args, **kwargs)

    def _init_optimization(self, optimizer: str, lr: float, weight_decay: float, **kwargs: Any) -> None:
        if optimizer != "S5AdamW":
            super()._init_optimization(optimizer=optimizer, lr=lr, weight_decay=weight_decay, **kwargs)
            return

        # Establish loss/metrics/early-stop exactly as the base implementation,
        # then replace only S5's documented optimizer groups.
        super()._init_optimization(optimizer="AdamW", lr=lr, weight_decay=weight_decay, **kwargs)
        ssm_params, regular_params = [], []
        ssm_tokens = ("lambda_", "b_", "c_", ".d", "log_step", ".norm")
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("network.") and any(token in name for token in ssm_tokens):
                ssm_params.append(param)
            else:
                regular_params.append(param)
        if not ssm_params or not regular_params:
            raise RuntimeError("S5 parameter grouping unexpectedly produced an empty group.")
        self.optimizer = optim.AdamW(
            [
                # Public S5 ``noBCdecay`` puts B/C/D, eigenvalues, learned
                # steps and BatchNorm in an Adam/no-decay SSM group.
                {"params": ssm_params, "lr": float(self.s5_ssm_lr), "weight_decay": 0.0},
                {"params": regular_params, "lr": lr, "weight_decay": self.s5_weight_decay},
            ]
        )
        self.hyper_paras["optimizer"] = "S5AdamW"
        self.hyper_paras["s5_ssm_lr"] = float(self.s5_ssm_lr)
        self.hyper_paras["s5_weight_decay"] = float(self.s5_weight_decay)
        self.hyper_paras["s5_warmup_epochs"] = int(self.s5_warmup_epochs)

    def _init_scheduler(self, loader_length: int) -> None:
        if self.s5_ssm_lr is None:
            self.scheduler = None
            return
        warmup_steps = max(1, int(self.s5_warmup_epochs) * loader_length)
        total_steps = max(warmup_steps + 1, self.max_epochs * loader_length)

        def scale(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            progress = min(1.0, (step - warmup_steps + 1) / float(total_steps - warmup_steps))
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda=[scale, scale])

    def _epoch_record(self, record: dict[str, object]) -> None:
        if self.epoch_metrics_path is None:
            return
        self.epoch_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.epoch_metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    @staticmethod
    def _capture_rng() -> dict[str, object]:
        return {
            "python": __import__("random").getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }

    @staticmethod
    def _restore_rng(state: dict[str, object]) -> None:
        __import__("random").setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"])
        if torch.cuda.is_available() and state.get("torch_cuda"):
            torch.cuda.set_rng_state_all(state["torch_cuda"])

    def _save_resume(self, epoch: int, best_res: dict[str, object]) -> None:
        payload = {
            "earlystop": self.early_stop.state_dict(),
            "model": self.state_dict(),
            "optim": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "epoch": epoch,
            "best_res": best_res,
            "best_params": self.best_params,
            "best_network_params": self.best_network_params,
            "rng": self._capture_rng(),
            "data_loader_generator": (
                self._data_loader_generator.get_state()
                if self._data_loader_generator is not None else None
            ),
        }
        torch.save(payload, self.checkpoint_dir / "resume.pth")

    def _load_resume(self) -> tuple[int, dict[str, object]]:
        path = self.checkpoint_dir / "resume.pth"
        if not path.exists():
            return 0, {}
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.early_stop.load_state_dict(checkpoint["earlystop"])
        self.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optim"])
        if self.scheduler is not None and checkpoint.get("scheduler") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        loader_state = checkpoint.get("data_loader_generator")
        if loader_state is not None and self._data_loader_generator is not None:
            self._data_loader_generator.set_state(loader_state)
        self.best_params = checkpoint["best_params"]
        self.best_network_params = checkpoint["best_network_params"]
        self._restore_rng(checkpoint["rng"])
        return int(checkpoint["epoch"]) + 1, checkpoint["best_res"]

    def _train_loader(self, trainset: Dataset) -> DataLoader:
        workers = int(self.num_workers or 0)
        kwargs: dict[str, object] = {
            "batch_size": self.batch_size,
            "shuffle": True,
            "pin_memory": True,
            "num_workers": workers,
            "generator": self._data_loader_generator,
        }
        if workers > 0:
            kwargs["persistent_workers"] = self.persistent_workers
            kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(trainset, **kwargs)

    def fit(self, trainset: Dataset, validset: Optional[Dataset] = None, testset: Optional[Dataset] = None) -> nn.Module:
        if validset is not None or not self.early_stop_on_train:
            raise ValueError("legacy_train_es requires no validation set and train-accuracy selection.")
        if self.checkpoint_dir is None:
            raise ValueError("Legacy baseline training requires a checkpoint directory.")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        trainset.load()
        loader_length = (len(trainset) + self.batch_size - 1) // self.batch_size
        self._init_scheduler(loader_length)
        self._data_loader_generator = torch.Generator()
        self._data_loader_generator.set_state(torch.get_rng_state())
        self.best_params = copy.deepcopy(self.state_dict())
        self.best_network_params = copy.deepcopy(self.network.state_dict())
        start_epoch, best_res = self._load_resume()
        loader = self._train_loader(trainset)
        best_epoch = int(best_res.pop("best_epoch", 0))
        epoch_times = []
        iteration_times = []
        if use_cuda() and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for epoch in range(start_epoch, self.max_epochs):
            self.train()
            tracker = GlobalTracker(self.metrics, self.metric_fn)
            train_loss = AverageMeter()
            if use_cuda() and torch.cuda.is_available():
                torch.cuda.synchronize()
            epoch_start = time.perf_counter()
            for _, (data, label) in enumerate(loader):
                if use_cuda():
                    data, label = (
                        to_torch(data, device="cuda", non_blocking=self.non_blocking_transfer),
                        to_torch(label, device="cuda", non_blocking=self.non_blocking_transfer),
                    )
                if use_cuda() and torch.cuda.is_available():
                    torch.cuda.synchronize()
                iteration_start = time.perf_counter()
                pred = self(data)
                loss = self.loss_fn(label.squeeze(-1), pred.squeeze(-1))
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                if use_cuda() and torch.cuda.is_available():
                    torch.cuda.synchronize()
                iteration_times.append(time.perf_counter() - iteration_start)
                train_loss.update(loss.item(), int(np.prod(label.shape)))
                tracker.update(label, pred)
            if use_cuda() and torch.cuda.is_available():
                torch.cuda.synchronize()
            epoch_time = time.perf_counter() - epoch_start
            epoch_times.append(epoch_time)
            tracker.concat()
            metrics = tracker.performance()
            metrics["loss"] = train_loss.performance()
            improved = self.early_stop.step(metrics[self.observe]) == EarlyStopStatus.BEST
            if improved:
                best_epoch = epoch
                self.best_params = copy.deepcopy(self.state_dict())
                self.best_network_params = copy.deepcopy(self.network.state_dict())
                best_res = {"train": metrics.copy()}
                torch.save(self.best_params, self.checkpoint_dir / "model_best.pkl")
                torch.save(self.best_network_params, self.checkpoint_dir / "network_best.pkl")
            self._epoch_record({
                "epoch": epoch,
                "train": metrics,
                "epoch_time_sec": epoch_time,
                "mean_iteration_time_sec": float(np.mean(iteration_times[-len(loader):])),
                "lr": [group["lr"] for group in self.optimizer.param_groups],
                "selected": improved,
            })
            printt(f"{epoch}\t'train'\tTime:{epoch_time:.2f}")
            for metric, value in metrics.items():
                printt(f"{metric}: {value:.4f}")
            print(f"{datetime.datetime.today()}")
            self._save_resume(epoch, {**best_res, "best_epoch": best_epoch})
            if self.interrupt_after_epoch is not None and epoch >= self.interrupt_after_epoch:
                raise RuntimeError("Test-only requested interruption after checkpoint save.")
            if not improved and self.early_stop.num_bad_epochs >= self.early_stop.patience:
                break

        del loader
        trainset.freeup()
        self._load_weight(self.best_params)
        if testset is not None:
            testset.load()
            with torch.no_grad():
                test_res = self.evaluate(testset)
            best_res["test"] = test_res
            testset.freeup()
        best_res["best_epoch"] = best_epoch
        best_res["runtime"] = {
            "epochs_ran": len(epoch_times),
            "sec_per_epoch": float(np.mean(epoch_times)) if epoch_times else 0.0,
            "epoch_times_sec": epoch_times,
            "steady_state_iteration_time_sec": float(np.mean(iteration_times[1:])) if len(iteration_times) > 1 else 0.0,
            "throughput_samples_per_sec": float(self.batch_size / np.mean(iteration_times[1:])) if len(iteration_times) > 1 else 0.0,
            "gpu_peak_memory_mb": float(torch.cuda.max_memory_allocated() / (1024 ** 2)) if use_cuda() and torch.cuda.is_available() else 0.0,
        }
        with (self.checkpoint_dir / "res.json").open("w", encoding="utf-8") as handle:
            json.dump(best_res, handle, indent=2, sort_keys=True)
        return self
