# src/engine/trainer.py
import os
import torch
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm

from .dist import is_dist, is_main_process
from .checkpoint import CheckpointManager

class Trainer:
    def __init__(self, model, criterion, cfg, device="cpu"):
        self.cfg = cfg
        self.device = device
        self.use_amp = bool(cfg.train.amp and device.startswith("cuda") and torch.cuda.is_available())

        self.model = model.to(device)
        if is_dist():
            self.model = DDP(
                self.model,
                device_ids=[torch.cuda.current_device()] if device.startswith("cuda") else None,
                find_unused_parameters=bool(cfg.ddp.find_unused_parameters),
                broadcast_buffers=bool(cfg.ddp.broadcast_buffers),
            )

        self.criterion = criterion
        self.optimizer = AdamW(self.model.parameters(), lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay))
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

        os.makedirs(cfg.train.save_dir, exist_ok=True)
        self.ckpt = CheckpointManager(
            save_dir=cfg.train.save_dir,
            monitor=str(getattr(cfg.train, "monitor", "map50")),
            mode=str(getattr(cfg.train, "monitor_mode", "max")).lower(),
        )
        self.start_epoch = 0  # can be updated by resume()

    def _maybe_ddp_module(self):
        return self.model.module if isinstance(self.model, DDP) else self.model

    def resume(self, path: str):
        """Load checkpoint (model+opt+scaler) and set start_epoch."""
        if not os.path.isfile(path):
            if is_main_process():
                print(f"[resume] file not found: {path}")
            return
        se, _ = self.ckpt.load(path, self._maybe_ddp_module(), self.optimizer, self.scaler, map_location=self.device)
        self.start_epoch = max(0, se)
        if is_main_process():
            print(f"[resume] loaded '{path}' (start_epoch={self.start_epoch}, best={self.ckpt.best_value:.6f})")

    def fit(self, train_loader, val_loader, evaluator, train_sampler=None):
        epochs = int(self.cfg.train.epochs)
        soft_warmup = int(self.cfg.loss.soft_obj_warmup_epochs)

        for epoch in range(self.start_epoch, epochs):
            if is_dist() and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            self.model.train()
            bar = tqdm(total=len(train_loader), ncols=120, ascii=True, desc=f"{epoch+1}/{epochs}") if is_main_process() else None
            agg = {"loss":0.0, "box":0.0, "obj":0.0, "ang":0.0, "kpt":0.0, "pos":0.0}

            for it, batch in enumerate(train_loader):
                imgs = batch["image"].to(self.device, non_blocking=True)
                outs = self.model(imgs)
                det_maps = outs["det"]
                feats = outs["feats"]
                b2 = {k: ([t.to(self.device, non_blocking=True) for t in v] if isinstance(v, list) else v.to(self.device, non_blocking=True))
                      for k, v in batch.items() if k != "image"}

                with autocast(device_type='cuda', enabled=self.use_amp):
                    loss, logs = self.criterion(det_maps, feats, {"image": imgs, **b2}, model=self._maybe_ddp_module(),
                                                epoch=epoch)
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                if is_main_process():
                    agg["loss"] += logs["loss"]; agg["box"] += logs["loss_box"]
                    agg["obj"] += logs["loss_obj"]; agg["ang"] += logs["loss_ang"]
                    agg["kpt"] += logs["loss_kpt"]
                    agg["pos"] += logs.get("num_pos", 0.0)
                    if bar:
                        bar.set_postfix({
                            "box": f"{logs['loss_box']:.3f}",
                            "obj": f"{logs['loss_obj']:.3f}",
                            "ang": f"{logs['loss_ang']:.3f}",
                            "kpt": f"{logs['loss_kpt']:.3f}",
                            "IoU": f"{logs.get('mean_iou', 0.0):.3f}",
                            "Pos": int(logs.get("num_pos", 0)),
                        })
                        bar.update(1)

            if is_main_process() and bar:
                bar.close()
                n = max(1, len(train_loader))
                gpu_mem = 0.0
                if self.device.startswith("cuda") and torch.cuda.is_available():
                    gpu_mem = torch.cuda.max_memory_allocated() / (1024**3)
                    torch.cuda.reset_peak_memory_stats()

                print(f"{epoch+1:>5}/{epochs:<5}  GPU_mem {gpu_mem:5.2f}G  "
                      f"box_loss {agg['box']/n:.4f}  obj_loss {agg['obj']/n:.4f}  "
                      f"ang_loss {agg['ang']/n:.4f}  kpt_loss {agg['kpt']/n:.4f}  "
                      f"Pos {agg['pos']/n:.1f}")

                # ---- EVAL (rank 0 only)
                metrics = evaluator.evaluate(self._maybe_ddp_module(), val_loader, device=self.device)
                print(f"                 all    Images {metrics['images']:5d}  "
                      f"mAP50 {metrics['map50']:.6f}  PCK@0.05 {metrics['pck@0.05']:.6f}  "
                      f"PCK_any@0.05 {metrics['pck_any@0.05']:.6f}  "
                      f"TPs {metrics['tp_count']}  pred/img {metrics['pred_per_img_avg']:.1f}  "
                      f"R@0.1 {metrics['recall@0.1']:.2f}  R@0.3 {metrics['recall@0.3']:.2f}  R@0.5 {metrics['recall@0.5']:.2f}  "
                      f"bestIoU {metrics['best_iou_mean']:.3f}")

                # ---- CHECKPOINTS (rank 0 only)
                # save last
                last_path = self.ckpt.save_last(epoch, self._maybe_ddp_module(), self.optimizer, self.scaler, self.cfg, metrics)
                # save best if improved
                improved, best_path = self.ckpt.save_best_if_improved(epoch, self._maybe_ddp_module(), self.optimizer, self.scaler, self.cfg, metrics)
                if improved:
                    print(f"[ckpt] best improved ({self.ckpt.monitor}={metrics.get(self.ckpt.monitor):.6f}) -> {best_path}")
                else:
                    print(f"[ckpt] saved last -> {last_path}")
