# src/engine/trainer.py
import torch
from torch.optim import AdamW
from torch.amp import autocast, GradScaler   # NEW API
from tqdm.auto import tqdm

class Trainer:
    def __init__(self, model, criterion, device="cpu",
                 lr=1e-3, weight_decay=5e-4, use_amp=False, cfg=None):
        self.model = model.to(device)
        self.criterion = criterion
        self.device = device
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        # AMP
        self.use_amp = use_amp and torch.cuda.is_available() and device.startswith("cuda")
        self.scaler = GradScaler('cuda', enabled=self.use_amp)
        self.cfg = cfg or {}

    def fit(self, train_loader, val_loader, evaluator, epochs=1, soft_warmup=2):
        for epoch in range(epochs):
            self.model.train()
            bar = tqdm(total=len(train_loader), ncols=120, ascii=True, desc=f"{epoch+1}/{epochs}")
            agg = {"loss":0.0, "box":0.0, "obj":0.0, "ang":0.0, "kpt":0.0, "kc":0.0, "pos":0.0}
            for it, batch in enumerate(train_loader):
                imgs = batch["image"].to(self.device)
                outs = self.model(imgs)
                det_maps = outs["det"]; kpt_maps = outs["kpt"]
                b2 = {k: ([t.to(self.device) for t in v] if isinstance(v, list) else v.to(self.device))
                      for k, v in batch.items() if k != "image"}
                with autocast(device_type='cuda', enabled=self.use_amp):
                    loss, logs = self.criterion(det_maps, kpt_maps, {"image": imgs, **b2},
                                                epoch=epoch, soft_warmup_epochs=soft_warmup)
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                agg["loss"] += logs["loss"]
                agg["box"]  += logs["loss_box"]
                agg["obj"]  += logs["loss_obj"]
                agg["ang"]  += logs["loss_ang"]
                agg["kpt"]  += logs["loss_kpt"]
                agg["kc"]   += logs["loss_kc"]
                agg["pos"]  += logs.get("num_pos", 0.0)

                bar.set_postfix({
                    "box": f"{logs['loss_box']:.3f}",
                    "obj": f"{logs['loss_obj']:.3f}",
                    "ang": f"{logs['loss_ang']:.3f}",
                    "kpt": f"{logs['loss_kpt']:.3f}",
                    "IoU": f"{logs.get('mean_iou', 0.0):.3f}",
                    "Pos": int(logs.get("num_pos", 0)),
                })
                bar.update(1)
            bar.close()

            n = max(1, len(train_loader))
            gpu_mem = 0.0
            if self.device.startswith("cuda") and torch.cuda.is_available():
                gpu_mem = torch.cuda.max_memory_allocated() / (1024**3)
                torch.cuda.reset_peak_memory_stats()

            # Training epoch summary
            print(f"{epoch+1:>5}/{epochs:<5}  GPU_mem {gpu_mem:5.2f}G  "
                  f"box_loss {agg['box']/n:.4f}  obj_loss {agg['obj']/n:.4f}  "
                  f"ang_loss {agg['ang']/n:.4f}  kpt_loss {agg['kpt']/n:.4f}  "
                  f"kc_loss {agg['kc']/n:.4f}  Pos {agg['pos']/n:.1f}")

            # ---- Evaluation ----
            metrics = evaluator.evaluate(self.model, val_loader, device=self.device)
            print(f"                 all    Images {metrics.get('images', 0):>5}  "
                  f"mAP50 {metrics.get('map50', 0.0):.3f}  PCK@0.05 {metrics.get('pck@0.05', 0.0):.3f}")
            # switch back to train next epoch (done at loop top)

