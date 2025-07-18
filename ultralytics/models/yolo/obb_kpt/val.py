# In ultralytics/models/yolo/obb_kpt/val.py
import numpy as np
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.metrics import OBBMetrics, PoseMetrics

class OBBKeypointValidator(DetectionValidator):
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.kpt_metrics = PoseMetrics(save_dir=self.save_dir, plot=self.args.plots)
        self.metrics = OBBMetrics(save_dir=self.save_dir, plot=self.args.plots)

    def init_metrics(self, model):
        super().init_metrics(model)
        self.kpt_metrics.kpt_shape = self.data['kpt_shape']
        nkpt = self.kpt_metrics.kpt_shape
        self.kpt_metrics.sigma = np.ones(nkpt) / nkpt

    def get_stats(self):
        stats = self.metrics.get_stats()
        kpt_stats = self.kpt_metrics.get_stats()
        stats.update(kpt_stats)
        return stats

    def update_metrics(self, preds, batch):
        self.metrics.process(preds, batch)
        self.kpt_metrics.process(preds, batch)