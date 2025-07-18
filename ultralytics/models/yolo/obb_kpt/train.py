# created by dbasavegowda
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import OBB_KPTModel

class OBBKeypointTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = OBB_KPTModel(cfg, ch=3, nc=self.data['nc'], verbose=verbose)
        if weights: model.load(weights)
        return model