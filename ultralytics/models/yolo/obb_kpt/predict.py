# In ultralytics/models/yolo/obb_kpt/predict.py
import torch
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import ops

class OBBKeypointPredictor(DetectionPredictor):
    def postprocess(self, preds, img, orig_imgs):
        preds = ops.non_max_suppression(preds, self.args.conf, self.args.iou,
                                        agnostic=self.args.agnostic_nms, max_det=self.args.max_det,
                                        classes=self.args.classes, nc=self.model.nc, nk=self.model.kpt_shape)
        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            if not pred.shape:
                results.append(self.results_type(orig_img=orig_img, path=self.batch, names=self.model.names, boxes=torch.empty(0, 7)))
                continue
            kpts = pred[:, 7:]
            pred = pred[:, :7]
            results.append(self.results_type(orig_img=orig_img, path=self.batch, names=self.model.names, boxes=pred, keypoints=kpts))
        return results