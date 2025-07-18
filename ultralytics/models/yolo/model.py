# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# +++++++++++++++++++++ START OF MODIFICATION (Imports) +++++++++++++++++++++
# Add these imports to the top of the file to support the new method.
# These provide the necessary functions like yaml_load, check_yaml, and RANK.
import os
from pathlib import Path

# ++++++++++++++++++++++ END OF MODIFICATION (Imports) ++++++++++++++++++++++
from typing import Any, Dict, List, Optional, Union

from ultralytics.data.build import load_inference_source
from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import (
    WorldModel,
    YOLOEModel,
    YOLOESegModel,
    yaml_model_load,
)
from ultralytics.utils import ROOT, YAML, checks

# added by dbasavegowda
# Replace the entire existing 'YOLO' class with this one.


class YOLO(Model):
    """
    A class for loading and running YOLO models for various tasks.
    This version has a modified constructor to correctly handle new model
    creation from custom YAML files.
    """

    def __init__(self, model: Union[str, Path] = "yolo11n.pt", task: Optional[str] = None, verbose: bool = False):
        """
        Initialize a YOLO model.

        This constructor initializes a YOLO model, automatically switching to specialized model types
        (YOLOWorld or YOLOE) based on the model filename.

        Args:
            model (str | Path): Model name or path to model file, i.e. 'yolo11n.pt', 'yolo11n.yaml'.
            task (str, optional): YOLO task specification, i.e. 'detect', 'segment', 'classify', 'pose', 'obb'.
                Defaults to auto-detection based on model.
            verbose (bool): Display model info on load.

        Examples:
            >>> from ultralytics import YOLO
            >>> model = YOLO("yolo11n.pt")  # load a pretrained YOLOv11n detection model
            >>> model = YOLO("yolo11n-seg.pt")  # load a pretrained YOLO11n segmentation model
        """
        path = Path(model if isinstance(model, (str, Path)) else "")
        if "-world" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:  # if YOLOWorld PyTorch model
            new_instance = YOLOWorld(path, verbose=verbose)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        elif "yoloe" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:  # if YOLOE PyTorch model
            new_instance = YOLOE(path, task=task, verbose=verbose)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        else:
            # Continue with default YOLO initialization
            super().__init__(model=model, task=task, verbose=verbose)
            if hasattr(self.model, "model") and "RTDETR" in self.model.model[-1]._get_name():  # if RTDETR head
                from ultralytics import RTDETR

                new_instance = RTDETR(self)
                self.__class__ = type(new_instance)
                self.__dict__ = new_instance.__dict__

    # added by dbasavegowda
    # +++++++++++++++++++++ START OF MODIFICATION (_new method) +++++++++++++++++++++
    # --- Rationale for Change ---
    # We override the _new method to correctly handle model creation from our custom YAML.
    # The base Model._new() method fails because it doesn't know about our custom task_map.
    # This overridden method intercepts the process, looks up our custom model class
    # from the task_map, and builds it correctly before the base logic can fail.

    def _new(self, cfg: Union[str, Path], task=None, verbose=True):
        """Initializes a new model and infers the task type from the model head."""
        cfg_dict = yaml_model_load(checks.check_yaml(cfg))
        self.task = cfg_dict.get("task") or task

        # Look up the correct model class from our task_map
        model_class_path = self.task_map[self.task]["model"]

        if isinstance(model_class_path, str):
            parts = model_class_path.split(".")
            module_path, class_name = ".".join(parts[:-1]), parts[-1]
            module = __import__(module_path, fromlist=[class_name])
            model_class = getattr(module, class_name)
        else:
            model_class = model_class_path

        RANK = int(os.getenv("RANK", -1))

        # Build the model instance using the locally defined RANK
        self.model = model_class(cfg_dict, verbose=verbose and RANK == -1)
        self.overrides["model"] = str(cfg)

    # ++++++++++++++++++++++ END OF MODIFICATION (_new method) ++++++++++++++++++++++

    @property
    def task_map(self):
        """Returns a dictionary mapping tasks to their corresponding model classes."""
        return {
            "detect": {
                "model": "ultralytics.nn.tasks.DetectionModel",
                "trainer": "ultralytics.models.yolo.detect.DetectionTrainer",
                "validator": "ultralytics.models.yolo.detect.DetectionValidator",
                "predictor": "ultralytics.models.yolo.detect.DetectionPredictor",
            },
            "segment": {
                "model": "ultralytics.nn.tasks.SegmentationModel",
                "trainer": "ultralytics.models.yolo.segment.SegmentationTrainer",
                "validator": "ultralytics.models.yolo.segment.SegmentationValidator",
                "predictor": "ultralytics.models.yolo.segment.SegmentationPredictor",
            },
            "pose": {
                "model": "ultralytics.nn.tasks.PoseModel",
                "trainer": "ultralytics.models.yolo.pose.PoseTrainer",
                "validator": "ultralytics.models.yolo.pose.PoseValidator",
                "predictor": "ultralytics.models.yolo.pose.PosePredictor",
            },
            "classify": {
                "model": "ultralytics.nn.tasks.ClassificationModel",
                "trainer": "ultralytics.models.yolo.classify.ClassificationTrainer",
                "validator": "ultralytics.models.yolo.classify.ClassificationValidator",
                "predictor": "ultralytics.models.yolo.classify.ClassificationPredictor",
            },
            "obb": {
                "model": "ultralytics.nn.tasks.OBBModel",
                "trainer": "ultralytics.models.yolo.obb.OBBTrainer",
                "validator": "ultralytics.models.yolo.obb.OBBValidator",
                "predictor": "ultralytics.models.yolo.obb.OBBPredictor",
            },
            # Our new, complete task definition
            "obb_kpt": {
                "model": "ultralytics.nn.tasks.OBB_KPTModel",
                "trainer": "ultralytics.models.yolo.obb_kpt.OBBKeypointTrainer",
                "validator": "ultralytics.models.yolo.obb_kpt.OBBKeypointValidator",
                "predictor": "ultralytics.models.yolo.obb_kpt.OBBKeypointPredictor",
            },
        }


class YOLOWorld(Model):
    """
    YOLO-World object detection model.

    YOLO-World is an open-vocabulary object detection model that can detect objects based on text descriptions
    without requiring training on specific classes. It extends the YOLO architecture to support real-time
    open-vocabulary detection.

    Attributes:
        model: The loaded YOLO-World model instance.
        task: Always set to 'detect' for object detection.
        overrides: Configuration overrides for the model.

    Methods:
        __init__: Initialize YOLOv8-World model with a pre-trained model file.
        task_map: Map tasks to their corresponding model, trainer, validator, and predictor classes.
        set_classes: Set the model's class names for detection.

    Examples:
        Load a YOLOv8-World model
        >>> model = YOLOWorld("yolov8s-world.pt")

        Set custom classes for detection
        >>> model.set_classes(["person", "car", "bicycle"])
    """

    def __init__(self, model: Union[str, Path] = "yolov8s-world.pt", verbose: bool = False) -> None:
        """
        Initialize YOLOv8-World model with a pre-trained model file.

        Loads a YOLOv8-World model for object detection. If no custom class names are provided, it assigns default
        COCO class names.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """
        super().__init__(model=model, task="detect", verbose=verbose)

        # Assign default COCO class names when there are no custom names
        if not hasattr(self.model, "names"):
            self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self) -> Dict[str, Dict[str, Any]]:
        """Map head to model, validator, and predictor classes."""
        return {
            "detect": {
                "model": WorldModel,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
                "trainer": yolo.world.WorldTrainer,
            }
        }

    def set_classes(self, classes: List[str]) -> None:
        """
        Set the model's class names for detection.

        Args:
            classes (List[str]): A list of categories i.e. ["person"].
        """
        self.model.set_classes(classes)
        # Remove background if it's given
        background = " "
        if background in classes:
            classes.remove(background)
        self.model.names = classes

        # Reset method class names
        if self.predictor:
            self.predictor.model.names = classes


class YOLOE(Model):
    """
    YOLOE object detection and segmentation model.

    YOLOE is an enhanced YOLO model that supports both object detection and instance segmentation tasks with
    improved performance and additional features like visual and text positional embeddings.

    Attributes:
        model: The loaded YOLOE model instance.
        task: The task type (detect or segment).
        overrides: Configuration overrides for the model.

    Methods:
        __init__: Initialize YOLOE model with a pre-trained model file.
        task_map: Map tasks to their corresponding model, trainer, validator, and predictor classes.
        get_text_pe: Get text positional embeddings for the given texts.
        get_visual_pe: Get visual positional embeddings for the given image and visual features.
        set_vocab: Set vocabulary and class names for the YOLOE model.
        get_vocab: Get vocabulary for the given class names.
        set_classes: Set the model's class names and embeddings for detection.
        val: Validate the model using text or visual prompts.
        predict: Run prediction on images, videos, directories, streams, etc.

    Examples:
        Load a YOLOE detection model
        >>> model = YOLOE("yoloe-11s-seg.pt")

        Set vocabulary and class names
        >>> model.set_vocab(["person", "car", "dog"], ["person", "car", "dog"])

        Predict with visual prompts
        >>> prompts = {"bboxes": [[10, 20, 100, 200]], "cls": ["person"]}
        >>> results = model.predict("image.jpg", visual_prompts=prompts)
    """

    def __init__(
        self, model: Union[str, Path] = "yoloe-11s-seg.pt", task: Optional[str] = None, verbose: bool = False
    ) -> None:
        """
        Initialize YOLOE model with a pre-trained model file.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            task (str, optional): Task type for the model. Auto-detected if None.
            verbose (bool): If True, prints additional information during initialization.
        """
        super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self) -> Dict[str, Dict[str, Any]]:
        """Map head to model, validator, and predictor classes."""
        return {
            "detect": {
                "model": YOLOEModel,
                "validator": yolo.yoloe.YOLOEDetectValidator,
                "predictor": yolo.detect.DetectionPredictor,
                "trainer": yolo.yoloe.YOLOETrainer,
            },
            "segment": {
                "model": YOLOESegModel,
                "validator": yolo.yoloe.YOLOESegValidator,
                "predictor": yolo.segment.SegmentationPredictor,
                "trainer": yolo.yoloe.YOLOESegTrainer,
            },
        }

    def get_text_pe(self, texts):
        """Get text positional embeddings for the given texts."""
        assert isinstance(self.model, YOLOEModel)
        return self.model.get_text_pe(texts)

    def get_visual_pe(self, img, visual):
        """
        Get visual positional embeddings for the given image and visual features.

        This method extracts positional embeddings from visual features based on the input image. It requires
        that the model is an instance of YOLOEModel.

        Args:
            img (torch.Tensor): Input image tensor.
            visual (torch.Tensor): Visual features extracted from the image.

        Returns:
            (torch.Tensor): Visual positional embeddings.

        Examples:
            >>> model = YOLOE("yoloe-11s-seg.pt")
            >>> img = torch.rand(1, 3, 640, 640)
            >>> visual_features = torch.rand(1, 1, 80, 80)
            >>> pe = model.get_visual_pe(img, visual_features)
        """
        assert isinstance(self.model, YOLOEModel)
        return self.model.get_visual_pe(img, visual)

    def set_vocab(self, vocab: List[str], names: List[str]) -> None:
        """
        Set vocabulary and class names for the YOLOE model.

        This method configures the vocabulary and class names used by the model for text processing and
        classification tasks. The model must be an instance of YOLOEModel.

        Args:
            vocab (List[str]): Vocabulary list containing tokens or words used by the model for text processing.
            names (List[str]): List of class names that the model can detect or classify.

        Raises:
            AssertionError: If the model is not an instance of YOLOEModel.

        Examples:
            >>> model = YOLOE("yoloe-11s-seg.pt")
            >>> model.set_vocab(["person", "car", "dog"], ["person", "car", "dog"])
        """
        assert isinstance(self.model, YOLOEModel)
        self.model.set_vocab(vocab, names=names)

    def get_vocab(self, names):
        """Get vocabulary for the given class names."""
        assert isinstance(self.model, YOLOEModel)
        return self.model.get_vocab(names)

    def set_classes(self, classes: List[str], embeddings) -> None:
        """
        Set the model's class names and embeddings for detection.

        Args:
            classes (List[str]): A list of categories i.e. ["person"].
            embeddings (torch.Tensor): Embeddings corresponding to the classes.
        """
        assert isinstance(self.model, YOLOEModel)
        self.model.set_classes(classes, embeddings)
        # Verify no background class is present
        assert " " not in classes
        self.model.names = classes

        # Reset method class names
        if self.predictor:
            self.predictor.model.names = classes

    def val(
        self,
        validator=None,
        load_vp: bool = False,
        refer_data: Optional[str] = None,
        **kwargs,
    ):
        """
        Validate the model using text or visual prompts.

        Args:
            validator (callable, optional): A callable validator function. If None, a default validator is loaded.
            load_vp (bool): Whether to load visual prompts. If False, text prompts are used.
            refer_data (str, optional): Path to the reference data for visual prompts.
            **kwargs (Any): Additional keyword arguments to override default settings.

        Returns:
            (dict): Validation statistics containing metrics computed during validation.
        """
        custom = {"rect": not load_vp}  # method defaults
        args = {**self.overrides, **custom, **kwargs, "mode": "val"}  # highest priority args on the right

        validator = (validator or self._smart_load("validator"))(args=args, _callbacks=self.callbacks)
        validator(model=self.model, load_vp=load_vp, refer_data=refer_data)
        self.metrics = validator.metrics
        return validator.metrics

    def predict(
        self,
        source=None,
        stream: bool = False,
        visual_prompts: Dict[str, List] = {},
        refer_image=None,
        predictor=None,
        **kwargs,
    ):
        """
        Run prediction on images, videos, directories, streams, etc.

        Args:
            source (str | int | PIL.Image | np.ndarray, optional): Source for prediction. Accepts image paths,
                directory paths, URL/YouTube streams, PIL images, numpy arrays, or webcam indices.
            stream (bool): Whether to stream the prediction results. If True, results are yielded as a
                generator as they are computed.
            visual_prompts (Dict[str, List]): Dictionary containing visual prompts for the model. Must include
                'bboxes' and 'cls' keys when non-empty.
            refer_image (str | PIL.Image | np.ndarray, optional): Reference image for visual prompts.
            predictor (callable, optional): Custom predictor function. If None, a predictor is automatically
                loaded based on the task.
            **kwargs (Any): Additional keyword arguments passed to the predictor.

        Returns:
            (List | generator): List of Results objects or generator of Results objects if stream=True.

        Examples:
            >>> model = YOLOE("yoloe-11s-seg.pt")
            >>> results = model.predict("path/to/image.jpg")
            >>> # With visual prompts
            >>> prompts = {"bboxes": [[10, 20, 100, 200]], "cls": ["person"]}
            >>> results = model.predict("path/to/image.jpg", visual_prompts=prompts)
        """
        if len(visual_prompts):
            assert "bboxes" in visual_prompts and "cls" in visual_prompts, (
                f"Expected 'bboxes' and 'cls' in visual prompts, but got {visual_prompts.keys()}"
            )
            assert len(visual_prompts["bboxes"]) == len(visual_prompts["cls"]), (
                f"Expected equal number of bounding boxes and classes, but got {len(visual_prompts['bboxes'])} and "
                f"{len(visual_prompts['cls'])} respectively"
            )
            if not isinstance(self.predictor, yolo.yoloe.YOLOEVPDetectPredictor):
                self.predictor = (predictor or yolo.yoloe.YOLOEVPDetectPredictor)(
                    overrides={
                        "task": self.model.task,
                        "mode": "predict",
                        "save": False,
                        "verbose": refer_image is None,
                        "batch": 1,
                    },
                    _callbacks=self.callbacks,
                )

            num_cls = (
                max(len(set(c)) for c in visual_prompts["cls"])
                if isinstance(source, list) and refer_image is None  # means multiple images
                else len(set(visual_prompts["cls"]))
            )
            self.model.model[-1].nc = num_cls
            self.model.names = [f"object{i}" for i in range(num_cls)]
            self.predictor.set_prompts(visual_prompts.copy())
            self.predictor.setup_model(model=self.model)

            if refer_image is None and source is not None:
                dataset = load_inference_source(source)
                if dataset.mode in {"video", "stream"}:
                    # NOTE: set the first frame as refer image for videos/streams inference
                    refer_image = next(iter(dataset))[1][0]
            if refer_image is not None:
                vpe = self.predictor.get_vpe(refer_image)
                self.model.set_classes(self.model.names, vpe)
                self.task = "segment" if isinstance(self.predictor, yolo.segment.SegmentationPredictor) else "detect"
                self.predictor = None  # reset predictor
        elif isinstance(self.predictor, yolo.yoloe.YOLOEVPDetectPredictor):
            self.predictor = None  # reset predictor if no visual prompts

        return super().predict(source, stream, **kwargs)
