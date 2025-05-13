"""
GRES Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import os
import sys
from transformers import pipeline
from PIL import Image
import requests
from torchvision import transforms

OCCAM_ROOT = "/weka/oh/arubinstein17/github/OCCAM"
HORNET_CONFIG = os.path.join(OCCAM_ROOT, "configs", "cropformer", "cropformer_hornet.yaml")
HORNET_WEIGHTS = os.path.join(OCCAM_ROOT, "checkpoints", "CropFormer_hornet_3x_03823a.pth")
CONF_THRESHOLD = 0.5
RANDOM_SEED = 42
from stuned.utility.utils import (
    AttrDict,
    apply_random_seed
)
sys.path.insert(
    # 0, (os.path.dirname(os.path.dirname(__file__)))
    0, (OCCAM_ROOT)
)  # to allow importing from get_segments directly
from occam.get_segments.run_cropformer import EntitySegDecoder
sys.path.pop(0)
try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import copy
import itertools
import logging
# import os

from functools import reduce
import operator

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch
import torch.utils.data as torchdata

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import DatasetEvaluators, verify_results

from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

# MaskFormer
from gres_model import (
    RefCOCOMapper,
    ReferEvaluator,
    add_maskformer2_config,
    add_refcoco_config
)
from gres_model.data.samplers import RandomSubsetTrainingSampler

import torch.nn as nn
# import sys
import numpy as np

class Trainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            os.makedirs(output_folder, exist_ok=True)
        evaluator_list = []
        evaluator_list.append(
            ReferEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        assert cfg.INPUT.DATASET_MAPPER_NAME == "refcoco"
        mapper = RefCOCOMapper(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        assert cfg.INPUT.DATASET_MAPPER_NAME == "refcoco"
        mapper = RefCOCOMapper(cfg, False)

        data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        if cfg.DATALOADER.SAMPLER_TEST == "RandomSubsetTrainingSampler":
            apply_random_seed(RANDOM_SEED)
            subset_sampler = RandomSubsetTrainingSampler(len(data_loader.dataset), cfg.DATALOADER.RANDOM_SUBSET_RATIO)
            data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper, sampler=subset_sampler)

        return data_loader

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if "text_encoder" in module_name:
                    continue
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)

                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    hyperparams["weight_decay"] = 0.0

                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm

                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        hyperparams = copy.copy(defaults)
        params.append({"params": reduce(operator.concat,
                                        [[p for p in model.text_encoder.encoder.layer[i].parameters()
                                          if p.requires_grad] for i in range(10)]),
                        **hyperparams
                     })

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_refcoco_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="referring")
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        if "CLIP" in cfg.REFMODEL:
            assert "@" in cfg.REFMODEL
            mask_generator = cfg.REFMODEL.split("@")[1]
            if "-" in cfg.REFMODEL:
                prediction_method = cfg.REFMODEL.split("-")[1]
            else:
                prediction_method = "best_clip_score"
            model = build_clip_ref(mask_generator, prediction_method)
        else:
            assert args.model is None
            model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


def build_clip_ref(mask_generator, prediction_method="best_clip_score"):
    return ClipRefModel(mask_generator, prediction_method)


class ClipRefModel(nn.Module):
    def __init__(self, mask_generator, prediction_method="best_clip_score"):
        super().__init__()
        if mask_generator == "HQES":
            self.mask_generator = build_hqes()
        else:
            raise ValueError(f"Invalid mask generator: {mask_generator}")
        self.clip_model = build_clip_model()
        self.prediction_method = prediction_method

    def forward(self, x):
        assert len(x) == 1, "batch size must be 1"
        image = x[0]['image'].permute(1, 2, 0).unsqueeze(0)
        masks = self.mask_generator(image)
        num_masks = int(masks.max()) + 1
        masks_as_image = torch.zeros(
            (num_masks, masks.shape[0], masks.shape[1])
        ).to(torch.bool)
        for mask_value in range(num_masks):
            masks_as_image[mask_value] = torch.Tensor(masks == mask_value)

        image_ready_for_masking = image[0].permute(2, 0, 1)

        # applied_masks = []
        best_mask, best_clip_score = get_best_mask(
            image_ready_for_masking,
            masks_as_image,
            self.clip_model,
            x[0]['sentence'],
            prediction_method=self.prediction_method
        )

        assert best_mask is not None, "no mask found"

        # 0 - inverted best mask
        # 1 - best mask

        res = {}
        res['nt_label'] = torch.tensor([(1 - best_clip_score), (best_clip_score)])
        best_mask = best_mask.to(torch.int).unsqueeze(0)
        res['ref_seg'] = torch.cat([1 - best_mask, best_mask], dim=0)

        return [res]


def get_best_mask(
    image_ready_for_masking,
    masks_as_image,
    clip_model,
    sentence,
    prediction_method="best_clip_score"
):
    best_clip_score = 0
    best_mask = None
    num_masks = masks_as_image.shape[0]
    if prediction_method == "best_clip_score":

        for i in range(num_masks):
            cur_mask = masks_as_image[i]
            if filter_out(cur_mask):
                continue
            applied_mask = image_ready_for_masking * cur_mask + 0.5 * (~cur_mask)
            applied_mask = transforms.ToPILImage()(applied_mask)
            # applied_mask = applied_mask.permute(2, 0, 1).unsqueeze(0)
            # applied_masks.append(applied_mask)
            outputs = clip_model(
                # image=applied_masks,
                image=applied_mask,
                # candidate_labels=[x[0]['sentence']]
                candidate_labels=[sentence]
            )
            if outputs[0]['score'] > best_clip_score:
                best_clip_score = outputs[0]['score']
                best_mask = masks_as_image[i]
    else:
        raise ValueError(f"Invalid prediction method: {prediction_method}")

    # outputs_list = self.clip_model(
    #     image=applied_masks,
    #     candidate_labels=[x[0]['sentence']]
    # )

    # best_clip_score = 0
    # best_mask = None
    # for i, outputs in enumerate(outputs_list):
    #     # print(f"DEBUG: {i}: {outputs}")
    #     if outputs[0]['score'] > best_clip_score:
    #         best_clip_score = outputs[0]['score']
    #         best_mask = masks_as_image[i]
    return best_mask, best_clip_score


def filter_out(mask):
    # filter out masks that are too small
    if mask.sum() < 200:
        return True
    return False


def build_clip_model():

    # load pipe with proper text handling
    # <LIBS>/transformers/pipelines/zero_shot_image_classification.py
    # max_position_embeddings = 64 here: <LIBS>/transformers/models/siglip/modeling_siglip.py#329
    # inside clip.text_model.embeddings
    image_classifier = pipeline(
        task="zero-shot-image-classification",
        # model="google/siglip2-base-patch16-224",
        model="google/siglip2-so400m-patch14-384",
        hypothesis_template="{}", # default: "This is a photo of {}."
        tokenizer_kwargs={
            "truncation": True,
            "max_length": 64
        }
    )

    # inference demo
    # load image
    # url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    # image = Image.open(requests.get(url, stream=True).raw)
    # candidate_labels = ["2 cats", "a plane", "a remote"]
    # outputs = image_classifier(
    #     image,
    #     candidate_labels=candidate_labels,
    # )
    # outputs = [{"score": round(output["score"], 4), "label": output["label"] } for output in outputs]
    # print(outputs)
    # empty template b16: [{'score': 0.3104, 'label': '2 cats'}, {'score': 0.002, 'label': 'a remote'}, {'score': 0.0, 'label': 'a plane'}]
    # empty template s400: [{'score': 0.6312, 'label': '2 cats'}, {'score': 0.0002, 'label': 'a remote'}, {'score': 0.0, 'label': 'a plane'}]
    # default template b16: [{'score': 0.1719, 'label': '2 cats'}, {'score': 0.0241, 'label': 'a remote'}, {'score': 0.0, 'label': 'a plane'}]
    return image_classifier


def build_hqes():
    return EntitySegDecoder(
        config_path=HORNET_CONFIG,
        opts=["MODEL.WEIGHTS", HORNET_WEIGHTS],
        confidence_threshold=CONF_THRESHOLD
    )


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
