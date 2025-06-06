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
import torch
import subprocess
import re

OCCAM_ROOT = "/weka/oh/arubinstein17/github/OCCAM"
CUVLER_ROOT = "/home/oh/arubinstein17/github/CuVLER"
HORNET_CONFIG = os.path.join(OCCAM_ROOT, "configs", "cropformer", "cropformer_hornet.yaml")
HORNET_WEIGHTS = os.path.join(OCCAM_ROOT, "checkpoints", "CropFormer_hornet_3x_03823a.pth")
CONF_THRESHOLD = 0.5
NO_TARGET_THRESHOLD = 0.02
# NO_TARGET_THRESHOLD = 0.00
NO_TARGET_THRESHOLD_ALPHA_CLIP = 20.0 # 20.0
ALPHA_CLIP_MODEL_ID = "ViT-L/14"
RANDOM_SEED = 42
from stuned.utility.utils import (
    AttrDict,
    apply_random_seed
)
from ftdinosaur_inference import build_dinosaur, utils
# from stuned.utility.transforms import (
#     make_transforms
# )
sys.path.insert(
    # 0, (os.path.dirname(os.path.dirname(__file__)))
    0, (OCCAM_ROOT)
)  # to allow importing from get_segments directly
from occam.get_segments.run_cropformer import EntitySegDecoder
from occam.robust_classification.masking import is_background
from occam.robust_classification.eval import (
    BORDER_VS_AREA_RATIO,
    compute_border_touch
)
from occam.submodules.dino_ft_wrapper import (
    load_model,
    # get_masks_as_image,
    # overlay_masks_on_image,
    # prep_for_display,
    # get_cmap
)
from occam.submodules.alpha_clip_wrapper import (
    ALPHA_CLIP_IMAGE_PREPROCESS_CONFIG,
    make_alpha_clip_model
)
from occam.datasets.bboxed_dataset import (
    make_common_and_specific_transforms
)
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


# CUVLER BEGIN
sys.path.insert(0, CUVLER_ROOT)
sys.path.insert(0, os.path.join(CUVLER_ROOT, "cad"))
# /home/oh/arubinstein17/github/CuVLER/cad
import detectron2
from detectron2.modeling.roi_heads import (
    ROI_HEADS_REGISTRY
)
import cad
# /weka/oh/arubinstein17/github/OCCAM/envs/occam/lib/python3.10/site-packages/detectron2/modeling/roi_heads/roi_heads.py
from cad.modeling.roi_heads import (
    ROI_HEADS_REGISTRY as CUVLER_ROI_HEADS_REGISTRY,
    CustomCascadeROIHeads
)
sys.path.pop(0)
sys.path.pop(0)
# CUVLER END


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
        if cfg.REFMODEL is not None and "CLIP" in cfg.REFMODEL:
            assert "@" in cfg.REFMODEL
            clip_type = cfg.REFMODEL.split("@")[0]
            mask_generator = cfg.REFMODEL.split("@")[1]
            if "-" in mask_generator:
                mask_generator, prediction_method = mask_generator.split("-")
                # prediction_method = cfg.REFMODEL.split("-")[1]
            else:
                prediction_method = "best_clip_score"
            model = build_clip_ref(mask_generator, clip_type, prediction_method)
        else:
            # assert args.model is None
            set_device_with_most_free_memory()
            print("torch.cuda.current_device() ", torch.cuda.current_device())
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


def build_clip_ref(mask_generator, clip_type, prediction_method="best_clip_score"):
    return ClipRefModel(mask_generator, clip_type, prediction_method)


class ClipRefModel(nn.Module):
    def __init__(self, mask_generator, clip_type, prediction_method="best_clip_score"):
        super().__init__()
        set_device_with_most_free_memory()
        print("torch.cuda.current_device() ", torch.cuda.current_device())
        if mask_generator == "HQES":
            self.mask_generator = build_hqes()
        elif mask_generator == "Cuvler":
            self.mask_generator = build_cuvler()
        elif mask_generator == "FTdino":
            self.mask_generator = build_ftdino()
        else:
            raise ValueError(f"Invalid mask generator: {mask_generator}")
        # self.mask_generator.to(torch.device(torch.cuda.current_device()))
        self.mask_generator.cuda()
        self.mask_generator.eval()
        self.clip_model = build_clip_model(clip_type)
        # self.clip_model.cuda()
        # self.clip_model.eval()
        self.prediction_method = prediction_method

    def forward(self, x):
        with torch.no_grad():
            assert len(x) == 1, "batch size must be 1"
            image = x[0]['image'].permute(1, 2, 0).unsqueeze(0)
            masks = self.mask_generator(image.cuda())
            if isinstance(self.mask_generator, EntitySegDecoder):
                num_masks = int(masks.max()) + 1
                masks_as_image = torch.zeros(
                    (num_masks, masks.shape[0], masks.shape[1])
                ).to(torch.bool)
                for mask_value in range(num_masks):
                    masks_as_image[mask_value] = torch.Tensor(masks == mask_value)
                masks_as_image = masks_as_image.cuda()
            else:
                assert isinstance(self.mask_generator, (FTDino, Cuvler))
                masks_as_image = masks

            image_ready_for_masking = image[0].permute(2, 0, 1).cuda()

            # applied_masks = []
            sentence = x[0]['sentence']
            best_mask, best_clip_score = get_best_mask(
                image_ready_for_masking,
                masks_as_image,
                self.clip_model,
                sentence,
                prediction_method=self.prediction_method
            )
            # best_mask, best_clip_score = get_best_mask(
            #     image_ready_for_masking * best_mask + 1.0 * (~best_mask),
            #     masks_as_image,
            #     self.clip_model,
            #     x[0]['sentence'],
            #     prediction_method="iterative_removal"
            # )
            # print("DEBUG: remove removal")

            if best_mask is None:
                assert best_clip_score == 0.0, "no mask found"

            # 0 - inverted best mask
            # 1 - best mask

            res = {}
            if isinstance(self.clip_model, AlphaClipClassifier):
                no_target_threshold = NO_TARGET_THRESHOLD_ALPHA_CLIP
            else:
                no_target_threshold = NO_TARGET_THRESHOLD

            if best_clip_score > no_target_threshold:
                # res['nt_label'] = torch.tensor([(1 - best_clip_score), (best_clip_score)]).softmax(dim=0) # need softmax in case of clip scores outside of [0, 1]
                res['nt_label'] = torch.tensor([1.0, 0.0]) # need softmax in case of clip scores outside of [0, 1]
                best_mask = best_mask.to(torch.int).unsqueeze(0)
                res['ref_seg'] = torch.cat([1 - best_mask, best_mask], dim=0)
            else:
                # no target case
                res['nt_label'] = torch.tensor([0.0, 1.0])
                res['ref_seg'] = torch.cat([torch.zeros_like(best_mask).to(torch.int), torch.zeros_like(best_mask).to(torch.int)], dim=0)

            return [res]


def get_best_mask(
    image_ready_for_masking,
    masks_as_image,
    clip_model,
    sentence,
    prediction_method="best_clip_score"
):

    def make_applied_mask(image_ready_for_masking, mask, clip_model):
        if isinstance(clip_model, AlphaClipClassifier):
            image_for_alpha_clip = clip_model.post_normalize_transform(
                clip_model.before_normalize_transform(transforms.ToPILImage()(image_ready_for_masking))
            ).unsqueeze(0).cuda()
            mask_for_alpha_clip = clip_model.before_normalize_transform(mask.cpu().numpy()).unsqueeze(0).cuda()
            applied_mask = (image_for_alpha_clip, mask_for_alpha_clip)
        else:
            applied_mask = image_ready_for_masking * mask + 0.5 * (~mask)
            applied_mask = transforms.ToPILImage()(applied_mask)
        return applied_mask

    def predict_for_mask(image_ready_for_masking, mask, clip_model, sentence):
        # if isinstance(clip_model, AlphaClipClassifier):
        #     image_for_alpha_clip = clip_model.post_normalize_transform(
        #         clip_model.before_normalize_transform(transforms.ToPILImage()(image_ready_for_masking))
        #     ).unsqueeze(0).cuda()
        #     mask_for_alpha_clip = clip_model.before_normalize_transform(mask.numpy()).unsqueeze(0).cuda()
        #     applied_mask = (image_for_alpha_clip, mask_for_alpha_clip)
        # else:
        #     applied_mask = image_ready_for_masking * mask + 0.5 * (~mask)
        #     applied_mask = transforms.ToPILImage()(applied_mask)
        applied_mask = make_applied_mask(image_ready_for_masking, mask, clip_model)

        outputs = clip_model(
            image=applied_mask,
            candidate_labels=[sentence]
        )
        return outputs

    num_masks = masks_as_image.shape[0]
    if prediction_method == "iterative_removal":

        applied_mask = make_applied_mask(image_ready_for_masking, torch.ones_like(masks_as_image[0]).to(torch.bool), clip_model)
        outputs = clip_model(
            image=applied_mask,
            candidate_labels=[sentence]
        )
        best_clip_score = outputs[0]['score']
        cur_image = image_ready_for_masking
        best_mask = torch.ones_like(masks_as_image[0]).to(torch.bool)
    elif prediction_method == "iterative_addition":
        best_mask = torch.zeros_like(masks_as_image[0]).to(torch.bool)
        best_clip_score = 0
    else:
        assert prediction_method == "best_clip_score"
        best_mask = None
        best_clip_score = 0
    if prediction_method in ["best_clip_score", "iterative_removal", "iterative_addition"]:

        for i in range(num_masks):
            cur_mask = masks_as_image[i]
            if filter_out(cur_mask):
                continue
            if prediction_method == "best_clip_score":
                outputs = predict_for_mask(image_ready_for_masking, cur_mask, clip_model, sentence)

                if outputs[0]['score'] > best_clip_score:
                    best_clip_score = outputs[0]['score']
                    best_mask = masks_as_image[i]
            elif prediction_method == "iterative_removal":
                # iteratively remove masks if this removal increases score
                outputs = predict_for_mask(cur_image, ~cur_mask, clip_model, sentence)
                if outputs[0]['score'] >= best_clip_score:
                    best_clip_score = outputs[0]['score']
                    cur_image = cur_image * ~cur_mask + 0.5 * cur_mask

                    best_mask = best_mask & ~cur_mask
            else:
                assert prediction_method == "iterative_addition"
                cumulative_mask = best_mask | cur_mask
                outputs = predict_for_mask(image_ready_for_masking, cumulative_mask, clip_model, sentence)
                if outputs[0]['score'] > best_clip_score:
                    best_clip_score = outputs[0]['score']
                    best_mask = cumulative_mask
    else:
        raise ValueError(f"Invalid prediction method: {prediction_method}")

    return best_mask, best_clip_score


def filter_out(mask):
    to_filter = False
    # filter out masks that are too small
    if mask.sum() < 200:
        to_filter = True
    # if is_background(mask, to_extract=False, threshold=3):
    #     to_filter = True
    if is_background_v2(mask, threshold=1):
        to_filter = True
    # border_touch = compute_border_touch(mask)
    # mask_area = mask.sum() / (mask.shape[0] * mask.shape[1])
    # if (border_touch / mask_area) > BORDER_VS_AREA_RATIO:
    #     to_filter = True
    return to_filter


def is_background_v2(mask, threshold=3):
    sum = (
        mask[0, 0].item()
        + mask[0, -1].item()
        + mask[-1, 0].item()
        + mask[-1, -1].item()
    )
    if sum > threshold:
        return True
    mid_0 = mask.shape[0] // 2
    mid_1 = mask.shape[1] // 2
    sum = (
        mask[0, mid_1].item()
        + mask[mid_0, -1].item()
        + mask[-1, mid_1].item()
        + mask[mid_0, 0].item()
    )
    if sum > threshold:
        return True
    q_00 = mid_0 // 2
    q_10 = mid_1 // 2
    q_01 = mid_0 + q_00
    q_11 = mid_1 + q_10
    sum = (
        mask[0, q_10].item()
        + mask[0, q_11].item()
        + mask[q_00, 0].item()
        + mask[q_01, 0].item()
        + mask[q_00, -1].item()
        + mask[q_01, -1].item()
        + mask[-1, q_10].item()
        + mask[-1, q_11].item()

    )
    if sum > threshold * 2:
        return True
    return False


def build_clip_model(clip_type):

    # load pipe with proper text handling
    # <LIBS>/transformers/pipelines/zero_shot_image_classification.py
    # max_position_embeddings = 64 here: <LIBS>/transformers/models/siglip/modeling_siglip.py#329
    # inside clip.text_model.embeddings
    if clip_type == "CLIP":
        image_classifier = pipeline(
            task="zero-shot-image-classification",
            # model="google/siglip2-base-patch16-224",
            model="google/siglip2-so400m-patch14-384",
            hypothesis_template="{}", # default: "This is a photo of {}."
            tokenizer_kwargs={
                "truncation": True,
                "max_length": 64
            },
            device_map='cuda'
        )
    else:
        assert clip_type == "AlphaCLIP"
        image_classifier = AlphaClipClassifier()
        # image_classifier = make_alpha_clip_model(model_id="ViT-L/14")

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


class AlphaClipClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha_clip_model = None
        self.recent_text_features = None
        # self.transforms = make_transforms(
        #     ALPHA_CLIP_IMAGE_PREPROCESS_CONFIG
        # )
        self.before_normalize_transform, self.post_normalize_transform = make_common_and_specific_transforms(
            ALPHA_CLIP_IMAGE_PREPROCESS_CONFIG
        )

    def forward(self, image, candidate_labels):

        # outputs = clip_model(
        #     # image=applied_masks,
        #     image=applied_mask,
        #     # candidate_labels=[x[0]['sentence']]
        #     candidate_labels=[sentence]
        # )
        assert len(candidate_labels) > 0, "Empty candidate labels"
        if isinstance(candidate_labels[0], dict):
            # {'raw': 'The elephant head on the right.', 'sent_id': 245062, 'ref_id': 92459}
            candidate_labels = [x['raw'] for x in candidate_labels]
        # add text for background to have at least two texts
        assert len(candidate_labels) == 1, "Only one text is supported for AlphaCLIP"
        # candidate_labels += ["background"]
        make_model = False
        if self.alpha_clip_model is None:
            make_model = True
        if self.recent_text_features != candidate_labels:
            make_model = True
        if make_model:
            self.alpha_clip_model = make_alpha_clip_model(
                model_id=ALPHA_CLIP_MODEL_ID,
                category_list=candidate_labels
            )
            self.recent_text_features = candidate_labels

        outputs = self.alpha_clip_model(image, softmax=False)
        res = []
        # for output in outputs:
        #     res.append({
        #         'score': output,
        #     })
        assert len(outputs.shape) == 2, "AlphaCLIP outputs must be a 2D tensor"
        # first axis for inputs, second axis for texts
        for input_i in range(outputs.shape[0]):
            res_per_input = []
            for text_i in range(outputs.shape[1]):
                res_per_input.append({
                    'score': outputs[input_i, text_i],
                    'label': candidate_labels[text_i]
                })
            res.append(res_per_input)
        if len(res) == 1:
            res = res[0]
        # return res[0] # take prob of the first text
        return res


def build_hqes():
    return EntitySegDecoder(
        config_path=HORNET_CONFIG,
        opts=["MODEL.WEIGHTS", HORNET_WEIGHTS],
        confidence_threshold=CONF_THRESHOLD
    )


def build_cuvler():
    return Cuvler()


class Cuvler(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = torch.load(os.path.join(CUVLER_ROOT, "cuvler_cfg.pt"), weights_only=False)
        # Merge the registries by copying all entries from CUVLER_ROI_HEADS_REGISTRY to ROI_HEADS_REGISTRY
        for name, obj in CUVLER_ROI_HEADS_REGISTRY._obj_map.items():
            if name not in ROI_HEADS_REGISTRY:
                ROI_HEADS_REGISTRY._do_register(name, obj)
        model = Trainer.build_model(cfg)
        # cfg.MODEL.WEIGHTS = os.path.join(CUVLER_ROOT, "checkpoints", "zero_shot.pth")
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            # cfg.MODEL.WEIGHTS,
            os.path.join(CUVLER_ROOT, "checkpoints", "zero_shot.pth"),
            # resume=args.resume
            resume=False
        )
        self.model = model
        # self.preproc = None

    def __call__(self, image):
        image = image.squeeze(0)
        cuvler_input = [
            {
                'image': image.permute(2, 0, 1),
                # 'image': image.permute(2, 0, 1).unsqueeze(0),
                'height': image.shape[0],
                'width': image.shape[1]
            }
        ]
        output = self.model(cuvler_input)
        masks_as_image = output[0]['instances'].pred_masks
        return masks_as_image

class FTDino(nn.Module):
    def __init__(self):
        super().__init__()
        model_name = "dinosaur_base_patch14_518_topk3.coco_dv2_ft_s7_300k+10k"
        model = load_model(model_name)
        preproc = build_dinosaur.build_preprocessing(model_name)
        self.model = model
        self.preproc = preproc

    def __call__(self, image):
        image = image[0].permute(2, 0, 1)
        with torch.no_grad():
            inp = self.preproc(image).unsqueeze(0)
            outp = self.model(inp, num_slots=5)
        # mg_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1)  # C x H x W
        height, width = image.shape[1:]
        masks = outp['masks']

        # Need to resize masks to image (1 x K x P -> 1 x K x H x W)
        masks_as_image = utils.resize_patches_to_image(masks, size=(height, width))
        masks_as_image = utils.soft_masks_to_one_hot(masks_as_image).squeeze(0)
        return masks_as_image


def build_ftdino():
    return FTDino()


def get_gpu_memory_map():
    """Get the current gpu usage.
    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.free,memory.total',
         '--format=csv,nounits,noheader'
    ], encoding='utf-8')

    # Parse the output
    gpu_memory = {}
    for i, line in enumerate(result.strip().split('\n')):
        free_mem, total_mem = map(int, line.split(','))
        gpu_memory[i] = free_mem

    return gpu_memory

def get_device_with_most_free_memory():
    """Get the CUDA device with the most free memory.
    Returns
    -------
    device: torch.device
        The CUDA device with the most free memory
    """
    if not torch.cuda.is_available():
        return torch.device('cpu')

    gpu_memory = get_gpu_memory_map()
    if not gpu_memory:
        return torch.device('cpu')

    # Get the device with most free memory
    device_id = max(gpu_memory.items(), key=lambda x: x[1])[0]
    return torch.device(f'cuda:{device_id}')

def set_device_with_most_free_memory():
    """Set the CUDA device with the most free memory as the current device.
    Returns
    -------
    device: torch.device
        The CUDA device that was set
    """
    device = get_device_with_most_free_memory()
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    return device


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
