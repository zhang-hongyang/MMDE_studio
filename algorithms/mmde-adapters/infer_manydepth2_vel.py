#!/usr/bin/env python3
"""ManyDepth2-Vel multi-frame metric depth adapter for MMDE sequences.

The first frame of each sequence uses the checkpoint's monocular branch. All
later frames use the immediately preceding frame, learned pose, GMFlow, and
the multi-frame cost-volume branch. Predictions are therefore complete while
the temporal path remains identical to the training/evaluation implementation.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from common import (base_parser, finite_depth, grouped_frames, load_context,
                    prediction_path, resize_native, save_prediction,
                    valid_prediction)

DEFAULT_CODE_ROOT = Path(
    "/home/ZhangHongyang/ResearchHub-workspaces/Speed2MetricDepth/code/current")
DEFAULT_WEIGHTS = Path(
    "/home/ZhangHongyang/ResearchHub-scratch/runs/Speed2MetricDepth/"
    "velocity-main-seed1-20261002-v3/velocity_main_seed1/models/weights_19")
DEFAULT_GMFLOW = Path(
    "/home/ZhangHongyang/ResearchHub-scratch/models/Speed2MetricDepth/"
    "pretrained-v1/gmflow_sintel-0c07dcb3.pth")


def _tensor(image: np.ndarray, height: int, width: int,
            device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    value = torch.from_numpy(rgb).to(device=device, dtype=torch.float32)
    value = value.permute(2, 0, 1)[None] / 255.0
    return F.interpolate(value, (height, width), mode="bilinear",
                         align_corners=False)


def _load(module: torch.nn.Module, path: Path, *, filter_keys: bool = False):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if filter_keys:
        wanted = module.state_dict()
        state = {key: value for key, value in state.items() if key in wanted}
    module.load_state_dict(state, strict=True)
    return module


class Predictor:
    def __init__(self, code_root: Path, weights: Path, gmflow_path: Path,
                 device: torch.device):
        sys.path.insert(0, str(code_root))
        from manydepth2 import networks
        from manydepth2.layers import disp_to_depth, transformation_from_parameters
        from core_gm.gmflow.gmflow.gmflow import GMFlow

        self.disp_to_depth = disp_to_depth
        self.transform = transformation_from_parameters
        self.device = device
        encoder_state = torch.load(weights / "encoder.pth", map_location="cpu",
                                   weights_only=True)
        self.height = int(encoder_state.get("height", 192))
        self.width = int(encoder_state.get("width", 640))
        self.min_depth = 0.1
        self.max_depth = 100.0

        self.pose_encoder = networks.ResnetEncoder(18, False, num_input_images=2)
        self.pose_decoder = networks.PoseDecoder(
            self.pose_encoder.num_ch_enc, num_input_features=1,
            num_frames_to_predict_for=2)
        _load(self.pose_encoder, weights / "pose_encoder.pth")
        _load(self.pose_decoder, weights / "pose.pth")

        self.mono_encoder = networks.hrnet18(pretrained=False)
        _load(self.mono_encoder, weights / "mono_encoder.pth", filter_keys=True)
        self.mono_encoder.num_ch_enc = [64, 18, 36, 72, 144]
        self.mono_decoder = networks.HRDepthDecoder(self.mono_encoder.num_ch_enc)
        _load(self.mono_decoder, weights / "mono_depth.pth")

        encoder_options = dict(
            num_layers=18, pretrained=False, input_width=self.width,
            input_height=self.height, adaptive_bins=True,
            min_depth_bin=float(encoder_state.get("min_depth_bin", 0.1)),
            max_depth_bin=float(encoder_state.get("max_depth_bin", 20.0)),
            depth_binning="linear", num_depth_bins=96)
        self.encoder = networks.multihrnet18_flow00(**encoder_options)
        self.encoder.num_ch_enc = [64, 18, 36, 72, 144]
        _load(self.encoder, weights / "encoder.pth", filter_keys=True)
        self.decoder = networks.HRDepthDecoder(self.encoder.num_ch_enc)
        _load(self.decoder, weights / "depth.pth")
        self.min_depth_bin = encoder_options["min_depth_bin"]
        self.max_depth_bin = encoder_options["max_depth_bin"]

        self.flow = GMFlow(
            feature_channels=128, num_scales=1, upsample_factor=8,
            num_head=1, attention_type="swin", ffn_dim_expansion=4,
            num_transformer_layers=6)
        flow_state = torch.load(gmflow_path, map_location="cpu", weights_only=True)
        self.flow.load_state_dict(flow_state.get("model", flow_state), strict=True)

        for module in (self.pose_encoder, self.pose_decoder, self.mono_encoder,
                       self.mono_decoder, self.encoder, self.decoder, self.flow):
            # Some upstream HRNet modules override ``to`` without returning
            # self, so chaining ``.to(...).eval()`` is not portable here.
            module.to(device)
            module.eval()

    def mono(self, current: torch.Tensor) -> np.ndarray:
        output = self.mono_decoder(self.mono_encoder(current))
        _, depth = self.disp_to_depth(
            output[("disp", 0)], self.min_depth, self.max_depth)
        return depth[0, 0].detach().float().cpu().numpy()

    def multi(self, current: torch.Tensor, previous: torch.Tensor,
              intrinsic: np.ndarray, native_shape: tuple[int, int]) -> np.ndarray:
        pose_features = self.pose_encoder(torch.cat([previous, current], 1))
        axisangle, translation = self.pose_decoder([pose_features])
        relative_pose = self.transform(
            axisangle[:, 0], translation[:, 0], invert=True)[:, None]

        result = self.flow(
            current * 255.0, previous * 255.0,
            attn_splits_list=[2], corr_radius_list=[-1],
            prop_radius_list=[-1])
        flow = F.interpolate(result["flow_preds"][-1], scale_factor=0.25,
                             mode="bilinear", align_corners=False) / 4.0

        mono_output = self.mono_decoder(self.mono_encoder(current))
        _, coarse = self.disp_to_depth(
            mono_output[("disp", 0)], self.min_depth, self.max_depth)
        coarse = F.interpolate(coarse, (self.height // 4, self.width // 4),
                               mode="bilinear", align_corners=False)

        native_h, native_w = native_shape
        K = np.eye(4, dtype=np.float32)
        K[:3, :3] = np.asarray(intrinsic, dtype=np.float32)
        K[0, :] *= (self.width / native_w) / 4.0
        K[1, :] *= (self.height / native_h) / 4.0
        K = torch.from_numpy(K)[None].to(self.device)
        inv_K = torch.linalg.pinv(K)
        features, _, _ = self.encoder(
            current, previous[:, None], relative_pose, flow, coarse,
            K, inv_K, self.min_depth_bin, self.max_depth_bin, True)
        output = self.decoder(features)
        _, depth = self.disp_to_depth(
            output[("disp", 0)], self.min_depth, self.max_depth)
        return depth[0, 0].detach().float().cpu().numpy()


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--code-root", default=os.environ.get(
        "MMDE_SPEED2METRIC_ROOT", str(DEFAULT_CODE_ROOT)))
    ap.add_argument("--weights", default=os.environ.get(
        "MANYDEPTH2_VEL_WEIGHTS", str(DEFAULT_WEIGHTS)))
    ap.add_argument("--gmflow", default=os.environ.get(
        "MANYDEPTH2_GMFLOW_WEIGHTS", str(DEFAULT_GMFLOW)))
    args = ap.parse_args()
    frames, out = load_context(args)
    device = torch.device(args.device)
    predictor = Predictor(Path(args.code_root), Path(args.weights),
                          Path(args.gmflow), device)

    with torch.inference_mode():
        groups = grouped_frames(frames)
        for group in groups[args.group_offset::args.group_stride]:
            previous_tensor = None
            for frame in group:
                image = cv2.imread(str(frame["image_path"]), cv2.IMREAD_COLOR)
                if image is None:
                    raise FileNotFoundError(frame["image_path"])
                current = _tensor(image, predictor.height, predictor.width, device)
                path = prediction_path(out, frame)
                context = previous_tensor
                context_path = frame.get("prev_image_path")
                if context_path:
                    context_image = cv2.imread(str(context_path), cv2.IMREAD_COLOR)
                    if context_image is None:
                        raise FileNotFoundError(context_path)
                    context = _tensor(context_image, predictor.height,
                                      predictor.width, device)
                if context is None:
                    depth = predictor.mono(current)
                else:
                    K = np.asarray(frame.get("K"), dtype=np.float32)
                    if K.shape != (3, 3):
                        raise ValueError("ManyDepth2-Vel requires a 3x3 K")
                    depth = predictor.multi(current, context, K,
                                            image.shape[:2])
                if not valid_prediction(path):
                    save_prediction(out, frame, resize_native(
                        finite_depth(depth), frame))
                previous_tensor = current
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
