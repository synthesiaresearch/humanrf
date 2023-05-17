import csv
import os
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import List

import cv2
import lpips
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity
from tqdm import tqdm

import actorshq.evaluation.presets as presets
from actorshq.dataset.volumetric_dataset import VolumetricDataset


def load_image(image_path):
    image_np = np.array(Image.open(image_path))[..., :3]
    image_pt = torch.from_numpy(image_np).permute([2, 0, 1])[:3] / 255.0
    return image_np, image_pt.cuda()


def crop_images_with_roi(images, roi):
    x, y, w, h = roi

    cropped_images = []
    for image in images:
        if isinstance(image, np.ndarray):
            if image.shape[0:2] == (h, w):
                cropped_images.append(image)
                continue
            crop_slice = slice(y, y + h), slice(x, x + w), slice(None)
        elif isinstance(image, torch.Tensor):
            if image.shape[1:3] == (h, w):
                cropped_images.append(image)
                continue
            crop_slice = slice(None), slice(y, y + h), slice(x, x + w)
        cropped_images.append(image[crop_slice])

    return cropped_images


def render_y4m(input_pattern, output):
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_pattern, "-pix_fmt", "yuv444p", "-loglevel", "error", output],
        check=True,
        stdout=subprocess.DEVNULL,
    )


__LPIPS__ = {}


def init_lpips(net_name: str, device):
    assert net_name in ['alex', 'vgg']
    if net_name not in __LPIPS__:
        print(f'init_lpips: lpips_{net_name}')
        __LPIPS__[net_name] = lpips.LPIPS(pretrained=True, net=net_name, version='0.1').eval().to(device)
    return __LPIPS__[net_name]


def compute_lpips(im1: torch.Tensor, im2: torch.Tensor, model: str = 'alex'):
    if isinstance(im1, np.ndarray):
        im1 = torch.from_numpy(im1).permute([2, 0, 1]).contiguous().cuda()
    if isinstance(im2, np.ndarray):
        im2 = torch.from_numpy(im2).permute([2, 0, 1]).contiguous().cuda()
    device = im1.device
    lpips_model = init_lpips(model, device)
    return lpips_model(im1.unsqueeze(0), im2.unsqueeze(0), normalize=True).item()


def compute_ssim(im1: np.ndarray, im2: np.ndarray):
    return structural_similarity(im1, im2, channel_axis=2)


def compute_psnr(im1: torch.Tensor, im2: torch.Tensor, mask: torch.Tensor = None):
    mse = torch.square(im1 - im2).mean(0).view(-1)
    if mask is not None:
        mse = mse[mask.reshape(-1) > 0]
    mse = mse.mean()
    return -10 * np.log10(mse.item())


def evaluate(
    results_directory: Path,
    output_directory: Path,
    coverage: str,
    camera_preset: str,
    frame_numbers: List[int],
    data_folder: Path,
    result_suffix: str,
):
    cameras_frames = presets.get_render_sequence(
        coverage,
        camera_preset,
        frame_numbers,
    )

    dataset = VolumetricDataset(data_folder)
    results = defaultdict(list)
    for camera_idx, frame_idx in tqdm(cameras_frames):
        camera = dataset.cameras[camera_idx]
        filename = dataset.filepaths.get_rgb_path(camera.name, frame_idx)

        # Get the mask and its bounding rect
        mask_path = dataset.filepaths.get_mask_path(camera.name, frame_idx)
        mask_np = cv2.imread(str(mask_path))[..., 0:1]

        groundtruth_np, groundtruth_pt = load_image(filename)
        prediction_np, prediction_pt = load_image(results_directory / "test_frames" / (filename.stem + result_suffix))

        # Get the region of interest
        mask_roi = cv2.boundingRect(mask_np)
        groundtruth_np, groundtruth_pt, prediction_np, prediction_pt, mask_np = crop_images_with_roi(
            [groundtruth_np, groundtruth_pt, prediction_np, prediction_pt, mask_np],
            mask_roi,
        )

        results["PSNR"].append(compute_psnr(groundtruth_pt, prediction_pt, torch.from_numpy(mask_np)))
        results["LPIPS"].append(compute_lpips(groundtruth_pt, prediction_pt))
        results["SSIM"].append(compute_ssim(groundtruth_np, prediction_np))

    averages = {metric: np.mean(values) for metric, values in results.items()}
    print(f"== Evaluating with {len(results['PSNR'])} frames ==")
    for metric, average in averages.items():
        print(f"{metric}: {average}")

    compute_vmaf = coverage == "siggraph_test"
    if compute_vmaf:
        assert len(presets.camera_configs["siggraph_vmaf"]) == 1
        cameras_frames_vmaf = presets.get_vmaf_test_sequence(frame_numbers)
        perform_vmaf = all(
            [
                (results_directory / "test_frames" / f"Cam{c+1:03d}_rgb{f:06d}{result_suffix}").exists()
                for c, f in cameras_frames_vmaf
            ]
        )
        if not perform_vmaf:
            print(
                f"No frames for VMAF computation available, skipping eval for camera {presets.camera_configs['siggraph_vmaf']}."
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            path_tmp = Path(tmpdir)
            path_frames_gt = path_tmp / "gt"
            path_frames_pred = path_tmp / "pred"
            path_frames_gt.mkdir()
            path_frames_pred.mkdir()
            for i, (c, f) in enumerate(cameras_frames_vmaf):
                cam_name = f"Cam{c+1:03d}"
                df = data_folder / "rgbs" / cam_name
                filename_gt_src = f"{cam_name}_rgb{f:06d}.jpg"
                filename_gt_tgt = f"{i:06d}.jpg"
                # * ffmpeg can't handle the original path that contains non-wildcard `%`
                # * depending on the settings we sub-sample the frames used for vmaf computation
                os.symlink(df.resolve() / filename_gt_src, path_frames_gt / filename_gt_tgt)
                os.symlink(
                    results_directory.resolve() / "test_frames" / f"Cam{c+1:03d}_rgb{f:06d}{result_suffix}",
                    path_frames_pred / f"{i:06d}{result_suffix}",
                )

            path_video_pred = results_directory / f"{cam_name}.y4m"
            path_video_gt = path_tmp / f"{cam_name}.y4m"
            render_y4m(path_frames_pred / f"%06d{result_suffix}", path_video_pred)
            render_y4m(path_frames_gt / f"%06d.jpg", path_video_gt)
            subprocess.run(
                ["vmaf", "-d", path_video_pred, "-r", path_video_gt, "--output", output_directory / "vmaf.xml"],
                check=True,
            )

    output_directory.mkdir(exist_ok=True, parents=True)
    with open(output_directory / 'metrics.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["camera", "frame", *results.keys()])
        writer.writeheader()
        for i, (camera_idx, frame_idx) in enumerate(cameras_frames):
            writer.writerow(
                {"camera": camera_idx + 1, "frame": frame_idx, **{k: results[k][i] for k in results.keys()}}
            )

    with open(output_directory / 'averages.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=averages.keys())
        writer.writeheader()
        writer.writerow(averages)
