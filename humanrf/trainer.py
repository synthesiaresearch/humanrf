# Some parts of this code is based on torch-ngp repository whose license can be found below.
# MIT License

# Copyright (c) 2022 hawkey

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import subprocess
from pathlib import Path
from typing import Optional, Tuple

import cv2
import lpips
import numpy as np
import tensorboardX
import torch
import tqdm
from rich.console import Console
from rich.table import Table
from rich.theme import Theme
from skimage.metrics import structural_similarity

from actorshq.dataset.data_loader import DataLoader
from actorshq.dataset.input_batch import InputBatch
from humanrf.args.run_args import _run_args
from humanrf.input import merge_input_batches
from humanrf.scene_representation.humanrf import HumanRF
from humanrf.utils.loss import bce_loss
from humanrf.utils.memory import collect_and_free_memory, to_device
from humanrf.volume_rendering import RenderOutput, prune_samples, render


class Trainer:
    def __init__(
        self,
        config: _run_args,
        workspace: Path,
        checkpoint: Optional[str],
        model: HumanRF,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        max_num_checkpoints: int = 2,
        store_visualization_hd: bool = True,
        store_visualization_tb: bool = True,
    ) -> None:
        self.config = config
        self.workspace = workspace
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.max_num_checkpoints = max_num_checkpoints
        self.store_visualization_hd = store_visualization_hd
        self.store_visualization_tb = store_visualization_tb
        self.console = Console()

        self.model.to(config.device)

        self.scaler = torch.cuda.amp.GradScaler(growth_interval=config.training.scaler_growth_interval)
        self.lpips_alex = lpips.LPIPS(net='alex')

        self.step = 0
        self.val_step = 0
        self.stats = {
            "lpips_vals": [],
            "psnr_vals": [],
            "ssim_vals": [],
            "checkpoints": [],
            "best_lpips": np.inf,
            "best_psnr": 0,
            "best_ssim": 0,
        }

        self.photometric_loss = torch.nn.HuberLoss(reduction="mean", delta=0.01)
        self.mask_loss = bce_loss

        checkpoints_folder_name = "checkpoints"
        best_checkpoint_name = "best.pth"
        latest_checkpoint_name = "latest.pth"
        self.checkpoints_dir = self.workspace / checkpoints_folder_name
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.best_checkpoint_path = self.checkpoints_dir / best_checkpoint_name
        self.latest_checkpoint_path = self.checkpoints_dir / latest_checkpoint_name

        self._log_info(
            f"# parameters: {(sum([p.numel() for p in model.parameters() if p.requires_grad]) / 1e6):.3f} million"
        )

        self.load_checkpoint(full_state=True, checkpoint=checkpoint)

    def _log_info(self, text: str) -> None:
        self.console.print(f"[INFO] {text}", style="blue")

    def _log_warning(self, text: str) -> None:
        self.console.print(f"[WARNING] {text}", style="yellow")

    def train(self, training_data_loader: DataLoader, validation_data_loader: DataLoader, max_steps: int) -> None:
        """Performs training and validation, and saves the 'best' and 'latest' checkpoints regularly.

        Args:
            training_data_loader (DataLoader): Data loader for the training.
            validation_data_loader (DataLoader): Data loader for the validation.
            max_steps (int): Total number of training iterations to perform.
        """
        tb_path = self.workspace / "run"
        tb_path.mkdir(exist_ok=True)
        self.writer = tensorboardX.SummaryWriter(tb_path)
        pbar = tqdm.tqdm(
            unit=' total training iterations',
            total=self.config.training.max_steps,
        )
        pbar.update(self.step)

        # exponential moving average of loss
        loss_ema = 0
        self.model.train()

        training_data_loader_iter = iter(training_data_loader)

        for _ in range(self.step, max_steps + 1):
            self.step += 1

            # collect input batch
            training_data_loader.batch_size = self.config.training.rays_initial_batch_size
            total_num_rays = 0
            total_num_samples = 0
            input_batches = []
            while True:
                current_batch = next(training_data_loader_iter)
                with torch.cuda.amp.autocast():
                    prune_samples(
                        input_batch=current_batch,
                        scene_representation=self.model,
                        is_training=True,
                    )

                input_batches.append(current_batch)

                total_num_rays += training_data_loader.batch_size
                total_num_samples += current_batch.num_samples
                if total_num_samples < 0.9 * self.config.training.samples_max_batch_size:
                    average_num_samples_per_ray = total_num_samples / total_num_rays
                    assert average_num_samples_per_ray > 0, "There is probably a problem with the predicted geometry."

                    missing_num_samples = self.config.training.samples_max_batch_size - total_num_samples
                    training_data_loader.batch_size = int(missing_num_samples / average_num_samples_per_ray)
                else:
                    break

            # The procedure above does not guarantee that maximum number of samples will be upper bounded by
            # 'config.training.samples_max_batch_size'. This might introduce having excessive number of samples
            # which would lead to out of memory (OOM) issues. In merge_input_batches function, we
            # set max_num_samples to be at most 10% larger than 'config.training.samples_max_batch_size' to
            # prevent OOM.
            input_batch = merge_input_batches(
                input_batches=input_batches, max_num_samples=int(self.config.training.samples_max_batch_size * 1.1)
            )

            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                loss, losses_info = self.train_step(input_batch)

            step_loss = loss.item()
            loss_ema = 0.95 * loss_ema + 0.05 * step_loss

            for key in losses_info:
                self.writer.add_scalar(f"{key}/training", losses_info[key], self.step)
            self.writer.add_scalar("lr", self.optimizer.param_groups[0]['lr'], self.step)

            lr = self.optimizer.param_groups[0]['lr']
            pbar.set_description(f"loss={step_loss:.5f} (exp avg loss={loss_ema:.5f}), lr={lr:.6f}")
            pbar.update(1)

            if self.step > 0:
                training_data_loader.pause_replacing()
                if self.step % self.config.training.save_checkpoint_every_n_steps == 0:
                    self.save_checkpoint(full_state=True, best=False)
                    collect_and_free_memory()

                if self.step % self.config.validation.every_n_steps == 0:
                    self.validate(validation_data_loader)
                    self.save_checkpoint(full_state=True, best=True)
                    collect_and_free_memory()

                training_data_loader.continue_replacing()

        self.writer.close()
        pbar.close()

    def _calculate_losses(
        self, render_output: RenderOutput, gt_rgb: torch.Tensor, gt_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        # Photometric loss
        photometric_loss = self.photometric_loss(render_output.color, gt_rgb)
        total_loss = photometric_loss

        # Mask regularization loss
        if self.config.training.bce_loss_weight is not None:
            mask_loss = self.mask_loss(render_output.weights_sum, gt_mask).mean() * self.config.training.bce_loss_weight
            total_loss = total_loss + mask_loss

        with torch.no_grad():
            mse = torch.square(render_output.color - gt_rgb).mean()

            losses_info = {
                f"photometric": photometric_loss.item(),
                f"psnr": -10 * np.log10(mse.item()),
            }
            if self.config.training.bce_loss_weight is not None:
                losses_info["mask_loss"] = mask_loss.item()

        return total_loss, losses_info

    def train_step(self, input_batch: InputBatch) -> Tuple[torch.Tensor, dict]:
        _, num_color_channels = input_batch.rgba.shape
        if num_color_channels == 3:
            background_rgb = 1
            gt_rgb = input_batch.rgba
        elif num_color_channels == 4:
            gt_rgb = input_batch.rgba[..., 0:3]
            gt_mask = input_batch.rgba[..., 3:4]
            background_rgb = torch.rand_like(gt_rgb)
            gt_rgb = gt_rgb * gt_mask + background_rgb * (1 - gt_mask)
        else:
            raise RuntimeError("The ground truth image can be either RGB or RGBA!")

        render_output = render(
            input_batch=input_batch,
            scene_representation=self.model,
            background_rgb=background_rgb,
            is_training=True,
        )
        loss, losses_info = self._calculate_losses(render_output, gt_rgb, gt_mask)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.lr_scheduler.step()

        return loss, losses_info

    @torch.no_grad()
    def validate(self, data_loader: DataLoader) -> None:
        """Renders full images and performs evaluation.

        Args:
            data_loader (DataLoader): Validation data loader that provides input configurations to generate full images.
        """
        self.console.rule(f"[bold green]Validation at Step {self.step}")
        pbar = tqdm.tqdm(
            total=data_loader.num_camera_frame_pairs,
        )

        total_loss = {}
        self.model.eval()

        comparison_image_paths = []
        path_validation = self.workspace / "validation"
        path_validation.mkdir(exist_ok=True)
        log_path_validation = self.workspace / "validation.txt"
        with open(log_path_validation, "a") as f:
            f.write(f"\Step: {self.step}\n")

        background_rgb = 0
        val_img_step = 0
        partial_render_outputs = []
        partial_image_batches = []
        for data_idx, current_batch in enumerate(data_loader):
            with torch.cuda.amp.autocast():
                prune_samples(
                    input_batch=current_batch,
                    scene_representation=self.model,
                    is_training=False,
                )

            partial_image_batch = InputBatch(
                ray_masks=current_batch.ray_masks,
                rgba=current_batch.rgba,
                width=current_batch.width,
                height=current_batch.height,
            )
            to_device(partial_image_batch, "cpu")
            partial_image_batches.append(partial_image_batch)

            with torch.cuda.amp.autocast():
                partial_render_output = render(
                    input_batch=current_batch,
                    scene_representation=self.model,
                    background_rgb=background_rgb,
                    is_training=False,
                )
            to_device(partial_render_output, "cpu")
            partial_render_outputs.append(partial_render_output)

            if (data_idx + 1) % data_loader.num_batches_per_full_image != 0:
                continue

            full_image_batch = merge_input_batches(partial_image_batches)
            full_render_output = RenderOutput.merge_render_outputs(partial_render_outputs)

            full_pred_rgb, full_comparison_img, losses_info = self.evaluate_one_image(
                full_image_batch, full_render_output, background_rgb
            )

            partial_render_outputs = []
            partial_image_batches = []
            val_img_step += 1

            for key in losses_info:
                if key not in total_loss:
                    total_loss[key] = losses_info[key]
                else:
                    total_loss[key] += losses_info[key]

            global_step_info = f"step_{self.step:04d}"
            val_img_step_info = f"{val_img_step:04d}"
            save_path_rgb = path_validation / f"{global_step_info}_{val_img_step_info}_rgb.png"
            save_path_comparison = path_validation / f"{global_step_info}_{val_img_step_info}_comparison.png"
            comparison_image_paths.append(save_path_comparison)

            full_pred_rgb = (full_pred_rgb[0].cpu().numpy() * 255).astype(np.uint8)
            full_comparison_img = (full_comparison_img.cpu().numpy() * 255).astype(np.uint8)

            if self.store_visualization_hd:
                cv2.imwrite(str(save_path_rgb), cv2.cvtColor(full_pred_rgb, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(save_path_comparison), cv2.cvtColor(full_comparison_img, cv2.COLOR_RGB2BGR))

            if self.store_visualization_tb:
                self.writer.add_image(f"comp_{val_img_step_info}", full_comparison_img.transpose(2, 0, 1), self.step)

            description = ""
            average_description = ""
            for key in losses_info:
                if key not in ["mask_loss", "photometric"]:
                    description += f"{key}={losses_info[key]:.4f} "
                    average_description += f"{key}={total_loss[key] / val_img_step:.4f} "
            step_description = f"CURRENT: {description} --- AVERAGE: {average_description}"
            pbar.set_description(step_description, refresh=False)
            pbar.update(1)
            with open(log_path_validation, "a") as f:
                f.write(f"image_id: {val_img_step} --- {step_description}\n")

        for key in total_loss:
            total_loss[key] /= val_img_step

        self.stats["lpips_vals"].append(total_loss["lpips"])
        self.stats["psnr_vals"].append(total_loss["psnr"])
        self.stats["ssim_vals"].append(total_loss["ssim"])

        pbar.close()

        for key in total_loss:
            self.writer.add_scalar(f"{key}/validation", total_loss[key], self.step)

        self.val_step += 1

    @torch.no_grad()
    def evaluate_one_image(
        self, full_image_batch: InputBatch, full_render_output: RenderOutput, background_rgb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Combines the rendered rays into an output image, selects the area of interest (foreground) and computes
        the evaluation metrics.
        """
        _, num_color_channels = full_image_batch.rgba.shape
        if num_color_channels == 3:
            gt_rgb = full_image_batch.rgba
        elif num_color_channels == 4:
            gt_rgb = full_image_batch.rgba[..., 0:3]
            gt_mask = full_image_batch.rgba[..., 3:4]
            gt_rgb = gt_rgb * gt_mask + background_rgb * (1 - gt_mask)
        else:
            raise RuntimeError("The ground truth image can be either RGB or RGBA!")

        _, losses_info = self._calculate_losses(full_render_output, gt_rgb, gt_mask)

        full_pred_rgb = self.combine_rays_to_image(full_image_batch, full_render_output, background_rgb)

        ray_mask = full_image_batch.ray_masks.squeeze(1)
        image_width, image_height = full_image_batch.width, full_image_batch.height

        full_gt = torch.full((image_width * image_height, 3), background_rgb, dtype=torch.float)
        full_gt[ray_mask] = gt_rgb
        full_gt = full_gt.view(1, image_height, image_width, 3)

        full_mask = torch.zeros((image_width * image_height), dtype=torch.float)
        full_mask[ray_mask] = gt_mask.squeeze(1)
        full_mask = full_mask.view(image_height, image_width)

        x, y, w, h = cv2.boundingRect(((full_mask > 0) * 255).numpy().astype(np.uint8))
        full_pred_roi = full_pred_rgb[:, y : y + h, x : x + w, :]
        full_gt_roi = full_gt[:, y : y + h, x : x + w, :]

        losses_info["lpips"] = self.lpips_alex(
            full_pred_roi.permute(0, 3, 1, 2),
            full_gt_roi.permute(0, 3, 1, 2),
            normalize=True,
        ).item()

        losses_info["ssim"] = structural_similarity(
            full_pred_roi[0].numpy(), full_gt_roi[0].numpy(), channel_axis=2, data_range=1.0
        )

        full_comparison_img = torch.cat((full_pred_roi[0], full_gt_roi[0]), dim=1)
        return full_pred_rgb, full_comparison_img, losses_info

    @torch.no_grad()
    def test(self, data_loader: DataLoader, save_path: Path, render_video: bool = False) -> None:
        """Renders and saves full images without performing evaluation.

        Args:
            data_loader (DataLoader): Test data loader that provides input configurations to generate full images.
            save_path (Path): Outputs will be saved in this directory.
            render_video (bool, optional): Whether to combine the rendered images into a video. Defaults to False.
        """
        self.console.rule(f"[bold red]Test")
        self._log_info(f"Results are saved to {save_path}")
        pbar = tqdm.tqdm(
            total=data_loader.num_camera_frame_pairs,
        )
        self.model.eval()

        save_path.mkdir(exist_ok=True, parents=True)

        background_rgb = 0
        self.test_img_step = 0
        partial_render_outputs = []
        partial_image_batches = []
        for data_idx, current_batch in enumerate(data_loader):
            with torch.cuda.amp.autocast():
                prune_samples(
                    input_batch=current_batch,
                    scene_representation=self.model,
                    is_training=False,
                )

            partial_image_batch = InputBatch(
                ray_masks=current_batch.ray_masks,
                width=current_batch.width,
                height=current_batch.height,
            )
            to_device(partial_image_batch, "cpu")
            partial_image_batches.append(partial_image_batch)

            with torch.cuda.amp.autocast():
                partial_render_output = render(
                    input_batch=current_batch,
                    scene_representation=self.model,
                    background_rgb=background_rgb,
                    is_training=False,
                )
            to_device(partial_render_output, "cpu")
            partial_render_outputs.append(partial_render_output)

            if (data_idx + 1) % data_loader.num_batches_per_full_image != 0:
                continue

            full_image_batch = merge_input_batches(partial_image_batches)
            full_render_output = RenderOutput.merge_render_outputs(partial_render_outputs)
            partial_render_outputs = []
            partial_image_batches = []

            full_pred_rgb = self.combine_rays_to_image(full_image_batch, full_render_output, background_rgb)

            camera_number, frame_number = data_loader.render_sequence[self.test_img_step]
            if render_video:
                filename = f"{self.test_img_step:06d}"
            else:
                filename = data_loader.dataset.filepaths.get_rgb_path(
                    data_loader.cameras[camera_number].name,
                    frame_number,
                ).stem

            full_pred_rgb = (full_pred_rgb[0].cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(str(save_path / f"{filename}.png"), cv2.cvtColor(full_pred_rgb, cv2.COLOR_RGB2BGR))

            pbar.update(1)
            self.test_img_step += 1

        pbar.close()
        if render_video:
            subprocess.run(
                # fmt: off
                [
                    "ffmpeg",
                    "-r", "25",
                    "-i", str(Path(save_path) / "%06d.png"),
                    "-filter_complex", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-loglevel", "error",
                    "-c:v", "libx264",
                    "-crf", "14",
                    "-profile:v", "baseline",
                    "-level", "3.0",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "faststart",
                    "-y",
                    str(Path(save_path).parent / f"video_{save_path.stem}.mp4"),
                ]
                # fmt: on
            )

    @torch.no_grad()
    def combine_rays_to_image(
        self, full_image_batch: InputBatch, full_render_output: RenderOutput, background_rgb: torch.Tensor
    ) -> torch.Tensor:
        """Combines the partial render outputs into a full image."""
        image_width, image_height = full_image_batch.width, full_image_batch.height
        full_pred_rgb = torch.full((image_width * image_height, 3), background_rgb, dtype=torch.float)
        full_pred_rgb[full_image_batch.ray_masks.squeeze(1)] = full_render_output.color
        full_pred_rgb = full_pred_rgb.view(1, image_height, image_width, 3)

        return full_pred_rgb

    def save_checkpoint(self, full_state: bool, best: bool) -> None:
        state = {
            "step": self.step,
            "stats": self.stats,
            "val_step": self.val_step,
        }

        if full_state:
            state["optimizer"] = self.optimizer.state_dict()
            state["lr_scheduler"] = self.lr_scheduler.state_dict()
            state["scaler"] = self.scaler.state_dict()

        if not best:
            filepath = f"{self.checkpoints_dir}/step_{self.step:08d}.pth"
            self.stats["checkpoints"].append(filepath)

            if len(self.stats["checkpoints"]) > self.max_num_checkpoints:
                oldest_checkpoint = Path(self.stats["checkpoints"].pop(0))
                if oldest_checkpoint.exists():
                    oldest_checkpoint.unlink()

            state["model"] = self.model.state_dict()
            torch.save(state, filepath)

        elif len(self.stats["lpips_vals"]) > 0:
            self.stats["best_lpips"] = min(self.stats["best_lpips"], self.stats["lpips_vals"][-1])
            self.stats["best_psnr"] = max(self.stats["best_psnr"], self.stats["psnr_vals"][-1])
            self.stats["best_ssim"] = max(self.stats["best_ssim"], self.stats["ssim_vals"][-1])

            table = Table(title="")
            table.add_column("-", justify="center", style="white")
            table.add_column("LPIPS", style="red")
            table.add_column("PSNR", style="magenta")
            table.add_column("SSIM", style="cyan")
            table.add_row(
                f"Step {self.step:08d}",
                f"{self.stats['lpips_vals'][-1]:.4f}",
                f"{self.stats['psnr_vals'][-1]:.2f}",
                f"{self.stats['ssim_vals'][-1]:.4f}",
            )
            table.add_row(
                "Best",
                f"{self.stats['best_lpips']:.4f}",
                f"{self.stats['best_psnr']:.2f}",
                f"{self.stats['best_ssim']:.4f}",
                style="bold",
            )
            self.console.print(table, justify="center")

            if self.stats["lpips_vals"][-1] == self.stats["best_lpips"]:
                self._log_info(f"New best LPIPS is achieved, saving the checkpoint...")

                state["model"] = self.model.state_dict()
                torch.save(state, self.best_checkpoint_path)

    def load_checkpoint(self, full_state: bool, checkpoint: Optional[str]) -> None:
        if checkpoint is None:
            self._log_warning("No checkpoint is specified! If desired, do it via '--checkpoint' parameter from CLI.")
            return

        if checkpoint == "latest":
            checkpoint_list = sorted(Path(f"{self.checkpoints_dir}").glob("step_*.pth"))
            if len(checkpoint_list) > 0:
                checkpoint = checkpoint_list[-1]
            else:
                self._log_warning("No checkpoint is found, model is randomly initialized.")
                return
        elif checkpoint == "best":
            checkpoint = self.best_checkpoint_path

        self._log_info(f"Loading the checkpoint from {checkpoint} ...")

        checkpoint_dict = torch.load(checkpoint, map_location=self.config.device)
        self.model.load_state_dict(checkpoint_dict["model"])

        if not full_state:
            self._log_info(f"The model is loaded at step {self.step}")
            return

        self.val_step = checkpoint_dict["val_step"]
        self.step = checkpoint_dict["step"]
        self.stats = checkpoint_dict["stats"]

        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint_dict["optimizer"])

        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint_dict["lr_scheduler"])

        if self.scaler is not None:
            self.scaler.load_state_dict(checkpoint_dict["scaler"])

        self._log_info(f"The full state is loaded at step {self.step}")
