import copy
import functools
import os
import time
from types import SimpleNamespace
import numpy as np

import blobfile as bf
import torch
from torch.optim import AdamW

from diffusion import logger
from utils import dist_util
from diffusion.fp16_util import MixedPrecisionTrainer
from diffusion.resample import LossAwareSampler, UniformSampler
from tqdm import tqdm
from diffusion.resample import create_named_schedule_sampler
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from eval import eval_humanml, eval_humanact12_uestc
from data_loaders.get_data import get_dataset_loader


# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop_VAE:
    def __init__(self, args, train_platform, model, data):
        self.args = args
        self.dataset = args.dataset
        self.train_platform = train_platform
        self.model = model
        self.data = data
        self.batch_size = args.batch_size
        self.microbatch = args.batch_size  # deprecating this option
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint
        self.use_fp16 = False  # deprecating this option
        self.fp16_scale_growth = 1e-3  # deprecating this option
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size  # * dist.get_world_size()
        self.num_steps = args.num_steps
        self.num_epochs = self.num_steps // len(self.data) + 1

        self.sync_cuda = torch.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )

        self.save_dir = args.save_dir
        self.overwrite = args.overwrite

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.

        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != "cpu":
            self.device = torch.device(dist_util.dev())

        self.eval_gt_data = None
        if args.dataset in ["kit", "humanml"] and args.eval_during_training:
            self.eval_gt_data = get_dataset_loader(
                name=args.dataset,
                batch_size=args.eval_batch_size,
                num_frames=None,
                split=args.eval_split,
                hml_mode="gt",
            )

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                dist_util.load_state_dict(
                    resume_checkpoint, map_location=dist_util.dev()
                )
            )

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:09}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        for epoch in range(self.num_epochs):
            print(f"Starting epoch {epoch}")
            for motion, cond in tqdm(self.data):
                if not (
                    not self.lr_anneal_steps
                    or self.step + self.resume_step < self.lr_anneal_steps
                ):
                    break

                motion = motion.to(self.device)
                cond["y"] = {
                    key: val.to(self.device) if torch.is_tensor(val) else val
                    for key, val in cond["y"].items()
                }

                self.run_step(motion, cond)
                if self.step % self.log_interval == 0:
                    for k, v in logger.get_current().name2val.items():
                        if k == "loss":
                            print(
                                "step[{}]: loss[{:0.5f}]".format(
                                    self.step + self.resume_step, v
                                )
                            )
                        
                        if k == "klloss":
                            print(
                                "step[{}]: klloss[{:0.5f}]".format(
                                    self.step + self.resume_step, v
                                )
                            )
                        
                        if k == "reploss":
                            print(
                                "step[{}]: reploss[{:0.5f}]".format(
                                    self.step + self.resume_step, v
                                )
                            )

                        if k in ["step", "samples"] or "_q" in k:
                            continue
                        else:
                            self.train_platform.report_scalar(
                                name=k, value=v, iteration=self.step, group_name="Loss"
                            )

                if self.step % self.save_interval == 0:
                    self.save()
                    self.model.eval()
                    self.evaluate()
                    self.model.train()

                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return
                self.step += 1
            if not (
                not self.lr_anneal_steps
                or self.step + self.resume_step < self.lr_anneal_steps
            ):
                break
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
            self.evaluate()

    def evaluate(self):
        if not self.args.eval_during_training:
            return
        start_eval = time.time()
        if self.eval_gt_data:
            for motion, cond in tqdm(self.data):
                if not (
                    not self.lr_anneal_steps
                    or self.step + self.resume_step < self.lr_anneal_steps
                ):
                    break

                motion = motion.to(self.device)
                cond["y"] = {
                    key: val.to(self.device) if torch.is_tensor(val) else val
                    for key, val in cond["y"].items()
                }

                loss = self.compute_loss(motion)
                print(f"Validation Loss: {loss}")
        end_eval = time.time()
        print(f"Evaluation time: {round(end_eval-start_eval)/60}min")

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        self.mp_trainer.optimize(self.opt)
        self._anneal_lr()
        self.log_step()

    def compute_loss(self, motion):
        def loss_function(x, x_hat, mean, log_var):
            mseloss = torch.nn.MSELoss(reduction='mean')
            reproduction_loss = mseloss(x, x_hat)
            KLD_loss = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())

            return reproduction_loss , KLD_loss

        # motion = [bs, joint_num, 1, timesteps]

        bs, njoints, nfeats, nframes = motion.shape
        motion = motion.reshape(bs, njoints*nfeats, nframes)
        motion = motion.permute(0,2,1)

        terms = {}

        x_hat, mean, log_var = self.model(motion)
        rep_loss, kl_loss = loss_function(motion, x_hat, mean, log_var)
        terms["loss"] = rep_loss+kl_loss
        terms["klloss"] = kl_loss
        terms["reploss"] = rep_loss
        return terms

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        losses = self.compute_loss(batch)

        loss = losses["loss"].mean()
        log_loss_dict({k: v for k, v in losses.items()})
        self.mp_trainer.backward(loss)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def ckpt_file_name(self):
        return f"model{(self.step+self.resume_step):09d}.pt"

    def save(self):
        def save_checkpoint(params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)

            # Do not save CLIP weights
            clip_weights = [e for e in state_dict.keys() if e.startswith("clip_model.")]
            for e in clip_weights:
                del state_dict[e]

            logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint(self.mp_trainer.master_params)

        with bf.BlobFile(
            bf.join(self.save_dir, f"opt{(self.step+self.resume_step):09d}.pt"),
            "wb",
        ) as f:
            torch.save(self.opt.state_dict(), f)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def log_loss_dict(losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
