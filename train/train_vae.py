# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import os
import json
from utils.fixseed import fixseed
from utils.parser_util import train_args
from utils import dist_util
from train.training_loop_vae import TrainLoop_VAE
from data_loaders.get_data import get_dataset_loader
from utils.model_util import create_model_and_diffusion
from train.train_platforms import (
    ClearmlPlatform,
    TensorboardPlatform,
    NoPlatform,
)  # required for the eval operation
from model.vae import KLAutoEncoder


def main():
    args = train_args()
    fixseed(args.seed)
    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name="Args")

    if args.save_dir is None:
        raise FileNotFoundError("save_dir was not specified.")
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError("save_dir [{}] already exists.".format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, "args.json")
    with open(args_path, "w") as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device)

    print("creating data loader...")
    data = get_dataset_loader(
        name=args.dataset, batch_size=args.batch_size, num_frames=args.num_frames
    )

    print("creating model and diffusion...")
    # SMPL defaults
    njoints = 25
    nfeats = 6

    if args.dataset == 'humanml':
        njoints = 263
        nfeats = 1
    elif args.dataset == 'kit':
        njoints = 251
        nfeats = 1
    #1013 wonjae
    if args.joint_position:
        assert args.dataset == 'humanml'
        njoints = 67
        nfeats = 1
    model = KLAutoEncoder(device=args.device, 
                          nfeats=njoints*nfeats,
                          latent_dim=64, 
                          num_heads=4, 
                          num_layers=7, 
                          ff_size=1024, 
                          dropout=0.1).to(
        args.device
    )

    print("Training...")
    TrainLoop_VAE(args, train_platform, model, data).run_loop()
    train_platform.close()


if __name__ == "__main__":
    main()
