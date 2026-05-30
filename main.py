import os
import sys
import json

import numpy as np
import sklearn.metrics
import torch
import argparse

from datasets.echonet_dynamic import EchoRiskMultiModal, multimodal_collate_fn
from models.multimodal_uniformer import MultiModalEchoCoTr
from utils import get_optimizer, get_lr_scheduler, get_mean_and_sd, run_train, run_test, set_seed


def flatten_config(config_dict):
    flat = {}
    section_map = {
        'data': ['csv_train', 'csv_test', 'dicom_root', 'frames', 'frequency', 'resize', 'train_split_ratio', 'cache_dir'],
        'model': ['model_name', 'pretrained', 'weights'],
        'training': ['epochs', 'batch_size', 'num_workers', 'modal_dropout'],
        'optimization': ['optimizer_name', 'lr', 'weight_decay', 'lr_scheduler', 'lr_step_period'],
        'device': ['device'],
        'output': ['output'],
        'exp': ['exp_no', 'exp_name', 'seed'],
    }
    for section, keys in section_map.items():
        if section in config_dict:
            for k in keys:
                if k in config_dict[section]:
                    flat[k] = config_dict[section][k]
    return flat


def load_config_defaults(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return flatten_config(config)


def build_parser(defaults):
    parser = argparse.ArgumentParser(description='EchoRisk Multimodal EF Prediction')

    parser.add_argument('--config',
                        type=str,
                        default=None,
                        help='Path to JSON config file (overrides defaults)')

    parser.add_argument('--exp_no',
                        type=str,
                        default=defaults.get('exp_no', ''),
                        help='Experiment number')

    parser.add_argument('--exp_name',
                        type=str,
                        default=defaults.get('exp_name', ''),
                        help='Experiment name')

    parser.add_argument('--seed',
                        type=int,
                        default=defaults.get('seed', 0),
                        help='Random seed')

    parser.add_argument('--csv_train',
                        type=str,
                        default=defaults.get('csv_train'),
                        help='Path to training CSV')

    parser.add_argument('--csv_test',
                        type=str,
                        default=defaults.get('csv_test'),
                        help='Path to test CSV (optional, for final evaluation)')

    parser.add_argument('--dicom_root',
                        type=str,
                        default=defaults.get('dicom_root'),
                        help='Root directory for DICOM files')

    parser.add_argument('--output',
                        type=str,
                        default=defaults.get('output'),
                        help='Path to output directory')

    parser.add_argument('--model_name',
                        choices=['uniformer_small', 'uniformer_base'],
                        default=defaults.get('model_name', 'uniformer_small'),
                        help='Backbone model name')

    parser.add_argument('--pretrained',
                        type=lambda x: x.lower() in ('true', '1', 'yes'),
                        default=defaults.get('pretrained', False),
                        help='Whether to load pretrained weights')

    parser.add_argument('--weights',
                        type=str,
                        default=defaults.get('weights'),
                        help='Path to pretrained weights')

    parser.add_argument('--epochs',
                        type=int,
                        default=defaults.get('epochs', 45),
                        help='Number of epochs to train')

    parser.add_argument('--optimizer_name',
                        type=str,
                        default=defaults.get('optimizer_name', 'adamW'),
                        choices=['SGD', 'adamW', 'adam'],
                        help='Optimizer name')

    parser.add_argument('--lr_scheduler',
                        choices=['step', 'cosine'],
                        default=defaults.get('lr_scheduler', 'step'),
                        help='Learning rate scheduler')

    parser.add_argument('--lr',
                        type=float,
                        default=defaults.get('lr', 1e-4),
                        help='Learning rate')

    parser.add_argument('--weight_decay',
                        type=float,
                        default=defaults.get('weight_decay', 1e-4),
                        help='Weight decay')

    parser.add_argument('--lr_step_period',
                        type=int,
                        default=defaults.get('lr_step_period', 15),
                        help='Learning rate decay period')

    parser.add_argument('--frames',
                        type=int,
                        default=defaults.get('frames', 32),
                        help='Number of frames to sample')

    parser.add_argument('--frequency',
                        type=int,
                        default=defaults.get('frequency', 2),
                        help='Period between frames')

    parser.add_argument('--num_workers',
                        type=int,
                        default=defaults.get('num_workers', 4),
                        help='Number of workers')

    parser.add_argument('--batch_size',
                        type=int,
                        default=defaults.get('batch_size', 8),
                        help='Batch size')

    parser.add_argument('--resize',
                        type=int,
                        default=defaults.get('resize', 224),
                        help='Resize spatial dimensions to this size')

    parser.add_argument('--modal_dropout',
                        type=float,
                        default=defaults.get('modal_dropout', 0.15),
                        help='Probability of dropping one view during training')

    parser.add_argument('--train_split_ratio',
                        type=float,
                        default=defaults.get('train_split_ratio', 0.8),
                        help='Ratio of training data (by patient), rest used as val')

    parser.add_argument('--cache_dir',
                        type=str,
                        default=defaults.get('cache_dir'),
                        help='Path to preprocessed .pt cache dir (optional)')

    parser.add_argument('--device',
                        type=str,
                        default=defaults.get('device'),
                        help='Device to use')

    return parser


def main():
    prelim_parser = argparse.ArgumentParser(add_help=False)
    prelim_parser.add_argument('--config', type=str, default=None)
    prelim_args, remaining = prelim_parser.parse_known_args()

    defaults = {}
    if prelim_args.config:
        defaults = load_config_defaults(prelim_args.config)
        print(f"Loaded config from: {prelim_args.config}")

    parser = build_parser(defaults)
    args = parser.parse_args(remaining)

    assert args.csv_train is not None, "csv_train is required (provide via --config or --csv_train)"
    assert args.dicom_root is not None, "dicom_root is required (provide via --config or --dicom_root)"

    print("Exp Name: ", args.exp_name)
    print("Exp No.: ", args.exp_no)
    print("Seed: ", args.seed)
    print("Model Name: ", args.model_name)
    print("Pretrained: ", args.pretrained)
    print("Epochs: ", args.epochs)
    print("Modal Dropout: ", args.modal_dropout)
    print("Batch Size: ", args.batch_size)

    set_seed(args.seed)

    if args.output is None:
        output = os.path.join("output", "video", "{}_{}_{}_{}".format(
            args.model_name, args.frames, args.resize,
            "pretrained" if args.pretrained else "random"))
    else:
        output = args.output
    os.makedirs(output, exist_ok=True)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiModalEchoCoTr(
        model_name=args.model_name,
        pretrained=args.pretrained,
        weights=args.weights,
    )

    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)

    optimizer = get_optimizer(model, args)

    lr_scheduler = get_lr_scheduler(optimizer, args)

    dummy_ds = EchoRiskMultiModal(
        csv_path=args.csv_train,
        dicom_root=args.dicom_root,
        split="train",
        frames=args.frames,
        resize=args.resize,
        train_split_ratio=args.train_split_ratio,
        split_seed=args.seed,
        cache_dir=args.cache_dir,
    )
    args.mean, args.std = get_mean_and_sd(dummy_ds)

    print("Dataset mean: ", args.mean)
    print("Dataset std: ", args.std)

    with open(os.path.join(output, "log.csv"), "a") as f:
        epoch_resume = 0
        bestLoss = float("inf")
        try:
            checkpoint = torch.load(os.path.join(output, "checkpoint.pt"), weights_only=False)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['opt_dict'])
            lr_scheduler.load_state_dict(checkpoint['scheduler_dict'])
            epoch_resume = checkpoint["epoch"] + 1
            bestLoss = checkpoint["best_loss"]
            f.write("Resuming from epoch {}\n".format(epoch_resume))
            print("Epochs to resume: ", epoch_resume)
        except FileNotFoundError:
            f.write("Starting run from scratch\n")

        if epoch_resume < args.epochs:
            run_train(output, device, model, optimizer, lr_scheduler, bestLoss, epoch_resume, f, args)

        print(model)

        run_test(output, device, model, f, args)


if __name__ == "__main__":
    main()
