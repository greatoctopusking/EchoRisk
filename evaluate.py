import os
import json
import csv
import argparse

import numpy as np
import sklearn.metrics
import torch
import tqdm

from datasets.echonet_dynamic import EchoRiskMultiModal, multimodal_collate_fn
from models.multimodal_uniformer import MultiModalEchoCoTr
from utils import set_seed, bootstrap_metric


def parse_eval_args():
    prelim_parser = argparse.ArgumentParser(add_help=False)
    prelim_parser.add_argument('--config', type=str, default=None)
    prelim_args, remaining = prelim_parser.parse_known_args()

    defaults = {}
    if prelim_args.config:
        with open(prelim_args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
        for section in ['exp', 'data', 'model', 'eval', 'device', 'output']:
            if section in config:
                for k, v in config[section].items():
                    defaults[k] = v

    parser = argparse.ArgumentParser(description='EchoRisk Model Evaluation')
    parser.add_argument('--config', type=str, default=None, help='Path to eval config JSON')
    parser.add_argument('--exp_name', type=str, default=defaults.get('exp_name', 'eval'))
    parser.add_argument('--seed', type=int, default=defaults.get('seed', 0))
    parser.add_argument('--csv_test', type=str, default=defaults.get('csv_test'))
    parser.add_argument('--dicom_root', type=str, default=defaults.get('dicom_root'))
    parser.add_argument('--frames', type=int, default=defaults.get('frames', 32))
    parser.add_argument('--frequency', type=int, default=defaults.get('frequency', 2))
    parser.add_argument('--resize', type=int, default=defaults.get('resize', 224))
    parser.add_argument('--mean', type=str, default=None, help='Comma-separated 3 floats')
    parser.add_argument('--std', type=str, default=None, help='Comma-separated 3 floats')
    parser.add_argument('--model_name', type=str, default=defaults.get('model_name', 'uniformer_small'))
    parser.add_argument('--checkpoint', type=str, default=defaults.get('checkpoint'))
    parser.add_argument('--batch_size', type=int, default=defaults.get('batch_size', 8))
    parser.add_argument('--num_workers', type=int, default=defaults.get('num_workers', 4))
    parser.add_argument('--bootstrap_samples', type=int, default=defaults.get('bootstrap_samples', 10000))
    parser.add_argument('--save_predictions', type=lambda x: x.lower() in ('true', '1', 'yes'),
                        default=defaults.get('save_predictions', True))
    parser.add_argument('--save_plot', type=lambda x: x.lower() in ('true', '1', 'yes'),
                        default=defaults.get('save_plot', True))
    parser.add_argument('--device', type=str, default=defaults.get('device'))
    parser.add_argument('--output_dir', type=str, default=defaults.get('output_dir'))

    args = parser.parse_args(remaining)

    assert args.csv_test is not None, "csv_test is required"
    assert args.dicom_root is not None, "dicom_root is required"
    assert args.checkpoint is not None, "checkpoint is required"

    if args.mean is not None:
        args.mean = np.array([float(x) for x in args.mean.split(',')], dtype=np.float32)
    if args.std is not None:
        args.std = np.array([float(x) for x in args.std.split(',')], dtype=np.float32)

    return args


def run_eval_inference(model, dataloader, device):
    model.eval()
    y, yhat = [], []

    with torch.no_grad():
        with tqdm.tqdm(total=len(dataloader)) as progressbar:
            for batch in dataloader:
                a4c_video, a2c_video, ef, a4c_mask, a2c_mask = batch

                ef = ef.to(device)
                a4c_video = a4c_video.to(device) if a4c_video.numel() > 0 else None
                a2c_video = a2c_video.to(device) if a2c_video.numel() > 0 else None
                a4c_mask = a4c_mask.to(device)
                a2c_mask = a2c_mask.to(device)

                outputs = model(a4c_video, a2c_video, a4c_mask, a2c_mask)

                y.append(ef.cpu().numpy())
                yhat.append(outputs.view(-1).to("cpu").detach().numpy())

                progressbar.update()

    y = np.concatenate(y) if y else np.array([])
    yhat = np.concatenate(yhat) if yhat else np.array([])
    return y, yhat

def write_report(output_dir, y, yhat, view_cats, bootstrap_samples):
    report_path = os.path.join(output_dir, 'report.txt')
    lines = []
    def w(s):
        lines.append(s)
        print(s)

    w('=' * 60)
    w('  EchoRisk Evaluation Report')
    w('=' * 60)

    r2 = bootstrap_metric(y, yhat, sklearn.metrics.r2_score, bootstrap_samples)
    mae = bootstrap_metric(y, yhat, sklearn.metrics.mean_absolute_error, bootstrap_samples)
    rmse = tuple(map(np.sqrt, bootstrap_metric(y, yhat, sklearn.metrics.mean_squared_error, bootstrap_samples)))
    bias = float(np.mean(yhat - y))

    w(f'\n--- Overall (N={len(y)}) ---')
    w(f'  MAE:   {mae[0]:.2f} ({mae[1]:.2f} - {mae[2]:.2f})')
    w(f'  RMSE:  {rmse[0]:.2f} ({rmse[1]:.2f} - {rmse[2]:.2f})')
    w(f'  R²:    {r2[0]:.3f} ({r2[1]:.3f} - {r2[2]:.3f})')
    w(f'  Bias:  {bias:.2f}')

    w('\n--- By View ---')
    for cat in ['both', 'a4c_only', 'a2c_only']:
        idx = [i for i, c in enumerate(view_cats) if c == cat]
        if not idx:
            continue
        y_cat = y[idx]
        yh_cat = yhat[idx]
        r2_cat = bootstrap_metric(y_cat, yh_cat, sklearn.metrics.r2_score, bootstrap_samples)
        mae_cat = bootstrap_metric(y_cat, yh_cat, sklearn.metrics.mean_absolute_error, bootstrap_samples)
        w(f'  {cat}:  N={len(idx)}, MAE={mae_cat[0]:.2f}, R²={r2_cat[0]:.3f}')

    w('\n--- EF Distribution ---')
    w(f'  True:  mean={np.mean(y):.1f}, std={np.std(y):.1f}, '
      f'min={np.min(y):.1f}, max={np.max(y):.1f}')
    w(f'  Pred:  mean={np.mean(yhat):.1f}, std={np.std(yhat):.1f}, '
      f'min={np.min(yhat):.1f}, max={np.max(yhat):.1f}')

    w('=' * 60)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'\nReport saved to {report_path}')


def save_predictions_csv(output_dir, y, yhat, view_cats, dataset):
    csv_path = os.path.join(output_dir, 'predictions.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'patient_id', 'timepoint', 'view_category',
                          'y_true', 'y_pred', 'error'])
        for i in range(len(y)):
            pid, tp = dataset.samples[i][0], dataset.samples[i][1]
            writer.writerow([i, pid, tp, view_cats[i],
                              f'{y[i]:.2f}', f'{yhat[i]:.2f}', f'{yhat[i] - y[i]:.2f}'])
    print(f'Predictions saved to {csv_path}')


def save_plots(output_dir, y, yhat, view_cats):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not installed, skipping plots')
        return

    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(y, yhat, alpha=0.4, s=8, c='steelblue')
    lims = [min(y.min(), yhat.min()) - 5, max(y.max(), yhat.max()) + 5]
    ax.plot(lims, lims, 'r--', linewidth=1, alpha=0.7)
    ax.set_xlabel('True EF')
    ax.set_ylabel('Predicted EF')
    ax.set_title(f'EF Prediction (N={len(y)}, R²={sklearn.metrics.r2_score(y, yhat):.3f})')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'scatter.png'), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    errors = yhat - y
    ax.hist(errors, bins=30, edgecolor='white', alpha=0.8, color='steelblue')
    ax.axvline(0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('Error (Predicted - True)')
    ax.set_ylabel('Count')
    ax.set_title(f'Error Distribution (Bias={np.mean(errors):.2f})')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'errors_dist.png'), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, cat in zip(axes, ['both', 'a4c_only', 'a2c_only']):
        idx = [i for i, c in enumerate(view_cats) if c == cat]
        if not idx:
            ax.set_title(f'{cat} (N=0)')
            continue
        yc, yhc = y[idx], yhat[idx]
        ax.scatter(yc, yhc, alpha=0.4, s=8, c='steelblue')
        ax.plot([yc.min(), yc.max()], [yc.min(), yc.max()], 'r--', linewidth=1, alpha=0.7)
        ax.set_title(f'{cat} (N={len(idx)}, MAE={sklearn.metrics.mean_absolute_error(yc, yhc):.2f})')
        ax.set_xlabel('True EF')
        ax.set_ylabel('Predicted EF')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'view_breakdown.png'), dpi=150)
    plt.close(fig)
    print(f'Plots saved to {output_dir}')


def main():
    args = parse_eval_args()

    set_seed(args.seed)

    if args.output_dir is None:
        output_dir = os.path.join('eval_results', args.exp_name)
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = EchoRiskMultiModal(
        csv_path=args.csv_test,
        dicom_root=args.dicom_root,
        split="test",
        frames=args.frames,
        frequency=args.frequency,
        resize=args.resize,
        mean=args.mean if args.mean is not None else 0.,
        std=args.std if args.std is not None else 1.,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        collate_fn=multimodal_collate_fn,
    )

    model = MultiModalEchoCoTr(
        model_name=args.model_name,
        pretrained=False,
        weights=None,
    )
    model.to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint['state_dict']
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: {args.checkpoint} (epoch {checkpoint.get('epoch', '?')})")

    y, yhat = run_eval_inference(model, dataloader, device)

    a4c_mask = [dataset.samples[i][2] is not None for i in range(len(dataset))]
    a2c_mask = [dataset.samples[i][3] is not None for i in range(len(dataset))]
    view_cats = []
    for am, bm in zip(a4c_mask, a2c_mask):
        if am and bm:
            view_cats.append('both')
        elif am:
            view_cats.append('a4c_only')
        elif bm:
            view_cats.append('a2c_only')
        else:
            view_cats.append('none')

    write_report(output_dir, y, yhat, view_cats, args.bootstrap_samples)

    if args.save_predictions:
        save_predictions_csv(output_dir, y, yhat, view_cats, dataset)

    if args.save_plot:
        save_plots(output_dir, y, yhat, view_cats)


if __name__ == "__main__":
    main()
