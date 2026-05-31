import os
import csv
import json
import argparse
import math

import numpy as np
import sklearn.metrics
import torch
import pydicom
import torchvision
import tqdm

from models.uniformer import uniformer_small
from utils import set_seed, bootstrap_metric


def parse_args():
    parser = argparse.ArgumentParser(description='EchoRisk Phase 1 Single-View Evaluation')
    parser.add_argument('--config', type=str, required=True, help='Path to eval config JSON')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    class Args:
        pass
    a = Args()
    a.exp_name = cfg.get('exp', {}).get('exp_name', 'phase1_eval')
    a.seed = cfg.get('exp', {}).get('seed', 0)

    data = cfg.get('data', {})
    a.csv_val = data.get('csv_val')
    a.dicom_root = data.get('dicom_root')
    a.frames = data.get('frames', 36)
    a.frequency = data.get('frequency', 4)
    a.resize = data.get('resize', 112)

    model_cfg = cfg.get('model', {})
    a.model_name = model_cfg.get('model_name', 'uniformer_small')
    a.checkpoint = model_cfg.get('checkpoint')

    ev = cfg.get('eval', {})
    a.batch_size = ev.get('batch_size', 16)
    a.num_workers = ev.get('num_workers', 4)
    a.bootstrap_samples = ev.get('bootstrap_samples', 10000)
    a.save_predictions = ev.get('save_predictions', True)
    a.save_plot = ev.get('save_plot', True)

    a.device = cfg.get('device', {}).get('device', None)
    a.output_dir = cfg.get('output', {}).get('output_dir', 'eval_results/phase1_eval')

    assert a.csv_val, "csv_val is required"
    assert a.dicom_root, "dicom_root is required"
    assert a.checkpoint, "checkpoint is required"

    return a


class SingleViewDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, dicom_root, frames=36, frequency=4, resize=112):
        self.frames = frames
        self.frequency = frequency
        self.resize_size = resize
        self.samples = []
        self.transform = torchvision.transforms.Resize((resize, resize), antialias=True)

        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row['patient_id'].strip()
                tp = row['timepoint'].strip()
                lvef = float(row['lvef']) if row['lvef'].strip() else None
                if lvef is None:
                    continue

                for view_key, view_name in [('video_a4c', 'A4C'), ('video_a2c', 'A2C')]:
                    fname = row.get(view_key, '').strip()
                    if not fname:
                        continue
                    dcm_path = os.path.join(dicom_root, pid, tp, fname)
                    if not os.path.exists(dcm_path):
                        continue
                    self.samples.append((pid, tp, view_name, dcm_path, lvef))

        print(f"Loaded {len(self.samples)} single-view samples from {csv_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pid, tp, view, dcm_path, lvef = self.samples[idx]
        video = self._load_dicom(dcm_path)
        video = self._preprocess(video)
        return video, np.float32(lvef), pid, tp, view

    def _load_dicom(self, path):
        ds = pydicom.dcmread(path)
        video = ds.pixel_array
        video = np.ascontiguousarray(video, dtype=np.float32)
        video = video.transpose(3, 0, 1, 2)
        return video

    def _preprocess(self, video):
        video = torch.from_numpy(video)
        video = self.transform(video)
        video = video.numpy()
        video = self._sample_frames(video)
        return video

    def _sample_frames(self, video):
        c, f, h, w = video.shape
        target_frames = min(self.frames, f)

        if f >= target_frames:
            indices = np.linspace(0, f - 1, target_frames, dtype=int)
        else:
            indices = np.arange(f)
            pad_len = target_frames - f
            indices = np.concatenate([indices, np.full(pad_len, f - 1)])

        return video[:, indices, :, :].astype(np.float32)


def collate_fn(batch):
    videos = torch.stack([torch.from_numpy(b[0]) for b in batch])
    efs = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    pids = [b[2] for b in batch]
    tps = [b[3] for b in batch]
    views = [b[4] for b in batch]
    return videos, efs, pids, tps, views


def run_inference(model, dataloader, device):
    model.eval()
    all_y, all_yhat = [], []
    all_pids, all_tps, all_views = [], [], []

    with torch.no_grad():
        with tqdm.tqdm(total=len(dataloader)) as pb:
            for videos, efs, pids, tps, views in dataloader:
                videos = videos.to(device)
                preds = model(videos)

                all_y.append(efs.cpu().numpy())
                all_yhat.append(preds.view(-1).cpu().detach().numpy())
                all_pids.extend(pids)
                all_tps.extend(tps)
                all_views.extend(views)
                pb.update()

    y = np.concatenate(all_y)
    yhat = np.concatenate(all_yhat)
    return y, yhat, all_pids, all_tps, all_views


def write_report(output_dir, y, yhat, views, bootstrap_samples):
    report_path = os.path.join(output_dir, 'report.txt')
    lines = []
    def w(s):
        lines.append(s)
        print(s)

    w('=' * 60)
    w('  EchoRisk Phase 1 Evaluation Report')
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
    for cat in ['A4C', 'A2C']:
        idx = [i for i, v in enumerate(views) if v == cat]
        if not idx:
            continue
        y_cat = y[idx]
        yh_cat = yhat[idx]
        r2_cat = bootstrap_metric(y_cat, yh_cat, sklearn.metrics.r2_score, bootstrap_samples)
        mae_cat = bootstrap_metric(y_cat, yh_cat, sklearn.metrics.mean_absolute_error, bootstrap_samples)
        rmse_cat = tuple(map(np.sqrt, bootstrap_metric(
            y_cat, yh_cat, sklearn.metrics.mean_squared_error, bootstrap_samples)))
        w(f'  {cat}: N={len(idx)}, MAE={mae_cat[0]:.2f}, RMSE={rmse_cat[0]:.2f}, R²={r2_cat[0]:.3f}')

    w('\n--- EF Distribution ---')
    w(f'  True:  mean={np.mean(y):.1f}, std={np.std(y):.1f}, '
      f'min={np.min(y):.1f}, max={np.max(y):.1f}')
    w(f'  Pred:  mean={np.mean(yhat):.1f}, std={np.std(yhat):.1f}, '
      f'min={np.min(yhat):.1f}, max={np.max(yhat):.1f}')

    w('=' * 60)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'\nReport saved to {report_path}')


def save_predictions(output_dir, y, yhat, pids, tps, views):
    csv_path = os.path.join(output_dir, 'predictions.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['patient_id', 'timepoint', 'view', 'y_true', 'y_pred', 'error'])
        for i in range(len(y)):
            writer.writerow([pids[i], tps[i], views[i],
                             f'{y[i]:.2f}', f'{yhat[i]:.2f}', f'{yhat[i] - y[i]:.2f}'])
    print(f'Predictions saved to {csv_path}')


def save_plots(output_dir, y, yhat, views):
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
    ax.set_title(f'Phase 1 EF Prediction (N={len(y)}, R²={sklearn.metrics.r2_score(y, yhat):.3f})')
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

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, cat in zip(axes, ['A4C', 'A2C']):
        idx = [i for i, v in enumerate(views) if v == cat]
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
    args = parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = SingleViewDataset(
        csv_path=args.csv_val,
        dicom_root=args.dicom_root,
        frames=args.frames,
        frequency=args.frequency,
        resize=args.resize,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device.type == 'cuda'),
        collate_fn=collate_fn,
    )

    model = uniformer_small()
    model.head = torch.nn.Linear(model.embed_dim[-1], 1)
    model.head.bias.data[0] = 55.6
    model.to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint['state_dict']
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: {args.checkpoint} (epoch {checkpoint.get('epoch', '?')})")

    y, yhat, pids, tps, views = run_inference(model, dataloader, device)

    write_report(args.output_dir, y, yhat, views, args.bootstrap_samples)

    if args.save_predictions:
        save_predictions(args.output_dir, y, yhat, pids, tps, views)

    if args.save_plot:
        save_plots(args.output_dir, y, yhat, views)


if __name__ == '__main__':
    main()
