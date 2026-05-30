import os
import json
import argparse
import time

import numpy as np
import sklearn.metrics
import torch
import tqdm

from models.uniformer import uniformer_small, uniformer_base
from datasets.echonet_dynamic import EchoNet
from utils import set_seed, get_optimizer, get_lr_scheduler, bootstrap_metric


def get_model(model_name, pretrained, weights_path):
    if model_name == 'uniformer_small':
        model = uniformer_small()
    elif model_name == 'uniformer_base':
        model = uniformer_base()
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    if pretrained and weights_path is not None:
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
        result = model.load_state_dict(state_dict, strict=False)
        loaded = len(state_dict) - len(result.unexpected_keys) - len(result.missing_keys)
        print(f"[Pretrain] Loaded {loaded}/{len(state_dict)} keys from K400")

    model.head = torch.nn.Linear(in_features=model.head.in_features, out_features=1)
    model.head.bias.data[0] = 55.6
    return model


def run_epoch(model, dataloader, train, optimizer, device):
    model.train(train)
    total_loss, n = 0, 0
    y, yhat = [], []
    scaler = torch.amp.GradScaler('cuda', enabled=train)

    with torch.set_grad_enabled(train):
        with tqdm.tqdm(total=len(dataloader)) as pb:
            for video, ef in dataloader:
                video, ef = video.to(device), ef.to(device)

                with torch.amp.autocast('cuda', enabled=train):
                    outputs = model(video)
                    loss = torch.nn.functional.mse_loss(outputs.view(-1), ef)

                y.append(ef.cpu().numpy())
                yhat.append(outputs.view(-1).to("cpu").detach().numpy())

                if train:
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                total_loss += loss.item() * ef.size(0)
                n += ef.size(0)
                pb.set_postfix_str("{:.2f} ({:.2f})".format(total_loss / max(n, 1), loss.item()))
                pb.update()

    yhat = np.concatenate(yhat) if yhat else np.array([])
    y = np.concatenate(y) if y else np.array([])
    return total_loss / max(n, 1), yhat, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = json.load(f)

    exp = cfg.get('exp', {})
    data_cfg = cfg.get('data', {})
    model_cfg = cfg.get('model', {})
    train_cfg = cfg.get('training', {})
    opt_cfg = cfg.get('optimization', {})

    root = data_cfg['root']
    frames = data_cfg.get('frames', 36)
    frequency = data_cfg.get('frequency', 4)
    batch_size = train_cfg.get('batch_size', 20)
    epochs = train_cfg.get('epochs', 45)
    num_workers = train_cfg.get('num_workers', 4)
    lr = opt_cfg.get('lr', 1e-4)
    wd = opt_cfg.get('weight_decay', 1e-4)
    model_name = model_cfg.get('model_name', 'uniformer_small')
    pretrained = model_cfg.get('pretrained', True)
    weights_path = model_cfg.get('weights', None)
    seed = exp.get('seed', 0)
    output_dir = cfg.get('output', {}).get('output', 'output/echonet_pretrain')

    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = EchoNet(root=root, split="train", frames=frames, frequency=frequency)
    val_ds = EchoNet(root=root, split="val", frames=frames, frequency=frequency)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    dataloader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, drop_last=True, **dataloader_kwargs)
    val_loader = torch.utils.data.DataLoader(val_ds, shuffle=False, **dataloader_kwargs)

    model = get_model(model_name, pretrained, weights_path)
    model.to(device)
    if device.type == "cuda":
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt_cfg.get('lr_step_period', 15))

    best_loss = float('inf')
    with open(os.path.join(output_dir, 'log.csv'), 'a') as log:
        log.write("epoch,phase,loss,r2,time,n_samples\n")

        for epoch in range(epochs):
            print(f"\nEpoch #{epoch}")
            for phase, loader in [('train', train_loader), ('val', val_loader)]:
                t0 = time.time()
                loss, yhat, y = run_epoch(model, loader, phase == 'train', optimizer, device)
                r2 = sklearn.metrics.r2_score(y, yhat) if len(y) > 1 else 0.0
                elapsed = time.time() - t0
                log.write(f"{epoch},{phase},{loss:.6f},{r2:.6f},{elapsed:.1f},{len(y)}\n")
                print(f"  {phase} loss: {loss:.4f}, r2: {r2:.4f}")

            scheduler.step()

            save_dict = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'loss': loss,
                'r2': r2,
            }
            torch.save(save_dict, os.path.join(output_dir, "checkpoint.pt"))
            if loss < best_loss:
                best_loss = loss
                torch.save(save_dict, os.path.join(output_dir, "best.pt"))

    best_ckpt = torch.load(os.path.join(output_dir, "best.pt"), weights_only=False, map_location='cpu')
    encoder_state = best_ckpt['state_dict']
    encoder_state = {k.replace('module.', '') if k.startswith('module.') else k: v
                     for k, v in encoder_state.items()
                     if not k.startswith(('module.head', 'head'))}
    torch.save(encoder_state, os.path.join(output_dir, "encoder.pt"))
    print(f"Encoder weights saved to {os.path.join(output_dir, 'encoder.pt')}")


if __name__ == '__main__':
    main()
