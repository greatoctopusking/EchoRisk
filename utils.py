import math
import os
import time
import random

import numpy as np
import sklearn.metrics
import torch
import tqdm

from models.uniformer import uniformer_small, uniformer_base
from datasets.echonet_dynamic import EchoNet, EchoRiskMultiModal, multimodal_collate_fn


def set_seed(s):
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(s)
    random.seed(s)
    os.environ['PYTHONHASHSEED'] = str(s)


def get_optimizer(model, args):
    if args.optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer_name == "adamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer_name}")
    return optimizer


def get_lr_scheduler(optimizer, args):
    if args.lr_scheduler == "step":
        lr_step_period = args.lr_step_period if args.lr_step_period is not None else 15
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_period)
    elif args.lr_scheduler == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        raise ValueError(f"Unknown lr_scheduler: {args.lr_scheduler}")
    return lr_scheduler


def get_model(model_name, args):
    if model_name in ["r2plus1d_18", "mc3_18", "r3d_18"]:
        import torchvision
        model = torchvision.models.video.__dict__[model_name](pretrained=args.pretrained)
        model.fc = torch.nn.Linear(model.fc.in_features, 1)
        model.fc.bias.data[0] = 55.6
    elif model_name == "uniformer_small":
        model = uniformer_small()
        if args.pretrained and args.weights is not None:
            state_dict = torch.load(args.weights, map_location='cpu', weights_only=True)
            model.load_state_dict(state_dict, strict=False)
        model.head = torch.nn.Linear(in_features=model.head.in_features, out_features=1)
        model.head.bias.data[0] = 55.6
    elif model_name == "uniformer_base":
        model = uniformer_base()
        if args.pretrained and args.weights is not None:
            state_dict = torch.load(args.weights, map_location='cpu', weights_only=True)
            model.load_state_dict(state_dict, strict=False)
        model.head = torch.nn.Linear(in_features=model.head.in_features, out_features=1)
        model.head.bias.data[0] = 55.6
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    return model


def get_mean_and_sd(dataset, num_samples=128, batch_size=8, num_workers=4):
    if num_samples is not None and len(dataset) > num_samples:
        indices = np.random.choice(len(dataset), num_samples, replace=False).tolist()
        dataset = torch.utils.data.Subset(dataset, indices)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,
        collate_fn=multimodal_collate_fn)

    samples, sum1, sum2 = 0, 0., 0.
    for (a4c, a2c, ef, a4c_mask, a2c_mask) in tqdm.tqdm(dataloader):
        videos = []
        if a4c_mask.any():
            videos.append(a4c[a4c_mask])
        if a2c_mask.any():
            videos.append(a2c[a2c_mask])
        if not videos:
            continue
        combined = torch.cat(videos, dim=0)
        combined = combined.transpose(0, 1).contiguous().view(3, -1)
        sum1 += torch.sum(combined, dim=1).numpy()
        sum2 += torch.sum(combined ** 2, dim=1).numpy()
        samples += combined.shape[1]

    mean = np.float32(sum1 / max(samples, 1))
    sd = np.float32(np.sqrt(sum2 / max(samples, 1) - mean ** 2))

    return mean, sd


def bootstrap_metric(arg1, arg2, fun, num_samples=10000):
    results = []
    arg1, arg2 = np.array(arg1), np.array(arg2)

    for _ in range(num_samples):
        index = np.random.choice(len(arg1), len(arg1))
        results.append(fun(arg1[index], arg2[index]))

    results = sorted(results)
    percentile_05 = results[round(0.05 * len(results))]
    percentile_95 = results[round(0.95 * len(results))]

    return fun(arg1, arg2), percentile_05, percentile_95


def run_epoch(model, dataloader, train, optimizer, device, modal_dropout=0.0):
    model.train(train)

    total_loss, samples = 0, 0
    y, yhat = [], []
    scaler = torch.amp.GradScaler('cuda', enabled=train)

    with torch.set_grad_enabled(train):
        with tqdm.tqdm(total=len(dataloader)) as progressbar:
            for batch in dataloader:
                a4c_video, a2c_video, ef, a4c_mask, a2c_mask = batch

                ef = ef.to(device)
                a4c_video = a4c_video.to(device) if a4c_video.numel() > 0 else None
                a2c_video = a2c_video.to(device) if a2c_video.numel() > 0 else None
                a4c_mask = a4c_mask.to(device)
                a2c_mask = a2c_mask.to(device)

                if train and modal_dropout > 0:
                    if a4c_mask.any():
                        drop_a4c = torch.rand(a4c_mask.sum(), device=device) < modal_dropout
                        a4c_mask_dropped = a4c_mask.clone()
                        drop_indices = a4c_mask.nonzero(as_tuple=True)[0][drop_a4c]
                        a4c_mask_dropped[drop_indices] = False
                    else:
                        a4c_mask_dropped = a4c_mask

                    if a2c_mask.any():
                        drop_a2c = torch.rand(a2c_mask.sum(), device=device) < modal_dropout
                        a2c_mask_dropped = a2c_mask.clone()
                        drop_indices = a2c_mask.nonzero(as_tuple=True)[0][drop_a2c]
                        a2c_mask_dropped[drop_indices] = False
                    else:
                        a2c_mask_dropped = a2c_mask

                    mask_a4c = a4c_mask_dropped
                    mask_a2c = a2c_mask_dropped
                else:
                    mask_a4c = a4c_mask
                    mask_a2c = a2c_mask

                with torch.amp.autocast('cuda', enabled=train):
                    outputs = model(a4c_video, a2c_video, mask_a4c, mask_a2c)
                    loss = torch.nn.functional.mse_loss(outputs.view(-1), ef)

                y.append(ef.cpu().numpy())
                yhat.append(outputs.view(-1).to("cpu").detach().numpy())

                if train:
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                total_loss += loss.item() * ef.size(0)
                samples += ef.size(0)

                progressbar.set_postfix_str("{:.2f} ({:.2f})".format(total_loss / max(samples, 1), loss.item()))
                progressbar.update()

    yhat = np.concatenate(yhat) if yhat else np.array([])
    y = np.concatenate(y) if y else np.array([])

    return total_loss / max(samples, 1), yhat, y


def run_train(output, device, model, optimizer, lr_scheduler, bestLoss, epoch_resume, f, args):
    kwargs = dict(
        csv_path=args.csv_train,
        dicom_root=args.dicom_root,
        mean=args.mean,
        std=args.std,
        frames=args.frames,
        frequency=args.frequency,
        resize=args.resize,
        train_split_ratio=args.train_split_ratio,
        split_seed=args.seed,
        cache_dir=getattr(args, 'cache_dir', None),
    )
    train_ds = EchoRiskMultiModal(split="train", **kwargs)
    val_ds = EchoRiskMultiModal(split="val", **kwargs)

    loss, r2 = 0.0, 0.0
    for epoch in range(epoch_resume, args.epochs):
        print("Epoch #{}".format(epoch), flush=True)
        for phase in ['train', 'val']:
            start_time = time.time()
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)

            dataset = train_ds if phase == "train" else val_ds
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=True,
                pin_memory=(device.type == "cuda"),
                drop_last=(phase == "train"),
                collate_fn=multimodal_collate_fn,
            )

            modal_dropout = args.modal_dropout if phase == "train" else 0.0
            loss, yhat, y = run_epoch(model, dataloader, phase == "train", optimizer, device,
                                      modal_dropout=modal_dropout)
            r2 = sklearn.metrics.r2_score(y, yhat) if len(y) > 1 else 0.0
            f.write("{},{},{},{},{},{},{},{},{}\n".format(epoch,
                                                            phase,
                                                            loss,
                                                            r2,
                                                            time.time() - start_time,
                                                            y.size,
                                                            sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count())),
                                                            sum(torch.cuda.max_memory_reserved() for i in range(torch.cuda.device_count())),
                                                            args.batch_size))
            if phase == "train":
                print("  train loss: {:.4f}, r2: {:.4f}".format(loss, r2))
            else:
                print("  val loss: {:.4f}, r2: {:.4f}".format(loss, r2))

            f.flush()
        lr_scheduler.step()

        save = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'frequency': args.frequency,
            'frames': args.frames,
            'best_loss': bestLoss,
            'loss': loss,
            'r2': r2,
            'opt_dict': optimizer.state_dict(),
            'scheduler_dict': lr_scheduler.state_dict(),
        }
        torch.save(save, os.path.join(output, "checkpoint.pt"))
        if loss < bestLoss:
            torch.save(save, os.path.join(output, "best.pt"))
            bestLoss = loss


def run_test(output, device, model, f, args):
    if not args.csv_test:
        return

    if args.epochs != 0:
        checkpoint = torch.load(os.path.join(output, "best.pt"), weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        f.write("Best validation loss {} from epoch {}\n".format(checkpoint["loss"], checkpoint["epoch"]))
        f.flush()

    set_seed(0)
    dataset = EchoRiskMultiModal(
        csv_path=args.csv_test,
        dicom_root=args.dicom_root,
        split="test",
        mean=args.mean,
        std=args.std,
        frames=args.frames,
        frequency=args.frequency,
        resize=args.resize,
        cache_dir=getattr(args, 'cache_dir', None),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        collate_fn=multimodal_collate_fn,
    )
    loss, yhat, y = run_epoch(model, dataloader, False, None, device)

    if len(yhat) == 0:
        return

    r2 = bootstrap_metric(y, yhat, sklearn.metrics.r2_score)
    mae = bootstrap_metric(y, yhat, sklearn.metrics.mean_absolute_error)
    rmse = tuple(map(math.sqrt, bootstrap_metric(y, yhat, sklearn.metrics.mean_squared_error)))

    f.write("test R2:   {:.3f} ({:.3f} - {:.3f})\n".format(*r2))
    f.write("test MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(*mae))
    f.write("test RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(*rmse))
    f.flush()

    print("test R2: {:.3f} ({:.3f} - {:.3f})".format(*r2))
    print("test MAE: {:.2f} ({:.2f} - {:.2f})".format(*mae))
    print("test RMSE: {:.2f} ({:.2f} - {:.2f})".format(*rmse))
