import os
import csv
import collections

import cv2
import numpy as np
import torch
import torchvision
import tqdm


def _defaultdict_of_lists():
    return collections.defaultdict(list)


class EchoNet(torchvision.datasets.VisionDataset):
    def __init__(self, root=None,
                 split="train",
                 mean=0.,
                 std=1.,
                 frames=16,
                 frequency=2,
                 max_frames=250,
                 pad=None):

        assert(root is not None)

        super().__init__(root)

        self.split = split
        self.mean = mean
        self.std = std
        self.frames = frames
        self.max_frames = max_frames
        self.frequency = frequency
        self.pad = pad

        self.vnames, self.outcome = [], []
        self.read_filelist()

        self.frames_list = collections.defaultdict(list)
        self.trace = collections.defaultdict(_defaultdict_of_lists)

        self.read_volumetracings()

        self.filter_videos()

        self._validate_readable()

        print("{} dataset size: {}".format(split, len(self.vnames)))

    def read_filelist(self):
        with open(os.path.join(self.root, "FileList.csv")) as f:
            self.file_header = f.readline().strip().split(",")
            filename_index = self.file_header.index("FileName")
            split_index = self.file_header.index("Split")

            for line in f:
                line_split = line.strip().split(',')

                filename = os.path.splitext(line_split[filename_index])[0] + ".avi"
                file_split = line_split[split_index].lower()

                if self.split in ["all", file_split] and os.path.exists(os.path.join(self.root, "Videos", filename)):
                    self.vnames.append(filename)
                    self.outcome.append(line_split)

        self.check_missing_videos()

    def check_missing_videos(self):
        missing_videos = set(self.vnames) - set(os.listdir(os.path.join(self.root, "Videos")))
        if len(missing_videos) != 0:
            print("{} videos are missing in {}:".format(len(missing_videos), os.path.join(self.root, "Videos")))
            for f in sorted(missing_videos):
                print("\t", f)
            raise FileNotFoundError(os.path.join(self.root, "Videos", sorted(missing_videos)[0]))

    def read_volumetracings(self):
        with open(os.path.join(self.root, "VolumeTracings.csv")) as f:
            header = f.readline().strip().split(",")
            assert header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]

            for line in f:
                filename, x1, y1, x2, y2, frame = line.strip().split(',')
                x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                frame = int(frame)
                if frame not in self.trace[filename]:
                    self.frames_list[filename].append(frame)
                self.trace[filename][frame].append((x1, y1, x2, y2))

        for filename in self.frames_list:
            for frame in self.frames_list[filename]:
                 self.trace[filename][frame] = np.array(self.trace[filename][frame])

    def filter_videos(self):
        min_frames = 2
        videos_to_keep = [len(self.frames_list[f]) >= min_frames for f in self.vnames]
        self.vnames = [f for (f, k) in zip(self.vnames, videos_to_keep) if k]
        self.outcome = [f for (f, k) in zip(self.outcome, videos_to_keep) if k]

    def __getitem__(self, index):
        video = os.path.join(self.root, "Videos", self.vnames[index])

        video = self.load_video(video).astype(np.float32)

        video = self.normalize_video(video)

        video = self.sample_video(video)

        if self.pad is not None:
            video = self.pad_video(video)

        ef = np.float32(self.outcome[index][self.file_header.index("EF")])

        return video, ef

    def __len__(self):
        return len(self.vnames)

    def normalize_video(self, video):
        if isinstance(self.mean, (float, int)):
            video -= self.mean
        else:
            video -= self.mean.reshape(3, 1, 1, 1)

        if isinstance(self.std, (float, int)):
            video /= self.std
        else:
            video /= self.std.reshape(3, 1, 1, 1)

        return video

    def sample_video(self, video):
        c, f, h, w = video.shape
        frames = self.frames
        frames = min(frames, self.max_frames)

        if f < frames * self.frequency:
            video = np.concatenate((video, np.zeros((c, frames * self.frequency - f, h, w), video.dtype)), axis=1)
            c, f, h, w = video.shape

        start = np.random.choice(f - (frames - 1) * self.frequency, 1)

        video = tuple(video[:, s + self.frequency * np.arange(frames), :, :] for s in start)[0]

        return video

    def pad_video(self, video):
        if self.pad is None:
            return video

        c, l, h, w = video.shape
        tvideo = np.zeros((c, l, h + 2 * self.pad, w + 2 * self.pad), dtype=video.dtype)
        tvideo[:, :, self.pad:-self.pad, self.pad:-self.pad] = video
        i, j = np.random.randint(0, 2 * self.pad, 2)

        return tvideo[:, :, i:(i + h), j:(j + w)]

    def load_video(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        cap = None
        for backend in [cv2.CAP_ANY, cv2.CAP_FFMPEG]:
            cap = cv2.VideoCapture(path, backend)
            if cap.isOpened():
                break
        if cap is None or not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")

        frames = []
        while True:
            out, frame = cap.read()
            if not out:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()

        if not frames:
            raise ValueError(f"No frames could be read from {path}")

        video = np.stack(frames, axis=0)
        return video.transpose((3, 0, 1, 2))

    def _validate_readable(self):
        cache_path = os.path.join(self.root, f".validated_{self.split.upper()}.txt")

        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                valid_set = set(line.strip() for line in f if line.strip())
            keep = [i for i, v in enumerate(self.vnames) if v in valid_set]
            if len(keep) < len(self.vnames):
                self.vnames = [self.vnames[i] for i in keep]
                self.outcome = [self.outcome[i] for i in keep]
            print(f"Loaded cached validation: {len(self.vnames)} valid videos from {cache_path}")
            return

        bad_indices = []
        for i, v in enumerate(tqdm.tqdm(self.vnames, desc="Validating videos")):
            path = os.path.join(self.root, "Videos", v)
            cap = None
            for backend in [cv2.CAP_ANY, cv2.CAP_FFMPEG]:
                cap = cv2.VideoCapture(path, backend)
                if cap.isOpened():
                    break
            if cap is None or not cap.isOpened():
                bad_indices.append(i)
                continue
            out, _ = cap.read()
            cap.release()
            if not out:
                bad_indices.append(i)

        if bad_indices:
            self.vnames = [v for i, v in enumerate(self.vnames) if i not in bad_indices]
            self.outcome = [o for i, o in enumerate(self.outcome) if i not in bad_indices]
            print(f"\nFiltered {len(bad_indices)} unreadable videos "
                  f"(remaining: {len(self.vnames)})")

        with open(cache_path, 'w', encoding='utf-8') as f:
            for v in self.vnames:
                f.write(v + '\n')
        print(f"Cached validation results to {cache_path}")


import pydicom

DICOM_CMAP = None

try:
    _ = pydicom.dcmread
except AttributeError:
    raise ImportError("pydicom is required. Install with: pip install pydicom")


class EchoRiskMultiModal(torchvision.datasets.VisionDataset):
    def __init__(self, csv_path, dicom_root, split="train",
                 mean=0., std=1.,
                 frames=32, frequency=2,
                 max_frames=250,
                 resize=224,
                 train_split_ratio=0.8,
                 split_seed=42,
                 cache_dir=None):
        self.split = split
        self.mean = mean
        self.std = std
        self.frames = frames
        self.frequency = frequency
        self.max_frames = max_frames
        self.dicom_root = dicom_root
        self.resize_size = resize
        self.cache_dir = cache_dir

        all_samples = self._load_csv(csv_path, dicom_root)

        if split in ("train", "val") and train_split_ratio < 1.0:
            self.samples = self._patient_split(all_samples, split, train_split_ratio, split_seed)
        else:
            self.samples = all_samples

        if self.cache_dir is None:
            self.transform = torchvision.transforms.Resize((resize, resize), antialias=True) if resize else None
        else:
            self.transform = None

        print("{} dataset size: {} (from {} total){}".format(
            split, len(self.samples), len(all_samples),
            " [cached]" if cache_dir else ""))

    @staticmethod
    def _patient_split(all_samples, split, train_split_ratio, split_seed):
        patients = sorted(set(s[0] for s in all_samples))
        rng = np.random.RandomState(split_seed)
        rng.shuffle(patients)
        n_train = int(len(patients) * train_split_ratio)
        train_patients = set(patients[:n_train])

        if split == "train":
            return [s for s in all_samples if s[0] in train_patients]
        else:
            return [s for s in all_samples if s[0] not in train_patients]

    @staticmethod
    def _load_csv(csv_path, dicom_root):
        samples = []
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row["patient_id"].strip()
                tp = row["timepoint"].strip()
                lvef = float(row["lvef"]) if row["lvef"].strip() else None
                if lvef is None:
                    continue

                a4c_fname = row.get("video_a4c", "").strip()
                a2c_fname = row.get("video_a2c", "").strip()

                a4c_path = os.path.join(dicom_root, pid, tp, a4c_fname) if a4c_fname else None
                a2c_path = os.path.join(dicom_root, pid, tp, a2c_fname) if a2c_fname else None

                if a4c_path and not os.path.exists(a4c_path):
                    a4c_path = None
                if a2c_path and not os.path.exists(a2c_path):
                    a2c_path = None

                if a4c_path is None and a2c_path is None:
                    continue

                samples.append((pid, tp, a4c_path, a2c_path, lvef))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        pid, tp, a4c_path, a2c_path, lvef = self.samples[index]

        if self.cache_dir:
            a4c_video = self._load_cached(pid, tp, "A4C") if a4c_path else None
            a2c_video = self._load_cached(pid, tp, "A2C") if a2c_path else None
        else:
            a4c_video = self._load_dicom(a4c_path) if a4c_path else None
            a2c_video = self._load_dicom(a2c_path) if a2c_path else None

        if a4c_video is not None:
            a4c_video = self._preprocess(a4c_video)
        if a2c_video is not None:
            a2c_video = self._preprocess(a2c_video)

        ef = np.float32(lvef)

        return a4c_video, a2c_video, ef

    def _load_cached(self, pid, tp, view):
        cache_path = os.path.join(self.cache_dir, pid, f"{tp}_{view}.pt")
        return torch.load(cache_path, map_location='cpu', weights_only=True).numpy()

    def _load_dicom(self, path):
        ds = pydicom.dcmread(path)
        video = ds.pixel_array
        video = np.ascontiguousarray(video, dtype=np.float32)
        video = video.transpose(3, 0, 1, 2)
        return video

    def _preprocess(self, video):
        if self.transform is not None:
            video = torch.from_numpy(video)
            video = self.transform(video)
            video = video.numpy()

        video = self._normalize(video)
        video = self._sample_frames(video)

        return video

    def _normalize(self, video):
        if isinstance(self.mean, (float, int)):
            video -= self.mean
        else:
            video -= self.mean.reshape(3, 1, 1, 1)

        if isinstance(self.std, (float, int)):
            video /= self.std
        else:
            video /= self.std.reshape(3, 1, 1, 1)

        return video

    def _sample_frames(self, video):
        c, f, h, w = video.shape
        target_frames = min(self.frames, self.max_frames)

        if f >= target_frames:
            indices = np.linspace(0, f - 1, target_frames, dtype=int)
        else:
            indices = np.arange(f)
            pad_len = target_frames - f
            indices = np.concatenate([indices, np.full(pad_len, f - 1)])

        sampled = video[:, indices, :, :].astype(np.float32)

        return sampled


def multimodal_collate_fn(batch):
    batch_size = len(batch)
    a4c_list, a2c_list, ef_list = [], [], []
    a4c_mask_list, a2c_mask_list = [], []
    a4c_indices, a2c_indices = [], []

    for i, (a4c, a2c, ef) in enumerate(batch):
        ef_list.append(ef)
        if a4c is not None:
            a4c_list.append(torch.from_numpy(a4c))
            a4c_indices.append(i)
        a4c_mask_list.append(a4c is not None)

        if a2c is not None:
            a2c_list.append(torch.from_numpy(a2c))
            a2c_indices.append(i)
        a2c_mask_list.append(a2c is not None)

    a4c_mask = torch.tensor(a4c_mask_list, dtype=torch.bool)
    a2c_mask = torch.tensor(a2c_mask_list, dtype=torch.bool)
    ef_tensor = torch.tensor(ef_list, dtype=torch.float32)

    if a4c_list:
        sample_shape = a4c_list[0].shape
        a4c_tensor = torch.zeros(batch_size, *sample_shape)
        a4c_tensor[a4c_indices] = torch.stack(a4c_list)
    else:
        a4c_tensor = torch.zeros(batch_size, 3, 32, 224, 224)

    if a2c_list:
        sample_shape = a2c_list[0].shape
        a2c_tensor = torch.zeros(batch_size, *sample_shape)
        a2c_tensor[a2c_indices] = torch.stack(a2c_list)
    else:
        a2c_tensor = torch.zeros(batch_size, 3, 32, 224, 224)

    return a4c_tensor, a2c_tensor, ef_tensor, a4c_mask, a2c_mask
