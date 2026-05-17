"""
Preprocess Sleep-EDF-20 for EEG + EOG.

直接运行:
    python preprocess_sleepedf20_eeg_eog.py

输出:
    x_EEG:      [N, 25, 1, 3000]
    x_EOG:      [N, 25, 1, 3000]
    x_EEG_miss: [N, 25, 1, 3000]
    x_EOG_miss: [N, 25, 1, 3000]
    y:          [N, 25]
"""

import argparse
import glob
import math
import ntpath
import os
import shutil
from datetime import datetime

import numpy as np
from mne.io import read_raw_edf

import dhedfreader


DATA_DIR = "../data/physionet_sleep/physionet_sleep/sleep_cassette/"
OUTPUT_DIR = "../data/output_data/physionet_20_eeg_eog/"

SELECT_CH_EEG = "EEG Fpz-Cz"
SELECT_CH_EOG = "EOG horizontal"

MISSING_RATE = 0.5
MISSING_CHUNK_EPOCHS = 25
CONTEXT_LEN = 25

SEED = 2024
W_EDGE_MINS = 30

EPOCH_SEC_SIZE = 30

EEG_MODAL_LABEL = 1
EOG_MODAL_LABEL = 2

W = 0
N1 = 1
N2 = 2
N3 = 3
REM = 4
UNKNOWN = 5

stage_dict = {
    "W": W,
    "N1": N1,
    "N2": N2,
    "N3": N3,
    "REM": REM,
    "UNKNOWN": UNKNOWN,
}

ann2label = {
    "Sleep stage W": W,
    "Sleep stage 1": N1,
    "Sleep stage 2": N2,
    "Sleep stage 3": N3,
    "Sleep stage 4": N3,
    "Sleep stage R": REM,
    "Sleep stage ?": UNKNOWN,
    "Movement time": UNKNOWN,
}


def read_edf_header(edf_path):
    with open(edf_path, "r", errors="ignore") as f:
        reader = dhedfreader.BaseEDFReader(f)
        reader.read_header()
        return reader.header


def read_hypnogram(ann_path):
    with open(ann_path, "r", errors="ignore") as f:
        reader = dhedfreader.BaseEDFReader(f)
        reader.read_header()
        header = reader.header
        _, _, ann = zip(*reader.records())
    return header, ann[0]


def make_chunk_missing_mask(n_epochs, missing_rate, chunk_epochs, seed):
    """
    缺失单位是 epoch。

    eeg_mask / eog_mask:
        shape: [n_epochs]
        1 表示该 epoch 的该模态存在
        0 表示该 epoch 的该模态缺失

    missing_rate=0.5 且两个模态时:
        总模态槽位数 = 2 * n_epochs
        缺失槽位数 = 0.5 * 2 * n_epochs = n_epochs

    保证同一个 epoch 不会 EEG 和 EOG 同时缺失。
    """
    rng = np.random.RandomState(seed)

    eeg_mask = np.ones(n_epochs, dtype=np.int64)
    eog_mask = np.ones(n_epochs, dtype=np.int64)

    if missing_rate <= 0:
        return eeg_mask, eog_mask

    total_missing = int(round(missing_rate * 2 * n_epochs))
    total_missing = min(total_missing, n_epochs)

    chunk_starts = np.arange(0, n_epochs, chunk_epochs)
    rng.shuffle(chunk_starts)

    missing_count = 0

    for start in chunk_starts:
        if missing_count >= total_missing:
            break

        end = min(start + chunk_epochs, n_epochs)
        cur_len = min(end - start, total_missing - missing_count)

        if rng.rand() < 0.5:
            eeg_mask[start:start + cur_len] = 0
            eog_mask[start:start + cur_len] = 1
        else:
            eog_mask[start:start + cur_len] = 0
            eeg_mask[start:start + cur_len] = 1

        missing_count += cur_len

    assert np.all((eeg_mask + eog_mask) >= 1), "存在 EEG 和 EOG 同时缺失的 epoch"

    return eeg_mask, eog_mask


def apply_missing_signal(x, mask, seed):

    rng = np.random.RandomState(seed)

    x_miss = x.copy()

    missing_idx = np.where(mask == 0)[0]
    valid_idx = np.where(mask == 1)[0]

    if len(missing_idx) == 0:
        return x_miss

    if len(valid_idx) == 0:
        raise RuntimeError("该模态没有可用于替换的未缺失 epoch")

    replace_idx = rng.choice(
        valid_idx,
        size=len(missing_idx),
        replace=True,
    )

    x_miss[missing_idx, :, :] = x[replace_idx, :, :]

    return x_miss


def reshape_epochs_to_context(x, context_len):
    """
    输入:
        x: [n_epochs, 3000, 1]

    输出:
        x: [N, 25, 1, 3000]
    """
    n_epochs, n_samples, n_channels = x.shape
    n_seq = n_epochs // context_len

    x = x.reshape(n_seq, context_len, n_samples, n_channels)
    x = np.transpose(x, (0, 1, 3, 2))

    return x


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--select_ch_eeg", type=str, default=SELECT_CH_EEG)
    parser.add_argument("--select_ch_eog", type=str, default=SELECT_CH_EOG)
    parser.add_argument("--missing_rate", type=float, default=MISSING_RATE)
    parser.add_argument("--missing_chunk_epochs", type=int, default=MISSING_CHUNK_EPOCHS)
    parser.add_argument("--context_len", type=int, default=CONTEXT_LEN)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--w_edge_mins", type=int, default=W_EDGE_MINS)

    args = parser.parse_args()

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)

    psg_fnames = sorted(glob.glob(os.path.join(args.data_dir, "*PSG.edf")))
    ann_fnames = sorted(glob.glob(os.path.join(args.data_dir, "*Hypnogram.edf")))

    psg_fnames = np.asarray(psg_fnames)
    ann_fnames = np.asarray(ann_fnames)

    print("Found PSG files:", len(psg_fnames))
    print("Found annotation files:", len(ann_fnames))

    assert len(psg_fnames) == len(ann_fnames), "PSG 和 Hypnogram 文件数量不一致"

    for i in range(len(psg_fnames)):
        print("\nProcessing:", psg_fnames[i])

        raw = read_raw_edf(
            psg_fnames[i],
            preload=True,
            stim_channel=None,
            verbose=False,
        )

        sampling_rate = raw.info["sfreq"]
        samples_per_epoch = int(EPOCH_SEC_SIZE * sampling_rate)

        print("Sampling rate:", sampling_rate)
        print("Samples per epoch:", samples_per_epoch)
        print("Available channels:", raw.ch_names)

        if args.select_ch_eeg not in raw.ch_names:
            raise ValueError(
                f"找不到 EEG 通道: {args.select_ch_eeg}\n"
                f"当前文件可用通道: {raw.ch_names}"
            )

        if args.select_ch_eog not in raw.ch_names:
            raise ValueError(
                f"找不到 EOG 通道: {args.select_ch_eog}\n"
                f"当前文件可用通道: {raw.ch_names}"
            )

        raw_df = raw.to_data_frame(scalings=100.0)

        raw_eeg_df = raw_df[args.select_ch_eeg].to_frame()
        raw_eog_df = raw_df[args.select_ch_eog].to_frame()

        h_raw = read_edf_header(psg_fnames[i])
        raw_start_dt = datetime.strptime(h_raw["date_time"], "%Y-%m-%d %H:%M:%S")

        h_ann, ann = read_hypnogram(ann_fnames[i])
        ann_start_dt = datetime.strptime(h_ann["date_time"], "%Y-%m-%d %H:%M:%S")

        assert raw_start_dt == ann_start_dt, "PSG 和 Hypnogram 起始时间不一致"

        remove_idx = []
        labels = []
        label_idx = []

        for a in ann:
            onset_sec, duration_sec, ann_char = a
            ann_str = "".join(ann_char)
            ann_key = ann_str[2:-1]

            label = ann2label[ann_key]

            idx = int(onset_sec * sampling_rate) + np.arange(
                int(duration_sec * sampling_rate),
                dtype=np.int64,
            )

            if label != UNKNOWN:
                if duration_sec % EPOCH_SEC_SIZE != 0:
                    raise RuntimeError("Annotation duration 不能被 30 秒整除")

                duration_epoch = int(duration_sec / EPOCH_SEC_SIZE)
                labels.append(np.ones(duration_epoch, dtype=np.int64) * label)
                label_idx.append(idx)
            else:
                remove_idx.append(idx)

        labels = np.hstack(labels)

        n_total_samples = len(raw_eeg_df)
        all_idx = np.arange(n_total_samples)

        if len(remove_idx) > 0:
            remove_idx = np.hstack(remove_idx)
            select_idx = np.setdiff1d(all_idx, remove_idx)
        else:
            select_idx = all_idx

        label_idx = np.hstack(label_idx)
        select_idx = np.intersect1d(select_idx, label_idx)

        if len(label_idx) > len(select_idx):
            extra_idx = np.setdiff1d(label_idx, select_idx)

            if len(extra_idx) > 0 and np.all(extra_idx > select_idx[-1]):
                n_label_trims = int(math.ceil(len(extra_idx) / samples_per_epoch))
                if n_label_trims != 0:
                    labels = labels[:-n_label_trims]

        raw_eeg = raw_eeg_df.values[select_idx]
        raw_eog = raw_eog_df.values[select_idx]

        if len(raw_eeg) % samples_per_epoch != 0:
            raise RuntimeError("EEG 数据不能完整切分成 30 秒 epoch")

        if len(raw_eog) % samples_per_epoch != 0:
            raise RuntimeError("EOG 数据不能完整切分成 30 秒 epoch")

        n_epochs = int(len(raw_eeg) / samples_per_epoch)

        x_EEG = np.asarray(np.split(raw_eeg, n_epochs)).astype(np.float32)
        x_EOG = np.asarray(np.split(raw_eog, n_epochs)).astype(np.float32)

        y = labels.astype(np.int64)

        assert len(x_EEG) == len(y), f"x_EEG 和 y 数量不一致: {len(x_EEG)} vs {len(y)}"
        assert len(x_EOG) == len(y), f"x_EOG 和 y 数量不一致: {len(x_EOG)} vs {len(y)}"

        print("Before sleep selection:", x_EEG.shape, x_EOG.shape, y.shape)

        nw_idx = np.where(y != stage_dict["W"])[0]

        start_idx = nw_idx[0] - args.w_edge_mins * 2
        end_idx = nw_idx[-1] + args.w_edge_mins * 2

        start_idx = max(start_idx, 0)
        end_idx = min(end_idx, len(y) - 1)

        keep_idx = np.arange(start_idx, end_idx + 1)

        x_EEG = x_EEG[keep_idx]
        x_EOG = x_EOG[keep_idx]
        y = y[keep_idx]

        print("After sleep selection:", x_EEG.shape, x_EOG.shape, y.shape)

        n_valid = (len(y) // args.context_len) * args.context_len

        if n_valid <= 0:
            raise RuntimeError(
                f"有效 epoch 数量 {len(y)} 不足一个 context_len={args.context_len}"
            )

        if n_valid < len(y):
            print(f"Trim tail epochs: {len(y)} -> {n_valid}")
            x_EEG = x_EEG[:n_valid]
            x_EOG = x_EOG[:n_valid]
            y = y[:n_valid]

        print("After context trim:", x_EEG.shape, x_EOG.shape, y.shape)

        eeg_mask, eog_mask = make_chunk_missing_mask(
            n_epochs=len(y),
            missing_rate=args.missing_rate,
            chunk_epochs=args.missing_chunk_epochs,
            seed=args.seed + i,
        )

        x_EEG_miss = apply_missing_signal(
            x_EEG,
            eeg_mask,
            seed=args.seed + i * 2,
        )

        x_EOG_miss = apply_missing_signal(
            x_EOG,
            eog_mask,
            seed=args.seed + i * 2 + 1,
        )

        EEG_miss_labels = eeg_mask.astype(np.int64)
        EOG_miss_labels = eog_mask.astype(np.int64)

        EEG_modal_labels = np.ones(len(y), dtype=np.int64) * EEG_MODAL_LABEL
        EOG_modal_labels = np.ones(len(y), dtype=np.int64) * EOG_MODAL_LABEL

        n_seq = len(y) // args.context_len

        x_EEG = reshape_epochs_to_context(x_EEG, args.context_len)
        x_EOG = reshape_epochs_to_context(x_EOG, args.context_len)
        x_EEG_miss = reshape_epochs_to_context(x_EEG_miss, args.context_len)
        x_EOG_miss = reshape_epochs_to_context(x_EOG_miss, args.context_len)

        y = y.reshape(n_seq, args.context_len)

        EEG_miss_labels = EEG_miss_labels.reshape(n_seq, args.context_len)
        EOG_miss_labels = EOG_miss_labels.reshape(n_seq, args.context_len)

        EEG_modal_labels = EEG_modal_labels.reshape(n_seq, args.context_len)
        EOG_modal_labels = EOG_modal_labels.reshape(n_seq, args.context_len)

        assert x_EEG.shape == (n_seq, args.context_len, 1, samples_per_epoch)
        assert x_EOG.shape == (n_seq, args.context_len, 1, samples_per_epoch)
        assert x_EEG_miss.shape == (n_seq, args.context_len, 1, samples_per_epoch)
        assert x_EOG_miss.shape == (n_seq, args.context_len, 1, samples_per_epoch)
        assert y.shape == (n_seq, args.context_len)

        filename = ntpath.basename(psg_fnames[i]).replace("-PSG.edf", ".npz")
        save_path = os.path.join(args.output_dir, filename)

        np.savez(
            save_path,
            x_EEG=x_EEG.astype(np.float32),
            x_EOG=x_EOG.astype(np.float32),
            x_EEG_miss=x_EEG_miss.astype(np.float32),
            x_EOG_miss=x_EOG_miss.astype(np.float32),
            y=y.astype(np.int64),
            EEG_miss_labels=EEG_miss_labels.astype(np.int64),
            EOG_miss_labels=EOG_miss_labels.astype(np.int64),
            EEG_modal_labels=EEG_modal_labels.astype(np.int64),
            EOG_modal_labels=EOG_modal_labels.astype(np.int64),
            fs=sampling_rate,
            ch_label_EEG=args.select_ch_eeg,
            ch_label_EOG=args.select_ch_eog,
            header_raw=h_raw,
            header_annotation=h_ann,
        )

        print("Saved:", save_path)
        print("x_EEG:", x_EEG.shape)
        print("x_EOG:", x_EOG.shape)
        print("x_EEG_miss:", x_EEG_miss.shape)
        print("x_EOG_miss:", x_EOG_miss.shape)
        print("y:", y.shape)
        print("EEG_miss_labels:", EEG_miss_labels.shape)
        print("EOG_miss_labels:", EOG_miss_labels.shape)
        print("EEG missing ratio:", 1.0 - EEG_miss_labels.mean())
        print("EOG missing ratio:", 1.0 - EOG_miss_labels.mean())
        print(
            "Total missing rate:",
            1.0
            - (EEG_miss_labels.sum() + EOG_miss_labels.sum())
            / (2.0 * EEG_miss_labels.size),
        )
        print("=======================================")


if __name__ == "__main__":
    main()