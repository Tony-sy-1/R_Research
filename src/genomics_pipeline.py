from __future__ import annotations
import os
import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cyvcf2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, Dataset

GENO_CHARS = ["H", "M", "L", "X"]
LD_CHARS = ["Y", "J", "N"]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def calculate_ld_labels(df_with_chrom: pd.DataFrame) -> np.ndarray:
    sample_cols = [c for c in df_with_chrom.columns if c != "chrom"]
    values = np.select(
        [
            df_with_chrom[sample_cols].values == "H",
            df_with_chrom[sample_cols].values == "M",
            df_with_chrom[sample_cols].values == "L",
            df_with_chrom[sample_cols].values == "X",
        ],
        [0, 1, 2, -9],
        default=-9,
    )
    n_snps = values.shape[0]
    if n_snps < 2:
        return np.array(["N"] * n_snps)

    r2 = np.full(n_snps, -1.0)
    for i in range(n_snps - 1):
        if df_with_chrom["chrom"].iloc[i] != df_with_chrom["chrom"].iloc[i + 1]:
            continue
        a, b = values[i], values[i + 1]
        m = (a != -9) & (b != -9)
        if m.sum() < 2:
            continue
        av, bv = a[m], b[m]
        r2[i] = 0.0 if np.var(av) == 0 or np.var(bv) == 0 else np.corrcoef(av, bv)[0, 1] ** 2
    return np.where(r2 == -1.0, "N", np.where(r2 >= 0.8, "Y", "J"))


def vcf_to_encoded_sequences(
    vcf_path: str,
    output_sequence_file: str,
    min_maf: float = 0.05,
    max_missing: float = 0.1,
) -> Dict[str, object]:
    vcf = cyvcf2.VCF(vcf_path)
    samples = vcf.samples
    if len(samples) == 0:
        raise ValueError("VCF 中没有样本")

    rows = []
    stats = {"records_scanned": 0, "retained_snps": 0, "filtered_non_snp": 0, "filtered_missing": 0, "filtered_maf": 0}
    for rec in vcf:
        stats["records_scanned"] += 1
        if not rec.is_snp or len(rec.ALT) > 1:
            stats["filtered_non_snp"] += 1
            continue

        gt = rec.gt_types
        if np.mean(gt == 3) > max_missing:
            stats["filtered_missing"] += 1
            continue
        aaf = rec.aaf
        if aaf is None or min(aaf, 1 - aaf) < min_maf:
            stats["filtered_maf"] += 1
            continue

        ref, alt = rec.REF, rec.ALT[0]
        major, minor = (alt, ref) if aaf > 0.5 else (ref, alt)
        hml = []
        for i, g in enumerate(rec.genotypes):
            if gt[i] == 3:
                hml.append("X")
                continue
            a1 = ref if g[0] == 0 else alt if g[0] == 1 else None
            a2 = ref if g[1] == 0 else alt if g[1] == 1 else None
            if a1 is None or a2 is None:
                hml.append("X")
            elif a1 == major and a2 == major:
                hml.append("H")
            elif a1 == minor and a2 == minor:
                hml.append("L")
            else:
                hml.append("M")

        row = {"chrom": rec.CHROM, "pos": rec.POS}
        row.update({s: g for s, g in zip(samples, hml)})
        rows.append(row)

    vcf.close()
    if not rows:
        raise ValueError("过滤后无 SNP")

    stats["retained_snps"] = len(rows)
    df = pd.DataFrame(rows).sort_values(["chrom", "pos"]).reset_index(drop=True)
    ld = calculate_ld_labels(df[["chrom"] + samples])

    os.makedirs(os.path.dirname(output_sequence_file) or ".", exist_ok=True)
    with open(output_sequence_file, "w", encoding="utf-8") as f:
        for sid in samples:
            f.write("".join([f"{g}{l}" for g, l in zip(df[sid], ld)]) + "\n")

    sid_file = os.path.splitext(output_sequence_file)[0] + "_sample_ids.txt"
    with open(sid_file, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(s + "\n")

    stats_path = os.path.splitext(output_sequence_file)[0] + "_stats.csv"
    pd.DataFrame([stats]).to_csv(stats_path, index=False)
    return {"sequence_file": output_sequence_file, "sample_ids_file": sid_file, "stats_file": stats_path, "stats": stats}


class SequenceDataset(Dataset):
    def __init__(self, genome_file: str, phenotype_file: str, split: str = "train", seed: int = 42, max_seq_len: int = 256):
        self.max_seq_len = max_seq_len
        self.vocab = {"[PAD]": 0, "[CLS]": 1}
        idx = 2
        for g in GENO_CHARS:
            for l in LD_CHARS:
                self.vocab[g + l] = idx
                idx += 1
        self.pad_id = self.vocab["[PAD]"]
        self.cls_id = self.vocab["[CLS]"]

        with open(genome_file, encoding="utf-8") as f:
            self.sequences = [line.strip() for line in f]

        pheno = pd.read_csv(phenotype_file, sep=None, engine="python", header=None)
        vals = pheno.iloc[:, -1].astype(float).values
        if len(self.sequences) != len(vals):
            raise ValueError(f"基因组样本数({len(self.sequences)})与表型样本数({len(vals)})不一致")

        all_idx = list(range(len(vals)))
        random.Random(seed).shuffle(all_idx)
        cut = int(0.8 * len(vals))
        self.indices = all_idx[:cut] if split == "train" else all_idx[cut:]
        self.y = RobustScaler().fit_transform(vals[self.indices].reshape(-1, 1)).flatten().astype(np.float32)

    def __len__(self):
        return len(self.indices)

    def _tokenize(self, seq: str) -> List[int]:
        return [self.vocab.get(seq[i:i + 2], self.pad_id) for i in range(0, len(seq), 2) if i + 1 < len(seq)]

    def __getitem__(self, idx: int):
        tokens = self._tokenize(self.sequences[self.indices[idx]])
        x = [self.cls_id] + tokens[: self.max_seq_len - 1]
        x += [self.pad_id] * (self.max_seq_len - len(x))
        m = [1 if t != self.pad_id else 0 for t in x]
        return torch.tensor(x), torch.tensor(m), torch.tensor(self.y[idx], dtype=torch.float32)


class SNPRegressor(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 128, num_layers: int = 2, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden_size)
        enc_layer = nn.TransformerEncoderLayer(hidden_size, nhead=nhead, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.scorer = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Linear(hidden_size // 2, 1))
        self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_size, 1))

    def forward(self, x, mask):
        h = self.emb(x)
        h = self.encoder(h, src_key_padding_mask=(mask == 0))
        s = self.scorer(h).squeeze(-1).masked_fill(mask == 0, -1e9)
        w = torch.softmax(s, dim=1).unsqueeze(-1)
        z = (h * w).sum(dim=1)
        return self.head(z).squeeze(-1), s


def evaluate_model(model: nn.Module, dataloader: DataLoader) -> Dict[str, object]:
    model.eval()
    ys, ps, snp_positions = [], [], []
    with torch.no_grad():
        for x, m, y in dataloader:
            pred, score = model(x, m)
            ys.extend(y.numpy())
            ps.extend(pred.numpy())
            snp_positions.append(torch.topk(score[:, 1:], k=min(50, score.shape[1] - 1), dim=1).indices.numpy())
    ys_arr, ps_arr = np.array(ys), np.array(ps)
    corr = float(np.corrcoef(ps_arr, ys_arr)[0, 1]) if len(ps_arr) > 1 else 0.0
    rmse = float(np.sqrt(np.mean((ps_arr - ys_arr) ** 2))) if len(ps_arr) > 0 else 0.0
    return {"y_true": ys_arr, "y_pred": ps_arr, "corr": corr, "rmse": rmse, "top_positions": np.concatenate(snp_positions, axis=0)}


def train_finetune(
    genome_file: str,
    phenotype_file: str,
    out_dir: str,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-3,
    hidden_size: int = 128,
    num_layers: int = 2,
    nhead: int = 4,
    dropout: float = 0.1,
    max_seq_len: int = 256,
    seed: int = 42,
) -> Dict[str, object]:
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    train_ds = SequenceDataset(genome_file, phenotype_file, split="train", seed=seed, max_seq_len=max_seq_len)
    val_ds = SequenceDataset(genome_file, phenotype_file, split="val", seed=seed, max_seq_len=max_seq_len)
    tr = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    va = DataLoader(val_ds, batch_size=batch_size)

    model = SNPRegressor(len(train_ds.vocab), hidden_size=hidden_size, num_layers=num_layers, nhead=nhead, dropout=dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    best_corr = -1e9
    history = []
    best_eval = None
    for ep in range(1, epochs + 1):
        model.train()
        batch_losses = []
        for x, m, y in tr:
            pred, _ = model(x, m)
            loss = F.mse_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            batch_losses.append(loss.item())

        eval_res = evaluate_model(model, va)
        history.append({"epoch": ep, "train_loss": float(np.mean(batch_losses)), "corr": eval_res["corr"], "rmse": eval_res["rmse"]})
        if eval_res["corr"] > best_corr:
            best_corr = eval_res["corr"]
            best_eval = eval_res
            torch.save(model.state_dict(), os.path.join(out_dir, "best_finetuned_model.pt"))
            np.save(os.path.join(out_dir, "selected_top_positions.npy"), eval_res["top_positions"])

    history_path = os.path.join(out_dir, "train_history.csv")
    pd.DataFrame(history).to_csv(history_path, index=False)

    pred_path = os.path.join(out_dir, "val_predictions.csv")
    if best_eval is not None:
        pd.DataFrame({"y_true": best_eval["y_true"], "y_pred": best_eval["y_pred"]}).to_csv(pred_path, index=False)

    return {
        "history": history,
        "history_file": history_path,
        "pred_file": pred_path,
        "snp_file": os.path.join(out_dir, "selected_top_positions.npy"),
        "best_corr": best_corr,
    }


def summarize_selected_snps(npy_path: str, top_n: int = 100) -> pd.DataFrame:
    arr = np.load(npy_path)
    counts = Counter(arr.flatten().tolist())
    df = pd.DataFrame(sorted(counts.items(), key=lambda x: x[1], reverse=True), columns=["snp_pos", "freq"])
    return df.head(top_n)
