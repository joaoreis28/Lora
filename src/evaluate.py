

import os
import re
import argparse
import random
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from peft import PeftModel
from transformers import CLIPModel, CLIPProcessor

from dataset import CLASS_TEMPLATES




COLORS = {
    "finetuned": "#4C72B0",
    "baseline":  "#DD8452",
    "correct":   "#2CA02C",
    "wrong":     "#D62728",
    "distractor":"#AAAAAA",
}

CANONICAL_QUERIES = {cls: templates[0] for cls, templates in CLASS_TEMPLATES.items()}
OPERATOR_QUERIES  = {
    "Abuse":         "abuse",
    "Arrest":        "arrest",
    "Arson":         "arson",
    "Assault":       "assault",
    "Burglary":      "break in",
    "Explosion":     "explosion",
    "Fighting":      "people fighting",
    "RoadAccidents": "car crash",
    "Robbery":       "robbery",
    "Shooting":      "shooting",
    "Shoplifting":   "shoplifting",
    "Stealing":      "theft",
    "Vandalism":     "vandalism",
    "Normal":        "normal scene",
}

DISTRACTOR_LABEL = "__distractor__"



def extract_class(filename):
    match = re.search(r'video_([a-zA-Z]+)\d+', filename)
    if match:
        return match.group(1)
    parts = filename.split('_')
    if len(parts) > 1:
        candidate = ''.join([c for c in parts[1] if not c.isdigit()])
        if candidate:
            return candidate
    return "Unknown"


def load_finetuned(checkpoint_dir, base_model_name, device):
    processor = CLIPProcessor.from_pretrained(checkpoint_dir)
    base      = CLIPModel.from_pretrained(base_model_name)
    model     = PeftModel.from_pretrained(base, checkpoint_dir)
    model.to(device).eval()
    return model, processor


def load_baseline(base_model_name, device):
    processor = CLIPProcessor.from_pretrained(base_model_name)
    model     = CLIPModel.from_pretrained(base_model_name)
    model.to(device).eval()
    return model, processor


def encode_text(model, processor, text, device):
    inputs = processor(
        text=text, return_tensors="pt",
        padding=True, truncation=True, max_length=77
    )
    with torch.no_grad():
        emb = model.get_text_features(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
        )
        if not isinstance(emb, torch.Tensor):
            emb = emb.pooler_output
    return F.normalize(emb, dim=-1).cpu()


def build_embeddings(model, processor, image_paths_and_labels, device, batch_size=64):
    """
    Recebe lista de (filepath, label) e retorna (embeddings, filenames, labels).
    Label = DISTRACTOR_LABEL para imagens externas.
    """
    all_embeddings, all_filenames, all_labels = [], [], []

    for start in tqdm(range(0, len(image_paths_and_labels), batch_size),
                      desc="    Encodando imagens", leave=False):
        batch = image_paths_and_labels[start : start + batch_size]
        images, valid_files, valid_labels = [], [], []

        for fpath, label in batch:
            try:
                img = Image.open(fpath).convert("RGB")
                images.append(img)
                valid_files.append(os.path.basename(fpath))
                valid_labels.append(label)
            except Exception:
                pass

        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt", padding=True)
        with torch.no_grad():
            emb = model.get_image_features(pixel_values=inputs["pixel_values"].to(device))

            if not isinstance(emb, torch.Tensor):
                emb = emb.pooler_output
            emb = F.normalize(emb, dim=-1)

        all_embeddings.append(emb.cpu())
        all_filenames.extend(valid_files)
        all_labels.extend(valid_labels)

    return torch.cat(all_embeddings, dim=0), all_filenames, all_labels


def _extract_video_id(filename):
    """
    Extrai o identificador do vídeo a partir do nome do frame.
    Deve produzir o MESMO id usado em dataset.py para garantir split consistente.
    Ex: 'video_Abuse001_x264_scene0_frame0.jpg' -> 'video_Abuse001'
    """
    match = re.search(r'(video_[a-zA-Z]+\d+)', filename)
    if match:
        return match.group(1)
    return filename


def collect_ucf_paths(frames_dir, split="val", val_ratio=0.2, seed=42, max_frames=None):
    """
    Coleta (filepath, classe) de frames UCF-Crime, filtrando por split.

    CRÍTICO: usa a MESMA lógica de split determinístico do dataset.py
    (split por vídeo, seed=42, val_ratio=0.2) para garantir que a avaliação
    seja feita em frames NÃO VISTOS durante o treino.

    split:
        "val"   → apenas vídeos de validação (recomendado — avaliação honesta)
        "train" → apenas vídeos de treino (útil para medir memorização)
        "all"   → todos os frames (NÃO recomendado — contém data leakage)
    """
    valid_ext = ('.jpg', '.jpeg', '.png')
    all_files = sorted([
        f for f in os.listdir(frames_dir)
        if f.lower().endswith(valid_ext)
    ])

    unique_videos = sorted(set(_extract_video_id(f) for f in all_files))
    rng = random.Random(seed)
    rng.shuffle(unique_videos)

    val_size     = int(len(unique_videos) * val_ratio)
    val_videos   = set(unique_videos[:val_size])
    train_videos = set(unique_videos[val_size:])

    if split == "train":
        allowed = train_videos
    elif split == "val":
        allowed = val_videos
    elif split == "all":
        allowed = set(unique_videos)
    else:
        raise ValueError(f"split inválido: '{split}'. Use 'train', 'val' ou 'all'.")

    files = [f for f in all_files if _extract_video_id(f) in allowed]


    if max_frames and max_frames < len(files):
        files = random.sample(files, max_frames)

    return [(os.path.join(frames_dir, f), extract_class(f)) for f in files]


def collect_distractor_paths(distractors_dir, n_distractors):
    """
    Coleta até n_distractors imagens do diretório externo.
    Busca recursivamente em subpastas — compatível com a estrutura
    do COCO (flat) e OpenImages (com subpastas por categoria).
    """
    valid_ext = ('.jpg', '.jpeg', '.png')
    all_files = []

    for root, _, files in os.walk(distractors_dir):
        for f in files:
            if f.lower().endswith(valid_ext):
                all_files.append(os.path.join(root, f))

    if not all_files:
        raise FileNotFoundError(
            f"Nenhuma imagem encontrada em '{distractors_dir}'."
        )

    sampled = random.sample(all_files, min(n_distractors, len(all_files)))
    return [(f, DISTRACTOR_LABEL) for f in sampled]


def build_index(model, processor, ucf_paths, distractor_paths, device, batch_size):
    """
    Monta o índice completo: frames UCF-Crime + distratores externos.
    Retorna (embeddings, filenames, labels) com os distratores embaralhados.
    """
    all_paths = ucf_paths + distractor_paths
    random.shuffle(all_paths)
    return build_embeddings(model, processor, all_paths, device, batch_size)


def compute_retrieval_metrics(model, processor, embeddings, labels,
                               device, k_values=(1, 5, 10), query_style="canonical"):
    """
    Calcula Precision@K e Average Precision para cada classe usando sua
    query canônica (ou estilo operador) como texto de busca.

    Distratores ocupam rank no índice mas nunca contam como relevantes —
    eles só aumentam a dificuldade em open set.
    """
    class_names  = sorted(set(labels) - {"Unknown", DISTRACTOR_LABEL})
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    label_tensor = torch.tensor([class_to_idx.get(l, -1) for l in labels])

    query_map = OPERATOR_QUERIES if query_style == "operator" else CANONICAL_QUERIES

    per_class_metrics = defaultdict(lambda: defaultdict(float))

    for class_name in class_names:
        class_idx = class_to_idx[class_name]
        relevant_mask = (label_tensor == class_idx)
        n_relevant = relevant_mask.sum().item()
        if n_relevant == 0:
            continue

        query_text = query_map.get(class_name, CLASS_TEMPLATES.get(class_name, [""])[0])
        text_emb   = encode_text(model, processor, query_text, device).squeeze(0)  # [D]

        sims = embeddings @ text_emb  # [N]

        sorted_indices = torch.argsort(sims, descending=True)
        sorted_labels  = label_tensor[sorted_indices]
        is_relevant    = (sorted_labels == class_idx).float()  

      
        for k in k_values:
            top_k_hits = is_relevant[:k].sum().item()
            per_class_metrics[class_name][f"P@{k}"] = top_k_hits / k

        cumulative_hits  = torch.cumsum(is_relevant, dim=0)           
        ranks            = torch.arange(1, len(sims) + 1).float()      
        precision_at_i   = cumulative_hits / ranks                     
        ap = (precision_at_i * is_relevant).sum().item() / n_relevant
        per_class_metrics[class_name]["AP"] = ap

    results = {}
    for k in k_values:
        scores = [per_class_metrics[c][f"P@{k}"] for c in class_names
                  if f"P@{k}" in per_class_metrics[c]]
        results[f"P@{k}"] = sum(scores) / len(scores) if scores else 0.0

    ap_scores = [per_class_metrics[c]["AP"] for c in class_names
                 if "AP" in per_class_metrics[c]]
    results["mAP"] = sum(ap_scores) / len(ap_scores) if ap_scores else 0.0

    return results, per_class_metrics


compute_recall_at_k = compute_retrieval_metrics




def build_confusion_matrix(model, processor, embeddings, labels, device, k=5):
    """
    Para cada classe (query), conta a distribuição de classes nos top-K.
    Distratores são agrupados numa coluna separada para mostrar
    o quanto o modelo "se perde" com ruído externo.
    """
    class_names  = sorted(set(labels) - {"Unknown", DISTRACTOR_LABEL})
    has_distractors = DISTRACTOR_LABEL in labels

    col_names    = class_names + ([DISTRACTOR_LABEL] if has_distractors else [])
    class_to_idx = {c: i for i, c in enumerate(col_names)}
    label_tensor = torch.tensor([class_to_idx.get(l, -1) for l in labels])

    n_rows = len(class_names)
    n_cols = len(col_names)
    confusion = np.zeros((n_rows, n_cols), dtype=np.float32)

    for i, class_name in enumerate(class_names):
        query_text = CANONICAL_QUERIES.get(class_name, class_name)
        text_emb   = encode_text(model, processor, query_text, device)
        sims       = (text_emb @ embeddings.T).squeeze(0)

        top_k_indices = sims.topk(min(k, len(sims))).indices
        for lbl in label_tensor[top_k_indices].tolist():
            if lbl >= 0:
                confusion[i, lbl] += 1

        row_sum = confusion[i].sum()
        if row_sum > 0:
            confusion[i] /= row_sum

    return confusion, class_names, col_names


def plot_confusion_matrix(confusion, row_names, col_names, title, output_path):
    display_cols = [
        "distrators" if c == DISTRACTOR_LABEL else c
        for c in col_names
    ]

    fig, ax = plt.subplots(figsize=(max(12, len(col_names)), max(10, len(row_names) * 0.8)))
    sns.heatmap(
        confusion,
        xticklabels=display_cols,
        yticklabels=row_names,
        annot=True, fmt=".2f",
        cmap="Blues", linewidths=0.5,
        ax=ax, vmin=0, vmax=1,
        annot_kws={"size": 7},
    )

    if DISTRACTOR_LABEL in col_names:
        dist_col_idx = col_names.index(DISTRACTOR_LABEL)
        ax.add_patch(plt.Rectangle(
            (dist_col_idx, 0), 1, len(row_names),
            fill=False, edgecolor="red", lw=2, clip_on=False
        ))

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Classe Recuperada (Top-K)", fontsize=10)
    ax.set_ylabel("Query (Classe Real)", fontsize=10)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', rotation=0,  labelsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


QUERIES_TO_SHOW = {
    "Fighting":      "people fighting",
    "Arson":         "person setting fire",
    "RoadAccidents": "car crash",
    "Normal":        "normal scene",
    "Robbery":       "armed robbery",
    "Shoplifting":   "shoplifting",
}


def plot_topk_results(model, processor, embeddings, filenames, labels,
                       frames_dir, distractors_dir, device, output_dir,
                       top_k=5):
    """
    Plota os top-K frames para queries representativas.
    Distratores recuperados aparecem com borda cinza e label "DISTRATOR".
    """
    os.makedirs(output_dir, exist_ok=True)

    # Mapa de filename → filepath para exibição
    filepath_map = {}
    if frames_dir:
        for f in os.listdir(frames_dir):
            filepath_map[f] = os.path.join(frames_dir, f)
    if distractors_dir:
        for root, _, files in os.walk(distractors_dir):
            for f in files:
                filepath_map[f] = os.path.join(root, f)

    for target_class, query_text in QUERIES_TO_SHOW.items():
        text_emb    = encode_text(model, processor, query_text, device)
        sims        = (text_emb @ embeddings.T).squeeze(0)
        top_indices = sims.topk(min(top_k, len(sims))).indices.tolist()
        top_scores  = sims[top_indices]

        fig, axes = plt.subplots(1, top_k, figsize=(3.5 * top_k, 4.5))
        fig.suptitle(
            f'Query: "{query_text}"   (esperado: {target_class})',
            fontsize=12, fontweight="bold", y=1.01
        )

        for rank, (ax, idx, score) in enumerate(zip(axes, top_indices, top_scores)):
            fname      = filenames[idx]
            pred_label = labels[idx]
            is_distractor = (pred_label == DISTRACTOR_LABEL)
            is_correct    = (pred_label == target_class)

            fpath = filepath_map.get(fname)
            if fpath and os.path.exists(fpath):
                try:
                    ax.imshow(Image.open(fpath).convert("RGB"))
                except Exception:
                    ax.set_facecolor("#eeeeee")
            else:
                ax.set_facecolor("#eeeeee")

            if is_distractor:
                border_color = COLORS["distractor"]
                marker, label_text = "⬜", "DISTRATOR"
            elif is_correct:
                border_color = COLORS["correct"]
                marker, label_text = "✅", pred_label
            else:
                border_color = COLORS["wrong"]
                marker, label_text = "❌", pred_label

            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(4)

            ax.set_title(
                f"#{rank+1}  {score:.3f}\n{marker} {label_text}",
                fontsize=8.5, color=border_color
            )
            ax.axis("off")

        plt.tight_layout()
        out_path = os.path.join(output_dir, f"topk_{target_class.lower()}.png")
        plt.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close()



def plot_recall_comparison(results_ft, results_bl, title, output_path):
    k_labels  = list(results_ft.keys())
    ft_values = [results_ft[k] for k in k_labels]
    bl_values = [results_bl[k] for k in k_labels]

    x, width = np.arange(len(k_labels)), 0.35
    fig, ax  = plt.subplots(figsize=(7, 5))

    bars_ft = ax.bar(x - width/2, ft_values, width,
                     label="CLIP + LoRA (fine-tuned)", color=COLORS["finetuned"], alpha=0.9)
    bars_bl = ax.bar(x + width/2, bl_values, width,
                     label="CLIP base (zero-shot)",    color=COLORS["baseline"],  alpha=0.9)

    for bar in list(bars_ft) + list(bars_bl):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f"{h:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(k_labels, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score (macro-average entre classes)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_per_class_recall(per_class_ft, per_class_bl, k, output_path, metric="P"):
    """
    metric: "P" para Precision@k, "AP" para Average Precision (ignora k).
    """
    classes = sorted(per_class_ft.keys())
    key = f"{metric}@{k}" if metric == "P" else "AP"

    ft_vals = [per_class_ft[c].get(key, 0) for c in classes]
    bl_vals = [per_class_bl[c].get(key, 0) for c in classes]

    fig, ax = plt.subplots(figsize=(8, max(5, len(classes) * 0.55)))
    y, h    = np.arange(len(classes)), 0.35

    ax.barh(y + h/2, ft_vals, h, label="Fine-tuned", color=COLORS["finetuned"], alpha=0.9)
    ax.barh(y - h/2, bl_vals, h, label="Baseline",   color=COLORS["baseline"],  alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel(key, fontsize=11)
    ax.set_title(f"{key} por Classe — Fine-tuned vs Baseline", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_closed_vs_open(closed_ft, closed_bl, open_ft, open_bl, output_path):
    """
    Compara as métricas em closed set vs open set num mesmo gráfico.
    Deixa evidente o impacto dos distratores na performance.
    """
    k_labels = list(closed_ft.keys())
    x = np.arange(len(k_labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(x - 1.5*width, [closed_ft[k] for k in k_labels], width,
           label="Fine-tuned — Closed Set", color=COLORS["finetuned"], alpha=0.9)
    ax.bar(x - 0.5*width, [open_ft[k]   for k in k_labels], width,
           label="Fine-tuned — Open Set",   color=COLORS["finetuned"], alpha=0.5, hatch="//")
    ax.bar(x + 0.5*width, [closed_bl[k] for k in k_labels], width,
           label="Baseline — Closed Set",   color=COLORS["baseline"],  alpha=0.9)
    ax.bar(x + 1.5*width, [open_bl[k]   for k in k_labels], width,
           label="Baseline — Open Set",     color=COLORS["baseline"],  alpha=0.5, hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels(k_labels, fontsize=11)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Recall médio", fontsize=11)
    ax.set_title("Closed Set vs Open Set — Impacto dos Distratores",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
  
def print_and_save_report(
    closed_ft, closed_bl, open_ft, open_bl,
    per_class_closed_ft, per_class_closed_bl,
    per_class_open_ft,   per_class_open_bl,
    n_ucf, n_distractors, output_path, split="val"
):
    lines = []
    def log(s=""):
        print(s)
        lines.append(s)


    section_label = "Open Set" if has_distractors else "Closed Set"
    per_class_ft_to_use = per_class_open_ft if has_distractors else per_class_closed_ft
    per_class_bl_to_use = per_class_open_bl if has_distractors else per_class_closed_bl

   


    with open(output_path, "w") as f:
        f.write("\n".join(lines))
   


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    default="./lora_checkpoints/best")
    p.add_argument("--base_model",    default="openai/clip-vit-base-patch32")
    p.add_argument("--frames_dir",    required=True,
                   help="Diretório com os frames UCF-Crime de teste")
    p.add_argument("--distractors",   default=None,
                   help="Diretório com imagens externas (COCO, OpenImages, etc.)")
    p.add_argument("--n_distractors", type=int, default=5000,
                   help="Quantas imagens distratoras incluir no índice open set")
    p.add_argument("--output_dir",    default="./evaluation_results")
    p.add_argument("--top_k",         type=int, default=5)
    p.add_argument("--batch_size",    type=int, default=64)
    p.add_argument("--max_frames",    type=int, default=None,
                   help="Limita frames UCF-Crime (útil para testes rápidos)")
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--split",         type=str, default="val",
                   choices=["train", "val", "all"],
                   help="Qual split usar para avaliação. 'val' (padrão) usa só "
                        "frames NÃO VISTOS no treino — avaliação honesta. "
                        "'all' inclui treino — apenas para comparação.")
    p.add_argument("--val_ratio",     type=float, default=0.2,
                   help="Proporção de vídeos no split de validação "
                        "(deve coincidir com o usado no train.py)")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    open_set = args.distractors is not None
  
    model_ft, proc_ft = load_finetuned(args.checkpoint, args.base_model, device)
    model_bl, proc_bl = load_baseline(args.base_model, device)

    ucf_paths = collect_ucf_paths(
        args.frames_dir,
        split=args.split,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_frames=args.max_frames,
    )
    print(f"  📹 {len(ucf_paths)} frames UCF-Crime no split '{args.split}'")

    distractor_paths = []
    if open_set:
        distractor_paths = collect_distractor_paths(args.distractors, args.n_distractors)

    closed_paths = ucf_paths
    open_paths   = ucf_paths + distractor_paths


    emb_ft_c, fnames_c, labels_c = build_index(
        model_ft, proc_ft, closed_paths, [], device, args.batch_size
    )
    if open_set:
        print("  [Open Set]")
        emb_ft_o, fnames_o, labels_o = build_index(
            model_ft, proc_ft, ucf_paths, distractor_paths, device, args.batch_size
        )


    emb_bl_c, _, _ = build_index(
        model_bl, proc_bl, closed_paths, [], device, args.batch_size
    )
    if open_set:
        print("  [Open Set]")
        emb_bl_o, _, _ = build_index(
            model_bl, proc_bl, ucf_paths, distractor_paths, device, args.batch_size
        )



    print("  Closed Set — Fine-tuned")
    res_ft_c, pc_ft_c = compute_recall_at_k(model_ft, proc_ft, emb_ft_c, labels_c, device)
    print("  Closed Set — Baseline")
    res_bl_c, pc_bl_c = compute_recall_at_k(model_bl, proc_bl, emb_bl_c, labels_c, device)

    res_ft_o = res_bl_o = pc_ft_o = pc_bl_o = {}
    if open_set:
        print("  Open Set — Fine-tuned")
        res_ft_o, pc_ft_o = compute_recall_at_k(model_ft, proc_ft, emb_ft_o, labels_o, device)
        print("  Open Set — Baseline")
        res_bl_o, pc_bl_o = compute_recall_at_k(model_bl, proc_bl, emb_bl_o, labels_o, device)


    print_and_save_report(
        res_ft_c, res_bl_c,
        res_ft_o if open_set else res_ft_c,
        res_bl_o if open_set else res_bl_c,
        pc_ft_c,  pc_bl_c,
        pc_ft_o if open_set else pc_ft_c,
        pc_bl_o if open_set else pc_bl_c,
        n_ucf=len(ucf_paths),
        n_distractors=len(distractor_paths),
        output_path=os.path.join(args.output_dir, "report.txt"),
        split=args.split,
    )



    plot_recall_comparison(
        res_ft_c, res_bl_c,
        title="Métricas de Retrieval — Closed Set",
        output_path=os.path.join(args.output_dir, "metrics_closed_set.png")
    )
    plot_per_class_recall(pc_ft_c, pc_bl_c, k=5, metric="P",
        output_path=os.path.join(args.output_dir, "per_class_p5_closed.png"))
    plot_per_class_recall(pc_ft_c, pc_bl_c, k=None, metric="AP",
        output_path=os.path.join(args.output_dir, "per_class_ap_closed.png"))

    if open_set:
        plot_recall_comparison(
            res_ft_o, res_bl_o,
            title="Métricas de Retrieval — Open Set (com distratores)",
            output_path=os.path.join(args.output_dir, "metrics_open_set.png")
        )
        plot_per_class_recall(pc_ft_o, pc_bl_o, k=5, metric="P",
            output_path=os.path.join(args.output_dir, "per_class_p5_open.png"))
        plot_per_class_recall(pc_ft_o, pc_bl_o, k=None, metric="AP",
            output_path=os.path.join(args.output_dir, "per_class_ap_open.png"))
        plot_closed_vs_open(
            res_ft_c, res_bl_c, res_ft_o, res_bl_o,
            output_path=os.path.join(args.output_dir, "closed_vs_open.png")
        )

    for tag, model, proc, emb, lbs in [
        ("closed_finetuned", model_ft, proc_ft, emb_ft_c, labels_c),
        ("closed_baseline",  model_bl, proc_bl, emb_bl_c, labels_c),
        *([
            ("open_finetuned", model_ft, proc_ft, emb_ft_o, labels_o),
            ("open_baseline",  model_bl, proc_bl, emb_bl_o, labels_o),
        ] if open_set else [])
    ]:
        conf, row_names, col_names = build_confusion_matrix(
            model, proc, emb, lbs, device, k=args.top_k
        )
        title = f"Confusão — {tag.replace('_', ' ').title()} (Top-{args.top_k})"
        plot_confusion_matrix(
            conf, row_names, col_names, title,
            output_path=os.path.join(args.output_dir, f"confusion_{tag}.png")
        )

    for tag, model, proc, emb, fnames, lbs in [
        ("finetuned_closed", model_ft, proc_ft, emb_ft_c, fnames_c, labels_c),
        ("baseline_closed",  model_bl, proc_bl, emb_bl_c, fnames_c, labels_c),
        *([
            ("finetuned_open", model_ft, proc_ft, emb_ft_o, fnames_o, labels_o),
            ("baseline_open",  model_bl, proc_bl, emb_bl_o, fnames_o, labels_o),
        ] if open_set else [])
    ]:
        plot_topk_results(
            model, proc, emb, fnames, lbs,
            frames_dir=args.frames_dir,
            distractors_dir=args.distractors,
            device=device,
            output_dir=os.path.join(args.output_dir, f"topk_{tag}"),
            top_k=args.top_k,
        )

 


if __name__ == "__main__":
    main()