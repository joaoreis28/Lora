import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from torchvision import transforms

from dataset_captions import UCFCrimeDataset, CLASS_TEMPLATES
from lora_setup import get_clip_lora_model


def set_seed(seed):
    random_state = np.random.RandomState(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    random.seed(seed)
    return random_state


def compute_recall_at_k(model, val_dataset, processor, device,
                         k_values=(1, 5, 10), max_samples=512):
    model.eval()
    indices = torch.randperm(len(val_dataset))[:max_samples].tolist()

    class_names  = sorted(CLASS_TEMPLATES.keys())
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    all_image_embeddings = []
    all_labels = []

    with torch.no_grad():
        for idx in tqdm(indices, desc="  [R@K] Encodando imagens", leave=False):
            item = val_dataset[idx]
            pixel_values = item["pixel_values"].unsqueeze(0).to(device)

            image_outputs = model.get_image_features(pixel_values=pixel_values)
            image_emb = image_outputs if isinstance(image_outputs, torch.Tensor) \
                        else image_outputs.pooler_output
            image_emb = F.normalize(image_emb, dim=-1)

            all_image_embeddings.append(image_emb.cpu())
            all_labels.append(class_to_idx.get(item["class_name"], -1))

    image_embeddings = torch.cat(all_image_embeddings, dim=0)
    labels = torch.tensor(all_labels)

    text_embeddings_by_class = {}
    for class_name, class_idx in class_to_idx.items():
        query_text = CLASS_TEMPLATES[class_name][0]
        inputs = processor(text=query_text, return_tensors="pt",
                           padding=True, truncation=True, max_length=77)
        with torch.no_grad():
            text_outputs = model.get_text_features(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
            )
            text_emb = text_outputs if isinstance(text_outputs, torch.Tensor) \
                       else text_outputs.pooler_output
            text_emb = F.normalize(text_emb, dim=-1).cpu()
        text_embeddings_by_class[class_idx] = text_emb.squeeze(0)

    recalls = defaultdict(list)
    for img_idx in range(len(labels)):
        img_label = labels[img_idx].item()
        if img_label < 0:
            continue
        text_emb = text_embeddings_by_class[img_label]
        sims = image_embeddings @ text_emb
        rank = (sims > sims[img_idx]).sum().item() + 1
        for k in k_values:
            recalls[f"R@{k}"].append(1.0 if rank <= k else 0.0)

    return {key: sum(vals) / len(vals) for key, vals in recalls.items() if vals}


def run_epoch_train(model, loader, optimizer, lr_scheduler, scaler, device):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="  [Treino]")
    for batch in pbar:
        optimizer.zero_grad()
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values   = batch["pixel_values"].to(device)
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask,
                      pixel_values=pixel_values, return_loss=True)

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(**kwargs)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() == scale_before:
                lr_scheduler.step()
        else:
            outputs = model(**kwargs)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(loader)


def run_epoch_val(model, loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc="  [Val Loss]"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values   = batch["pixel_values"].to(device)
            kwargs = dict(input_ids=input_ids, attention_mask=attention_mask,
                          pixel_values=pixel_values, return_loss=True)
            if device == "cuda":
                with torch.amp.autocast('cuda'):
                    outputs = model(**kwargs)
            else:
                outputs = model(**kwargs)
            total_loss += outputs.loss.item()
    return total_loss / len(loader)


def verify_no_leakage(train_dataset, val_dataset, test_dataset):
    """Confirma que os três splits não compartilham vídeos."""
    train_vids = set(train_dataset._extract_video_id(f) for f in train_dataset.frames)
    val_vids   = set(val_dataset._extract_video_id(f)   for f in val_dataset.frames)
    test_vids  = set(test_dataset._extract_video_id(f)  for f in test_dataset.frames)

    overlaps = {
        "treino ∩ val":  train_vids & val_vids,
        "treino ∩ test": train_vids & test_vids,
        "val ∩ test":    val_vids & test_vids,
    }

    print(f"🔍 Sanity check de leakage:")
    print(f"   Vídeos treino={len(train_vids)} | val={len(val_vids)} | test={len(test_vids)}")
    for name, overlap in overlaps.items():
        if overlap:
            raise RuntimeError(f"LEAKAGE: {name} = {len(overlap)} vídeos!")
    print("  Sem leakage entre os três splits.\n")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--caption_mode", default="templates",
                   choices=["templates", "captions"])
    p.add_argument("--captions_file", default="./data/captions_clean.json")
    p.add_argument("--data_dir", default="./data/UCF-Crime")
    p.add_argument("--save_dir", required=True,
                   help="Diretório para salvar checkpoints (ex: ./checkpoints/templates_seed42)")
    p.add_argument("--seed",       type=int, default=42,
                   help="Seed do experimento. Use diferentes seeds para medir variância.")
    p.add_argument("--epochs",     type=int, default=15)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr",         type=float, default=5e-5)
    p.add_argument("--patience",   type=int, default=4)
    p.add_argument("--val_ratio",  type=float, default=0.15)
    p.add_argument("--test_ratio", type=float, default=0.15)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "openai/clip-vit-base-patch32"

    model, processor = get_clip_lora_model(model_name=model_name)
    model.to(device)

    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"⚠️ Diretório não encontrado: {args.data_dir}")

    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(degrees=5),
    ])

    dataset_kwargs = dict(
        root_dir=args.data_dir,
        processor=processor,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        caption_mode=args.caption_mode,
        captions_file=args.captions_file if args.caption_mode == "captions" else None,
    )

    train_dataset = UCFCrimeDataset(split="train", transform=train_transforms, **dataset_kwargs)
    val_dataset   = UCFCrimeDataset(split="val",   **dataset_kwargs)
    test_dataset  = UCFCrimeDataset(split="test",  **dataset_kwargs)

    verify_no_leakage(train_dataset, val_dataset, test_dataset)

    def collate_fn(batch):
        class_names = [item.pop("class_name") for item in batch]
        collated = torch.utils.data.dataloader.default_collate(batch)
        collated["class_name"] = class_names
        return collated

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=4, collate_fn=collate_fn)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=0.05)
    num_training_steps = args.epochs * len(train_loader)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )
    scaler = torch.amp.GradScaler('cuda') if device == "cuda" else None

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss     = float("inf")
    epochs_no_improve = 0

    for epoch in range(args.epochs):
 

        avg_train_loss = run_epoch_train(model, train_loader, optimizer,
                                         lr_scheduler, scaler, device)
        avg_val_loss = run_epoch_val(model, val_loader, device)
        recall_metrics = compute_recall_at_k(model, val_dataset, processor, device)


        recall_str = " | ".join(f"{k}: {v:.3f}" for k, v in recall_metrics.items())

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            model.save_pretrained(os.path.join(args.save_dir, "best"))
            processor.save_pretrained(os.path.join(args.save_dir, "best"))
            print(f"   Melhor modelo salvo (val_loss={best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"   Sem melhora há {epochs_no_improve}/{args.patience} épocas.")

        if epochs_no_improve >= args.patience:
            print(f"\nEarly stopping após {epoch+1} épocas.")
            break

 


if __name__ == "__main__":
    main()