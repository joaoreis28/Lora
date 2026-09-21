

import os
import re
import json
import random
from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict



CLASS_TEMPLATES = {
    "Abuse": [
        "security footage showing physical abuse",
        "surveillance camera recording an abuse incident",
        "abuse",
        "person being abused",
        "violence against a person",
    ],
    "Arrest": [
        "police arrest captured on security camera",
        "surveillance footage of a person being arrested",
        "arrest",
        "police detaining someone",
        "person being handcuffed",
    ],
    "Arson": [
        "security camera footage of a fire being set deliberately",
        "surveillance video showing arson in progress",
        "arson",
        "person setting fire",
        "deliberate fire",
    ],
    "Assault": [
        "security footage of a physical assault",
        "surveillance camera recording a violent assault",
        "assault",
        "physical attack",
        "person being attacked",
    ],
    "Burglary": [
        "security camera footage of a burglary",
        "surveillance video showing someone breaking in",
        "burglary",
        "break in",
        "intruder entering building",
    ],
    "Explosion": [
        "security camera footage of an explosion",
        "surveillance video recording an explosion event",
        "explosion",
        "blast caught on camera",
        "building exploding",
    ],
    "Fighting": [
        "security footage of people fighting",
        "surveillance camera recording a fight",
        "fight",
        "people fighting",
        "physical altercation",
        "brawl",
    ],
    "RoadAccidents": [
        "security camera footage of a road accident",
        "surveillance video of a traffic collision",
        "road accident",
        "car crash",
        "traffic collision on camera",
        "vehicle accident",
    ],
    "Robbery": [
        "security footage of a robbery taking place",
        "surveillance camera recording an armed robbery",
        "robbery",
        "armed robbery",
        "person being robbed",
    ],
    "Shooting": [
        "security camera footage of a shooting incident",
        "surveillance video showing gunfire",
        "shooting",
        "gunshot on camera",
        "person shot",
    ],
    "Shoplifting": [
        "security footage of shoplifting in a store",
        "surveillance camera recording someone stealing merchandise",
        "shoplifting",
        "person stealing from store",
        "theft in shop",
    ],
    "Stealing": [
        "security camera footage of theft",
        "surveillance video showing someone stealing",
        "stealing",
        "theft",
        "person taking something",
    ],
    "Vandalism": [
        "security footage of vandalism",
        "surveillance camera recording property damage",
        "vandalism",
        "property damage",
        "person damaging property",
    ],
    "Normal": [
        "a normal security camera scene with no incident",
        "routine surveillance footage showing everyday activity",
        "normal scene",
        "no incident",
        "empty area",
        "people walking normally",
    ],
}

_DEFAULT_TEMPLATE = "security camera footage of {class_name}"


def get_split_videos(unique_videos, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Divide a lista de vídeos em três conjuntos de forma determinística.
    Mesma seed → mesmos conjuntos sempre.

    Retorna: (train_videos, val_videos, test_videos) como sets.
    """
    videos = sorted(unique_videos) 
    rng = random.Random(seed)
    rng.shuffle(videos)

    n = len(videos)
    n_val   = int(n * val_ratio)
    n_test  = int(n * test_ratio)
    n_train = n - n_val - n_test

    train_videos = set(videos[:n_train])
    val_videos   = set(videos[n_train : n_train + n_val])
    test_videos  = set(videos[n_train + n_val :])

    return train_videos, val_videos, test_videos


class UCFCrimeDataset(Dataset):
    def __init__(self, root_dir, processor,
                 split="train", val_ratio=0.15, test_ratio=0.15, seed=42,
                 transform=None,
                 caption_mode="templates", captions_file=None):
        """
        :param split        : 'train' (70%), 'val' (15%), 'test' (15%), ou 'all'.
        :param val_ratio    : proporção do conjunto de validação.
        :param test_ratio   : proporção do conjunto de teste.
        :param seed         : semente para split determinístico.
        :param caption_mode : "templates" (sorteia da CLASS_TEMPLATES) ou
                              "captions" (usa captions.json gerado por VLM).
        :param captions_file: caminho para captions.json, obrigatório se mode="captions".
        """
        self.root_dir     = root_dir
        self.processor    = processor
        self.transform    = transform
        self.split        = split
        self.caption_mode = caption_mode

        self.captions = None
        if caption_mode == "captions":
            if not captions_file or not os.path.exists(captions_file):
                raise FileNotFoundError(
                    f"caption_mode='captions' exige captions_file válido. "
                    f"Recebido: {captions_file}"
                )
            with open(captions_file, "r", encoding="utf-8") as f:
                self.captions = json.load(f)
            print(f"[Dataset] Captions carregados: {len(self.captions)} entradas")
        elif caption_mode != "templates":
            raise ValueError(f"caption_mode inválido: '{caption_mode}'. "
                             f"Use 'templates' ou 'captions'.")

        valid_extensions = ('.jpg', '.jpeg', '.png')
        all_frames = [
            f for f in os.listdir(root_dir)
            if f.lower().endswith(valid_extensions)
        ]

        if self.captions is not None:
            missing = [f for f in all_frames if f not in self.captions]
            if missing:
                print(f"[Dataset] ⚠️  {len(missing)} frames sem caption — serão ignorados.")
            all_frames = [f for f in all_frames if f in self.captions]

        video_to_frames = defaultdict(list)
        for f in all_frames:
            vid_id = self._extract_video_id(f)
            video_to_frames[vid_id].append(f)

        train_vids, val_vids, test_vids = get_split_videos(
            list(video_to_frames.keys()),
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )

        if split == "train":
            allowed = train_vids
        elif split == "val":
            allowed = val_vids
        elif split == "test":
            allowed = test_vids
        elif split == "all":
            allowed = train_vids | val_vids | test_vids
        else:
            raise ValueError(f"split inválido: '{split}'. "
                             f"Use 'train', 'val', 'test' ou 'all'.")

        self.frames = [f for f in all_frames if self._extract_video_id(f) in allowed]

        print(f"[Dataset] Modo='{caption_mode}' | Split='{split}' | seed={seed} | "
              f"{len(allowed)} vídeos | {len(self.frames)} frames")

    def _extract_video_id(self, filename):
        match = re.search(r'(video_[a-zA-Z]+\d+)', filename)
        return match.group(1) if match else filename

    def _extract_class(self, filename):
        match = re.search(r'video_([a-zA-Z]+)\d+', filename)
        if match:
            return match.group(1)
        parts = filename.split('_')
        if len(parts) > 1:
            candidate = ''.join([c for c in parts[1] if not c.isdigit()])
            if candidate:
                return candidate
        return "Unknown"

    def _get_text(self, filename, class_name):
        if self.caption_mode == "captions":
            return self.captions[filename]
        else:
            templates = CLASS_TEMPLATES.get(class_name)
            if templates:
                return random.choice(templates)
            return _DEFAULT_TEMPLATE.format(class_name=class_name)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx, _attempts=0):
        if _attempts >= 5:
            raise RuntimeError(
                f"Falha ao carregar 5 frames consecutivos a partir de {idx}."
            )

        filename = self.frames[idx]
        img_path = os.path.join(self.root_dir, filename)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Erro carregando '{img_path}': {e}")
            return self.__getitem__((idx + 1) % len(self), _attempts + 1)

        if self.transform:
            image = self.transform(image)

        class_name = self._extract_class(filename)
        text       = self._get_text(filename, class_name)

        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77
        )

        item = {k: v.squeeze(0) for k, v in inputs.items()}
        item["class_name"] = class_name
        return item