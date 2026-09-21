import os
import re
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


class UCFCrimeDataset(Dataset):
    def __init__(self, root_dir, processor, split="train", val_ratio=0.2, seed=42, transform=None):
        """
        Custom Dataset for UCF-Crime frames — otimizado para fine-tuning
        voltado à busca semântica de imagem por texto, com separação por vídeo.

        :param root_dir: Caminho para a pasta de frames do UCF-Crime.
        :param processor: CLIPProcessor (Hugging Face).
        :param split: 'train', 'val', ou 'all'.
        :param val_ratio: Proporção de vídeos para validação.
        :param seed: Semente para garantir que a divisão seja determinística.
        :param transform: Transforms opcionais para as imagens.
        """
        self.root_dir = root_dir
        self.processor = processor
        self.transform = transform
        self.split = split

        valid_extensions = ('.jpg', '.jpeg', '.png')
        all_frames = [
            f for f in os.listdir(root_dir)
            if f.lower().endswith(valid_extensions)
        ]

        video_to_frames = defaultdict(list)
        for f in all_frames:
            vid_id = self._extract_video_id(f)
            video_to_frames[vid_id].append(f)

        unique_videos = sorted(list(video_to_frames.keys()))

        rng = random.Random(seed)
        rng.shuffle(unique_videos)

        val_size = int(len(unique_videos) * val_ratio)
        val_videos = set(unique_videos[:val_size])
        train_videos = set(unique_videos[val_size:])

        if split == "train":
            allowed_videos = train_videos
        elif split == "val":
            allowed_videos = val_videos
        else:
            allowed_videos = set(unique_videos)  

        self.frames = [f for f in all_frames if self._extract_video_id(f) in allowed_videos]

        print(f"[Dataset] Split '{split}' carregado: "
              f"{len(allowed_videos)} vídeos | {len(self.frames)} frames.")

    def _extract_video_id(self, filename):
        """
        Extrai o identificador único do vídeo a partir do frame.
        Ex: 'video_Abuse001_x264_scene0_frame0.jpg' -> 'video_Abuse001'
        """
        match = re.search(r'(video_[a-zA-Z]+\d+)', filename)
        if match:
            return match.group(1)
        return filename

    def _extract_class(self, filename):
        """
        Extrai o nome da classe a partir do nome do arquivo.
        Ex: 'video_Abuse001_x264_scene0_frame0.jpg' -> 'Abuse'
        """
        match = re.search(r'video_([a-zA-Z]+)\d+', filename)
        if match:
            return match.group(1)

        parts = filename.split('_')
        if len(parts) > 1:
            candidate = ''.join([c for c in parts[1] if not c.isdigit()])
            if candidate:
                return candidate

        print(f"[WARN] Classe não identificada para '{filename}'. Usando 'Unknown'.")
        return "Unknown"

    def _get_text(self, class_name):
        """Retorna um template aleatório para a classe."""
        templates = CLASS_TEMPLATES.get(class_name)
        if templates:
            return random.choice(templates)
        return _DEFAULT_TEMPLATE.format(class_name=class_name)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx, _attempts=0):
        if _attempts >= 5:
            raise RuntimeError(
                f"Falha ao carregar 5 frames consecutivos a partir do índice {idx}. "
                f"Verifique a integridade do dataset."
            )

        filename = self.frames[idx]
        img_path = os.path.join(self.root_dir, filename)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Erro carregando '{img_path}': {e}. Tentativa {_attempts+1}/5.")
            return self.__getitem__((idx + 1) % len(self), _attempts + 1)

        if self.transform:
            image = self.transform(image)

        class_name = self._extract_class(filename)
        text = self._get_text(class_name)

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