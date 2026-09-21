

import os
import re
import json
import argparse
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from tqdm import tqdm


CLASS_HINTS = {
    "Abuse":         "physical or verbal mistreatment of a person",
    "Arrest":        "police officers detaining or handcuffing someone",
    "Arson":         "deliberate fire being set to property",
    "Assault":       "physical attack between people",
    "Burglary":      "someone breaking into or entering a building illegally",
    "Explosion":     "an explosive blast or detonation",
    "Fighting":      "people engaged in a physical fight",
    "RoadAccidents": "a traffic accident or vehicle collision",
    "Robbery":       "someone being robbed, often with force or threat",
    "Shooting":      "a shooting incident with firearms",
    "Shoplifting":   "someone stealing merchandise in a retail store",
    "Stealing":      "theft of property (not in a store)",
    "Vandalism":     "deliberate damage to property",
    "Normal":        "a routine scene without any incident",
}


def extract_class(filename):
    """Mesma lógica do dataset.py — extrai classe do nome do arquivo."""
    match = re.search(r'video_([a-zA-Z]+)\d+', filename)
    if match:
        return match.group(1)
    parts = filename.split('_')
    if len(parts) > 1:
        candidate = ''.join([c for c in parts[1] if not c.isdigit()])
        if candidate:
            return candidate
    return "Unknown"



def build_prompt(class_name):
    """
    Prompt class-aware: o modelo recebe a classe como contexto, mas é instruído
    a descrever o que VÊ — não a forçar a narrativa da classe.
    Isso reduz alucinação em frames ambíguos.
    """
    hint = CLASS_HINTS.get(class_name, class_name.lower())

    return (
        f"This is a frame from a security camera video classified as '{class_name}' "
        f"(scenes involving {hint}).\n\n"
        f"Describe what you visually observe in this specific frame in ONE concise sentence "
        f"(6 to 15 words). Focus on observable visual elements: people, objects, actions, "
        f"and scene context. Do NOT speculate about intent or events beyond what is visible. "
        f"Do NOT start with 'The image shows' or similar — describe directly."
    )



def load_model(model_name, device="cuda"):


    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    return model, processor




def generate_caption(model, processor, image, class_name, device,
                     max_new_tokens=60):
    """
    Gera caption para uma única imagem.
    """
    prompt_text = build_prompt(class_name)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": prompt_text},
        ],
    }]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text], images=[image],
        return_tensors="pt", padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,         
            temperature=1.0,
            pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    caption = processor.decode(generated_ids, skip_special_tokens=True).strip()

    caption = re.sub(r"^(the image shows|this image shows|in this image,?|i can see)\s*",
                     "", caption, flags=re.IGNORECASE).strip()

    caption = caption.rstrip(".").strip()

    return caption




def load_existing_captions(output_path):
    """Carrega captions já gerados — permite retomar após interrupção."""
    if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            return existing
    return {}


def save_captions(captions, output_path):
    """Salvamento atômico: escreve em tmp, depois renomeia. Evita corrupção."""
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(captions, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_dir", required=True,
                        help="Diretório com os frames UCF-Crime")
    parser.add_argument("--output", default="./captions.json",
                        help="Arquivo JSON de saída")
    parser.add_argument("--model_name", default="Qwen/Qwen2-VL-7B-Instruct",
                        help="Modelo vision-language a usar")
    parser.add_argument("--save_every", type=int, default=100,
                        help="Salva o progresso a cada N frames (proteção contra crash)")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Limita número de frames (útil para teste rápido)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print(" CUDA não disponível.")
        return


    valid_ext = ('.jpg', '.jpeg', '.png')
    all_frames = sorted([
        f for f in os.listdir(args.frames_dir)
        if f.lower().endswith(valid_ext)
    ])

    if args.max_frames:
        all_frames = all_frames[:args.max_frames]


    # Retomada: carrega captions já existentes
    captions = load_existing_captions(args.output)
    remaining = [f for f in all_frames if f not in captions]

    if not remaining:
        return


    model, processor = load_model(args.model_name, device)



    failures = 0
    pbar = tqdm(remaining, desc="Gerando captions")

    for i, fname in enumerate(pbar):
        fpath = os.path.join(args.frames_dir, fname)
        class_name = extract_class(fname)

        try:
            image = Image.open(fpath).convert("RGB")
            caption = generate_caption(model, processor, image, class_name, device)

            if not caption or len(caption.split()) < 3:
                caption = f"security camera footage of {CLASS_HINTS.get(class_name, class_name.lower())}"
                failures += 1

            captions[fname] = caption
            pbar.set_postfix({"class": class_name, "preview": caption[:40]})

        except Exception as e:
            print(f"\n⚠️  Erro em '{fname}': {e}")
            failures += 1
            continue

        if (i + 1) % args.save_every == 0:
            save_captions(captions, args.output)

    save_captions(captions, args.output)

   

    
    seen_classes = set()
    for fname, caption in captions.items():
        cls = extract_class(fname)
        if cls not in seen_classes:
            seen_classes.add(cls)
            print(f"  [{cls:<14}] {caption}")
        if len(seen_classes) >= 14:
            break


if __name__ == "__main__":
    main()