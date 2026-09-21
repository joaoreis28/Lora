import os
import re
import json
import argparse
from collections import defaultdict



DISCARD_PATTERNS = [
    r"\bblack screen\b",
    r"\bwarning text\b",
    r"\bviewer upset\b",
    r"\bblurred,?\s+white,?\s+and\s+(indistinct|grey)\b",
    r"\bno discernible (objects|people|content)\b",
    r"\bblank (screen|frame)\b",
    r"\bdark,?\s+with (a )?warning\b",
    r"\bindistinct scene with no\b",
    r"\btext indicating potential\b",
]


OVERLAY_TEXT_PATTERNS = [
    r'\bthe text\s*["\u201c][^"\u201d]*["\u201d]',          # the text "..."
    r'\bwith (the )?text\s*["\u201c][^"\u201d]*["\u201d]',  # with text "..."
    r'\blabeled\s*["\u201c][^"\u201d]*["\u201d]',           # labeled "..."
    r'\bdisplay(?:ing|s)?\s*(?:the text\s*)?["\u201c][^"\u201d]*["\u201d]',
    r'\breads?\s*["\u201c][^"\u201d]*["\u201d]',            # reads "..."
    r'["\u201c][^"\u201d]{0,60}(?:CCTV|Camera \d+|Evidence)[^"\u201d]*["\u201d]',
]

POLICE_BADGE_PATTERN = r"\bpolice badge with text\b"

PLACE_PATTERN = r"\b[A-Z][a-z]+(?:ville|town|burg|field|port)?,\s*[A-Z]{2}\b"




def should_discard(caption):
    """Retorna True se o caption indica um frame inútil (tela de aviso, etc.)."""
    low = caption.lower()
    for pat in DISCARD_PATTERNS:
        if re.search(pat, low):
            return True
    if re.search(POLICE_BADGE_PATTERN, low):
        return True
    return False


def clean_caption(caption, max_words=20):
    """
    Aplica limpezas no caption. Retorna o caption limpo, ou None se ficar
    irrecuperável (vazio ou curto demais após limpeza).
    """
    text = caption.strip()

    for pat in OVERLAY_TEXT_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)

    text = re.sub(PLACE_PATTERN, "", text)

    text = text.replace('"', "").replace("\u201c", "").replace("\u201d", "")
    text = re.sub(r"\s*,\s*,", ",", text)      # vírgulas duplas
    text = re.sub(r"\s{2,}", " ", text)        # espaços múltiplos
    text = re.sub(r"\s+([,.])", r"\1", text)   # espaço antes de pontuação
    text = text.strip(" ,.-")

    text = re.sub(
        r"^(the frame (displays|depicts|captures|shows)|the image shows|"
        r"a security camera captures|this frame)\s*",
        "", text, flags=re.IGNORECASE
    ).strip()

    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])

    if len(text.split()) < 3:
        return None

    text = text[0].upper() + text[1:] if text else text

    return text


def extract_class(filename):
    match = re.search(r'video_([a-zA-Z]+)\d+', filename)
    if match:
        return match.group(1)
    return "Unknown"



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="./captions.json",
                        help="Arquivo de captions brutos")
    parser.add_argument("--output", default="./captions_clean.json",
                        help="Arquivo de saída com captions limpos")
    parser.add_argument("--max_words", type=int, default=20,
                        help="Número máximo de palavras por caption")
    parser.add_argument("--report", default="./cleaning_report.txt",
                        help="Relatório do processo de limpeza")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Arquivo não encontrado: {args.input}")

    with open(args.input, "r", encoding="utf-8") as f:
        raw_captions = json.load(f)

    print(f" Captions carregados: {len(raw_captions)}")

    cleaned         = {}
    discarded       = []
    modified        = []
    unchanged       = 0

    stats = defaultdict(lambda: {"total": 0, "kept": 0, "discarded": 0, "modified": 0})

    for fname, caption in raw_captions.items():
        cls = extract_class(fname)
        stats[cls]["total"] += 1

        if should_discard(caption):
            discarded.append((fname, caption))
            stats[cls]["discarded"] += 1
            continue

        clean = clean_caption(caption, max_words=args.max_words)

        if clean is None:
            discarded.append((fname, caption))
            stats[cls]["discarded"] += 1
            continue

        cleaned[fname] = clean
        stats[cls]["kept"] += 1

        if clean != caption:
            modified.append((fname, caption, clean))
            stats[cls]["modified"] += 1
        else:
            unchanged += 1

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    report_lines = []
    def log(s=""):
        print(s)
        report_lines.append(s)

    
    with open(args.report, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

   

if __name__ == "__main__":
    main()