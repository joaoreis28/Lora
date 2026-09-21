

import os
import json
import argparse
import itertools
from collections import defaultdict, Counter

import numpy as np


VERDICTS = ["CORRECT", "VAGUE", "INCORRECT"]



def cohen_kappa(labels_a, labels_b, categories=VERDICTS):
    """
    Calcula o coeficiente de Cohen κ entre duas listas de rótulos categóricos.

    κ = (p_o - p_e) / (1 - p_e)
      p_o = concordância observada
      p_e = concordância esperada ao acaso
    """
    assert len(labels_a) == len(labels_b), "Listas de tamanhos diferentes"
    n = len(labels_a)
    if n == 0:
        return None

    cat_to_idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)

    # Matriz de confusão
    confusion = np.zeros((k, k), dtype=float)
    for a, b in zip(labels_a, labels_b):
        if a in cat_to_idx and b in cat_to_idx:
            confusion[cat_to_idx[a]][cat_to_idx[b]] += 1

    total = confusion.sum()
    if total == 0:
        return None

    # Concordância observada (diagonal)
    p_o = np.trace(confusion) / total

    # Concordância esperada (produto das marginais)
    row_marginals = confusion.sum(axis=1) / total
    col_marginals = confusion.sum(axis=0) / total
    p_e = np.sum(row_marginals * col_marginals)

    if abs(1 - p_e) < 1e-10:
        return 1.0 if p_o == 1.0 else 0.0

    return (p_o - p_e) / (1 - p_e)


def interpret_kappa(k):
    """Interpretação textual segundo Landis & Koch (1977)."""
    if k is None:
        return "indefinido"
    if k < 0.00:
        return "discordância"
    elif k <= 0.20:
        return "leve (slight)"
    elif k <= 0.40:
        return "razoável (fair)"
    elif k <= 0.60:
        return "moderada (moderate)"
    elif k <= 0.80:
        return "substancial (substantial)"
    else:
        return "quase perfeita (almost perfect)"


def majority_vote(verdicts):
    """
    Retorna o voto majoritário. Em caso de empate (cada avaliador diz uma
    coisa diferente), usa a hierarquia de severidade: INCORRECT > VAGUE >
    CORRECT (prioriza marcar problema, conservador).
    """
    if not verdicts:
        return None
    counts = Counter(verdicts)
    max_count = max(counts.values())
    winners = [v for v, c in counts.items() if c == max_count]

    if len(winners) == 1:
        return winners[0]

    # Empate — usa hierarquia de severidade
    for severity in ["INCORRECT", "VAGUE", "CORRECT"]:
        if severity in winners:
            return severity
    return winners[0]




def load_validations(files):
    """
    Carrega N arquivos de validação. Retorna:
      - common_keys: filenames validados por TODOS os avaliadores
      - data: {evaluator_name: {fname: record}}
      - model_verdicts: {fname: verdict_do_LLaVA}
    """
    data = {}
    model_verdicts = {}

    for filepath in files:
        name = os.path.basename(filepath).replace("validation_", "").replace(".json", "")
        with open(filepath, "r", encoding="utf-8") as f:
            content = json.load(f)
        data[name] = content

        for fname, record in content.items():
            if "verdict" in record:
                model_verdicts[fname] = record["verdict"]

    all_keys = None
    for name, content in data.items():
        validated = {
            fname for fname, rec in content.items()
            if rec.get("human_verdict") in VERDICTS
        }
        all_keys = validated if all_keys is None else (all_keys & validated)

    common_keys = sorted(all_keys) if all_keys else []
    return common_keys, data, model_verdicts



def extract_class_from_record(data, fname):
    for content in data.values():
        if fname in content and "class" in content[fname]:
            return content[fname]["class"]
    return "Unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", required=True,
                        help="Arquivos de validação dos avaliadores")
    parser.add_argument("--output_dir", default="./results/validation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    common_keys, data, model_verdicts = load_validations(args.files)
    evaluators = list(data.keys())

    if not common_keys:
        print(" Nenhum caption foi validado por todos os avaliadores.")
        return

    report = []
    def log(s=""):
        print(s)
        report.append(s)

    log("=" * 74)
    log("  RELATÓRIO DE VALIDAÇÃO HUMANA — CAMADA 2")
    log("  Concordância entre Avaliadores e com o Modelo Validador (LLaVA)")
    log("=" * 74)
    log(f"  Avaliadores          : {', '.join(evaluators)}")
    log(f"  Captions em comum     : {len(common_keys)}")
    log(f"  (validados por todos os {len(evaluators)} avaliadores)")

    log("\n" + "─" * 74)
    log("  1. DISTRIBUIÇÃO DE VEREDITOS")
    log("─" * 74)
    log(f"{'Avaliador':<20} {'CORRECT':>12} {'VAGUE':>12} {'INCORRECT':>12}")
    log("-" * 60)

    for name in evaluators:
        verdicts = [data[name][k]["human_verdict"] for k in common_keys]
        counts = Counter(verdicts)
        n = len(verdicts)
        log(f"{name:<20} "
            f"{counts.get('CORRECT',0):>5} ({100*counts.get('CORRECT',0)/n:>4.1f}%) "
            f"{counts.get('VAGUE',0):>5} ({100*counts.get('VAGUE',0)/n:>4.1f}%) "
            f"{counts.get('INCORRECT',0):>4} ({100*counts.get('INCORRECT',0)/n:>4.1f}%)")

    model_v = [model_verdicts[k] for k in common_keys]
    counts = Counter(model_v)
    n = len(model_v)
    log("-" * 60)
    log(f"{'MODELO (LLaVA)':<20} "
        f"{counts.get('CORRECT',0):>5} ({100*counts.get('CORRECT',0)/n:>4.1f}%) "
        f"{counts.get('VAGUE',0):>5} ({100*counts.get('VAGUE',0)/n:>4.1f}%) "
        f"{counts.get('INCORRECT',0):>4} ({100*counts.get('INCORRECT',0)/n:>4.1f}%)")

    log("\n" + "─" * 74)
    log("  2. CONCORDÂNCIA INTER-AVALIADOR (Cohen κ entre humanos)")
    log("─" * 74)

    for a, b in itertools.combinations(evaluators, 2):
        labels_a = [data[a][k]["human_verdict"] for k in common_keys]
        labels_b = [data[b][k]["human_verdict"] for k in common_keys]
        k_val = cohen_kappa(labels_a, labels_b)
        # Concordância bruta (% de vezes que deram o mesmo veredito)
        agreement = sum(1 for x, y in zip(labels_a, labels_b) if x == y) / len(labels_a)
        log(f"  {a} vs {b}:")
        log(f"     Cohen κ         : {k_val:.3f} ({interpret_kappa(k_val)})")
        log(f"     Concordância bruta: {100*agreement:.1f}%")

    log("\n" + "─" * 74)
    log("  3. VOTO MAJORITÁRIO HUMANO")
    log("─" * 74)

    majority = {}
    for k in common_keys:
        verdicts = [data[name][k]["human_verdict"] for name in evaluators]
        majority[k] = majority_vote(verdicts)

    maj_counts = Counter(majority.values())
    log(f"  Distribuição do voto majoritário:")
    for v in VERDICTS:
        n_v = maj_counts.get(v, 0)
        log(f"     {v:<12}: {n_v:>5} ({100*n_v/len(common_keys):.1f}%)")

  

    log("\n" + "─" * 74)
    log("  4. CONCORDÂNCIA HUMANO-MODELO (Cohen κ)")
    log("─" * 74)

    for name in evaluators:
        labels_h = [data[name][k]["human_verdict"] for k in common_keys]
        labels_m = [model_verdicts[k] for k in common_keys]
        k_val = cohen_kappa(labels_h, labels_m)
        agreement = sum(1 for x, y in zip(labels_h, labels_m) if x == y) / len(labels_h)
        log(f"  {name} vs Modelo:")
        log(f"     Cohen κ         : {k_val:.3f} ({interpret_kappa(k_val)})")
        log(f"     Concordância bruta: {100*agreement:.1f}%")

    labels_maj = [majority[k] for k in common_keys]
    labels_m   = [model_verdicts[k] for k in common_keys]
    k_main = cohen_kappa(labels_maj, labels_m)
    agreement_main = sum(1 for x, y in zip(labels_maj, labels_m) if x == y) / len(labels_maj)

    log(f"\n  ★ MÉTRICA PRINCIPAL — Voto Majoritário vs Modelo:")
    log(f"     Cohen κ         : {k_main:.3f} ({interpret_kappa(k_main)})")
    log(f"     Concordância bruta: {100*agreement_main:.1f}%")

    log("\n" + "─" * 74)
    log("  5. MATRIZ DE CONFUSÃO — Voto Majoritário (linhas) vs Modelo (colunas)")
    log("─" * 74)

    cat_to_idx = {c: i for i, c in enumerate(VERDICTS)}
    confusion = np.zeros((3, 3), dtype=int)
    for k in common_keys:
        h = majority[k]
        m = model_verdicts[k]
        if h in cat_to_idx and m in cat_to_idx:
            confusion[cat_to_idx[h]][cat_to_idx[m]] += 1

    log(f"{'':>12} {'CORRECT':>10} {'VAGUE':>10} {'INCORRECT':>12}  (modelo)")
    for i, v in enumerate(VERDICTS):
        row = confusion[i]
        log(f"{v:>12} {row[0]:>10} {row[1]:>10} {row[2]:>12}")
    log(f"  (humano majoritário nas linhas)")

    log("\n" + "─" * 74)
    log("  6. DISCORDÂNCIA HUMANO-MODELO POR CLASSE")
    log("─" * 74)
    log(f"{'Classe':<15} {'Concordância':>14} {'Discordâncias':>15}")
    log("-" * 46)

    by_class = defaultdict(lambda: {"agree": 0, "total": 0})
    for k in common_keys:
        cls = extract_class_from_record(data, k)
        by_class[cls]["total"] += 1
        if majority[k] == model_verdicts[k]:
            by_class[cls]["agree"] += 1

    for cls in sorted(by_class):
        stats = by_class[cls]
        agree_pct = 100 * stats["agree"] / stats["total"] if stats["total"] else 0
        disagreements = stats["total"] - stats["agree"]
        log(f"{cls:<15} {agree_pct:>13.1f}% {disagreements:>10}/{stats['total']}")

    disagreements = {}
    for k in common_keys:
        if majority[k] != model_verdicts[k]:
            disagreements[k] = {
                "class":          extract_class_from_record(data, k),
                "caption":        data[evaluators[0]][k].get("caption", ""),
                "model_verdict":  model_verdicts[k],
                "human_majority": majority[k],
                "human_votes":    [data[name][k]["human_verdict"] for name in evaluators],
            }

    disagreement_path = os.path.join(args.output_dir, "disagreements.json")
    with open(disagreement_path, "w", encoding="utf-8") as f:
        json.dump(disagreements, f, ensure_ascii=False, indent=2)



    report_path = os.path.join(args.output_dir, "validation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))


if __name__ == "__main__":
    main()