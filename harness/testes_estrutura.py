"""Testes de estrutura: simetria, Toeplitz e suavidade (NAO ENVIAR no .zip oficial).

Uso: python harness/testes_estrutura.py [--rapido]

Parte A (diagnostico): mede, so com as celulas OBSERVADAS, quanto de cada estrutura
existe nas 10 matrizes publicas:
  simetria   erro relativo entre x[i,j] e x[j,i] nos pares em que os dois sao conhecidos
             (0 = perfeitamente simetrica; ~1.4 = sem relacao). So para matrizes quadradas.
  R2 diag    fracao da variancia explicada pela media de cada diagonal (Toeplitz)
  R2 lin/col fracao explicada pela media de cada linha / coluna (referencia)
  R2 poliC2  fracao explicada por um polinomio de grau 2 em cada coluna (suavidade em i)
  R2 poliL2  idem em cada linha (suavidade em j)

Parte B (candidatos): coloca candidatos novos no MESMO concurso de validacao cruzada da
V2 (harness/v2_configuravel.py) e compara o NRMSE com a V2 pura em tres lotes: banco
publico, sinteticas com estrutura (sinteticas.FAMILIAS_ESTRUTURA) e o lote sintetico
padrao. Tambem confere NaN/Inf, celulas conhecidas e o tempo maximo por matriz.
"""
import os
import sys
import time

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, AQUI)
import v2_configuravel as v2  # noqa: E402
from comum import carregar_banco, nrmse, problemas_da_saida  # noqa: E402
from sinteticas import FAMILIAS_ESTRUTURA, gerar_lote  # noqa: E402

SUAVIDADE = ("poliC1", "poliC2", "poliC3", "poliL1", "poliL2", "poliL3")
CONFIGS = [
    ("V2 (entrega)", v2.config()),
    ("V2 + simetrica", v2.config(extras=("simetrica",))),
    ("V2 + toeplitz", v2.config(extras=("toeplitz",))),
    ("V2 + suavidade (poli 1-3)", v2.config(extras=SUAVIDADE)),
    ("V2 + suavidade, penal 2%", v2.config(extras=SUAVIDADE, penal_extra=0.02)),
    ("V2 + superficie (grau 1-2)", v2.config(extras=("superf1", "superf2"))),
    ("V2 + todas as estruturas", v2.config(extras=("simetrica", "toeplitz") + SUAVIDADE + ("superf1", "superf2"))),
]


# ----------------------------------------------------------------------------
# Parte A: diagnostico
# ----------------------------------------------------------------------------

def _r2_grupos(x, m, rotulos):
    """1 - SQ dentro dos grupos / SQ total, so com observadas (grupos com >= 2 obs)."""
    v = x[m]
    g = rotulos[m]
    sqt = float(np.sum((v - v.mean()) ** 2))
    if sqt == 0:
        return float("nan")
    sqd = 0.0
    for r in np.unique(g):
        w = v[g == r]
        sqd += float(np.sum((w - w.mean()) ** 2))
    return 1.0 - sqd / sqt


def _r2_poli(x, m, grau, eixo):
    if eixo == 1:
        return _r2_poli(x.T, m.T, grau, 0)
    n, p = x.shape
    t = np.linspace(-1, 1, n)
    v = x[m]
    sqt = float(np.sum((v - v.mean()) ** 2))
    sqr = 0.0
    for j in range(p):
        k = m[:, j]
        if k.sum() <= grau + 1:  # sem graus de liberdade: nao conta como evidencia
            sqr += float(np.sum((x[k, j] - x[k, j].mean()) ** 2)) if k.sum() > 1 else 0.0
            continue
        V = np.vander(t[k], grau + 1)
        c = np.linalg.lstsq(V, x[k, j], rcond=None)[0]
        sqr += float(np.sum((V @ c - x[k, j]) ** 2))
    return 1.0 - sqr / sqt if sqt > 0 else float("nan")


def diagnostico(o, m):
    m = np.asarray(m, dtype=bool)  # as mascaras do banco vem como 0/1 inteiros
    x = np.where(m, o, 0.0)
    n, p = x.shape
    I, J = np.indices((n, p))
    sim = float("nan")
    if n == p:
        par = m & m.T & (I < J)
        if par.any():
            a, b = x[par], x.T[par]
            c = x[m].mean()
            sim = float(np.sqrt(np.sum((a - b) ** 2) / max(1e-12, np.sum((a - c) ** 2) + np.sum((b - c) ** 2)) * 2))
    return {
        "sim": sim,
        "diag": _r2_grupos(x, m, I - J),
        "lin": _r2_grupos(x, m, I),
        "col": _r2_grupos(x, m, J),
        "poliC2": _r2_poli(x, m, 2, 0),
        "poliL2": _r2_poli(x, m, 2, 1),
    }


# ----------------------------------------------------------------------------
# Parte B: candidatos no concurso de validacao cruzada
# ----------------------------------------------------------------------------

def rodar(cfg, itens):
    erros, tempos, invalidas, escolhas = [], [], 0, {}
    por_matriz = []
    for it in itens:
        info = {}
        t0 = time.perf_counter()
        saida = v2.principal_config(it["o"], it["m"], cfg, info)
        t = time.perf_counter() - t0
        probs = problemas_da_saida(saida, it["o"], it["m"])
        if probs:
            invalidas += 1
            por_matriz.append((float("nan"), info.get("escolha", "?")))
            continue
        e = nrmse(it["g"], saida, it["m"])[0]
        erros.append(e)
        tempos.append(t)
        nome = info.get("escolha", "?").split(" ")[0]
        escolhas[nome] = escolhas.get(nome, 0) + 1
        por_matriz.append((e, info.get("escolha", "?")))
    return {
        "media": float(np.mean(erros)) if erros else float("nan"),
        "tmax": max(tempos) if tempos else float("nan"),
        "invalidas": invalidas,
        "escolhas": escolhas,
        "por_matriz": por_matriz,
    }


def main():
    rapido = "--rapido" in sys.argv
    banco = carregar_banco(os.path.join(RAIZ, "banco_publico"))

    print("PARTE A - estrutura nas matrizes publicas (so celulas observadas)")
    print(f"{'mat':<5}{'forma':>9}{'simetria':>10}{'R2 diag':>9}{'R2 lin':>8}{'R2 col':>8}{'R2 poliC2':>11}{'R2 poliL2':>11}")
    for it in banco:
        d = diagnostico(it["o"], it["m"])
        sim = f"{d['sim']:.3f}" if np.isfinite(d["sim"]) else "-"
        print(f"{it['nome']:<5}{str(it['o'].shape):>9}{sim:>10}{d['diag']:>9.3f}{d['lin']:>8.3f}{d['col']:>8.3f}"
              f"{d['poliC2']:>11.3f}{d['poliL2']:>11.3f}")

    lotes = [
        ("banco publico (10)", banco),
        ("sinteticas com estrutura", gerar_lote(12 if rapido else 24, semente=11, familias=FAMILIAS_ESTRUTURA)),
        ("sinteticas padrao", gerar_lote(10 if rapido else 30, semente=7)),
    ]
    print("\nPARTE B - candidatos novos no concurso de validacao cruzada da V2")
    resultados = {}
    for nome_lote, itens in lotes:
        print(f"\n{nome_lote} ({len(itens)} matrizes)")
        print(f"{'configuracao':<30}{'NRMSE medio':>12}{'vs V2':>9}{'tempo max':>11}{'invalidas':>11}  escolhas")
        base = None
        for nome_cfg, cfg in CONFIGS:
            r = rodar(cfg, itens)
            resultados[(nome_lote, nome_cfg)] = r
            base = r["media"] if base is None else base
            delta = (r["media"] - base) / base * 100 if base else 0.0
            esc = ", ".join(f"{k}:{v}" for k, v in sorted(r["escolhas"].items(), key=lambda kv: -kv[1]))
            print(f"{nome_cfg:<30}{r['media']:>12.4f}{delta:>+8.1f}%{r['tmax']:>10.3f}s{r['invalidas']:>11}  {esc}")

    print("\nBanco publico, matriz a matriz (V2 x V2 + todas as estruturas)")
    a = resultados[("banco publico (10)", CONFIGS[0][0])]["por_matriz"]
    b = resultados[("banco publico (10)", CONFIGS[-1][0])]["por_matriz"]
    for it, (ea, sa), (eb, sb) in zip(banco, a, b):
        marca = "  <- muda" if abs(ea - eb) > 1e-9 else ""
        print(f"  {it['nome']}  V2 {ea:.4f} ({sa})  |  +estruturas {eb:.4f} ({sb}){marca}")

    ruins = [k for k, r in resultados.items() if r["invalidas"] or not np.isfinite(r["tmax"]) or r["tmax"] >= 2.0]
    print("\nCHECAGEM: " + ("ok, sem NaN/Inf/celula alterada e tudo < 2 s" if not ruins else f"PROBLEMA em {ruins}"))
    return 1 if ruins else 0


if __name__ == "__main__":
    sys.exit(main())
