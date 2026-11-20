"""Avaliador local

Roda um arquivo `principal.py` no banco publico e/ou em matrizes sinteticas e
mostra NRMSE, tempo e problemas de saida por matriz.

Uso:
    python harness/avaliar.py src/principal.py                       # banco publico
    python harness/avaliar.py src/principal.py --sinteticas 30       # lote sintetico
    python harness/avaliar.py src/principal.py --sinteticas 30 --semente 7 --tamanho-max 40
    python harness/avaliar.py src/principal.py --comparar src/outro.py   # dois arquivos lado a lado

A coluna NRMSE(ocultas) reproduziu o resultado oficial do Docker para a V2.
A coluna NRMSE(total) e mantida apenas como comparacao.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comum import carregar_banco, carregar_principal, executar, nrmse, problemas_da_saida  # noqa: E402
from sinteticas import gerar_lote  # noqa: E402


def avaliar(principal, itens):
    linhas = []
    for it in itens:
        try:
            saida, t = executar(principal, it["o"], it["m"])
            probs = problemas_da_saida(saida, it["o"], it["m"])
            fatais = probs  # inclui celula conhecida alterada
            if fatais:
                n_o, n_t = float("nan"), float("nan")
            else:
                n_o, n_t = nrmse(it["g"], saida, it["m"])
        except Exception as e:  # noqa: BLE001
            t, n_o, n_t = float("nan"), float("nan"), float("nan")
            fatais = [f"excecao {type(e).__name__}: {e}"]
        linhas.append({"nome": it["nome"], "forma": it["g"].shape, "n_o": n_o, "n_t": n_t, "t": t, "erro": fatais})
    return linhas


def imprimir(linhas, titulo):
    print(f"\n{titulo}")
    print(f"{'matriz':<28}{'forma':>9}  {'NRMSE(ocultas)':>14}  {'NRMSE(total)':>12}  {'tempo(s)':>8}")
    for l in linhas:
        extra = f"  !! {'; '.join(l['erro'])}" if l["erro"] else ""
        print(f"{l['nome']:<28}{str(l['forma']):>9}  {l['n_o']:>14.4f}  {l['n_t']:>12.4f}  {l['t']:>8.3f}{extra}")
    ok = [l for l in linhas if not l["erro"]]
    if ok:
        print(f"{'MEDIA (' + str(len(ok)) + ' validas)':<28}{'':>9}  {np.mean([l['n_o'] for l in ok]):>14.4f}  "
              f"{np.mean([l['n_t'] for l in ok]):>12.4f}  {np.mean([l['t'] for l in ok]):>8.3f}")
        print(f"tempo total do lote: {sum(l['t'] for l in ok):.2f} s   |   mais lenta: {max(l['t'] for l in ok):.3f} s")
    if len(ok) < len(linhas):
        print(f"ATENCAO: {len(linhas) - len(ok)} matriz(es) com saida invalida (desclassificaria a equipe).")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arquivo", help="caminho do principal.py")
    ap.add_argument("--banco", default="banco_publico", help="pasta do banco publico")
    ap.add_argument("--sinteticas", type=int, default=0, help="usar N matrizes sinteticas em vez do banco publico")
    ap.add_argument("--semente", type=int, default=0)
    ap.add_argument("--tamanho-max", type=int, default=40)
    ap.add_argument("--comparar", help="segundo arquivo para comparar lado a lado")
    a = ap.parse_args()

    if a.sinteticas:
        itens = gerar_lote(a.sinteticas, semente=a.semente, tamanho_max=a.tamanho_max)
        titulo = f"Lote sintetico ({len(itens)} matrizes, semente {a.semente}, ate {a.tamanho_max}x{a.tamanho_max})"
    else:
        itens = carregar_banco(a.banco)
        titulo = f"Banco publico ({len(itens)} matrizes)"

    imprimir(avaliar(carregar_principal(a.arquivo), itens), f"{titulo}: {a.arquivo}")
    if a.comparar:
        imprimir(avaliar(carregar_principal(a.comparar), itens), f"{titulo}: {a.comparar}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
