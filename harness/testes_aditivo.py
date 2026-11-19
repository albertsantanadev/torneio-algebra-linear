"""Etapa 3: modelo aditivo e aditivo + posto 1, com plano B para SVD instavel.
(NAO ENVIAR no .zip oficial.)

Uso: python harness/testes_aditivo.py

Modelos (em harness/v2_configuravel.py):
  aditivo        x_ij ~ mu + a_i + b_j, por minimos quadrados (o b_add do bench.py)
  aditivo + p1   x_ij ~ mu + a_i + b_j + u_i v_j, o termo de posto 1 por ALS com ridge,
                 sem usar SVD (por isso serve de plano B quando a SVD falha)

Parte A: cada modelo sozinho, matriz a matriz, no banco publico.
Parte B: os modelos entram no concurso de validacao cruzada da V2.
Parte C: plano B. A SVD e forcada a falhar em todas as chamadas e comparamos o que
         a V2 faz hoje (cai no aditivo regularizado) com o plano B (aditivo + posto 1).
Parte D: checagens (saida identica a entrega na config padrao, NaN/Inf, tempo, imports).
"""
import ast
import os
import sys
import time

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(RAIZ, "src"))
import v2_configuravel as v2  # noqa: E402
from comum import carregar_banco, carregar_principal, nrmse, problemas_da_saida  # noqa: E402
from sinteticas import FAMILIAS_ESTRUTURA, gerar_lote  # noqa: E402

PROBLEMAS = []  # (onde, o que) -> qualquer item reprova a checagem final
TEMPO_MAX = [0.0]


def b_add_bench(o, mb):
    """Copia literal do b_add de harness/bench.py (o bench le argv ao importar)."""
    n, m = o.shape
    I, J = np.where(mb)
    A = np.zeros((len(I), 1 + n + m))
    A[:, 0] = 1
    A[np.arange(len(I)), 1 + I] = 1
    A[np.arange(len(I)), 1 + n + J] = 1
    c = np.linalg.lstsq(A, o[I, J], rcond=None)[0]
    P = c[0] + c[1:1 + n][:, None] + c[1 + n:][None, :]
    s = o.copy()
    s[~mb] = P[~mb]
    return s


def sozinho(f, o, m):
    x = np.where(m, o, 0.0)
    return np.where(m, x, f(x, m))


def medir(saida, it, onde):
    probs = problemas_da_saida(saida, it["o"], it["m"])
    if probs:
        PROBLEMAS.append((onde, probs))
        return float("nan")
    return nrmse(it["g"], saida, it["m"])[0]


def rodar_cfg(cfg, itens, onde):
    erros, planos = [], 0
    for it in itens:
        info = {}
        t0 = time.perf_counter()
        s = v2.principal_config(it["o"], it["m"], cfg, info)
        t = time.perf_counter() - t0
        TEMPO_MAX[0] = max(TEMPO_MAX[0], t)
        if t >= 2.0:
            PROBLEMAS.append((f"{onde} {it['nome']}", [f"{t:.2f} s"]))
        erros.append(medir(s, it, f"{onde} {it['nome']}"))
        planos += 1 if info.get("plano_b") else 0
    return float(np.nanmean(erros)), erros, planos


def main():
    banco = carregar_banco(os.path.join(RAIZ, "banco_publico"))
    for it in banco:
        it["m"] = np.asarray(it["m"], dtype=bool)

    # ------------------------------------------------------------------ A
    print("PARTE A - cada modelo sozinho no banco publico (NRMSE nas ocultas)")
    print(f"{'mat':<5}{'forma':>9}{'b_add bench':>13}{'aditivo':>10}{'aditivo+p1':>12}{'V2 entrega':>12}")
    colA = {"b": [], "a": [], "p": [], "v": []}
    principal = carregar_principal(os.path.join(RAIZ, "src", "principal.py"))
    for it in banco:
        o, m = it["o"], it["m"]
        eb = medir(b_add_bench(np.where(m, o, 0.0), m), it, "A b_add")
        ea = medir(sozinho(v2.aditivo_lstsq, o, m), it, "A aditivo")
        t0 = time.perf_counter()
        ep = medir(sozinho(v2.aditivo_posto1, o, m), it, "A aditivo+p1")
        TEMPO_MAX[0] = max(TEMPO_MAX[0], time.perf_counter() - t0)
        ev = medir(np.asarray(principal(o.copy(), m.copy()), dtype=float), it, "A V2")
        for k, e in zip("bapv", (eb, ea, ep, ev)):
            colA[k].append(e)
        print(f"{it['nome']:<5}{str(o.shape):>9}{eb:>13.4f}{ea:>10.4f}{ep:>12.4f}{ev:>12.4f}")
    print(f"{'MEDIA':<14}" + "".join(f"{np.mean(colA[k]):>{w}.4f}" for k, w in zip("bapv", (13, 10, 12, 12))))
    if not np.allclose(colA["b"], colA["a"]):
        PROBLEMAS.append(("A", ["aditivo_lstsq difere do b_add do bench"]))

    # ------------------------------------------------------------------ B
    lotes = [
        ("publico", banco),
        ("sint. padrao s7", gerar_lote(30, semente=7)),
        ("sint. padrao s21", gerar_lote(30, semente=21)),
        ("sint. estrutura", gerar_lote(24, semente=11, familias=FAMILIAS_ESTRUTURA)),
    ]
    configs = [
        ("V2 (entrega)", v2.config()),
        ("V2 + aditivo", v2.config(extras=("aditivo_lstsq",))),
        ("V2 + aditivo+p1", v2.config(extras=("aditivo_p1",))),
        ("V2 + os dois", v2.config(extras=("aditivo_lstsq", "aditivo_p1"))),
        ("V2 + os dois, penal 2%", v2.config(extras=("aditivo_lstsq", "aditivo_p1"), penal_extra=0.02)),
    ]
    print("\nPARTE B - modelos no concurso de validacao cruzada da V2 (NRMSE medio)")
    print(f"{'configuracao':<26}" + "".join(f"{n:>18}" for n, _ in lotes))
    base = {}
    for nome, cfg in configs:
        linha = f"{nome:<26}"
        for nl, itens in lotes:
            media, _, _ = rodar_cfg(cfg, itens, f"B {nome} {nl}")
            base.setdefault(nl, media)
            linha += f"{media:>10.4f} ({(media - base[nl]) / base[nl] * 100:+5.1f}%)"
        print(linha)

    # ------------------------------------------------------------------ C
    print("\nPARTE C - plano B com a SVD forcada a falhar em todas as chamadas")
    cfgC = [
        ("V2 normal (referencia)", v2.config()),
        ("SVD falha, sem plano B", v2.config(forcar_falha_svd=True)),
        ("SVD falha, com plano B", v2.config(forcar_falha_svd=True, plano_b=True)),
    ]
    print(f"{'cenario':<26}" + "".join(f"{n:>18}" for n, _ in lotes[:3]) + "   plano B usado")
    for nome, cfg in cfgC:
        linha, usos = f"{nome:<26}", 0
        for nl, itens in lotes[:3]:
            media, _, planos = rodar_cfg(cfg, itens, f"C {nome} {nl}")
            usos += planos
            linha += f"{media:>18.4f}"
        print(linha + f"   {usos} matriz(es)")
    print("\nPlano B no banco publico, matriz a matriz:")
    _, sem, _ = rodar_cfg(cfgC[1][1], banco, "C detalhe sem")
    _, com, _ = rodar_cfg(cfgC[2][1], banco, "C detalhe com")
    for it, a, b in zip(banco, sem, com):
        print(f"  {it['nome']}  sem plano B {a:.4f}  |  com plano B {b:.4f}{'  <- melhora' if b < a - 1e-6 else ''}")

    # ------------------------------------------------------------------ D
    print("\nPARTE D - checagens")
    iguais = all(
        np.array_equal(v2.principal_config(it["o"], it["m"], v2.config(plano_b=True)),
                       np.asarray(principal(it["o"].copy(), it["m"].copy()), dtype=float))
        for it in banco
    )
    print(f"  V2 com plano_b ligado e SVD sem falha = entrega, celula por celula: {'sim' if iguais else 'NAO'}")
    if not iguais:
        PROBLEMAS.append(("D", ["plano_b muda a saida sem falha de SVD"]))
    with open(os.path.join(AQUI, "v2_configuravel.py"), encoding="utf-8") as f:
        arvore = ast.parse(f.read())
    mods = set()
    for no in ast.walk(arvore):
        if isinstance(no, ast.Import):
            mods |= {a.name.split(".")[0] for a in no.names}
        elif isinstance(no, ast.ImportFrom):
            mods.add((no.module or "").split(".")[0])
    fora = mods - {"numpy", "scipy", "math", "random"}
    print(f"  imports de v2_configuravel.py: {sorted(mods)}{'' if not fora else '  <- FORA DA LISTA'}")
    if fora:
        PROBLEMAS.append(("D", [f"imports fora da lista: {fora}"]))
    print(f"  tempo maximo por matriz: {TEMPO_MAX[0]:.3f} s (limite 2 s)")
    print("\nCHECAGEM: " + ("ok, sem NaN/Inf/celula alterada e tudo < 2 s" if not PROBLEMAS else f"PROBLEMA: {PROBLEMAS[:5]}"))
    return 1 if PROBLEMAS else 0


if __name__ == "__main__":
    sys.exit(main())
