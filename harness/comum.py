"""Funcoes compartilhadas pelo verificador e pelo avaliador local.

Nada aqui e enviado na entrega: o harness roda no computador pessoal, fora do
avaliador oficial
"""
import importlib.util
import os
import time

import numpy as np


def carregar_principal(caminho):
    """Importa o arquivo .py indicado e devolve a funcao `principal`."""
    caminho = os.path.abspath(caminho)
    spec = importlib.util.spec_from_file_location("submissao", caminho)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    if not hasattr(modulo, "principal"):
        raise AttributeError("O arquivo nao define a funcao `principal`.")
    return modulo.principal


def carregar_banco(pasta):
    """Le as matrizes do banco publico: lista de dicts com g (gabarito),
    m (mascara booleana) e o (observada, com NaN nas ocultas)."""
    itens = []
    i = 1
    while os.path.exists(os.path.join(pasta, f"gabarito_{i:02d}.npy")):
        g = np.load(os.path.join(pasta, f"gabarito_{i:02d}.npy"))
        m = np.load(os.path.join(pasta, f"mascara_{i:02d}.npy"))
        o = np.load(os.path.join(pasta, f"observada_{i:02d}.npy"))
        itens.append({"nome": f"{i:02d}", "g": g, "m": m, "o": o})
        i += 1
    if not itens:
        raise FileNotFoundError(f"Nenhum gabarito_NN.npy em {pasta}")
    return itens


def nrmse(g, pred, m):
    """Devolve (nrmse com desvio das ocultas, nrmse com desvio da matriz inteira).

    A coluna que usa o desvio das ocultas reproduziu o NRMSE oficial da V2.
    A segunda coluna e mantida como referencia alternativa.
    """
    ocultas = ~np.asarray(m).astype(bool)
    erro = np.sqrt(np.mean((np.asarray(pred, dtype=float)[ocultas] - g[ocultas]) ** 2))
    sd_ocultas = g[ocultas].std()
    sd_total = g.std()
    return (
        erro / sd_ocultas if sd_ocultas > 0 else float("inf"),
        erro / sd_total if sd_total > 0 else float("inf"),
    )


def executar(principal, o, m):
    """Roda `principal` numa copia dos dados e devolve (saida, segundos)."""
    entrada_o, entrada_m = o.copy(), m.copy()
    t0 = time.perf_counter()
    saida = principal(entrada_o, entrada_m)
    return saida, time.perf_counter() - t0


def problemas_da_saida(saida, o, m):
    """Lista de problemas que desclassificariam a equipe (vazia = saida valida)."""
    probs = []
    try:
        arr = np.asarray(saida)
    except Exception as e:  # noqa: BLE001
        return [f"nao foi possivel converter a saida em array: {e}"]
    if arr.dtype == object or not (np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating)):
        return [f"saida com tipo nao numerico ({arr.dtype})"]
    if arr.shape != o.shape:
        probs.append(f"dimensao errada: {arr.shape} em vez de {o.shape}")
        return probs
    if np.isnan(arr).any():
        probs.append(f"{int(np.isnan(arr).sum())} valor(es) NaN na saida")
    if np.isinf(arr).any():
        probs.append(f"{int(np.isinf(arr).sum())} valor(es) Inf na saida")
    conhecidas = np.asarray(m).astype(bool)
    if not probs:
        alteradas = ~np.isclose(arr[conhecidas].astype(float), np.asarray(o, dtype=float)[conhecidas], rtol=1e-9, atol=1e-12)
        if alteradas.any():
            probs.append(f"{int(alteradas.sum())} celula(s) conhecida(s) alterada(s)")
    return probs
