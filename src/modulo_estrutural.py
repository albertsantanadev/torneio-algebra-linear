"""Metodos estruturais isolados para estudo e comparacoes locais.

A entrega utiliza uma implementacao autonoma dentro de src/principal.py;
este arquivo nao precisa ser enviado ao avaliador.
"""
import numpy as np


def estrutural(observada, mascara, regularizacao=0.5):
    """Ajusta media + efeito de linha + efeito de coluna por minimos quadrados com ridge.

    Iteracoes alternadas: em cada passo, melhora os efeitos de linha e coluna,
    aplicando uma pequena regularizacao para evitar extremos nas linhas esparsas.
    """
    o = np.asarray(observada, dtype=float)
    m = np.asarray(mascara, dtype=bool)
    x = np.where(m, o, 0.0)
    n, p = x.shape
    total = int(np.sum(m))
    if not total:
        return np.zeros((n, p), dtype=float)
    media = float(np.sum(x) / total)
    nr = np.sum(m, axis=1)
    nc = np.sum(m, axis=0)
    efeitos_linha = np.zeros(n)
    efeitos_coluna = np.zeros(p)
    for _ in range(18):
        efeitos_linha = np.sum(np.where(m, x - media - efeitos_coluna[None, :], 0.0), axis=1) / (nr + regularizacao)
        efeitos_coluna = np.sum(np.where(m, x - media - efeitos_linha[:, None], 0.0), axis=0) / (nc + regularizacao)
    pred = media + efeitos_linha[:, None] + efeitos_coluna[None, :]
    pred = np.where(m, x, pred)
    return np.nan_to_num(pred, nan=media, posinf=media, neginf=media)
