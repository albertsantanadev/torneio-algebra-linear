"""Arquivo de entrega (responsavel pela integracao: Albert).

O avaliador exige UMA funcao `principal(observada, mascara)`; funcoes de apoio
ficam DENTRO dela. Imports permitidos: numpy, scipy, math e random.
Esta versao base so preenche as celulas ocultas com a media das observadas.
"""
import numpy as np


def principal(observada, mascara):
    mascara = np.asarray(mascara).astype(bool)
    saida = np.array(observada, dtype=float)

    if mascara.any():
        media = saida[mascara].mean()
    else:
        media = 0.0
    saida[~mascara] = media

    # Garantia final: nunca devolver NaN/Inf
    saida[~np.isfinite(saida)] = media
    return saida
