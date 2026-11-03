"""Modulo de metodos estruturais (responsavel: Romildo).

Implementar aqui: regressao entre colunas (minimos quadrados com ridge), modelo
aditivo (media + linha + coluna) e variantes, alem de testes de casos de borda.

Mesmas regras do avaliador do modulo de baixo posto.
"""
import numpy as np


def estrutural(observada, mascara):
    mascara = np.asarray(mascara).astype(bool)
    saida = np.array(observada, dtype=float)
    saida[~mascara] = saida[mascara].mean()  # placeholder: media global
    return saida
