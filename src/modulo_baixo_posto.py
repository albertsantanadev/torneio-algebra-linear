"""Modulo de baixo posto (responsavel: Erlon).

Implementar aqui: hard-impute (SVD truncado sem centralizar), soft-impute e ALS,
alem da escolha de k por validacao cruzada com divisoes deterministicas.

Regras do avaliador: sem `class`, sem `seed`/`default_rng`/`RandomState`/`Random`,
sem `import time`, apenas numpy, scipy, math e random. Quando integrado, o corpo
desta funcao (e seus auxiliares) vai para dentro de `principal`.
"""
import numpy as np


def baixo_posto(observada, mascara):
    mascara = np.asarray(mascara).astype(bool)
    saida = np.array(observada, dtype=float)
    saida[~mascara] = saida[mascara].mean()  # placeholder: media global
    return saida
