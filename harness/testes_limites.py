"""Testes locais extras (NAO ENVIAR no .zip oficial).

Uso: python harness/testes_limites.py src/principal.py
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comum import carregar_principal, problemas_da_saida


def main():
    caminho = sys.argv[1] if len(sys.argv) > 1 else 'src/principal.py'
    principal = carregar_principal(caminho)
    rng = np.random.default_rng(20261117)  # Exclusivamente no harness local.
    casos = []

    for shape in [(1, 1), (1, 40), (40, 1), (2, 2), (5, 40), (40, 5), (40, 40)]:
        base = rng.normal(size=shape)
        for densidade in [0.0, 0.05, 0.25, 0.5, 0.8, 1.0]:
            mask = rng.random(shape) < densidade
            casos.append(('dim_{}_{}_dens_{:.2f}'.format(shape[0], shape[1], densidade), base, mask))

    n, p = 40, 40
    i, j = np.indices((n, p))
    familias = {
        'constante': np.full((n, p), 13.0),
        'negativa': rng.normal(loc=-100.0, scale=8, size=(n, p)),
        'tendencia_linha': i * 7.0 + rng.normal(size=(n, p)) * 0.1,
        'tendencia_coluna': j * 5.0 + rng.normal(size=(n, p)) * 0.1,
        'suave_2d': 2 * i + 3 * j + rng.normal(size=(n, p)) * 0.2,
        'posto_um': rng.normal(size=(n, 1)) @ rng.normal(size=(1, p)) + 10,
        'posto_quatro': rng.normal(size=(n, 4)) @ rng.normal(size=(4, p)) + 80,
        'ruido': rng.normal(size=(n, p)),
        'alternado': (-1.0) ** (i + j) * 5,
    }
    for nome, base in familias.items():
        for densidade in [0.15, 0.5, 0.9]:
            mask = rng.random((n, p)) < densidade
            casos.append((nome + '_dens_{:.2f}'.format(densidade), base, mask))
    for nome, base in [('linha_inteira', familias['posto_quatro']), ('coluna_inteira', familias['posto_um'])]:
        mask = rng.random((n, p)) < 0.65
        if nome == 'linha_inteira':
            mask[14, :] = False
        else:
            mask[:, 15] = False
        casos.append((nome, base, mask))

    total, pior = 0.0, 0.0
    falhas = []
    for nome, completa, mask in casos:
        observada = np.where(mask, completa, np.nan)
        entrada, mascara = observada.copy(), mask.copy()
        inicio = time.perf_counter()
        try:
            saida = principal(observada, mask)
            tempo = time.perf_counter() - inicio
            problemas = problemas_da_saida(saida, entrada, mascara)
            if not np.array_equal(observada, entrada, equal_nan=True) or not np.array_equal(mask, mascara):
                problemas.append('entrada alterada')
            if tempo > 2.0:
                problemas.append('limite de 2 segundos excedido')
        except Exception as ex:
            tempo = time.perf_counter() - inicio
            problemas = [type(ex).__name__ + ': ' + str(ex)]
        total += tempo
        pior = max(pior, tempo)
        if problemas:
            falhas.append((nome, problemas))
    print('Casos extremos:', len(casos))
    print('Tempo total: {:.3f} s; pior: {:.3f} s'.format(total, pior))
    print('Falhas:', len(falhas))
    for nome, problemas in falhas:
        print('ERRO:', nome, problemas)
    return 1 if falhas else 0


if __name__ == '__main__':
    sys.exit(main())
