"""Reconstrucao matricial por posto baixo, sem usar gabaritos.

A funcao baixo_posto pode ser copiada para dentro de principal na entrega.
Bibliotecas: somente NumPy. Sem geradores aleatorios.
"""
import numpy as np


def baixo_posto(observada, mascara):
    o = np.asarray(observada, dtype=float)
    m = np.asarray(mascara, dtype=bool)
    x = np.where(m, o, 0.0)
    if not m.any():
        return x
    mu = float(np.mean(x[m]))
    if m.all():
        return x
    n, p = x.shape

    def inicializar(valores, mask):
        z = np.where(mask, valores, 0.0)
        total = np.sum(mask)
        media = float(np.sum(z) / total) if total else 0.0
        cont_l = np.sum(mask, axis=1)
        cont_c = np.sum(mask, axis=0)
        ml = np.divide(np.sum(z, axis=1), cont_l, out=np.full(n, media), where=cont_l > 0)
        mc = np.divide(np.sum(z, axis=0), cont_c, out=np.full(p, media), where=cont_c > 0)
        # Estimativa aditiva: efeito medio de linha mais efeito medio de coluna
        estimativa = ml[:, None] + mc[None, :] - media
        return np.where(mask, valores, estimativa)

    def ajustar(valores, mask, posto, passos, relaxacao=1.0):
        z = inicializar(valores, mask)
        for _ in range(passos):
            u, s, vt = np.linalg.svd(z, full_matrices=False)
            pred = (u[:, :posto] * s[:posto]) @ vt[:posto, :]
            z = np.where(mask, valores, relaxacao * pred + (1.0 - relaxacao) * z)
        z = np.where(mask, valores, pred)
        return z

    max_k = min(6, n, p)
    # Reserva dobras de teste deterministicas: cada valor conhecido pode ser validado.
    ii, jj = np.indices((n, p))
    grupos = (ii * 3 + jj * 5 + ii * jj) % 3
    candidatos = list(range(1, max_k + 1))
    erros = []
    erro_basico = 0.0
    n_basico = 0
    for f in range(3):
        teste = m & (grupos == f)
        treino = m & ~teste
        if not teste.any() or not treino.any():
            continue
        base = inicializar(x, treino)
        d = base[teste] - x[teste]
        erro_basico += float(np.dot(d, d))
        n_basico += int(np.sum(teste))
    erro_basico = erro_basico / n_basico if n_basico else np.inf
    for k in candidatos:
        soma = 0.0
        conta = 0
        for f in range(3):
            teste = m & (grupos == f)
            treino = m & ~teste
            if not teste.any() or np.sum(treino) < k * 2:
                continue
            pred = ajustar(x, treino, k, 14)
            diferenca = pred[teste] - x[teste]
            soma += float(np.dot(diferenca, diferenca))
            conta += int(np.sum(teste))
        erros.append(soma / conta if conta else np.inf)
    if np.all(np.isinf(erros)):
        return inicializar(x, m)
    if min(erros) >= 0.95 * erro_basico:
        return inicializar(x, m)
    k = candidatos[int(np.argmin(erros))]
    saida = ajustar(x, m, k, 35)
    saida[~np.isfinite(saida)] = mu
    saida[m] = x[m]
    return saida
