"""Versao experimental de reconstrucao - somente NumPy, sem gabaritos."""
import numpy as np


def principal(observada, mascara):
    o = np.asarray(observada, dtype=float)
    m = np.asarray(mascara, dtype=bool)
    x = np.where(m, o, 0.0)
    n, p = x.shape
    conhecidos = int(np.sum(m))
    if conhecidos == 0:
        return np.zeros((n, p), dtype=float)
    if conhecidos == n * p:
        return x.copy()
    media = float(np.sum(x) / conhecidos)

    def basicos(mask):
        a = np.where(mask, x, 0.0)
        nr = np.sum(mask, axis=1)
        nc = np.sum(mask, axis=0)
        q = int(np.sum(mask))
        mu = float(np.sum(a) / q) if q else media
        lm = np.divide(np.sum(a, axis=1), nr, out=np.full(n, mu), where=nr > 0)
        cm = np.divide(np.sum(a, axis=0), nc, out=np.full(p, mu), where=nc > 0)
        glob = np.full((n, p), mu)
        linha = np.broadcast_to(lm[:, None], (n, p))
        coluna = np.broadcast_to(cm[None, :], (n, p))
        return a, mu, glob, linha, coluna, nr, nc

    def aditivo(a, mu, mask, nr, nc, reg):
        efeito_l = np.zeros(n)
        efeito_c = np.zeros(p)
        for t in range(18):
            efeito_l = np.sum(np.where(mask, a - mu - efeito_c[None, :], 0), axis=1) / (nr + reg)
            efeito_c = np.sum(np.where(mask, a - mu - efeito_l[:, None], 0), axis=0) / (nc + reg)
        return mu + efeito_l[:, None] + efeito_c[None, :]

    def vizinhos(a, mask, fallback, por_linha):
        pred = np.array(fallback, copy=True)
        if por_linha:
            for j in range(p):
                indices = np.flatnonzero(mask[:, j])
                if len(indices) == 0:
                    continue
                for i in range(n):
                    if mask[i, j]:
                        continue
                    antes = indices[indices < i]
                    depois = indices[indices > i]
                    if len(antes) and len(depois):
                        i0, i1 = int(antes[-1]), int(depois[0])
                        pred[i, j] = ((i1 - i) * a[i0, j] + (i - i0) * a[i1, j]) / (i1 - i0)
                    elif len(antes):
                        pred[i, j] = a[int(antes[-1]), j]
                    elif len(depois):
                        pred[i, j] = a[int(depois[0]), j]
        else:
            for i in range(n):
                indices = np.flatnonzero(mask[i, :])
                if len(indices) == 0:
                    continue
                for j in range(p):
                    if mask[i, j]:
                        continue
                    antes = indices[indices < j]
                    depois = indices[indices > j]
                    if len(antes) and len(depois):
                        j0, j1 = int(antes[-1]), int(depois[0])
                        pred[i, j] = ((j1 - j) * a[i, j0] + (j - j0) * a[i, j1]) / (j1 - j0)
                    elif len(antes):
                        pred[i, j] = a[i, int(antes[-1])]
                    elif len(depois):
                        pred[i, j] = a[i, int(depois[0])]
        return pred

    def svd_fit(a, mask, mu, inicio, k, centrado, iteracoes):
        z = np.where(mask, a, inicio)
        centro = mu if centrado else 0.0
        for t in range(iteracoes):
            try:
                u, singular, vt = np.linalg.svd(z - centro, full_matrices=False)
                aproximacao = centro + (u[:, :k] * singular[:k]) @ vt[:k, :]
                z = np.where(mask, a, aproximacao)
            except np.linalg.LinAlgError:
                break
        return z


    def polir_als(a, mask, pre, k, centrado, mu, reg, passos):
        centro = mu if centrado else 0.0
        try:
            U, s, Vt = np.linalg.svd(pre - centro, full_matrices=False)
            U = U[:, :k] * np.sqrt(s[:k])
            V = Vt[:k, :].T * np.sqrt(s[:k])
            dados = a - centro
            olho = reg * np.eye(k)
            for tt in range(passos):
                for i in range(n):
                    ids = mask[i, :]
                    if int(np.sum(ids)) > 0:
                        mat = V[ids, :]
                        U[i, :] = np.linalg.solve(mat.T @ mat + olho, mat.T @ dados[i, ids])
                for j in range(p):
                    ids = mask[:, j]
                    if int(np.sum(ids)) > 0:
                        mat = U[ids, :]
                        V[j, :] = np.linalg.solve(mat.T @ mat + olho, mat.T @ dados[ids, j])
            return centro + U @ V.T
        except np.linalg.LinAlgError:
            return pre

    # Uma divisao deterministica que usa apenas as posicoes conhecidas.
    ii, jj = np.indices((n, p))
    grupos = ((ii * 53 + jj * 97 + ii * jj * 29 + ii * ii * 7 + jj * jj * 11) % 5)
    postos = [k for k in (1, 2, 3, 4, 5, 6, 8) if k <= min(n, p) and conhecidos >= 3 * k]
    tipos = ['global', 'linha', 'coluna', 'aditivo', 'aditivo_reg', 'viz_linha', 'viz_coluna']
    cand = [(a, 0) for a in tipos] + [('svd', (k, cent)) for k in postos for cent in (False, True)]
    erros = np.zeros(len(cand), dtype=float)
    total = 0
    grupos_cv = (0, 2)
    iteracoes_cv = 28 if n * p <= 600 else 23
    for grupo in grupos_cv:
        teste = m & (grupos == grupo)
        treino = m & ~teste
        if int(np.sum(teste)) == 0 or int(np.sum(treino)) < 4:
            continue
        a, mu, glob, linha, coluna, nr, nc = basicos(treino)
        add = aditivo(a, mu, treino, nr, nc, 0.5)
        addreg = aditivo(a, mu, treino, nr, nc, 2.0)
        base = [glob, linha, coluna, add, addreg, vizinhos(a, treino, coluna, True), vizinhos(a, treino, linha, False)]
        for cid, pre in enumerate(base):
            d = pre[teste] - x[teste]
            erros[cid] = erros[cid] + float(d @ d)
        if postos:
            for cid in range(len(tipos), len(cand)):
                k, cent = cand[cid][1]
                pre = svd_fit(a, treino, mu, add, k, cent, iteracoes_cv)
                d = pre[teste] - x[teste]
                erros[cid] = erros[cid] + float(d @ d)
        total += int(np.sum(teste))
    if total:
        erros /= total
    else:
        erros[:] = np.inf
    # Penalidade leve para modelos complexos quando estao empatados.
    for cid in range(len(tipos), len(cand)):
        k, cent = cand[cid][1]
        erros[cid] = erros[cid] * (1.0 + 0.003 * k)
    melhor = int(np.argmin(erros))
    a, mu, glob, linha, coluna, nr, nc = basicos(m)
    if melhor == 0:
        pre = glob
    elif melhor == 1:
        pre = linha
    elif melhor == 2:
        pre = coluna
    elif melhor == 3:
        pre = aditivo(a, mu, m, nr, nc, 0.5)
    elif melhor == 4:
        pre = aditivo(a, mu, m, nr, nc, 2.0)
    elif melhor == 5:
        pre = vizinhos(a, m, coluna, True)
    elif melhor == 6:
        pre = vizinhos(a, m, linha, False)
    else:
        add = aditivo(a, mu, m, nr, nc, 0.5)
        k, cent = cand[melhor][1]
        pre = svd_fit(a, m, mu, add, k, cent, 85)
    if melhor >= len(tipos):
        k, cent = cand[melhor][1]
        opcoes = [0.0, 0.01, 0.1]
        for_reg = np.zeros(len(opcoes) + 1)
        nt = 0
        for grupo in grupos_cv:
            teste = m & (grupos == grupo)
            treino = m & ~teste
            ntg = int(np.sum(teste))
            if ntg == 0 or int(np.sum(treino)) < 4:
                continue
            ta, tmu, tg, tl, tc, tnr, tnc = basicos(treino)
            ad = aditivo(ta, tmu, treino, tnr, tnc, 0.5)
            p0 = svd_fit(ta, treino, tmu, ad, k, cent, 85)
            diff = p0[teste] - x[teste]
            for_reg[0] = for_reg[0] + float(diff @ diff)
            for ix, reg in enumerate(opcoes):
                p1 = polir_als(ta, treino, p0, k, cent, tmu, reg, 10)
                diff = p1[teste] - x[teste]
                for_reg[ix + 1] = for_reg[ix + 1] + float(diff @ diff)
            nt += ntg
        if nt > 0:
            for_reg = for_reg / nt
            for ix, reg in enumerate(opcoes):
                for_reg[ix + 1] = for_reg[ix + 1] * 1.01
            ind = int(np.argmin(for_reg))
            if ind > 0:
                pre = polir_als(a, m, pre, k, cent, mu, opcoes[ind - 1], 10)

    # Linhas/colunas completamente ocultas nao podem ser reconstruidas por SVD.
    # Recuamos a medias observaveis, evitando extrapolacoes sem fundamento.
    nr_total = np.sum(m, axis=1)
    nc_total = np.sum(m, axis=0)
    media_linha = np.divide(np.sum(x, axis=1), nr_total, out=np.full(n, media), where=nr_total > 0)
    media_coluna = np.divide(np.sum(x, axis=0), nc_total, out=np.full(p, media), where=nc_total > 0)
    pre = np.where((nr_total == 0)[:, None], media, pre)
    pre = np.where((nc_total == 0)[None, :], media, pre)
    pre = np.where((nr_total == 0)[:, None] & (nc_total == 0)[None, :], media, pre)
    desv = float(np.sqrt(np.mean((x[m] - media) ** 2)))
    if desv > 0.0:
        pre = np.clip(pre, media - 10.0 * desv, media + 10.0 * desv)
    saida = np.where(m, x, pre)
    saida = np.nan_to_num(saida, nan=media, posinf=media, neginf=media)
    saida[m] = x[m]
    return saida
