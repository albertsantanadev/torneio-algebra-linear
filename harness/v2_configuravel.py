"""Replica configuravel da V2 (src/principal.py) para estudos de ablacao.

NAO DEVERIA SER ENVIADO na entrega. Serve para medir o efeito de cada peca do metodo final:
liga/desliga candidatos, centralizacao da SVD, penalidade de posto, polimento ALS
e trava de valores extremos, e permite somar candidatos novos (suavidade, ridge,
aditivo + posto 1, simetria, Toeplitz) ao mesmo concurso de validacao cruzada.

Com a configuracao padrao (CONFIG_V2) a saida e identica, celula por celula, a de
src/principal.py. O script harness/ablacao.py confere isso antes de montar as tabelas.
"""
import numpy as np

TIPOS_V2 = ("global", "linha", "coluna", "aditivo", "aditivo_reg", "viz_linha", "viz_coluna")

CONFIG_V2 = {
    "tipos": TIPOS_V2,          # candidatos estruturais da V2
    "svd": True,                # candidatos SVD (hard-impute)
    "svd_centros": (False, True),
    "postos": (1, 2, 3, 4, 5, 6, 8),
    "penal_posto": 0.003,       # erro * (1 + 0.003 k)
    "als": True,                # polimento ALS quando a SVD vence
    "trava": True,              # limita a media +- 10 desvios
    "grupos_cv": (0, 2),        # dobras usadas (de 5)
    "extras": (),               # candidatos novos: ver EXTRAS abaixo
    "penal_extra": 0.0,         # penalidade relativa para candidatos novos
    "plano_b": False,           # se a SVD falhar/gerar NaN: aditivo + posto 1 por ALS
    "forcar_falha_svd": False,  # SO PARA TESTE: toda chamada de SVD falha
}


def config(**mudancas):
    c = dict(CONFIG_V2)
    c.update(mudancas)
    return c


# ----------------------------------------------------------------------------
# Candidatos novos (cada um: (x, mask) -> matriz prevista n x p)
# x tem 0 nas celulas fora de mask.
# ----------------------------------------------------------------------------

def _resolver(A, b, lam):
    """Equacoes normais com ridge: (A^T A + lam I) c = A^T b, com plano B por lstsq."""
    d = A.shape[1]
    M = A.T @ A + lam * np.eye(d)
    try:
        return np.linalg.solve(M, A.T @ b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(M, A.T @ b, rcond=None)[0]


def poli_eixo(x, mask, grau, eixo):
    """Suavidade: cada coluna (eixo=0) ou linha (eixo=1) vira um polinomio do indice."""
    if eixo == 1:
        return poli_eixo(x.T, mask.T, grau, 0).T
    n, p = x.shape
    t = np.linspace(-1.0, 1.0, n)
    V = np.vander(t, grau + 1, increasing=True)
    mu = float(np.sum(x[mask]) / max(1, int(np.sum(mask))))
    out = np.full((n, p), mu)
    for j in range(p):
        k = mask[:, j]
        q = int(np.sum(k))
        if q == 0:
            continue
        d = min(grau + 1, q)
        c = _resolver(V[k, :d], x[k, j], 1e-8)
        out[:, j] = V[:, :d] @ c
    return out


def superficie(x, mask, grau):
    """Superficie polinomial global: soma de c_ab * i^a * j^b com a + b <= grau."""
    n, p = x.shape
    ti = np.linspace(-1.0, 1.0, n)
    tj = np.linspace(-1.0, 1.0, p)
    I, J = np.indices((n, p))
    cols = []
    for a in range(grau + 1):
        for b in range(grau + 1 - a):
            cols.append((ti[I] ** a) * (tj[J] ** b))
    F = np.stack(cols, axis=-1)  # n x p x d
    A = F[mask]
    c = _resolver(A, x[mask], 1e-8)
    return F @ c


def aditivo_lstsq(x, mask):
    """b_add do bench.py: mu + a_i + b_j por minimos quadrados (lstsq)."""
    n, p = x.shape
    I, J = np.nonzero(mask)
    A = np.zeros((len(I), 1 + n + p))
    A[:, 0] = 1.0
    A[np.arange(len(I)), 1 + I] = 1.0
    A[np.arange(len(I)), 1 + n + J] = 1.0
    c = np.linalg.lstsq(A, x[I, J], rcond=None)[0]
    return c[0] + c[1:1 + n][:, None] + c[1 + n:][None, :]


def aditivo_posto1(x, mask, lam=0.05, passos=25):
    """mu + a_i + b_j + u_i v_j. Posto 1 por ALS com ridge (nao depende da SVD)."""
    n, p = x.shape
    base = aditivo_lstsq(x, mask)
    w = mask.astype(float)
    r = np.where(mask, x - base, 0.0)
    escala = float(np.sum(r * r) / max(1, int(np.sum(mask))))
    # inicio de v pela coluna de maior energia do residuo (deterministico)
    v = np.sqrt(np.sum(r * r, axis=0) + 1e-12)
    u = np.zeros(n)
    for t in range(passos):
        u = (r @ v) / (w @ (v * v) + lam * escala + 1e-12)
        v = (r.T @ u) / (w.T @ (u * u) + lam * escala + 1e-12)
        base = aditivo_lstsq(np.where(mask, x - np.outer(u, v), 0.0), mask)
        r = np.where(mask, x - base, 0.0)
    return base + np.outer(u, v)


def ridge_colunas(x, mask, alfa, passos=3, eixo=0):
    """Cada coluna e prevista pelas outras colunas: (A^T A + lam I) c = A^T y."""
    if eixo == 1:
        return ridge_colunas(x.T, mask.T, alfa, passos, 0).T
    n, p = x.shape
    z = np.where(mask, x, aditivo_lstsq(x, mask))
    for t in range(passos):
        novo = z.copy()
        for j in range(p):
            k = mask[:, j]
            q = int(np.sum(k))
            if q == n or q < 2 or p < 2:
                continue
            A = np.delete(z, j, axis=1)
            media = A[k].mean(axis=0)
            desvio = A[k].std(axis=0) + 1e-9
            As = (A - media) / desvio
            y = x[k, j]
            ym = float(np.mean(y))
            c = _resolver(As[k], y - ym, alfa * q)
            prev = As @ c + ym
            novo[~k, j] = prev[~k]
        z = novo
    return z


def simetrica(x, mask, base):
    """Se a matriz for quadrada: oculta (i,j) recebe a observada (j,i); o resto vem de base."""
    n, p = x.shape
    if n != p:
        return base
    par = mask.T & ~mask
    out = np.array(base, copy=True)
    out[par] = x.T[par]
    return out


def toeplitz(x, mask, base):
    """Media de cada diagonal observada (i - j constante); sem dados na diagonal, usa base."""
    n, p = x.shape
    I, J = np.indices((n, p))
    D = I - J
    out = np.array(base, copy=True)
    for d in range(-(p - 1), n):
        na_diag = D == d
        obs = na_diag & mask
        if np.any(obs):
            out[na_diag] = float(np.mean(x[obs]))
    return out


EXTRAS = {
    "poliC1": lambda x, k, ctx: poli_eixo(x, k, 1, 0),
    "poliC2": lambda x, k, ctx: poli_eixo(x, k, 2, 0),
    "poliC3": lambda x, k, ctx: poli_eixo(x, k, 3, 0),
    "poliL1": lambda x, k, ctx: poli_eixo(x, k, 1, 1),
    "poliL2": lambda x, k, ctx: poli_eixo(x, k, 2, 1),
    "poliL3": lambda x, k, ctx: poli_eixo(x, k, 3, 1),
    "superf1": lambda x, k, ctx: superficie(x, k, 1),
    "superf2": lambda x, k, ctx: superficie(x, k, 2),
    "aditivo_lstsq": lambda x, k, ctx: aditivo_lstsq(x, k),
    "aditivo_p1": lambda x, k, ctx: aditivo_posto1(x, k),
    "ridgeC": lambda x, k, ctx: ridge_colunas(x, k, 0.01, eixo=0),
    "ridgeL": lambda x, k, ctx: ridge_colunas(x, k, 0.01, eixo=1),
    "simetrica": lambda x, k, ctx: simetrica(x, k, ctx["aditivo"]),
    "toeplitz": lambda x, k, ctx: toeplitz(x, k, ctx["aditivo"]),
}


# ----------------------------------------------------------------------------
# Replica da V2 com chaves de configuracao
# ----------------------------------------------------------------------------

def principal_config(observada, mascara, cfg=None, info=None):
    cfg = CONFIG_V2 if cfg is None else cfg
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

    cache_plano_b = {}

    def svd_fit(a, mask, mu, inicio, k, centrado, iteracoes):
        z = np.where(mask, a, inicio)
        centro = mu if centrado else 0.0
        for t in range(iteracoes):
            try:
                if cfg.get("forcar_falha_svd"):
                    raise np.linalg.LinAlgError("falha simulada")
                u, singular, vt = np.linalg.svd(z - centro, full_matrices=False)
                aproximacao = centro + (u[:, :k] * singular[:k]) @ vt[:k, :]
                if cfg.get("plano_b") and not np.all(np.isfinite(aproximacao)):
                    raise np.linalg.LinAlgError("SVD gerou NaN/Inf")
                z = np.where(mask, a, aproximacao)
            except np.linalg.LinAlgError:
                if cfg.get("plano_b"):
                    # Plano B: aditivo + posto 1 por ALS, que nao usa SVD.
                    if info is not None:
                        info["plano_b"] = info.get("plano_b", 0) + 1
                    try:
                        chave = mask.tobytes()  # o plano B nao depende de k: calcula 1 vez por mascara
                        if chave not in cache_plano_b:
                            cache_plano_b[chave] = aditivo_posto1(a, mask)
                        alt = cache_plano_b[chave]
                        if np.all(np.isfinite(alt)):
                            return np.where(mask, a, alt)
                    except Exception:  # noqa: BLE001
                        pass
                break  # V2: fica com o ultimo z valido (no inicio, o aditivo)
        return z

    def polir_als(a, mask, pre, k, centrado, mu, reg, passos):
        centro = mu if centrado else 0.0
        try:
            if cfg.get("forcar_falha_svd"):
                raise np.linalg.LinAlgError("falha simulada")
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

    def estruturais(mask):
        """Previsoes dos candidatos estruturais da V2 treinados em `mask`."""
        a, mu, glob, linha, coluna, nr, nc = basicos(mask)
        tabela = {}
        tabela["global"] = glob
        tabela["linha"] = linha
        tabela["coluna"] = coluna
        add = aditivo(a, mu, mask, nr, nc, 0.5)
        tabela["aditivo"] = add
        tabela["aditivo_reg"] = aditivo(a, mu, mask, nr, nc, 2.0)
        tabela["viz_linha"] = vizinhos(a, mask, coluna, True)
        tabela["viz_coluna"] = vizinhos(a, mask, linha, False)
        return a, mu, add, tabela

    ii, jj = np.indices((n, p))
    grupos = ((ii * 53 + jj * 97 + ii * jj * 29 + ii * ii * 7 + jj * jj * 11) % 5)
    postos = [k for k in cfg["postos"] if k <= min(n, p) and conhecidos >= 3 * k] if cfg["svd"] else []
    tipos = list(cfg["tipos"])
    extras = list(cfg["extras"])
    cand = [(t, 0) for t in tipos] + [(e, 0) for e in extras] + [("svd", (k, c)) for k in postos for c in cfg["svd_centros"]]
    inicio_svd = len(tipos) + len(extras)
    erros = np.zeros(len(cand), dtype=float)
    total = 0
    iteracoes_cv = 28 if n * p <= 600 else 23
    for grupo in cfg["grupos_cv"]:
        teste = m & (grupos == grupo)
        treino = m & ~teste
        if int(np.sum(teste)) == 0 or int(np.sum(treino)) < 4:
            continue
        a, mu, add, tabela = estruturais(treino)
        ctx = {"aditivo": add}
        for cid in range(len(cand)):
            nome, par = cand[cid]
            if nome == "svd":
                k, cent = par
                pre = svd_fit(a, treino, mu, add, k, cent, iteracoes_cv)
            elif nome in tabela:
                pre = tabela[nome]
            else:
                try:
                    pre = EXTRAS[nome](a, treino, ctx)
                    if not np.all(np.isfinite(pre)):
                        raise FloatingPointError
                except Exception:  # noqa: BLE001
                    pre = None
            if pre is None:
                erros[cid] = np.inf
                continue
            d = pre[teste] - x[teste]
            erros[cid] = erros[cid] + float(d @ d)
        total += int(np.sum(teste))
    if total:
        erros /= total
    else:
        erros[:] = np.inf
    for cid in range(inicio_svd, len(cand)):
        k, cent = cand[cid][1]
        erros[cid] = erros[cid] * (1.0 + cfg["penal_posto"] * k)
    for cid in range(len(tipos), inicio_svd):
        erros[cid] = erros[cid] * (1.0 + cfg["penal_extra"])
    melhor = int(np.argmin(erros))
    if not np.isfinite(erros[melhor]):
        melhor = 0  # sem validacao possivel: primeiro candidato (media global na V2)
    a, mu, add, tabela = estruturais(m)
    nome, par = cand[melhor]
    if nome == "svd":
        k, cent = par
        pre = svd_fit(a, m, mu, add, k, cent, 85)
    elif nome in tabela:
        pre = tabela[nome]
    else:
        try:
            pre = EXTRAS[nome](a, m, {"aditivo": add})
            if not np.all(np.isfinite(pre)):
                raise FloatingPointError
        except Exception:  # noqa: BLE001
            pre = add
    escolha = nome if nome != "svd" else f"svd k={par[0]}{' c' if par[1] else ''}"
    if nome == "svd" and cfg["als"]:
        k, cent = par
        opcoes = [0.0, 0.01, 0.1]
        for_reg = np.zeros(len(opcoes) + 1)
        nt = 0
        for grupo in cfg["grupos_cv"]:
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
                escolha += f" +ALS({opcoes[ind - 1]})"

    nr_total = np.sum(m, axis=1)
    nc_total = np.sum(m, axis=0)
    pre = np.where((nr_total == 0)[:, None], media, pre)
    pre = np.where((nc_total == 0)[None, :], media, pre)
    pre = np.where((nr_total == 0)[:, None] & (nc_total == 0)[None, :], media, pre)
    if cfg["trava"]:
        desv = float(np.sqrt(np.mean((x[m] - media) ** 2)))
        if desv > 0.0:
            pre = np.clip(pre, media - 10.0 * desv, media + 10.0 * desv)
    saida = np.where(m, x, pre)
    saida = np.nan_to_num(saida, nan=media, posinf=media, neginf=media)
    saida[m] = x[m]
    if info is not None:
        info["escolha"] = escolha
    return saida
