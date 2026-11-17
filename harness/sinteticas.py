"""Gerador de matrizes sinteticas para simular o lote privado

As 10 matrizes publicas sao poucas para escolher metodos sem overfitting, e o
aviso do monitor (03/11) diz que o lote oficial pode ter matrizes ate 40x40.
Aqui geramos famílias parecidas com o que vimos no banco publico:

  baixo_posto     posto 1..8 + ruido pequeno, com nivel medio grande ou nulo
  posto_exato     posto 1..6 sem ruido (como as matrizes 09 e 10)
  so_ruido        valores independentes em torno de uma media (como a 01)
  espectro_linear posto cheio, valores singulares em progressao linear (como a 08)
  borda           baixo posto com 55% ocultas e linha/coluna inteira oculta

Uso isolado: python harness/sinteticas.py  (imprime um resumo do lote)
"""
import numpy as np

FAMILIAS = ["baixo_posto", "posto_exato", "so_ruido", "espectro_linear", "borda"]


def _mascara(rng, n, p, frac_oculta, minimo=2):
    """Mascara booleana (True = conhecida) com pelo menos `minimo` conhecidas por linha/coluna."""
    m = rng.random((n, p)) >= frac_oculta
    for i in range(n):
        if m[i].sum() < minimo:
            m[i, rng.choice(p, size=minimo, replace=False)] = True
    for j in range(p):
        if m[:, j].sum() < minimo:
            m[rng.choice(n, size=minimo, replace=False), j] = True
    return m


def _uma(rng, familia, n, p):
    frac = rng.uniform(0.12, 0.42)
    if familia == "baixo_posto":
        k = int(rng.integers(1, 9))
        k = min(k, min(n, p) - 1)
        sinal = rng.normal(size=(n, k)) @ rng.normal(size=(k, p))
        sinal *= rng.uniform(1, 10) / sinal.std()
        g = sinal + rng.choice([0.0, rng.uniform(10, 60)]) + rng.normal(size=(n, p)) * sinal.std() * rng.uniform(0.005, 0.05)
    elif familia == "posto_exato":
        k = int(rng.integers(1, 7))
        k = min(k, min(n, p) - 1)
        g = rng.normal(size=(n, k)) @ rng.normal(size=(k, p)) * rng.uniform(1, 5) + rng.uniform(5, 40)
    elif familia == "so_ruido":
        g = rng.normal(loc=rng.uniform(-50, 50), scale=rng.uniform(0.5, 5), size=(n, p))
    elif familia == "espectro_linear":
        r = min(n, p)
        U, _ = np.linalg.qr(rng.normal(size=(n, r)))
        V, _ = np.linalg.qr(rng.normal(size=(p, r)))
        s = np.linspace(50, 1, r)
        g = (U * s) @ V.T
    elif familia == "borda":
        k = int(rng.integers(1, 5))
        k = min(k, min(n, p) - 1)
        g = rng.normal(size=(n, k)) @ rng.normal(size=(k, p)) * rng.uniform(1, 5) + rng.uniform(5, 40)
        frac = 0.55
    else:
        raise ValueError(familia)

    m = _mascara(rng, n, p, frac, minimo=1 if familia == "borda" else 2)
    if familia == "borda":
        m[int(rng.integers(0, n))] = False   # uma linha inteira oculta
        m[:, int(rng.integers(0, p))] = False  # uma coluna inteira oculta
        if m.all() or not m.any():
            m[0, 0] = not m[0, 0]
    o = np.where(m, g, np.nan)
    return {"g": g, "m": m, "o": o}


def gerar_lote(quantidade=30, semente=0, tamanho_max=40, tamanho_min=6):
    """Gera `quantidade` matrizes alternando as familias. Cada item: nome, familia, g, m, o."""
    rng = np.random.default_rng(semente)  # harness local: aqui default_rng e permitido
    itens = []
    for i in range(quantidade):
        familia = FAMILIAS[i % len(FAMILIAS)]
        n = int(rng.integers(tamanho_min, tamanho_max + 1))
        p = int(rng.integers(tamanho_min, tamanho_max + 1))
        item = _uma(rng, familia, n, p)
        item.update({"nome": f"s{i + 1:02d}-{familia}", "familia": familia})
        itens.append(item)
    # garante pelo menos uma 40x40 de baixo posto no lote (pior caso de tempo)
    if quantidade >= 1 and tamanho_max >= 40:
        pior = _uma(rng, "baixo_posto", 40, 40)
        pior.update({"nome": "s00-pior-caso-40x40", "familia": "baixo_posto"})
        itens.append(pior)
    return itens


if __name__ == "__main__":
    lote = gerar_lote(10, semente=1)
    for it in lote:
        print(f"{it['nome']:<28} {str(it['g'].shape):>9}  ocultas={(~it['m']).mean():4.0%}")
