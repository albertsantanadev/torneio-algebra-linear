"""Gerador de matrizes sinteticas para simular o lote privado

As 10 matrizes publicas sao poucas para escolher metodos sem overfitting, e o
aviso do monitor (03/11) diz que o lote oficial pode ter matrizes ate 40x40.
Aqui geramos famílias parecidas com o que vimos no banco publico:

  baixo_posto     posto 1..8 + ruido pequeno, com nivel medio grande ou nulo
  posto_exato     posto 1..6 sem ruido (como as matrizes 09 e 10)
  so_ruido        valores independentes em torno de uma media (como a 01)
  espectro_linear posto cheio, valores singulares em progressao linear (como a 08)
  borda           baixo posto com 55% ocultas e linha/coluna inteira oculta

Familias com estrutura (opcionais, usadas nos testes de simetria, Toeplitz e
suavidade; nao entram no lote padrao, para manter os numeros antigos):

  tendencia_linha  cada linha e quase constante e o nivel cresce com i (como a 02)
  tendencia_coluna cada coluna e quase constante e o nivel cresce com j (como a 03)
  poli_coluna      cada coluna e um polinomio de grau 1 ou 2 no indice i (como a 04 e a 05)
  poli_linha       cada linha e um polinomio de grau 1 ou 2 no indice j
  simetrica        matriz simetrica (B B^T + ruido simetrico)
  toeplitz         valor depende so de i - j (cada diagonal e constante) + ruido

Uso isolado: python harness/sinteticas.py  (imprime um resumo do lote)
"""
import numpy as np

FAMILIAS = ["baixo_posto", "posto_exato", "so_ruido", "espectro_linear", "borda"]
FAMILIAS_ESTRUTURA = ["tendencia_linha", "tendencia_coluna", "poli_coluna", "poli_linha", "simetrica", "toeplitz"]


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
    elif familia in ("tendencia_linha", "tendencia_coluna"):
        a, b = rng.uniform(-20, 60), rng.uniform(1, 12) * rng.choice([-1.0, 1.0])
        t = np.arange(n if familia == "tendencia_linha" else p, dtype=float)
        nivel = a + b * t
        g = nivel[:, None] + np.zeros((n, p)) if familia == "tendencia_linha" else nivel[None, :] + np.zeros((n, p))
        g = g + rng.normal(size=(n, p)) * abs(b) * rng.uniform(0.03, 0.12) * max(n, p) / 3
    elif familia in ("poli_coluna", "poli_linha"):
        tam, outro = (n, p) if familia == "poli_coluna" else (p, n)
        t = np.linspace(-1, 1, tam)
        grau = int(rng.integers(1, 3))
        coef = rng.normal(size=(grau + 1, outro)) * rng.uniform(2, 10)
        g = np.vander(t, grau + 1, increasing=True) @ coef
        g = g + rng.normal(size=g.shape) * g.std() * rng.uniform(0.005, 0.04)
        if familia == "poli_linha":
            g = g.T
    elif familia == "simetrica":
        p = n
        k = int(rng.integers(1, 6))
        B = rng.normal(size=(n, k))
        ruido = rng.normal(size=(n, n)) * rng.uniform(0.02, 0.3)
        g = B @ B.T + (ruido + ruido.T) / 2 + rng.uniform(-5, 5)
    elif familia == "toeplitz":
        seq = np.cumsum(rng.normal(size=n + p - 1)) * rng.uniform(0.5, 3) + rng.uniform(-20, 20)
        I, J = np.indices((n, p))
        g = seq[I - J + p - 1] + rng.normal(size=(n, p)) * rng.uniform(0.02, 0.2)
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


def gerar_lote(quantidade=30, semente=0, tamanho_max=40, tamanho_min=6, familias=None):
    """Gera `quantidade` matrizes alternando as familias. Cada item: nome, familia, g, m, o.

    `familias=None` usa o lote padrao (FAMILIAS); passe FAMILIAS_ESTRUTURA ou uma lista
    propria para os testes de estrutura.
    """
    familias = FAMILIAS if familias is None else list(familias)
    rng = np.random.default_rng(semente)  # harness local: aqui default_rng e permitido
    itens = []
    for i in range(quantidade):
        familia = familias[i % len(familias)]
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
