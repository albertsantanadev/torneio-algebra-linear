"""Arquivo de entrega do grupo (integracao: Albert).

O avaliador exige UMA funcao `principal(observada, mascara)`; funcoes de apoio ficam
DENTRO dela. Imports permitidos: numpy, scipy, math e random.

Como funciona:
  1. Existem metodos candidatos: media (seguro), baixo posto (Erlon) e estrutural (Romildo).
  2. Para cada candidato, esconde-se de novo parte das celulas CONHECIDAS (dobras
     deterministicas, sem sorteio) e mede-se o erro nelas. Isso estima como o metodo
     vai se sair nas celulas ocultas de verdade, sem olhar o gabarito.
  3. Usa-se o candidato de menor erro, mas so se ele vencer a media com folga de 2%.
     Matrizes sem estrutura (como a 01 e a 08 do banco publico) caem na media.
  4. Qualquer excecao, formato errado ou NaN/Inf num candidato o descarta.
  5. A saida final sempre mantem as celulas conhecidas e nunca tem NaN/Inf.

Para integrar: cole o corpo dos metodos do Erlon e do Romildo nos dois encaixes marcados
com "SUBSTITUIR". Cada encaixe e uma funcao (o, m) -> matriz completa, onde `o` tem NaN nas
celulas ocultas e `m` e booleana (True = conhecida). Se um metodo tiver funcoes de apoio,
elas tambem ficam dentro do encaixe.

Orcamento de tempo: cada candidato roda (K + 1) vezes (K dobras + ajuste final), K = 5 em
matrizes pequenas e 3 nas maiores. Um metodo que gasta 0,2 s numa 40x40 custa cerca de 0,8 s
aqui. Meca com `python harness/avaliar.py src/principal.py --sinteticas 30`.
"""
import numpy as np


def principal(observada, mascara):
    o = np.array(observada, dtype=float)
    m = np.asarray(mascara).astype(bool)

    # ------------------------------------------------------------------
    # Candidatos: cada um recebe (o, m) e devolve a matriz completa
    # ------------------------------------------------------------------
    def media_global(o, m):
        s = np.array(o, dtype=float)
        mu = s[m].mean() if m.any() else 0.0
        s[~m] = mu
        return s

    def baixo_posto(o, m):
        # SUBSTITUIR pelo metodo de baixo posto do Erlon (hard-impute/ALS com escolha de k)
        return media_global(o, m)

    def estrutural(o, m):
        # SUBSTITUIR pelo metodo estrutural do Romildo (regressao entre colunas, aditivo)
        return media_global(o, m)

    # ------------------------------------------------------------------
    # Execucao segura e validacao cruzada deterministica
    # ------------------------------------------------------------------
    def completar_seguro(metodo, o, m):
        """Roda o metodo como o avaliador faria (NaN nas ocultas). Devolve None se falhar."""
        try:
            s = np.array(metodo(np.where(m, o, np.nan), m.copy()), dtype=float)
        except Exception:
            return None
        if s.shape != o.shape or not np.all(np.isfinite(s)):
            return None
        return s

    def erro_cv(metodo, o, m, K):
        """Erro quadratico medio nas celulas conhecidas escondidas, dobra a dobra."""
        i, j = np.indices(o.shape)
        soma = 0.0
        conta = 0
        for f in range(K):
            teste = m & (((i + 2 * j) % K) == f)
            treino = m & ~teste
            if teste.sum() == 0 or treino.sum() < 2:
                continue
            s = completar_seguro(metodo, o, treino)
            if s is None:
                return np.inf
            soma += float(np.sum((s[teste] - o[teste]) ** 2))
            conta += int(teste.sum())
        if conta == 0:
            return np.inf
        return soma / conta

    # ------------------------------------------------------------------
    # Escolha do metodo
    # ------------------------------------------------------------------
    K = 5 if o.size <= 600 else 3
    erro_media = erro_cv(media_global, o, m, K)
    melhor = media_global
    melhor_erro = erro_media
    for metodo in (baixo_posto, estrutural):
        e = erro_cv(metodo, o, m, K)
        if e < melhor_erro and e < 0.98 * erro_media:
            melhor = metodo
            melhor_erro = e

    # ------------------------------------------------------------------
    # Saida final, sempre valida
    # ------------------------------------------------------------------
    saida = completar_seguro(melhor, o, m)
    if saida is None:
        saida = media_global(o, m)
    saida[m] = o[m]
    mu = o[m].mean() if m.any() else 0.0
    saida[~np.isfinite(saida)] = mu
    return saida
