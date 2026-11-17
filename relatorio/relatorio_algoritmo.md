# Relatório técnico - Torneio de Álgebra Linear (CIN0138)

**Equipe:** Albert, Erlon e Romildo
**Versão de entrega:** V2, validada no avaliador oficial em 17/11/2026

## Objetivo

Implementar `principal(observada, mascara)` para estimar as posições desconhecidas de matrizes retangulares ou quadradas de até 40 × 40. A máscara booleana indica as observações válidas. A saída deve conservar dimensões e entradas conhecidas, conter somente valores finitos e reduzir o NRMSE nas posições ocultas.

## Algoritmo

O algoritmo, escrito somente com NumPy, compara sete soluções estruturais: média global; média por linha; média por coluna; dois modelos aditivos de efeitos de linha e coluna com regularização (`0,5` e `2,0`); e interpolação linear entre vizinhos em cada uma das duas direções. Além delas, avalia reconstruções de baixo posto com SVD truncada iterativa (*Hard-Impute*), com e sem centralização na média global. Os postos candidatos são `k ∈ {1, 2, 3, 4, 5, 6, 8}`, limitados pelas dimensões e pela quantidade de dados observados.

A seleção dos métodos usa apenas valores conhecidos. A máscara é dividida em cinco grupos determinísticos e, em duas rodadas, um grupo é ocultado temporariamente enquanto os outros servem de treinamento. Soma-se o erro quadrático nas posições retidas e escolhe-se o menor, aplicando uma penalização pequena ao posto da SVD para reduzir sobreajuste.

Se a SVD vence, o algoritmo testa um refinamento por *mínimos quadrados alternados* (ALS): ajusta sucessivamente os fatores de linhas e colunas com regularização de `0`, `0,01` ou `0,1`. O refinamento é escolhido por nova validação, com penalização de 1% para os candidatos ALS. A matriz definitiva é recalculada com todas as observações disponíveis.

## Casos especiais e segurança

As células conhecidas são restauradas exatamente. Linhas ou colunas sem dados observados recebem a média global. Previsões exageradas são limitadas usando a dispersão das observações, e eventuais valores não finitos são substituídos por um valor finito. O código não importa arquivos, não fixa sementes aleatórias e não consulta os gabaritos das matrizes.

## Resultados (avaliador oficial atualizado)

| Métrica | Resultado |
|---|---:|
| Matrizes públicas válidas | 10 de 10 |
| NRMSE médio agregado da V2 | **0,264132** |
| NRMSE médio agregado da V1 | 0,266612 |
| Tempo total do lote público (V2) | **0,4440 s** |
| Maior tempo por matriz (V2) | **0,1187 s** |

A V2 melhorou o NRMSE agregado público em aproximadamente 0,93% em relação à V1. As matrizes 01 e 08 continuam como casos difíceis. Em testes locais complementares, 71 de 71 casos extremos foram aprovados; o lote sintético de 186 matrizes apresentou uma pequena redução no NRMSE médio (0,656470 para 0,654948). Tais testes sintéticos não representam o lote privado.

## Limitações e conclusão

A validação cruzada pode selecionar uma abordagem inadequada quando há poucas observações ou ruído elevado. O desempenho no conjunto público não garante a classificação no lote privado. A SVD exige maior custo computacional, por isso há limites de postos e iterações. A aplicação oficial aprovou a V2 no lote público com tempo folgado frente aos limites de 2 segundos por matriz e 25 segundos por lote. Entregamos um único arquivo Python autossuficiente, acompanhado deste relatório.
