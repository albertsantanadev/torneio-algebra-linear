# Harness local

- `bench.py`: baselines (media global/linha/coluna/aditivo), SVD iterativo e validacao cruzada.
  `python harness/bench.py banco_publico`
- `bench2.py`: varredura de posto k sem centralizar.
  `python harness/bench2.py banco_publico harness`

O NRMSE usa o desvio padrao das celulas ocultas. Ainda nao foi conferido contra o
avaliador oficial (tarefa do Albert).
