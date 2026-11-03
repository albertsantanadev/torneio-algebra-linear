# Torneio de Álgebra Linear (CIN0138, CIn/UFPE)

Algoritmo em Python/NumPy que estima os valores ocultos de matrizes (completamento de matrizes), avaliado por NRMSE. Equipe: Albert, Erlon e Romildo.

## O desafio

Entregar um `.py` com uma única função `principal(observada, mascara)` que devolve a matriz completa.

- `observada`: matriz com `NaN` nas células ocultas.
- `mascara`: `0/1` (vem como `int64`; converter com `astype(bool)`), `True` onde a célula é conhecida.
- Saída: mesmas dimensões, só números, sem `NaN` nem `Inf`. Células conhecidas mantêm o valor.
- Métrica: NRMSE (RMSE dividido pelo desvio padrão) nas posições ocultas.

Regras completas em [`docs/Regras_Projeto_Algebra.pdf`](docs/Regras_Projeto_Algebra.pdf).

### Restrições do código

- Apenas `numpy`, `scipy`, `math` e `random`. Scikit-learn e PyTorch são proibidos.
- Sem `class`; sem `seed`, `default_rng`, `RandomState` ou `Random`; sem `import time`.
- Funções de apoio ficam **dentro** de `principal`.
- Limite de 2 s por matriz e 25 s por lote. Qualquer saída inválida desclassifica a equipe.

## Estrutura

```
banco_publico/             # 10 matrizes de treino: gabarito, mascara e observada (.npy)
historico_local_tentativas/  # usado pelo avaliador (conteudo ignorado pelo git)
src/
  modulo_baixo_posto.py    # Erlon: SVD truncado, soft-impute, ALS, escolha de k
  modulo_estrutural.py     # Romildo: regressao entre colunas, modelo aditivo, casos de borda
  principal.py             # Albert: integracao final (arquivo de entrega)
harness/                   # avaliacao local (bench.py, bench2.py)
relatorio/                 # relatorio curto exigido na entrega
docs/                      # regras do torneio e README do avaliador
docker-compose.yml         # avaliador oficial (Docker, porta 8501)
```

## Como rodar o avaliador oficial

```bash
touch competicao.db          # precisa ser um arquivo, nao uma pasta
docker compose up            # abre em http://localhost:8501
docker compose down          # para o servico
```

## Como rodar o harness local

```bash
pip install numpy scipy
python harness/bench.py banco_publico
python harness/bench2.py banco_publico harness
```

## Divisão da Etapa 1

| Pessoa | Foco |
|---|---|
| Albert | Avaliador Docker, harness, verificador de regras, integração e entrega |
| Erlon | Métodos de baixo posto e escolha de `k` por validação cruzada |
| Romildo | Métodos estruturais, casos de borda e relatório |

## Prazos

| Data | Marco |
|---|---|
| 10/11 e 12/11 | Aulas de acompanhamento |
| 19/11 | Congelar a versão |
| 20/11, 23:59 | Entrega do `Grupo[N].zip` no Classroom |
| 29/11 | Publicação dos classificados |
| 03/12 | Etapa 2: apresentação e defesa de autoria |

## Entrega

`Grupo[N].zip` com o `.py` final (`principal`) e o relatório curto. Um integrante envia e lista os nomes dos três no comentário da postagem.
