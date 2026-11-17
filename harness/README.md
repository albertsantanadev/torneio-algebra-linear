# Harness local

Ferramentas para medir e checar o codigo sem o Docker. Nada daqui vai no `.zip` da entrega.

| Arquivo | Para que serve |
|---|---|
| `verificador.py` | Reprova o `principal.py` que fere as regras do avaliador (imports, `class`, `seed`, `default_rng`, assinatura, NaN/Inf, dimensao, tempo). |
| `avaliar.py` | Roda um `principal.py` no banco publico ou em matrizes sinteticas e mostra NRMSE e tempo por matriz. |
| `sinteticas.py` | Gera matrizes de teste (ate 40x40) que simulam o lote privado. |
| `comum.py` | Funcoes usadas pelos tres arquivos acima (carregar dados, NRMSE, checar saida). |
| `bench.py` | Baselines (media global/linha/coluna/aditivo), SVD iterativo e validacao cruzada. |
| `bench2.py` | Varredura do posto k sem centralizar. |

## Como usar

```bash
pip install numpy scipy

# 1. Antes de qualquer envio: o codigo cumpre as regras?
python harness/verificador.py src/principal.py banco_publico

# 2. Qual o erro e o tempo no banco publico?
python harness/avaliar.py src/principal.py

# 3. E em matrizes que o modelo nunca viu (inclui 40x40, sem estrutura e casos de borda)?
python harness/avaliar.py src/principal.py --sinteticas 30 --semente 7

# 4. Comparar duas versoes lado a lado
python harness/avaliar.py src/principal.py --comparar src/outra_versao.py

# Scripts antigos de exploracao
python harness/bench.py banco_publico
python harness/bench2.py banco_publico harness
```

## Validacao no avaliador oficial

A versão final V2 foi aceita no Docker atualizado em 17/11/2026, com NRMSE público `0,264132` e 0,4440 s no lote. A coluna `NRMSE(ocultas)` do harness corresponde aos resultados observados no avaliador oficial. Continue a usar o Docker para confirmar qualquer alteração no código; os tempos locais variam conforme a máquina.

## Limites do harness

- O `verificador.py` le o codigo por AST. O RestrictedPython do avaliador pode bloquear mais coisas
  (por exemplo nomes que comecam com `_`); o verificador avisa quando ve isso, mas o teste final e no Docker.
- O tempo medido aqui depende da maquina. O avaliador oficial roda em container limitado, entao
  deixe folga (a meta e ficar abaixo de 1 s por matriz aqui).
