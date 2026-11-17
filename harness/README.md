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

## Calibracao com o avaliador oficial (fazer antes de confiar nos numeros)

`avaliar.py` imprime duas colunas de NRMSE, uma com o desvio padrao das celulas ocultas e
outra com o da matriz inteira, porque o monitor mudou a normalizacao em 03/11 sem dizer como.

1. Atualizar o container (`docker rm -f avlc_avaliador`, `docker compose pull`, `docker compose up -d`).
2. Enviar o `src/principal.py` base (que preenche com a media) na interface em `http://localhost:8501`.
3. Comparar o NRMSE por matriz da interface com as duas colunas do `avaliar.py`.
4. Se uma coluna bater, essa e a normalizacao oficial. Se nenhuma bater, perguntar no Discord e ajustar `nrmse` em `comum.py`.

## Limites do harness

- O `verificador.py` le o codigo por AST. O RestrictedPython do avaliador pode bloquear mais coisas
  (por exemplo nomes que comecam com `_`); o verificador avisa quando ve isso, mas o teste final e no Docker.
- O tempo medido aqui depende da maquina. O avaliador oficial roda em container limitado, entao
  deixe folga (a meta e ficar abaixo de 1 s por matriz aqui).
