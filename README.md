# Torneio de Reconstrução de Matrizes - CIN0138

**Versão final para o Classroom: V2 (17/11/2026)**
Equipe: Albert, Erlon e Romildo.

> **Código congelado:** [`src/principal.py`](src/principal.py) é uma cópia **idêntica byte a byte** do código V2 que passou no Docker oficial atualizado (NRMSE público `0,264132`, 10/10 matrizes, 0,4440 s no lote). Não mudar o algoritmo sem revalidar no Docker.

## Onde está cada parte

| Arquivo ou pasta | Finalidade |
|---|---|
| [`src/principal.py`](src/principal.py) | **Código a enviar ao avaliador**; implementa `principal(observada, mascara)` sem depender dos outros arquivos |
| [`relatorio/relatorio_algoritmo.pdf`](relatorio/relatorio_algoritmo.pdf) | **Relatório curto para o Classroom** com método e resultados |
| `src/modulo_baixo_posto.py` | Protótipo de estudo da SVD, **não** usado pelo arquivo de entrega |
| `src/modulo_estrutural.py` | Protótipo de estudo do modelo aditivo, **não** usado pelo arquivo de entrega |
| `harness/` | Scripts de verificação, benchmark e casos extremos locais |
| `banco_publico/` | As dez matrizes e respectivos gabaritos, **apenas para testes**, nunca para a previsão |
| `experimentos/` | Algoritmos e resultados históricos; não são a versão de entrega |
| `docs/` | Guia oficial e notas sobre o avaliador |
| `docker-compose.yml` | Configuração do avaliador oficial |
| `iniciar_avaliador.ps1` | Preparação do SQLite + atualização/execução do Docker no Windows |

## Como funciona a versão final

O algoritmo é adaptativo: constrói estimativas pela média global, média de linhas e colunas, dois modelos aditivos regularizados, interpolação linear entre vizinhos nas duas direções, e reconstrução iterativa por SVD truncada (`Hard-Impute`) com escolha de posto entre 1, 2, 3, 4, 5, 6 e 8, quando viável. Se a SVD é selecionada, testa um refinamento de mínimos quadrados alternados (ALS) e três intensidades de regularização.

A escolha é feita por validação cruzada em **dois de cinco grupos determinísticos** de entradas conhecidas. Cada candidato é avaliado pelo erro quadrático nas entradas retidas. O algoritmo restaura as observações reais na matriz retornada, trata linhas/colunas vazias e evita NaN/Inf.

## Resultado confirmado no Docker atualizado (17/11/2026)

| Indicador | V1 anterior | V2 final |
|---|---:|---:|
| NRMSE agregado público | 0,266612 | **0,264132** |
| Matrizes públicas aprovadas | 10/10 | **10/10** |
| Tempo total público | - | **0,4440 s** |
| Maior tempo individual | - | **0,1187 s** |

A redução relativa no NRMSE público foi de aproximadamente **0,93%**. A V2 também passou em 71/71 casos extremos em testes locais, incluindo tamanhos 40x40. O conjunto privado pode produzir outros resultados; não existe garantia de classificação.

## Testar localmente

Instale Python e NumPy (`python -m pip install numpy`), abra o terminal na raiz do repositório e execute:

```powershell
python harness/verificador.py src/principal.py banco_publico
python harness/testes_limites.py src/principal.py
python harness/avaliar.py src/principal.py
python harness/avaliar.py src/principal.py --sinteticas 31 --semente 31 --tamanho-max 40
```

Os scripts de teste utilizam dados públicos e geradores aleatórios **apenas fora da submissão**. O verificador local não substitui o avaliador oficial RestrictedPython.

## Iniciar o Docker no Windows (instalação nova)

1. Abra o Docker Desktop, entre na pasta deste repositório no PowerShell.
2. Execute `powershell -ExecutionPolicy Bypass -File .\iniciar_avaliador.ps1` (leia o script antes, caso prefira). Ele cria um **arquivo** SQLite vazio `competicao.db` caso não exista e inicia a imagem atualizada. Para atualizar manualmente: certifique-se de que `competicao.db` é um arquivo, então execute `docker rm -f avlc_avaliador`, `docker compose pull`, `docker compose up -d`.
3. Abra `http://localhost:8501` e envie `src/principal.py`.

**Erro SQLite conhecido:** se `competicao.db` for uma **pasta**, pare o Docker com `docker compose down`, renomeie essa pasta para backup e crie um arquivo vazio `competicao.db` antes de subir a aplicação. O `docker-compose.yml` foi configurado para não criar o caminho ausente automaticamente como pasta. O banco `competicao.db` e os históricos locais são ignorados pelo Git.

## Envio ao Classroom - até 20/11/2026 às 23:59

O pacote `GrupoNUMERO.zip` (gerado separadamente) contém **apenas**:

```text
principal.py
relatorio_algoritmo.pdf
```

1. Substitua `NUMERO` pelo número real da equipe (por exemplo `Grupo7.zip`).
2. Envie o ZIP no Classroom; somente **um** integrante precisa enviar.
3. Informe os nomes completos dos integrantes no **comentário particular** da atividade.

O repositório completo (incluindo os testes) é para GitHub. **Não use o ZIP do repositório como arquivo de entrega do Classroom.**

## Divisão das contribuições para a apresentação

- **Albert:** integração, seleção por validação cruzada e verificação no avaliador.
- **Erlon:** SVD truncada de posto baixo e escolha de `k`, com ALS na V2.
- **Romildo:** modelos estruturais, interpolação, tratamento de casos extremos.

Na apresentação, todos devem saber explicar a máscara, o erro quadrático, por que modelos de posto baixo são úteis, o que ALS refina e quais são as limitações nos dados ruidosos.
