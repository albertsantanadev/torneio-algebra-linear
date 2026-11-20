<div align="center">

# 🧮 Torneio de Reconstrução de Matrizes

**Algoritmo adaptativo de Álgebra Linear para estimar valores ocultos em matrizes, combinando métodos estruturais, decomposição SVD e validação cruzada.**

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Cálculo%20Matricial-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)
[![Avaliação](https://img.shields.io/badge/Docker%20Oficial-10%2F10%20Aprovadas-2EA44F?style=flat-square)](#resultados-e-validação)
[![Limite](https://img.shields.io/badge/Matrizes-até%2040×40-blueviolet?style=flat-square)](#regras-e-limites)
[![Projeto](https://img.shields.io/badge/Projeto-Educacional-64748B?style=flat-square)](#sobre-o-projeto)

**CIN0138 — Álgebra Linear e Vetorial para Computação**

</div>

---

## 📋 Sobre o Projeto

O **Torneio de Reconstrução de Matrizes** é um projeto acadêmico que desafia equipes a desenvolver um algoritmo capaz de **preencher entradas ocultas** de matrizes usando apenas os valores observados e sua organização matemática.

Nossa solução utiliza uma abordagem **adaptativa**: em vez de escolher uma técnica fixa, avalia diferentes estratégias com validação cruzada e usa a que apresenta menor erro estimado nos valores conhecidos reservados para teste.

A implementação final está em [`src/principal.py`](src/principal.py) e pode ser executada de forma independente dos demais módulos do repositório.

---

## 🚀 Principais Recursos

- **🧠 Seleção automática de modelos** — compara alternativas conforme a estrutura da matriz recebida.
- **📊 Métodos estatísticos básicos** — médias global, por linha e por coluna, com tratamentos para ausência de observações.
- **➕ Modelos aditivos regularizados** — combinam efeitos de linha e coluna, reduzindo estimativas instáveis.
- **↔️ Interpolação entre vizinhos** — aproveita valores próximos em linhas e colunas.
- **🧩 Reconstrução de baixo posto** — utiliza decomposição em valores singulares (*SVD truncada*) e reconstrução iterativa (*Hard-Impute*).
- **🔁 Refinamento ALS** — testa mínimos quadrados alternados para aprimorar reconstruções de baixo posto.
- **🎯 Validação cruzada interna** — compara os candidatos usando exclusivamente entradas originalmente conhecidas.
- **🛡️ Tratamento de casos extremos** — mantém dimensões e valores observados e evita saídas com `NaN` ou infinito nos testes realizados.

---

## 🏆 Resultados e Validação

<div align="center">

| Métrica | Resultado oficial — V2 |
|:--|:--:|
| **NRMSE agregado no lote público** | **0,264132** |
| **Matrizes públicas aprovadas** | **10/10** |
| **Tempo total das 10 matrizes** | **0,4440 s** |
| **Maior tempo individual** | **0,1187 s** |
| **Casos extremos locais** | **71/71 aprovados** |

</div>

A versão **V2**, validada em **17/11/2026** no avaliador Docker atualizado, reduziu o NRMSE público da versão V1 de **0,266612** para **0,264132** (aproximadamente **0,93%** de redução relativa).

> **Importante:** esses resultados são do **lote público** e não permitem prever a pontuação no lote privado. Os tempos registrados no Docker podem variar entre ambientes.

Detalhes da verificação: [`RESULTADO_FINAL_DOCKER_E_TESTES.md`](RESULTADO_FINAL_DOCKER_E_TESTES.md).

---

## ⚙️ Como o Algoritmo Funciona

```text
Matriz observada + máscara de valores conhecidos
                     │
                     ▼
         Construção dos candidatos
       ┌─────────────┼──────────────┐
       │             │              │
  Médias e       Vizinhos       SVD truncada
  modelo aditivo                  (Hard-Impute)
       └─────────────┼──────────────┘
                     ▼
       Validação cruzada interna
                     ▼
         Seleção do menor erro
                     ▼
          Refinamento ALS
       (se a SVD for escolhida
        e houver ganho estimado)
                     ▼
     Preservação dos dados conhecidos
                     ▼
            Matriz completa
```

A função obrigatória do torneio é:

```python
principal(observada, mascara)
```

- `observada`: matriz com os valores disponíveis.
- `mascara`: matriz booleana (`True` nas posições conhecidas).
- **Retorno:** matriz numérica completa, com as mesmas dimensões da entrada.

A escolha dos métodos é feita por validação cruzada em **dois grupos determinísticos** de posições conhecidas. Os gabaritos públicos são usados **somente para avaliação externa**, nunca para calcular previsões durante a execução da função.

---

## 🛠️ Tecnologias e Ferramentas

| Tecnologia | Utilização |
|:--|:--|
| **Python** | Implementação do algoritmo e dos testes |
| **NumPy** | Operações matriciais, álgebra linear e SVD |
| **Docker / Docker Compose** | Execução do avaliador oficial em contêiner |
| **RestrictedPython** | Ambiente restrito utilizado pelo avaliador |
| **Git / GitHub** | Versionamento e colaboração da equipe |

A versão submetida em `src/principal.py` usa **NumPy** e não requer bibliotecas externas de aprendizado de máquina.

---

## 📂 Estrutura do Repositório

```text
torneio-algebra-linear/
├── src/
│   ├── principal.py                 # Algoritmo final V2 (submissão)
│   ├── modulo_baixo_posto.py       # Protótipos de SVD / baixo posto
│   └── modulo_estrutural.py        # Protótipos estruturais
├── banco_publico/                  # 10 matrizes e gabaritos públicos
├── harness/
│   ├── avaliar.py                  # NRMSE e tempos em lotes de teste
│   ├── verificador.py              # Verificação local da submissão
│   ├── testes_limites.py           # Casos extremos até 40 × 40
│   └── README.md                   # Instruções dos testes
├── experimentos/                   # Experimentos e resultados anteriores
├── relatorio/
│   ├── relatorio_algoritmo.pdf     # Relatório para o Classroom
│   └── relatorio_algoritmo.md      # Fonte editável do relatório
├── docs/                           # Guia e documentação do avaliador
├── docker-compose.yml              # Avaliador oficial
├── iniciar_avaliador.ps1           # Inicialização no Windows
├── RESULTADO_FINAL_DOCKER_E_TESTES.md
└── README.md
```

> Os módulos de estudo em `src/` ajudam a explicar a evolução do projeto, mas **não são importados** pelo `src/principal.py` entregue.

---

## 📦 Pré-requisitos e Instalação

### Requisitos

- [Python 3](https://www.python.org/downloads/)
- [NumPy](https://numpy.org/)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) com Compose, para usar o avaliador oficial
- [Git](https://git-scm.com/), para clonar o repositório

### Clonar o repositório

```bash
git clone https://github.com/albertsantanadev/torneio-algebra-linear.git
cd torneio-algebra-linear
```

### Instalar a dependência de execução

```bash
python -m pip install numpy
```

Para executar ferramentas auxiliares que utilizam SciPy, instale também `scipy`.

---

## 🧪 Testes Locais

Execute os comandos a partir da raiz do repositório:

```bash
# Verificação do código e de sua saída no lote público
python harness/verificador.py src/principal.py banco_publico

# NRMSE e tempo por matriz pública
python harness/avaliar.py src/principal.py

# Casos extremos, incluindo matrizes 40 × 40
python harness/testes_limites.py src/principal.py

# Lote sintético para testar generalização
python harness/avaliar.py src/principal.py --sinteticas 31 --semente 31 --tamanho-max 40
```

A semente dos testes sintéticos é usada **somente no harness local**; a função submetida não fixa sementes aleatórias.

> A verificação local é complementar. **A aceitação pelo Docker oficial é a referência para compatibilidade com o ambiente restrito.**

---

## 🐳 Avaliador Oficial via Docker

No Windows, inicie o Docker Desktop e execute, na raiz do repositório:

```powershell
powershell -ExecutionPolicy Bypass -File .\iniciar_avaliador.ps1
```

O script prepara `competicao.db` como arquivo SQLite (quando necessário), verifica os diretórios locais, atualiza a imagem e inicia o contêiner. Em seguida, acesse:

**http://localhost:8501**

Na interface, envie **`src/principal.py`**.

Para iniciar manualmente após verificar que `competicao.db` **é um arquivo e não uma pasta**:

```powershell
docker compose pull
docker compose up -d
```

> Se o SQLite apresentar `unable to open database file`, confira se `competicao.db` não foi criado indevidamente como **diretório**. Preserve qualquer dado existente antes de alterar o caminho.

---

## 📐 Regras e Limites

| Restrição | Requisito |
|:--|:--|
| Função obrigatória | `principal(observada, mascara)` |
| Tamanho máximo | **40 × 40** |
| Tempo individual | **2 segundos por matriz** |
| Tempo global | **25 segundos por lote** |
| Bibliotecas permitidas | NumPy, SciPy, `math` e `random` |
| Saída | Mesmas dimensões, sem `NaN` ou infinito |
| Critério de classificação | **NRMSE nas posições ocultas** |

Consulte o guia em [`docs/Regras_Projeto_Algebra.pdf`](docs/Regras_Projeto_Algebra.pdf).

---

## 📄 Relatório e Entrega

A entrega ao Classroom é diferente do repositório completo. O ZIP do grupo precisa conter apenas:

```text
GrupoNUMERO.zip
├── principal.py
└── relatorio_algoritmo.pdf
```

O arquivo Python de entrega corresponde a [`src/principal.py`](src/principal.py). O relatório está em [`relatorio/relatorio_algoritmo.pdf`](relatorio/relatorio_algoritmo.pdf).

**Prazo informado no guia:** 20/11/2026, às 23h59. O número real do grupo deve substituir `NUMERO` no nome do ZIP.

---

## 👥 Equipe e Contribuições

| Integrante | Contribuição principal |
|:--|:--|
| **Albert** | Integração do algoritmo e validação cruzada |
| **Erlon** | SVD, reconstrução de baixo posto e refinamento ALS |
| **Romildo** | Métodos estruturais, interpolação e casos extremos |

A solução é um trabalho colaborativo desenvolvido na disciplina **CIN0138 — Álgebra Linear e Vetorial para Computação**.

---

<div align="center">

<sub>🧮 Torneio de Reconstrução de Matrizes · Versão V2 · Projeto acadêmico · 2026</sub>

</div>
