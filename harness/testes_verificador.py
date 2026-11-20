"""Testes de regressao do verificador de regras

Uso: python harness/testes_verificador.py [pasta_banco]

Cada caso e um principal.py "ruim" escrito numa pasta temporaria. O teste roda o
verificador (etapa estatica e dinamica) e exige que ele REPROVE com a mensagem
esperada. Tambem exige que um principal.py minimo valido e o src/principal.py da
entrega sejam APROVADOS. Codigo de saida: 0 = todos os casos passaram.
"""
import os
import sys
import tempfile
import textwrap

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, AQUI)
from verificador import checar_dinamico, checar_estatico  # noqa: E402

# Corpo valido usado como base: media das conhecidas nas ocultas.
VALIDO = """
import numpy as np

def principal(observada, mascara):
    m = np.asarray(mascara, dtype=bool)
    o = np.where(m, np.asarray(observada, dtype=float), 0.0)
    media = float(o[m].mean()) if m.any() else 0.0
    return np.where(m, o, media)
"""


def com_corpo(linhas_extras, retorno="np.where(m, o, media)", topo=""):
    """Principal valido com linhas extras antes do return e/ou codigo no topo."""
    corpo = textwrap.indent(textwrap.dedent(linhas_extras).strip("\n"), "    ") if linhas_extras else ""
    return f"""{topo}
import numpy as np

def principal(observada, mascara):
    m = np.asarray(mascara, dtype=bool)
    o = np.where(m, np.asarray(observada, dtype=float), 0.0)
    media = float(o[m].mean()) if m.any() else 0.0
{corpo}
    return {retorno}
"""


# (nome, codigo, etapa, trecho esperado na mensagem de erro)
CASOS_RUINS = [
    # --- regras do codigo (etapa estatica) -----------------------------------
    ("class no topo", VALIDO + "\nclass Modelo:\n    pass\n", "estatica", "`class` e proibido"),
    ("class dentro de principal", com_corpo("class Aux:\n    x = 1"), "estatica", "`class` e proibido"),
    ("import time", com_corpo("", topo="import time"), "estatica", "import nao permitido: time"),
    ("import time dentro da funcao", com_corpo("import time\nt0 = time.time()"), "estatica", "import nao permitido: time"),
    ("from time import ...", com_corpo("", topo="from time import perf_counter"), "estatica", "from time"),
    ("import os", com_corpo("", topo="import os"), "estatica", "import nao permitido: os"),
    ("import sklearn", com_corpo("", topo="import sklearn"), "estatica", "import nao permitido: sklearn"),
    ("import pandas como alias", com_corpo("", topo="import pandas as pd"), "estatica", "import nao permitido: pandas"),
    ("__import__", com_corpo("t = __import__('time')"), "estatica", "__import__"),
    ("np.random.seed", com_corpo("np.random.seed(0)"), "estatica", ".seed"),
    ("random.seed", com_corpo("random.seed(1)", topo="import random"), "estatica", ".seed"),
    ("np.random.default_rng", com_corpo("g = np.random.default_rng(0)"), "estatica", ".default_rng"),
    ("np.random.RandomState", com_corpo("g = np.random.RandomState(0)"), "estatica", ".RandomState"),
    ("random.Random", com_corpo("g = random.Random(3)", topo="import random"), "estatica", ".Random"),
    ("from numpy.random import default_rng", com_corpo("", topo="from numpy.random import default_rng"), "estatica", "import proibido: default_rng"),
    ("from random import seed", com_corpo("", topo="from random import seed"), "estatica", "import proibido: seed"),
    ("from random import Random", com_corpo("", topo="from random import Random"), "estatica", "import proibido: Random"),
    ("getattr com 'seed'", com_corpo("getattr(np.random, 'seed')(0)"), "estatica", "acesso indireto proibido"),
    ("sem funcao principal", "import numpy as np\n\ndef resolver(observada, mascara):\n    return observada\n", "estatica", "nao existe a funcao `principal`"),
    ("assinatura errada", "import numpy as np\n\ndef principal(obs, mask):\n    return obs\n", "estatica", "assinatura deve ser"),
    # --- saida (etapa dinamica, roda no banco publico) -----------------------
    ("saida com NaN", com_corpo("", retorno="np.where(m, o, np.nan)"), "dinamica", "NaN"),
    ("saida com Inf", com_corpo("", retorno="np.where(m, o, np.inf)"), "dinamica", "Inf"),
    ("saida com -Inf", com_corpo("", retorno="np.where(m, o, -np.inf)"), "dinamica", "Inf"),
    ("dimensao transposta", com_corpo("", retorno="np.where(m, o, media).T"), "dinamica", "dimensao errada"),
    ("linha a menos", com_corpo("", retorno="np.where(m, o, media)[:-1]"), "dinamica", "dimensao errada"),
    ("saida vetor", com_corpo("", retorno="np.where(m, o, media).ravel()"), "dinamica", "dimensao errada"),
    ("celula conhecida alterada", com_corpo("", retorno="np.where(m, o + 1e-3, media)"), "dinamica", "conhecida(s) alterada(s)"),
    ("todas as conhecidas substituidas", com_corpo("", retorno="np.full(o.shape, media)"), "dinamica", "conhecida(s) alterada(s)"),
    ("lista irregular", com_corpo("", retorno="[[1.0], [1.0, 2.0]]"), "dinamica", "converter a saida"),
    ("retorna None", com_corpo("", retorno="None"), "dinamica", "tipo nao numerico"),
    ("retorna texto", com_corpo("", retorno="np.where(m, o, media).astype(str)"), "dinamica", "tipo nao numerico"),
    ("excecao", com_corpo("raise ValueError('falhou')"), "dinamica", "excecao ValueError"),
]

CASOS_BONS = [
    ("principal minimo valido", VALIDO),
    ("retorno como lista de listas", com_corpo("", retorno="np.where(m, o, media).tolist()")),
    ("import permitido de scipy/math/random", com_corpo("", topo="import math\nimport random\nimport scipy.linalg")),
]


def rodar(caminho, banco):
    e1, _ = checar_estatico(caminho)
    e2, _, _ = checar_dinamico(caminho, banco) if not e1 else ([], [], [])
    return e1, e2


def main():
    banco = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RAIZ, "banco_publico")
    falhas = 0
    with tempfile.TemporaryDirectory() as pasta:
        print(f"Casos que devem ser REPROVADOS ({len(CASOS_RUINS)})")
        for i, (nome, codigo, etapa, esperado) in enumerate(CASOS_RUINS):
            caminho = os.path.join(pasta, f"ruim_{i:02d}.py")
            with open(caminho, "w", encoding="utf-8") as f:
                f.write(codigo)
            e1, e2 = rodar(caminho, banco)
            erros = e1 if etapa == "estatica" else e2
            ok = any(esperado in e for e in erros)
            if etapa == "dinamica" and e1:
                ok = False  # o caso dinamico tem que passar na estatica para testar a execucao
            falhas += not ok
            detalhe = "" if ok else f"  <- esperado '{esperado}' na etapa {etapa}; achei: {e1 + e2}"
            print(f"  {'ok   ' if ok else 'FALHA'} {nome}{detalhe}")

        print(f"\nCasos que devem ser APROVADOS ({len(CASOS_BONS) + 1})")
        bons = list(CASOS_BONS)
        for i, (nome, codigo) in enumerate(bons):
            caminho = os.path.join(pasta, f"bom_{i:02d}.py")
            with open(caminho, "w", encoding="utf-8") as f:
                f.write(codigo)
            e1, e2 = rodar(caminho, banco)
            ok = not (e1 or e2)
            falhas += not ok
            print(f"  {'ok   ' if ok else 'FALHA'} {nome}{'' if ok else '  <- ' + str(e1 + e2)}")
        entrega = os.path.join(RAIZ, "src", "principal.py")
        e1, e2 = rodar(entrega, banco)
        ok = not (e1 or e2)
        falhas += not ok
        print(f"  {'ok   ' if ok else 'FALHA'} src/principal.py (entrega){'' if ok else '  <- ' + str(e1 + e2)}")

    total = len(CASOS_RUINS) + len(CASOS_BONS) + 1
    print(f"\nRESULTADO: {total - falhas}/{total} casos ok")
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
