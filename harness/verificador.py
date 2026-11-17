"""Verificador de regras da entrega 

Uso:
    python harness/verificador.py src/principal.py banco_publico

Etapa 1 (estatica, le o codigo sem executar): confere as restricoes do
avaliador (RestrictedPython): imports permitidos, sem `class`, sem seed/default_rng/
RandomState/Random, sem `time`, uma unica funcao `principal(observada, mascara)`.
Etapa 2 (dinamica): roda `principal` no banco publico e confere dimensao, tipo,
NaN/Inf, celulas conhecidas, tempo por matriz (2 s) e tempo do lote (25 s).

Codigo de saida: 0 = tudo certo, 1 = ha ERRO que desclassifica.
"""
import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comum import carregar_banco, carregar_principal, executar, problemas_da_saida  # noqa: E402

PERMITIDOS = {"numpy", "scipy", "math", "random"}
PROIBIDOS_NOMES = {"seed", "default_rng", "RandomState", "Random"}
LIMITE_MATRIZ_S = 2.0
LIMITE_LOTE_S = 25.0


def checar_estatico(caminho):
    """Devolve (erros, avisos) como listas de strings 'linha N: ...'."""
    with open(caminho, encoding="utf-8") as f:
        codigo = f.read()
    try:
        arvore = ast.parse(codigo)
    except SyntaxError as e:
        return [f"erro de sintaxe na linha {e.lineno}: {e.msg}"], []

    erros, avisos = [], []

    # --- estrutura do arquivo -------------------------------------------------
    funcoes_topo = [n for n in arvore.body if isinstance(n, ast.FunctionDef)]
    nomes = [n.name for n in funcoes_topo]
    if "principal" not in nomes:
        erros.append("nao existe a funcao `principal` no nivel do arquivo")
    outras = [n for n in nomes if n != "principal"]
    if outras:
        erros.append(
            f"funcoes fora de `principal`: {', '.join(outras)} "
            "(funcoes de apoio devem ficar DENTRO de `principal`)"
        )
    for n in funcoes_topo:
        if n.name == "principal":
            args = [a.arg for a in n.args.args]
            if args != ["observada", "mascara"]:
                erros.append(
                    f"linha {n.lineno}: assinatura deve ser principal(observada, mascara), achei ({', '.join(args)})"
                )
    for n in arvore.body:
        if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            continue
        if isinstance(n, ast.Expr) and isinstance(getattr(n, "value", None), ast.Constant):
            continue  # docstring
        avisos.append(f"linha {n.lineno}: codigo solto fora de funcao/import ({type(n).__name__})")

    # --- percorre tudo --------------------------------------------------------
    for no in ast.walk(arvore):
        linha = getattr(no, "lineno", "?")
        if isinstance(no, ast.ClassDef):
            erros.append(f"linha {linha}: `class` e proibido (classe {no.name})")
        elif isinstance(no, ast.Import):
            for a in no.names:
                raiz = a.name.split(".")[0]
                if raiz not in PERMITIDOS:
                    dica = " (controle o tempo por numero fixo de iteracoes, nao por relogio)" if raiz == "time" else ""
                    erros.append(f"linha {linha}: import nao permitido: {a.name}{dica}")
        elif isinstance(no, ast.ImportFrom):
            raiz = (no.module or "").split(".")[0]
            if raiz not in PERMITIDOS:
                erros.append(f"linha {linha}: import nao permitido: from {no.module} ...")
            for a in no.names:
                if a.name in PROIBIDOS_NOMES:
                    erros.append(f"linha {linha}: import proibido: {a.name}")
        elif isinstance(no, ast.Attribute):
            if no.attr in PROIBIDOS_NOMES:
                erros.append(f"linha {linha}: acesso proibido a .{no.attr}")
            if no.attr.startswith("_"):
                avisos.append(f"linha {linha}: atributo '{no.attr}' comeca com '_' (RestrictedPython costuma bloquear)")
        elif isinstance(no, ast.Name):
            if no.id in PROIBIDOS_NOMES:
                erros.append(f"linha {linha}: uso proibido do nome {no.id}")
            if no.id in {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "input"}:
                erros.append(f"linha {linha}: funcao nao permitida: {no.id}")
            if no.id.startswith("_") and no.id != "_":
                avisos.append(f"linha {linha}: nome '{no.id}' comeca com '_' (RestrictedPython costuma bloquear)")
        elif isinstance(no, ast.AugAssign) and isinstance(no.target, (ast.Subscript, ast.Attribute)):
            erros.append(
                f"linha {linha}: atribuicao aumentada em indice/atributo e rejeitada pelo RestrictedPython; "
                "troque A[i] += b por A[i] = A[i] + b"
            )
        elif isinstance(no, (ast.Global, ast.Nonlocal)):
            avisos.append(f"linha {linha}: global/nonlocal pode nao funcionar no RestrictedPython")
        elif isinstance(no, ast.Try):
            pass  # permitido (usamos try/except como fallback)

    # remove duplicatas mantendo a ordem
    return list(dict.fromkeys(erros)), list(dict.fromkeys(avisos))


def checar_dinamico(caminho, pasta_banco):
    erros, avisos, linhas = [], [], []
    try:
        principal = carregar_principal(caminho)
    except Exception as e:  # noqa: BLE001
        return [f"nao foi possivel importar o arquivo: {type(e).__name__}: {e}"], [], []
    total = 0.0
    for item in carregar_banco(pasta_banco):
        o, m = item["o"], item["m"]
        try:
            saida, t = executar(principal, o, m)
        except Exception as e:  # noqa: BLE001
            erros.append(f"matriz {item['nome']}: excecao {type(e).__name__}: {e}")
            continue
        total += t
        probs = problemas_da_saida(saida, o, m)
        for p in probs:
            (avisos if "conhecidas" in p else erros).append(f"matriz {item['nome']}: {p}")
        if t > LIMITE_MATRIZ_S:
            erros.append(f"matriz {item['nome']}: {t:.2f}s passa do limite de {LIMITE_MATRIZ_S:.0f}s")
        elif t > 0.5 * LIMITE_MATRIZ_S:
            avisos.append(f"matriz {item['nome']}: {t:.2f}s (mais da metade do limite de {LIMITE_MATRIZ_S:.0f}s)")
        linhas.append(f"  matriz {item['nome']} {str(o.shape):>9}  {t:6.3f} s  {'ok' if not probs else 'PROBLEMA'}")
    if total > LIMITE_LOTE_S:
        erros.append(f"lote: {total:.2f}s passa do limite de {LIMITE_LOTE_S:.0f}s")
    linhas.append(f"  total do lote: {total:.3f} s (limite {LIMITE_LOTE_S:.0f} s)")
    return erros, avisos, linhas


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    caminho = sys.argv[1]
    pasta = sys.argv[2] if len(sys.argv) > 2 else "banco_publico"

    print(f"== Verificando {caminho}\n\n[1/2] Regras do codigo")
    e1, a1 = checar_estatico(caminho)
    for m in e1:
        print("  ERRO  ", m)
    for m in a1:
        print("  aviso ", m)
    if not e1 and not a1:
        print("  nenhum problema encontrado")

    print(f"\n[2/2] Execucao no banco publico ({pasta})")
    e2, a2, linhas = checar_dinamico(caminho, pasta)
    for l in linhas:
        print(l)
    for m in e2:
        print("  ERRO  ", m)
    for m in a2:
        print("  aviso ", m)

    erros = e1 + e2
    print("\nRESULTADO:", "REPROVADO" if erros else "APROVADO", f"({len(erros)} erro(s), {len(a1) + len(a2)} aviso(s))")
    if not erros:
        print("Lembrete: isto nao substitui rodar o avaliador oficial no Docker antes de entregar.")
    return 1 if erros else 0


if __name__ == "__main__":
    sys.exit(main())
