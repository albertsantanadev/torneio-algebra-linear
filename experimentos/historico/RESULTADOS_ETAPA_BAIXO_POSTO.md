# Continuidade do projeto — módulo de baixo posto

Esta cópia preserva `src/principal.py` e `src/modulo_estrutural.py` originais.

## Arquivos novos ou alterados

- `src/modulo_baixo_posto.py`: implementa preenchimento inicial aditivo, SVD truncada iterativa (hard-impute), escolha de posto de 1 a 6 por validação cruzada determinística com 3 dobras e retorno ao método aditivo quando ele ganha na validação.
- `src/principal_teste_baixo_posto.py`: arquivo autossuficiente para experimentar a integração, sem importar módulos próprios. **Não é a entrega final do grupo**.

## Teste público — NRMSE normalizado pelo desvio das ocultas

| matriz | principal original | teste integrado |
| --- | ---: | ---: |
| 01 | 1.2342 | 1.2342 |
| 02 | 1.0513 | 0.0348 |
| 03 | 1.0994 | 0.3344 |
| 04 | 1.0395 | 0.0384 |
| 05 | 1.0006 | 0.1822 |
| 06 | 1.0311 | 0.0844 |
| 07 | 1.0010 | 0.1014 |
| 08 | 1.0026 | 1.0026 |
| 09 | 1.0109 | 0.2236 |
| 10 | 1.0008 | 0.0390 |
| **média** | **1.0471** | **0.3275** |

`python harness/verificador.py src/principal_teste_baixo_posto.py`: aprovado (0 erros, 0 avisos).

`python harness/avaliar.py src/principal_teste_baixo_posto.py --sinteticas 30 --semente 7 --tamanho-max 40`: NRMSE médio 0.7658 em 31 casos, pior caso 40x40 com 0.186 s (medição local).

**Atenção:** os testes públicos não garantem o desempenho no lote privado. A validação interna usa apenas valores conhecidos; as métricas dos testes públicos usam os gabaritos para avaliação e não são consultadas pelo algoritmo.

## Como testar

Da raiz do projeto:

```bash
python harness/avaliar.py src/principal_teste_baixo_posto.py --comparar src/principal.py
python harness/verificador.py src/principal_teste_baixo_posto.py
python harness/avaliar.py src/principal_teste_baixo_posto.py --sinteticas 30 --semente 7 --tamanho-max 40
```

## Próximos passos

1. Rodar também no Docker oficial/RestrictedPython.
2. Testar diferentes máscaras, densidades e tipos de matrizes; verificar generalização.
3. Comparar com módulo estrutural assim que pronto.
4. Consolidar arquivo final `principal.py` e relatório do grupo.
