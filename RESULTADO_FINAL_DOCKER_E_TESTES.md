# Validação da entrega V2 - 17/11/2026

## Resultado do avaliador oficial (captura de tela fornecida pela equipe)

- **Status:** sucesso nas 10 matrizes públicas.
- **NRMSE agregado:** 0,264132.
- **Tempo total:** 0,4440 s.
- **Tempo individual máximo:** 0,1187 s (matriz 10).
- **Referência de comparação:** V1 oficial com NRMSE 0,266612.

## Como confirmar localmente

```bash
python harness/verificador.py src/principal.py banco_publico
python harness/testes_limites.py src/principal.py
python harness/avaliar.py src/principal.py
```

Os resultados locais são execuções em Python convencional; o Docker oficial é a prova de compatibilidade da V2, que foi executada sem alterações funcionais. Não usar gabaritos públicos para escolher valores a preencher na matriz durante a execução.

## Política de congelamento

`src/principal.py` é idêntico, byte por byte, a `principal_experimental_v2.py` aprovado. Mudanças posteriores ao arquivo invalidam essa equivalência e exigem novo teste oficial.
