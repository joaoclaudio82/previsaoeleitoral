# Validação temporal

A avaliação principal deve ser *rolling-origin*: para cada data de corte, o modelo é treinado apenas com eleições e observações disponíveis naquele instante e prevê o horizonte seguinte.

## Métricas

- Brier score para probabilidade de vitória.
- Log loss para penalizar excesso de confiança.
- MAE de participação de voto.
- Cobertura de intervalos de 50%, 80% e 95%.
- Erro absoluto por UF.
- Calibration error por faixas de probabilidade.

## Cortes

Resultados devem ser reportados por eleição, região, instituto, modo de coleta e distância até o pleito. A média global não pode esconder falhas sistemáticas em subgrupos.
