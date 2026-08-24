# Arquitetura do ElectionAI

O sistema é dividido em seis camadas: ingestão, validação, modelagem, simulação, governança e apresentação.

## Fluxo

1. Ingestão registra dados com data de corte e origem.
2. Validação rejeita duplicatas, datas futuras e domínios inválidos.
3. O agregador hierárquico estima intenção latente por UF, instituto, modo e população-alvo.
4. O nowcast de comparecimento produz distribuições por UF.
5. A simulação Monte Carlo combina intenção, comparecimento, indecisos e transferência de votos.
6. A governança registra linhagem, versão do modelo e status de publicação.

## Princípios

- Nenhuma observação posterior ao `as_of_date` pode entrar no treino ou previsão.
- Dados sintéticos nunca são publicados como previsão real.
- Probabilidades devem ser acompanhadas por calibração e cobertura de intervalos.
- Sinais digitais são auxiliares e não podem dominar pesquisas e fundamentos.
- Toda previsão deve ser reproduzível por versão dos dados, modelo, seed e configuração.
