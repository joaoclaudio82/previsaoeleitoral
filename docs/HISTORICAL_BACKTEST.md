# ElectionAI v0.3 — Backtesting histórico real

Esta camada reconstrói previsões presidenciais brasileiras em datas passadas sem usar informação posterior ao snapshot avaliado.

## Fontes

### Resultados e comparecimento

Fonte canônica: Portal de Dados Abertos do Tribunal Superior Eleitoral (TSE), via API CKAN.

Conjuntos configurados:

- Resultados - 2010
- Resultados - 2014
- Resultados - 2018
- Resultados - 2022
- Comparecimento e Abstenção - 2014/2018/2022 quando disponível

Cada download registra URL e SHA-256 em `data/historical/<ano>/manifest.json`.

### Pesquisas de intenção de voto

O cadastro PesqEle do TSE é usado como referência de registro e metodologia, mas não é tratado como uma série estruturada das porcentagens publicadas. As porcentagens históricas são extraídas de tabelas públicas de pesquisas e mantêm a URL de origem em cada observação.

Fontes atualmente configuradas:

- 2014: Wikipedia em português — pesquisas da eleição presidencial
- 2018: Wikipedia em inglês — opinion polling
- 2022: Wikipedia em inglês — opinion polling

Essas tabelas são uma fonte secundária. O pipeline mantém rastreabilidade para permitir substituição posterior por arquivos primários de cada instituto.

## Snapshots

O pipeline tenta reconstruir:

- D-180
- D-120
- D-90
- D-60
- D-30
- D-15
- D-7
- D-3
- D-1

Cada snapshot:

1. elimina pesquisas posteriores à data de corte;
2. limita a idade máxima da pesquisa a 90 dias;
3. seleciona uma única assinatura de candidatos, baseada no cenário mais recente disponível;
4. impede mistura de cenários incompatíveis na mesma matriz Bayesiana.

## Mudanças de candidatura

2014 e 2018 tiveram mudanças relevantes de candidatura durante a campanha. Por isso existem dois conceitos:

- `exploratory snapshot`: representa o que era pesquisado naquele momento, mas não entra na pontuação retrospectiva;
- `scorable snapshot`: ocorre após o marco de estabilização da candidatura e pode ser comparado ao resultado oficial.

Isso evita um tipo sutil de look-ahead bias: substituir retrospectivamente um pré-candidato pelo nome que terminou na urna.

## Priors estaduais

O backtest estadual não usa o resultado da própria eleição como prior.

A hierarquia de informação é:

1. mesmo candidato na eleição presidencial anterior;
2. mesmo partido;
3. média nacional histórica do candidato/partido;
4. prior neutro.

A força do prior diminui nos níveis de fallback.

## Métricas

São produzidas métricas nacionais e estaduais:

- Brier score do vencedor;
- log loss;
- erro absoluto médio da participação de votos;
- cobertura do intervalo posterior de 90%;
- Expected Calibration Error (ECE);
- bins de confiabilidade;
- intercepto e inclinação aproximados de calibração.

Relatórios por UF permitem identificar regiões sistematicamente mal modeladas.

## Execução

```bash
python scripts/fetch_historical_data.py --years 2010 2014 2018 2022
python scripts/run_historical_backtest.py --years 2014 2018 2022 --draws 4000
```

Saída padrão:

```text
reports/historical_backtest/
├── forecasts_all_snapshots.csv
├── forecasts_scoreable.csv
├── metrics_by_snapshot.csv
├── metrics_by_state.csv
├── calibration_by_election.csv
├── calibration_by_uf.csv
├── reliability_bins.csv
└── summary.json
```

## GitHub Actions

O workflow `.github/workflows/historical-backtest.yml` é acionado manualmente. Ele executa testes, baixa dados, roda o backtest e publica os relatórios como artifact.

O workflow é manual porque os arquivos oficiais do TSE podem ser grandes e não devem ser baixados a cada push comum.

## Interpretação

Uma previsão de 70% somente é considerada bem calibrada se, em um conjunto suficientemente grande de eventos comparáveis, eventos com previsão próxima de 70% ocorrerem aproximadamente 70% das vezes.

Acertar o nome do vencedor isoladamente não é critério suficiente para validar o modelo.
