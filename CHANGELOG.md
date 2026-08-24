# Changelog

## 0.3.0 - historical backtesting layer

- adiciona manifesto de eleições presidenciais brasileiras históricas de 2010, 2014, 2018 e 2022;
- adiciona download auditável de resultados oficiais e comparecimento via CKAN/TSE com SHA-256;
- normaliza resultados presidenciais por UF, candidato, partido, turno e participação de votos;
- adiciona adaptador separado para tabelas públicas de pesquisas históricas, preservando URL de origem;
- suporta tabelas de pesquisas em português e inglês;
- cria snapshots D-180, D-120, D-90, D-60, D-30, D-15, D-7, D-3 e D-1;
- impede pesquisas futuras e mistura de cenários de candidatos incompatíveis;
- separa snapshots exploratórios de snapshots pontuáveis após estabilização da candidatura;
- adiciona resolução de identidade de candidatos entre eleições;
- constrói priors estaduais somente com informação da eleição presidencial anterior;
- executa posterior Bayesiano por snapshot e gera probabilidades nacionais e estaduais;
- adiciona Brier, log loss, MAE de votos, cobertura, ECE, reliability bins e slope/intercept de calibração;
- produz relatórios por eleição, snapshot e UF;
- adiciona scripts de coleta e backtesting histórico ponta a ponta;
- adiciona workflow manual do GitHub Actions para execução reproduzível e publicação de artifacts;
- adiciona testes contra vazamento temporal e inconsistência da cédula;
- documenta metodologia, limitações e proveniência em `docs/HISTORICAL_BACKTEST.md`.

## 0.2.0

- substitui a média ponderada pelo posterior Bayesiano hierárquico em espaço log-ratio;
- adiciona efeitos parciais de instituto, modo de coleta, população-alvo e UF;
- estima a confiabilidade dos institutos por calibração histórica e atualização empírica;
- modela erro correlacionado entre dimensões de voto;
- modela indecisos explicitamente e os aloca probabilisticamente;
- adiciona posterior estadual e agregação nacional ponderada por eleitorado e comparecimento;
- adiciona nowcasting Bayesiano de comparecimento por UF;
- substitui a matriz manual de transferência por modelo Bayesiano binomial treinado;
- regulariza sinais de buscas e sentimento conforme confiabilidade e anomalias;
- registra snapshots imutáveis, hashes e linhagem das previsões em SQLite;
- bloqueia vencedor público e aplica marca d'água quando os dados são sintéticos;
- amplia a API, o dashboard, os scripts e a suíte de testes.
