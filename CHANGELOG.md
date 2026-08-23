# Changelog

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
