# Model Card — ElectionAI 0.2.0

## Finalidade

Estimar distribuições probabilísticas de apoio, classificação no primeiro turno e vitória em segundo turno. O sistema foi desenhado para pesquisa metodológica, ensino e desenvolvimento de infraestrutura de previsão.

## Estado de validação

- estágio: laboratório técnico;
- dados fornecidos: integralmente sintéticos;
- eleição identificada: `SYNTHETIC-LAB`;
- validação externa: inexistente;
- publicação como previsão real: bloqueada por código;
- recomendação política ou eleitoral: proibida pelo desenho do produto.

## Componentes

1. modelo Bayesiano hierárquico de pesquisas;
2. calibração histórica de confiabilidade dos institutos;
3. posterior por unidade federativa;
4. modelo hierárquico de indecisos;
5. modelo estrutural supervisionado com sinais digitais protegidos;
6. nowcast Bayesiano de comparecimento por UF;
7. Monte Carlo nacional ponderado por eleitorado;
8. modelo Bayesiano de transferência e abstenção no segundo turno;
9. registro imutável de dados e linhagem de execução;
10. guarda de publicação.

## Agregador de pesquisas

O apoio é modelado em coordenadas additive log-ratio. Os coeficientes possuem priors gaussianos por grupo e posterior matrix-normal. Efeitos de instituto, modo de coleta, população-alvo e UF são parcialmente agrupados.

A covariância residual multivariada é amostrada na geração do posterior. A precisão das observações também é não diagonal, combinando proximidade temporal, geografia e correlação histórica entre institutos. Isso representa erros compartilhados entre dimensões de voto, reduzindo a hipótese irreal de independência entre candidatos.

## Confiabilidade de institutos

A qualidade não é entrada do usuário. O artefato de calibração estima variância posterior por instituto, modo e população-alvo usando pesquisas históricas comparadas ao resultado. O cenário atual atualiza esse valor por empirical Bayes.

O escore exibido é um diagnóstico relativo, não um ranking público definitivo de institutos.

## Estados e comparecimento

Estados com poucas pesquisas são regularizados em direção a priors estaduais documentados. A agregação nacional utiliza eleitorado registrado e comparecimento amostrado. O modelo de turnout é Bayesiano na escala logit e inclui efeito regularizado de UF.

## Segundo turno

A transferência é aprendida com dados históricos agregados. A simulação inclui uma probabilidade separada de abstenção adicional. Não existe parâmetro de matriz manual na API 0.2.

## Sinais digitais

Buscas e sentimento podem refletir bots, campanhas coordenadas, mudanças de API, alterações de público e viés de cobertura. Por isso:

- recebem pesos de confiabilidade entre zero e um;
- sofrem penalização exponencial por anomalia;
- são reduzidos em direção à neutralidade;
- nunca substituem pesquisas amostrais;
- geram avisos e diagnósticos.

## Saídas

- média posterior e intervalo de 90% das pesquisas;
- probabilidade de liderança no primeiro turno;
- probabilidade de vitória na simulação;
- distribuição de voto esperado;
- liderança e comparecimento por UF;
- confiabilidade posterior dos institutos;
- correlação residual;
- status de publicação;
- versões e hashes dos dados;
- identificador imutável da execução.

## Guarda de publicação

Para `dataset_type=synthetic`:

- `likely_winner` é sempre `null`;
- o status é `BLOCKED_SYNTHETIC_DEMONSTRATION`;
- uma marca d'água é obrigatória;
- a decisão fica registrada no banco.

Para dados históricos ou operacionais, a publicação continua bloqueada enquanto `validation_status` não for `independently_validated`. Esse campo não substitui auditoria real; apenas impede publicação acidental no fluxo padrão.

## Principais riscos

- viés de seleção, cobertura e não resposta nas pesquisas;
- efeitos de instituto que mudam ao longo do tempo;
- correlação não estacionária entre erros;
- priors estaduais inadequados;
- choque de comparecimento não representado pelas covariáveis;
- transferência de votos dependente do contexto da campanha;
- ruptura de candidatura, coligação ou regra eleitoral;
- manipulação de métricas digitais;
- falsa precisão na comunicação de probabilidades;
- uso indevido de dados sintéticos como notícia ou propaganda.

## Requisitos mínimos antes de produção

1. ingestão de dados reais com licenças auditadas;
2. versionamento desde a coleta original;
3. backtesting por eleição e por data de corte;
4. validação de cobertura dos intervalos posteriores;
5. comparação contra baselines simples;
6. calibração de turnout e transferência em eleições reais;
7. análise de sensibilidade a priors;
8. auditoria independente de metodologia e software;
9. protocolo de correção e arquivamento;
10. revisão jurídica e ética da publicação.

## Métricas

`scripts/evaluate.py` calcula Brier score, log loss e acurácia do componente estrutural com `GroupKFold` por eleição. Os resultados distribuídos são sintéticos e não são evidência de desempenho no mundo real.
