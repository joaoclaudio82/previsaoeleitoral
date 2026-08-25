# Metodologia — ElectionAI 0.4

## 1. Agregação Bayesiana hierárquica

As participações dos candidatos são transformadas para coordenadas additive log-ratio, usando um candidato de referência. Para cada levantamento, o modelo estima:

- nível latente de apoio;
- tendência temporal;
- efeito parcial do instituto;
- efeito parcial do modo de coleta;
- efeito parcial da população-alvo;
- efeito parcial da unidade federativa.

Os coeficientes recebem priors gaussianos com regularização específica por grupo. A solução é obtida em forma fechada como um modelo matrix-normal. Isso permite gerar amostras posteriores sem depender de MCMC para cada requisição.

A matriz residual multivariada preserva correlação entre os erros das dimensões de voto. Uma segunda matriz de observação introduz correlação temporal entre levantamentos do mesmo instituto e entre institutos cuja correlação histórica de erro foi estimada em eleições comuns. A confiabilidade de cada instituto é atualizada iterativamente por empirical Bayes, combinando calibração histórica e resíduos do cenário corrente. O campo legado `institute_quality` é aceito apenas para compatibilidade e não entra no cálculo.

## 2. Pesquisas estaduais

Pesquisas nacionais usam `uf=BR`; pesquisas estaduais usam a respectiva sigla. Os efeitos estaduais são parcialmente agrupados em direção ao nível nacional. Estados com poucos levantamentos são regularizados por `state_priors.csv`, cuja força é explícita e auditável.

## 3. Indecisos

`undecided_share` recebe um modelo hierárquico próprio em escala logit. Em cada simulação, os indecisos são distribuídos por uma Dirichlet influenciada pelo apoio já declarado, rejeição e propensão de decisão tardia. Portanto, indecisos não são simplesmente rateados de forma fixa.

## 4. Nowcasting de comparecimento

O comparecimento é previsto por regressão linear Bayesiana na escala logit com efeitos regularizados por UF. As variáveis atuais incluem histórico de comparecimento, tendência de abstenção, crescimento do cadastro, mobilidade, severidade climática e competitividade. Cada simulação sorteia coeficientes e ruído residual, produzindo uma distribuição de comparecimento por estado.

## 5. Segundo turno

A versão 0.2 não recebe matriz de transferência. Um modelo binomial Bayesiano estima, para cada candidato eliminado e dupla de finalistas:

- probabilidade de transferência condicional para o finalista A;
- probabilidade adicional de abstenção.

As variáveis incluem distância ideológica, rejeição relativa, bloco político e incumbência. Na ausência do artefato treinado, há um fallback transparente baseado em ideologia e rejeição, acompanhado de aviso.

## 6. Sinais digitais

Buscas e sentimento nunca entram diretamente. Cada sinal é reduzido em direção a um valor neutro usando confiabilidade e penalização exponencial por anomalia. Diagnósticos indicam candidatos sinalizados e o peso médio efetivamente aplicado. Esses sinais permanecem auxiliares e não substituem pesquisas representativas.

## 7. Linhagem de dados

Cada dataframe de entrada é convertido em representação canônica, recebe SHA-256 e é armazenado como snapshot imutável. O banco registra versão, schema, origem, data de referência, classificação sintética e caminho do snapshot. Cada execução registra os IDs das versões de entrada, versão do modelo, hash do resultado e status de publicação.

## 8. Bloqueio de publicação

Dados classificados como `synthetic` geram obrigatoriamente:

- `likely_winner = null` na API;
- status `BLOCKED_SYNTHETIC_DEMONSTRATION`;
- marca d'água explícita;
- registro da decisão no histórico da execução.

Dados não sintéticos continuam bloqueados até receberem `independently_validated`. Esse campo é uma barreira técnica, não uma certificação automática de qualidade.

## 9. Cenários sociais multiagente com MiroFish

A versão 0.4 introduz uma camada experimental de simulação social. Ela não modifica o posterior Bayesiano e não é ativada na execução padrão. O MiroFish é tratado como gerador de **choques contrafactuais incertos**, não como estimador direto de vencedor.

O contrato permite dois tipos de intervenção:

- choque de intenção de voto por candidato e UF, expresso por média e desvio-padrão em pontos percentuais;
- choque estadual de comparecimento e participação de indecisos, também com distribuição explícita.

Antes de entrar no Monte Carlo, cada média é reduzida por `confiança × agent_scenario_strength`. O valor padrão de `agent_scenario_strength` é 0,35. O desvio-padrão é preservado, evitando transformar incerteza do simulador em falsa precisão.

Para cada draw, os choques são sorteados e aplicados ao suporte estadual antes da alocação probabilística de indecisos. O suporte é novamente projetado no simplex, o comparecimento é limitado ao intervalo físico usado pelo modelo e a simulação de segundo turno continua inalterada.

A execução sem cenário permanece identificada como `bayesian_baseline`. Qualquer execução com cenário recebe `forecast_mode=experimental_agent_scenario`, `agent_layer_enabled=true` e diagnósticos de evento, magnitude e intensidade aplicada.

A camada multiagente só deve adquirir peso operacional após replay histórico fora da amostra demonstrar ganho estável em métricas probabilísticas. Até lá, sua função é análise de sensibilidade, exploração de mecanismos sociais e construção de cenários condicionais. Detalhes operacionais estão em `docs/MIROFISH_INTEGRATION.md`.
