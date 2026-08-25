# Integração MiroFish — camada experimental de cenários sociais

## Objetivo

A integração adiciona ao ElectionAI uma camada **contrafactual e experimental** para modelar choques sociais emergentes. O MiroFish não substitui pesquisas, resultados do TSE, o posterior Bayesiano, o nowcast de comparecimento ou o backtesting histórico.

A separação é intencional:

```text
pesquisas + TSE + fundamentos ──► posterior Bayesiano ──► Monte Carlo ──► baseline
                                          │
                                          │ opcional
                                          ▼
                               cenário MiroFish validado
                                          │
                                          ▼
                         choques por candidato/UF e estado
                                          │
                                          ▼
                              Monte Carlo contrafactual
```

O baseline continua sendo a execução canônica. A camada multiagente só é ativada quando um `AgentScenario` é fornecido explicitamente.

## Contrato de troca

O arquivo `app/agents/schemas.py` define um contrato Pydantic estrito. O MiroFish deve produzir efeitos condicionais e incertos, e não uma previsão direta de vencedor.

### Choque por candidato e UF

- `candidate_id`: identificador já conhecido pelo ElectionAI;
- `uf`: UF existente no posterior;
- `vote_shift_mean`: deslocamento médio em pontos percentuais, limitado a ±10 pp;
- `vote_shift_sd`: desvio-padrão do choque, limitado a 5 pp;
- `confidence`: confiança do simulador entre 0 e 1;
- `rationale`: justificativa curta para auditoria.

### Choque estadual

- `turnout_shift_mean`: deslocamento absoluto do comparecimento em fração, limitado a ±0,10;
- `turnout_shift_sd`: incerteza do deslocamento de comparecimento;
- `undecided_shift_mean`: deslocamento de indecisos em pontos percentuais;
- `undecided_shift_sd`: incerteza do deslocamento de indecisos;
- `confidence`: confiança do simulador.

Valores fora dos limites, efeitos duplicados, candidatos inexistentes ou UFs inexistentes são rejeitados.

## Shrinkage de segurança

Por padrão, o ElectionAI não aplica integralmente a média produzida pela simulação. Antes da injeção, cada média é reduzida por:

```text
efeito_aplicado = efeito_médio × confiança × agent_scenario_strength
```

`agent_scenario_strength` vale `0.35` por padrão e deve permanecer entre 0 e 1. O desvio-padrão não é reduzido: incerteza não é artificialmente eliminada.

## Onde o choque entra

Os choques entram em `app/services/monte_carlo.py` depois do ajuste estrutural do suporte estadual e **antes da alocação probabilística de indecisos**.

Para cada draw:

1. sorteia-se um choque candidato/UF da distribuição normal especificada;
2. o suporte estadual é deslocado e renormalizado no simplex;
3. sorteiam-se deslocamentos estaduais de comparecimento e indecisos;
4. os limites físicos são reaplicados;
5. indecisos são alocados probabilisticamente;
6. votos estaduais e nacionais são agregados;
7. o segundo turno é simulado normalmente.

Assim, o MiroFish altera a distribuição de cenários, e não apenas o resultado final de uma tabela.

## Preparar um projeto no MiroFish

Com um backend MiroFish local em `http://localhost:5001`:

```bash
python scripts/prepare_mirofish_scenario.py \
  --event-id debate-01 \
  --title "Debate presidencial" \
  --description "Evento de campanha a ser simulado" \
  --as-of 2026-08-25 \
  --seed dados/evento.md
```

O script:

1. lê candidatos e UFs conhecidos pelo ElectionAI;
2. monta um `simulation_requirement` que proíbe declaração de vencedor;
3. envia os materiais-semente ao endpoint nativo de geração de ontologia do MiroFish;
4. inicia a construção do grafo;
5. retorna `project_id` e `task_id` para continuação da simulação no MiroFish.

A execução de longa duração do MiroFish permanece desacoplada porque a API do projeto ainda está em evolução. O ponto estável da integração é o **contrato JSON final**.

## Importar um relatório MiroFish

`app/adapters/mirofish.py` aceita:

- JSON puro;
- envelope com `data`;
- envelope com `scenario`;
- relatório textual contendo um objeto JSON válido.

O conteúdo é revalidado integralmente pelo Pydantic antes de chegar ao forecast.

## Comparar baseline e cenário

```bash
python scripts/run_agent_scenario.py \
  --scenario data/scenarios/mirofish_scenario.json \
  --as-of 2026-08-25 \
  --strength 0.35 \
  --simulations 10000
```

O comando roda duas vezes com a mesma semente:

- `bayesian_baseline` sem agentes;
- `experimental_agent_scenario` com o choque validado.

A saída mostra diferenças de participação esperada no primeiro turno e probabilidade de vitória. Essas diferenças são resultados de experimento contrafactual, não uma atualização automática do forecast oficial.

## Configuração

Variáveis disponíveis no `.env`:

```env
MIROFISH_BASE_URL=http://localhost:5001
ENABLE_AGENT_SCENARIOS=false
AGENT_SCENARIO_PATH=data/scenarios/mirofish_scenario.json
AGENT_SCENARIO_STRENGTH=0.35
```

`ENABLE_AGENT_SCENARIOS=false` é o padrão deliberado.

## Validação científica recomendada

Antes de atribuir peso operacional à camada multiagente, executar replay histórico de eventos conhecidos de 2014, 2018 e 2022, respeitando os mesmos cortes temporais do backtest. Para cada evento, comparar pelo menos:

- baseline Bayesiano;
- cenário MiroFish isolado como choque;
- modelo híbrido;
- erro de participação de votos;
- Brier score;
- log loss;
- cobertura dos intervalos;
- estabilidade do efeito por UF;
- sensibilidade a modelo LLM, seed e número de agentes.

O ganho só deve ser aceito quando aparecer fora da amostra e de forma estável. Caso contrário, o MiroFish deve permanecer ferramenta de análise de cenários e não componente calibrador.

## Governança

Todo resultado com `agent_layer_enabled=true` deve ser identificado como experimental. O sistema registra em diagnósticos:

- `forecast_mode`;
- `agent_layer_enabled`;
- `agent_layer_experimental`;
- `agent_scenario_strength`;
- `event_id`;
- quantidade e magnitude máxima dos choques.

Essa linhagem permite reconstruir exatamente quando uma previsão foi puramente Bayesiana e quando houve intervenção multiagente.
