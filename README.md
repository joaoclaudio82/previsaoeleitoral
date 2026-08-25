# ElectionAI 0.4

Plataforma experimental para **previsão probabilística de eleições presidenciais**, com agregação Bayesiana hierárquica, efeitos estaduais, nowcasting de comparecimento, simulação de segundo turno, backtesting temporal e uma camada opcional de cenários sociais multiagente.

> **Bloqueio de uso indevido:** os dados sintéticos distribuídos com a demonstração continuam bloqueados para publicação como previsão eleitoral. A API remove o campo de vencedor público, registra o bloqueio e aplica a marca d'água **“DEMONSTRAÇÃO SINTÉTICA — NÃO É PREVISÃO ELEITORAL”**. Os resultados sintéticos não representam a eleição presidencial brasileira de 2026.

## Novidade da versão 0.4: cenários sociais com MiroFish

A v0.4 adiciona uma camada **experimental e opt-in** para estudar como debates, notícias, escândalos, apoios e outros eventos podem produzir choques sociais condicionais. O MiroFish não substitui o modelo Bayesiano: ele produz distribuições de choque que são validadas, reduzidas conservadoramente e amostradas dentro do Monte Carlo.

```text
Dados reais ──► posterior Bayesiano ──► Monte Carlo ──► baseline
                       │
                       │ cenário opcional
                       ▼
                    MiroFish
                       │
             choque candidato/UF
             turnout + indecisos
                       │
                       ▼
             Monte Carlo experimental
```

Princípios da integração:

- o modo padrão continua sendo `bayesian_baseline`;
- a camada multiagente fica desabilitada por padrão;
- candidatos, UFs, duplicidades e magnitudes são validados por schema Pydantic;
- o efeito médio é reduzido por confiança e por `agent_scenario_strength` (padrão 0,35);
- a incerteza do MiroFish é preservada e sorteada em cada simulação;
- todo resultado híbrido é marcado como `experimental_agent_scenario`;
- o ganho científico deve ser verificado por replay histórico antes de qualquer peso operacional.

Fluxo básico:

```bash
python scripts/prepare_mirofish_scenario.py \
  --event-id debate-01 \
  --title "Debate presidencial" \
  --description "Evento de campanha a ser simulado" \
  --as-of 2026-08-25 \
  --seed dados/evento.md

python scripts/run_agent_scenario.py \
  --scenario data/scenarios/mirofish_scenario.json \
  --as-of 2026-08-25 \
  --strength 0.35 \
  --simulations 10000
```

A arquitetura e o contrato de troca estão documentados em [`docs/MIROFISH_INTEGRATION.md`](docs/MIROFISH_INTEGRATION.md).

## Novidade da versão 0.3: backtesting histórico real

A v0.3 acrescenta uma camada reproduzível para reconstruir previsões em eleições presidenciais passadas usando apenas informação disponível até cada data de corte.

Fontes e escopo:

- resultados oficiais por UF: Portal de Dados Abertos do TSE;
- comparecimento e abstenção: TSE quando disponível;
- registro e metodologia de pesquisas: PesqEle/TSE;
- percentuais históricos de intenção de voto: tabelas públicas versionadas por URL, mantidas separadas da fonte oficial de resultados;
- eleições usadas como base: 2010, 2014, 2018 e 2022;
- eleições pontuáveis atualmente: 2014, 2018 e 2022.

Snapshots temporais:

```text
D-180 · D-120 · D-90 · D-60 · D-30 · D-15 · D-7 · D-3 · D-1
```

Cada snapshot elimina pesquisas futuras, limita a idade das pesquisas, fixa uma única composição de candidatos e registra a origem dos dados. Em 2014 e 2018, snapshots anteriores à estabilização da candidatura são preservados como exploratórios, mas não entram nas métricas retrospectivas de calibração.

### Executar a reconstrução histórica

```bash
python scripts/fetch_historical_data.py --years 2010 2014 2018 2022
python scripts/run_historical_backtest.py --years 2014 2018 2022 --draws 4000
```

Os relatórios são gravados em:

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

Também existe o workflow manual `.github/workflows/historical-backtest.yml`, que baixa as bases, executa os testes, roda o backtest e publica os relatórios como artifact do GitHub Actions.

A metodologia completa está em [`docs/HISTORICAL_BACKTEST.md`](docs/HISTORICAL_BACKTEST.md).

## Modelo Bayesiano hierárquico

A média ponderada da versão 0.1 foi substituída por um modelo matrix-normal em espaço additive log-ratio. O posterior inclui:

- efeito de instituto;
- efeito de modo de coleta;
- efeito da população-alvo;
- efeito por unidade federativa;
- tendência temporal;
- participação de indecisos;
- matriz de erro correlacionado entre dimensões de voto e correlação temporal entre institutos.

A inferência é fechada e gera amostras posteriores sem executar MCMC em cada chamada.

### Qualidade dos institutos aprendida

`institute_quality` não é mais utilizado. A confiabilidade é estimada a partir de:

1. erros históricos contra resultados conhecidos;
2. regularização empírico-Bayesiana;
3. resíduos observados no conjunto corrente;
4. efeito hierárquico do método e da população-alvo.

### Previsão por UF e comparecimento

O modelo produz posterior pelas **27 UFs oficiais** e agrega votos usando:

- número de eleitores registrados;
- apoio estadual amostrado;
- indecisos alocados probabilisticamente;
- comparecimento amostrado por UF.

Nos backtests, os priors estaduais só usam eleições presidenciais anteriores: mesmo candidato, depois mesmo partido, depois informação nacional histórica e, por último, um prior neutro com menor força.

O nowcast de comparecimento é uma regressão Bayesiana na escala logit com efeitos regularizados por estado.

### Transferência de segundo turno aprendida

A API não recebe matriz de transferência. Um modelo binomial Bayesiano estima transferência e abstenção adicional a partir de distância ideológica, rejeição, bloco político e incumbência.

### Proteção de sinais digitais

Buscas e sentimento são reduzidos em direção a valores neutros conforme:

- confiabilidade declarada da coleta;
- risco de anomalia;
- cobertura insuficiente;
- possível mudança de plataforma.

Os valores brutos não entram diretamente no modelo estrutural.

### Versionamento e rastreabilidade

Cada conjunto de entrada recebe:

- snapshot imutável;
- hash SHA-256;
- versão incremental;
- schema e metadados;
- classificação sintética ou operacional;
- vínculo com cada execução de previsão.

Downloads históricos oficiais também registram URL e SHA-256 em `data/historical/<ano>/manifest.json`.

## Arquitetura

```text
Resultados TSE + pesquisas históricas ──► snapshots temporais ──► backtesting/calibração
                                                │
Pesquisas nacionais e estaduais                 │
            │                                   │
            ▼                                   │
Modelo Bayesiano hierárquico ◄──────────────────┘
instituto · modo · população · UF · indecisos · erro correlacionado
            │
            ├───────────────┐
            ▼               ▼
Posterior por UF      Modelo estrutural
            │          sinais digitais protegidos
            └───────┬───────┘
                    ▼
       Nowcast de comparecimento por UF
                    │
                    ▼
        Monte Carlo de primeiro turno
                    │
                    ├── cenário MiroFish opcional e experimental
                    │
                    ▼
 Modelo aprendido de transferência e abstenção
                    │
                    ▼
 Probabilidades, intervalos, diagnósticos e linhagem
```

## Execução rápida da demonstração sintética

Requer Python 3.11 ou superior.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/generate_demo_data.py
python scripts/train.py
python scripts/version_data.py
pytest
python scripts/predict_demo.py
```

No Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### API

```bash
uvicorn app.api.main:app --reload
```

Documentação Swagger: `http://localhost:8000/docs`

Para payloads sintéticos, `likely_winner` permanece `null` e `publication_status` permanece bloqueado.

### Dashboard

```bash
streamlit run app/dashboard/app.py
```

### Docker

```bash
docker compose up --build
```

- API: `http://localhost:8000/docs`
- Dashboard: `http://localhost:8501`

## Métricas de validação histórica

A v0.3 avalia mais do que acerto do vencedor:

- Brier score;
- log loss;
- erro absoluto médio da participação de votos;
- cobertura dos intervalos posteriores;
- Expected Calibration Error;
- reliability bins;
- inclinação e intercepto aproximados de calibração;
- métricas por eleição, snapshot e UF.

O objetivo é que uma previsão de 70% se comporte como uma probabilidade de 70% em eventos comparáveis, e não apenas produza um ranking correto em uma eleição isolada.

## Contrato dos dados de pesquisas

Campos principais:

| Campo | Uso |
|---|---|
| `poll_id` | levantamento e unidade de correlação |
| `field_date` | recência da observação |
| `institute` | efeito hierárquico e confiabilidade aprendida |
| `collection_mode` | telefone, presencial, online ou misto |
| `target_population` | eleitores registrados, prováveis eleitores ou adultos |
| `candidate_id` | identificador estável |
| `share` | percentual informado para o candidato |
| `undecided_share` | indecisos, brancos ou não resposta conforme o cenário modelado |
| `sample_size` | precisão amostral |
| `margin_error` | margem informada em pontos percentuais |
| `scope` | `national` ou `state` |
| `uf` | `BR` para nacional ou sigla estadual |

O campo legado `institute_quality` é ignorado.

## Scripts principais

| Script | Função |
|---|---|
| `fetch_historical_data.py` | baixa e normaliza resultados/comparecimento do TSE e tabelas históricas de pesquisas |
| `run_historical_backtest.py` | executa snapshots históricos e gera métricas nacionais/estaduais |
| `backtest_v03.py` | avalia CSVs de previsões já produzidos |
| `prepare_mirofish_scenario.py` | cria projeto/ontologia/grafo no MiroFish para um evento eleitoral |
| `run_agent_scenario.py` | compara o baseline contra um contrato multiagente validado |
| `validate_data.py` | valida contratos e qualidade dos datasets |
| `model_report.py` | gera relatório metodológico em Markdown |
| `generate_demo_data.py` | cria os dados sintéticos da demonstração |
| `train.py` | treina modelo estrutural, calibração de institutos, turnout e transferência |
| `version_data.py` | registra snapshots e hashes no banco |
| `predict_demo.py` | executa a simulação sintética com bloqueio de publicação |

## Testes implementados

Além da suíte da v0.3, a v0.4 cobre:

- validação e limites do contrato multiagente;
- shrinkage de efeitos por confiança;
- parsing de relatórios MiroFish e tratamento de erros HTTP;
- rejeição de candidatos e UFs desconhecidos;
- isolamento do baseline quando a camada multiagente não está ativa;
- injeção dos choques dentro dos draws do Monte Carlo;
- normalização de resultados históricos do TSE;
- preservação de partido e participação por UF;
- ausência de pesquisas posteriores ao snapshot;
- consistência da composição de candidatos em cada snapshot;
- priors estaduais derivados apenas de eleição anterior;
- restrição do nível estadual às 27 UFs;
- métricas probabilísticas, drift e release gate;
- bloqueio contínuo da demonstração sintética.

## Limitações e próximos passos

A v0.4 mantém explícitas as seguintes limitações:

- a camada MiroFish ainda deve ser validada retrospectivamente evento a evento antes de receber peso operacional;
- os efeitos podem depender do LLM, do seed, da construção dos agentes e do material-semente;
- a API de execução de longa duração do MiroFish ainda está em evolução, por isso o contrato JSON é o ponto estável de integração;
- tabelas históricas de intenção de voto são fonte secundária e devem ser gradualmente substituídas por arquivos primários dos institutos;
- cenários de pergunta e mudanças de candidatura exigem curadoria por eleição;
- o modelo espacial pode evoluir de partial pooling para CAR/ICAR;
- turnout pode incorporar composição demográfica e comportamento individual quando dados adequados estiverem disponíveis;
- segundo turno deve receber uma reconstrução histórica própria de pesquisas e transferências;
- nenhuma previsão real de 2026 deve ser publicada antes de os backtests produzirem calibração e cobertura aceitáveis e passarem pelo release gate.

Consulte `METHODOLOGY.md`, `MODEL_CARD.md`, `CHANGELOG.md`, `docs/HISTORICAL_BACKTEST.md` e `docs/MIROFISH_INTEGRATION.md` para detalhes.
