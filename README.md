# ElectionAI 0.2

Plataforma experimental para **previsão probabilística de eleições presidenciais**, com agregação Bayesiana hierárquica, efeitos estaduais, nowcasting de comparecimento e simulação de segundo turno.

> **Bloqueio de uso indevido:** todos os dados distribuídos neste repositório são sintéticos. A API remove o campo de vencedor público, registra o bloqueio e aplica a marca d'água **“DEMONSTRAÇÃO SINTÉTICA — NÃO É PREVISÃO ELEITORAL”**. Os resultados não representam a eleição presidencial brasileira de 2026.

## Evoluções da versão 0.2

### Agregador Bayesiano hierárquico

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

O modelo produz posterior por estado e agrega votos usando:

- número de eleitores registrados;
- apoio estadual amostrado;
- indecisos alocados probabilisticamente;
- comparecimento amostrado por UF.

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

- snapshot CSV imutável;
- hash SHA-256;
- versão incremental;
- schema e metadados;
- classificação sintética ou operacional;
- vínculo com cada execução de previsão.

A implementação local usa SQLite em `data/registry/election_ai.sqlite3` e snapshots em `data/snapshots/`.

## Arquitetura

```text
Pesquisas nacionais e estaduais
            │
            ▼
Modelo Bayesiano hierárquico
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
                    ▼
 Modelo aprendido de transferência e abstenção
                    │
                    ▼
 Probabilidades, intervalos, diagnósticos e linhagem
```

## Execução rápida

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

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  --data @data/sample_request.json
```

Para o payload sintético distribuído, a resposta contém:

```json
{
  "model_version": "0.2.0",
  "likely_winner": null,
  "publication_status": "BLOCKED_SYNTHETIC_DEMONSTRATION",
  "watermark": "DEMONSTRAÇÃO SINTÉTICA — NÃO É PREVISÃO ELEITORAL"
}
```

### Dashboard

```bash
streamlit run app/dashboard/app.py
```

A interface mostra o ranking apenas como **resultado interno da simulação sintética** e mantém a advertência visível.

### Docker

```bash
docker compose up --build
```

- API: `http://localhost:8000/docs`
- Dashboard: `http://localhost:8501`

## Contrato dos dados

### Pesquisas

Campos principais de `current_polls.csv`:

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

### Priors estaduais

`state_priors.csv` contém participação inicial por candidato, força do prior, região e eleitorado. Em produção, esses priors devem ser derivados de eleições anteriores, demografia e modelos documentados, nunca de opinião subjetiva.

### Comparecimento

`current_turnout.csv` contém covariáveis estaduais para o nowcast. O arquivo histórico de treino fica em `data/processed/historical_turnout.csv`.

### Fundamentos e sinais digitais

Além de rejeição, incumbência e economia, o modelo recebe atributos para transferência e decisão tardia. Buscas e sentimento possuem campos de confiabilidade e anomalia; o pipeline gera versões protegidas antes do uso.

## Scripts

| Script | Função |
|---|---|
| `generate_demo_data.py` | cria todos os dados sintéticos da versão 0.2 |
| `train.py` | treina modelo estrutural, calibração de institutos, turnout e transferência |
| `version_data.py` | registra snapshots e hashes no banco |
| `evaluate.py` | avalia o componente estrutural em eleições sintéticas agrupadas |
| `predict_demo.py` | executa a simulação completa com bloqueio de publicação |

## Testes implementados

- normalização e efeitos estaduais do posterior hierárquico;
- ausência de dependência de `institute_quality`;
- versionamento imutável e deduplicação por hash;
- nowcast de comparecimento em intervalo válido;
- aprendizado de transferência por proximidade;
- integração ponta a ponta com 27 UFs;
- bloqueio de publicação sintética;
- contrato da API e health check.

## Limitações que permanecem

A versão 0.2 resolve as limitações arquiteturais listadas, mas ainda é um laboratório. Antes de qualquer uso real, ainda são necessários:

- base histórica brasileira real, completa e legalmente utilizável;
- reconstrução dos dados disponíveis em cada data passada;
- tratamento de mudanças de candidatura e cenários de pergunta;
- correção explícita de não resposta por perfil demográfico;
- modelo espacial mais rico para estados sem pesquisa;
- validação de turnout contra eleições brasileiras reais;
- pesquisas específicas de segundo turno para calibrar transferência;
- detecção temporal de mudança de regime em plataformas digitais;
- backtesting temporal, calibração e revisão metodológica independente;
- política pública de revisão, correção e arquivamento de previsões.

Consulte `METHODOLOGY.md`, `MODEL_CARD.md` e `CHANGELOG.md` para detalhes.
