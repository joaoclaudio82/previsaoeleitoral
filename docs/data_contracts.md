# Contratos de dados

## Pesquisas

Campos mínimos: `poll_id`, `institute`, `fieldwork_start`, `fieldwork_end`, `sample_size`, `uf`, `mode`, `target_population`, `candidate_id`, `share`.

Regras: `fieldwork_start <= fieldwork_end <= as_of_date`; `sample_size > 0`; `share` em `[0, 100]`; uma observação por combinação pesquisa/candidato; UFs no domínio oficial ou `BR`.

## Comparecimento

Campos mínimos: `year`, `uf`, `eligible_voters`, `votes_cast`. Taxa observada é `votes_cast / eligible_voters` e deve pertencer a `[0, 1]`.

## Fundamentos

Toda variável deve conter `reference_date`, `release_date`, fonte e unidade. O pipeline usa `release_date`, não apenas o período econômico, para impedir look-ahead bias.

## Sinais digitais

Devem registrar plataforma, janela temporal, cobertura, método de normalização e indicador de anomalia. Ausência desses metadados reduz automaticamente a confiabilidade do sinal.
