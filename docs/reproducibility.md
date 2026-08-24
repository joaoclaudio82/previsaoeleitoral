# Protocolo de reprodutibilidade

Uma previsão reproduzível precisa registrar:

- commit Git;
- versão do dataset e SHA-256;
- `as_of_date`;
- versão dos artefatos de modelo;
- seed pseudoaleatória;
- número de draws posteriores;
- número de simulações;
- ambiente Python e dependências.

A reconstrução deve começar do snapshot imutável correspondente ao `run_id`. Nunca se deve substituir silenciosamente um snapshot antigo por uma versão corrigida; correções geram nova versão.

Para backtesting, cada eleição deve ser reconstruída como se o sistema estivesse naquele dia, usando apenas informações publicadas até a data de corte.
