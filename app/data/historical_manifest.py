from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class HistoricalElection:
    year: int
    first_round_date: date
    second_round_date: date | None
    tse_results_package: str
    tse_poll_registry_package: str | None
    tse_turnout_package: str | None
    polling_page: str | None


HISTORICAL_ELECTIONS: dict[int, HistoricalElection] = {
    2010: HistoricalElection(
        year=2010,
        first_round_date=date(2010, 10, 3),
        second_round_date=date(2010, 10, 31),
        tse_results_package="resultados-2010",
        tse_poll_registry_package=None,
        tse_turnout_package=None,
        polling_page=None,
    ),
    2014: HistoricalElection(
        year=2014,
        first_round_date=date(2014, 10, 5),
        second_round_date=date(2014, 10, 26),
        tse_results_package="resultados-2014",
        tse_poll_registry_package="pesquisas-eleitorais-2014",
        tse_turnout_package="comparecimento-e-abstencao-2014",
        polling_page="https://pt.wikipedia.org/wiki/Pesquisas_de_opini%C3%A3o_para_a_elei%C3%A7%C3%A3o_presidencial_no_Brasil_em_2014",
    ),
    2018: HistoricalElection(
        year=2018,
        first_round_date=date(2018, 10, 7),
        second_round_date=date(2018, 10, 28),
        tse_results_package="resultados-2018",
        tse_poll_registry_package="pesquisas-eleitorais-2018",
        tse_turnout_package="comparecimento-e-abstencao-2018",
        polling_page="https://en.wikipedia.org/wiki/Opinion_polling_for_the_2018_Brazilian_presidential_election",
    ),
    2022: HistoricalElection(
        year=2022,
        first_round_date=date(2022, 10, 2),
        second_round_date=date(2022, 10, 30),
        tse_results_package="resultados-2022",
        tse_poll_registry_package="pesquisas-eleitorais-2022",
        tse_turnout_package="comparecimento-e-abstencao-2022",
        polling_page="https://en.wikipedia.org/wiki/Opinion_polling_for_the_2022_Brazilian_presidential_election",
    ),
}


def get_election(year: int) -> HistoricalElection:
    try:
        return HISTORICAL_ELECTIONS[year]
    except KeyError as exc:
        raise ValueError(f"Unsupported historical presidential election: {year}") from exc
