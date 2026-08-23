from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datetime import date
import pandas as pd
import plotly.express as px
import streamlit as st

from app.core.config import settings
from app.governance.publication_guard import assess_publication
from app.services.predictor import predict

st.set_page_config(page_title="ElectionAI 0.2", layout="wide")
st.title("ElectionAI 0.2 — laboratório de previsão probabilística")
st.error("DEMONSTRAÇÃO SINTÉTICA — NÃO É PREVISÃO DA ELEIÇÃO PRESIDENCIAL DE 2026")
st.caption("Os nomes, pesquisas, efeitos estaduais e resultados deste painel são artificiais e servem apenas para validar o pipeline.")

polls = pd.read_csv(settings.polls_path)
fundamentals = pd.read_csv(settings.fundamentals_path)
state_priors = pd.read_csv(settings.state_priors_path)
turnout = pd.read_csv(settings.turnout_path)
polls["field_date"] = pd.to_datetime(polls["field_date"]).dt.date
max_date = max(polls["field_date"])
as_of = st.sidebar.date_input("Data de referência do laboratório", value=max_date, max_value=date.today())
sims = st.sidebar.slider("Simulações", 5_000, 100_000, min(settings.n_simulations, 50_000), 5_000)

bundle = predict(
    polls=polls,
    fundamentals=fundamentals,
    state_priors=state_priors,
    turnout=turnout,
    as_of_date=as_of,
    model_path=settings.model_path,
    pollster_calibration_path=settings.pollster_calibration_path,
    turnout_model_path=settings.turnout_model_path,
    transfer_model_path=settings.transfer_model_path,
    n_simulations=sims,
    posterior_draws=min(settings.posterior_draws, sims),
    seed=settings.random_seed,
)
publication = assess_publication("synthetic", 2026, "unvalidated")
leader = bundle.candidates.iloc[0]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Líder na simulação", leader["candidate_name"])
col2.metric("Vitórias simuladas", f"{leader['win_probability'] * 100:.1f}%")
col3.metric("Comparecimento esperado", f"{bundle.diagnostics['turnout_national_mean'] * 100:.1f}%")
col4.metric("Versão", bundle.model_version)
st.info(f"Status de publicação: {publication.status}. O campo público de vencedor permanece bloqueado.")

chart = bundle.candidates.copy()
chart["Probabilidade na simulação (%)"] = chart["win_probability"] * 100
fig = px.bar(
    chart,
    x="candidate_name",
    y="Probabilidade na simulação (%)",
    error_y=(chart["expected_first_round_share_high"] - chart["expected_first_round_share"]),
    title="Distribuição de vitórias no laboratório sintético",
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Posterior nacional e simulação")
view = bundle.candidates.rename(columns={
    "candidate_name": "Candidato sintético", "poll_mean": "Posterior médio (%)",
    "poll_lower": "P5 (%)", "poll_upper": "P95 (%)",
    "first_round_lead_probability": "Lidera 1º turno", "win_probability": "Vence simulação",
    "expected_first_round_share": "Voto esperado (%)",
})
view["Lidera 1º turno"] *= 100
view["Vence simulação"] *= 100
st.dataframe(
    view[["Candidato sintético", "Posterior médio (%)", "P5 (%)", "P95 (%)", "Voto esperado (%)", "Lidera 1º turno", "Vence simulação"]],
    use_container_width=True,
    hide_index=True,
)

st.subheader("Nowcast por unidade federativa")
state_view = bundle.states.copy()
state_view["Comparecimento esperado (%)"] = state_view["expected_turnout"] * 100
state_view["Confiança do líder (%)"] = state_view["leader_probability"] * 100
st.dataframe(
    state_view[["uf", "leading_candidate", "Confiança do líder (%)", "Comparecimento esperado (%)"]],
    use_container_width=True,
    hide_index=True,
)

st.subheader("Confiabilidade aprendida dos institutos")
st.caption("Os escores são estimados por calibração histórica sintética e atualizados pelo ajuste hierárquico; o parâmetro externo da versão 0.1 foi removido.")
st.dataframe(bundle.institute_reliability, use_container_width=True, hide_index=True)

with st.expander("Diagnósticos e riscos"):
    st.json(bundle.diagnostics)
    for warning in bundle.warnings:
        st.warning(warning)
