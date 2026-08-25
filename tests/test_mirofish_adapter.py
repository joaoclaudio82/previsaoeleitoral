import json

import httpx
import pytest

from app.adapters.mirofish import MiroFishClient, MiroFishError, parse_agent_scenario


VALID_SCENARIO = {
    "event_id": "evt-1",
    "title": "Evento",
    "source": "mirofish",
    "experimental": True,
    "simulation_runs": 20,
    "candidate_shocks": [
        {
            "candidate_id": "A",
            "uf": "CE",
            "vote_shift_mean": 0.8,
            "vote_shift_sd": 0.4,
            "confidence": 0.6,
            "rationale": "efeito moderado",
        }
    ],
    "state_shocks": [],
    "provenance": {"mirofish_simulation_id": "sim_123"},
}


def test_parse_agent_scenario_accepts_json_embedded_in_report_text():
    report = "Resumo textual\n```json\n" + json.dumps(VALID_SCENARIO) + "\n```\nFim"
    scenario = parse_agent_scenario(report)
    assert scenario.event_id == "evt-1"
    assert scenario.candidate_shocks[0].uf == "CE"


def test_client_unwraps_successful_mirofish_envelope():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/graph/project/proj_1"
        return httpx.Response(200, json={"success": True, "data": {"project_id": "proj_1"}})

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)
    client = MiroFishClient("http://mirofish.local", client=http_client)
    assert client.get_project("proj_1")["project_id"] == "proj_1"
    http_client.close()


def test_client_raises_domain_error_for_mirofish_failure():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"success": False, "error": "graph building"})

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)
    client = MiroFishClient("http://mirofish.local", client=http_client)
    with pytest.raises(MiroFishError, match="graph building"):
        client.build_graph("proj_1")
    http_client.close()
