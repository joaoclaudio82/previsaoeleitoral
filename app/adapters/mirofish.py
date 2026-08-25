from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import httpx

from app.agents.schemas import AgentScenario


class MiroFishError(RuntimeError):
    pass


def _extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first valid JSON object from a MiroFish report."""
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise MiroFishError("Nenhum objeto JSON válido encontrado no relatório do MiroFish")


def parse_agent_scenario(payload: str | bytes | dict[str, Any]) -> AgentScenario:
    """Parse a strict ElectionAI shock contract from a MiroFish output."""
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    if isinstance(payload, str):
        try:
            raw = json.loads(payload)
        except json.JSONDecodeError:
            raw = _extract_json_object(payload)
    else:
        raw = dict(payload)

    # Accept common envelope shapes without silently transforming the contract.
    if "data" in raw and isinstance(raw["data"], dict) and "event_id" in raw["data"]:
        raw = raw["data"]
    elif "scenario" in raw and isinstance(raw["scenario"], dict):
        raw = raw["scenario"]
    return AgentScenario.model_validate(raw)


class MiroFishClient:
    """Small native REST adapter for the public MiroFish backend.

    MiroFish has a long-running project/graph/simulation lifecycle. ElectionAI keeps
    that lifecycle outside its calibrated forecasting core and only imports the
    final validated shock contract. The methods here cover the stable setup calls
    needed to create and prepare a simulation.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:5001",
        *,
        timeout: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self.client = client or httpx.Client(base_url=self.base_url, timeout=timeout)

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> "MiroFishClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @staticmethod
    def _unwrap(response: httpx.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise MiroFishError(f"MiroFish retornou resposta não JSON (HTTP {response.status_code})") from exc
        if response.is_error or payload.get("success") is False:
            message = payload.get("error") or response.reason_phrase or "erro desconhecido"
            raise MiroFishError(f"MiroFish HTTP {response.status_code}: {message}")
        data = payload.get("data", payload)
        if not isinstance(data, dict):
            return {"value": data}
        return data

    def health(self) -> dict[str, Any]:
        response = self.client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def generate_ontology(
        self,
        files: Iterable[str | Path],
        *,
        simulation_requirement: str,
        project_name: str,
        additional_context: str = "",
    ) -> dict[str, Any]:
        opened = []
        multipart = []
        try:
            for item in files:
                path = Path(item)
                handle = path.open("rb")
                opened.append(handle)
                multipart.append(("files", (path.name, handle, "application/octet-stream")))
            response = self.client.post(
                f"{self.base_url}/api/graph/ontology/generate",
                data={
                    "simulation_requirement": simulation_requirement,
                    "project_name": project_name,
                    "additional_context": additional_context,
                },
                files=multipart,
            )
            return self._unwrap(response)
        finally:
            for handle in opened:
                handle.close()

    def build_graph(self, project_id: str, *, force: bool = False) -> dict[str, Any]:
        response = self.client.post(
            f"{self.base_url}/api/graph/build",
            json={"project_id": project_id, "force": force},
        )
        return self._unwrap(response)

    def get_project(self, project_id: str) -> dict[str, Any]:
        response = self.client.get(f"{self.base_url}/api/graph/project/{project_id}")
        return self._unwrap(response)

    def create_simulation(
        self,
        project_id: str,
        *,
        graph_id: str | None = None,
        enable_twitter: bool = True,
        enable_reddit: bool = True,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "project_id": project_id,
            "enable_twitter": enable_twitter,
            "enable_reddit": enable_reddit,
        }
        if graph_id:
            body["graph_id"] = graph_id
        response = self.client.post(f"{self.base_url}/api/simulation/create", json=body)
        return self._unwrap(response)

    def prepare_simulation(
        self,
        simulation_id: str,
        *,
        entity_types: list[str] | None = None,
        use_llm_for_profiles: bool = True,
        parallel_profile_count: int = 5,
        force_regenerate: bool = False,
    ) -> dict[str, Any]:
        response = self.client.post(
            f"{self.base_url}/api/simulation/prepare",
            json={
                "simulation_id": simulation_id,
                "entity_types": entity_types,
                "use_llm_for_profiles": use_llm_for_profiles,
                "parallel_profile_count": parallel_profile_count,
                "force_regenerate": force_regenerate,
            },
        )
        return self._unwrap(response)
