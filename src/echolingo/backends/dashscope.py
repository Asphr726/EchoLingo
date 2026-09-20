"""Shared DashScope (Alibaba Model Studio) endpoint resolution.

Both Qwen adapters share one region table. The Beijing region is the only one
that grants new accounts free quota, so a Beijing key sent to the Singapore
host answers HTTP 401 with no further hint; the error text therefore names the
region and host that were tried.

Two host families exist per region:

* the classic hosts (``dashscope.aliyuncs.com`` / ``dashscope-intl``) accept an
  API key on its own; the default workspace of the key is used;
* the workspace hosts (``{workspace}.cn-beijing.maas.aliyuncs.com``) scope the
  call to one workspace ID.

A workspace ID is optional: when present the workspace host is used, otherwise
the classic host, so a key alone is enough to test and run a session.
"""

from __future__ import annotations

from dataclasses import dataclass

REGIONS = ("singapore", "beijing")

_REGION_NAMES = {"singapore": "Singapore", "beijing": "Beijing"}
_WORKSPACE_HOSTS = {
    "singapore": "ap-southeast-1.maas.aliyuncs.com",
    "beijing": "cn-beijing.maas.aliyuncs.com",
}
_CLASSIC_HOSTS = {
    "singapore": "dashscope-intl.aliyuncs.com",
    "beijing": "dashscope.aliyuncs.com",
}

# The realtime WebSocket protocol is OpenAI-compatible and the official
# samples send this header; the classic hosts reject the upgrade without it.
REALTIME_HEADERS = {"OpenAI-Beta": "realtime=v1"}


@dataclass(frozen=True, slots=True)
class DashScopeEndpoint:
    region: str
    host: str
    workspace_scoped: bool

    @property
    def region_name(self) -> str:
        return _REGION_NAMES[self.region]

    @property
    def realtime_url(self) -> str:
        return f"wss://{self.host}/api-ws/v1/realtime"

    @property
    def compatible_base_url(self) -> str:
        return f"https://{self.host}/compatible-mode/v1"

    def describe(self) -> dict[str, object]:
        return {
            "region": self.region,
            "host": self.host,
            "workspace_scoped": self.workspace_scoped,
        }


def normalize_region(region: str | None, default: str = "singapore") -> str:
    value = (region or "").strip().lower()
    return value if value in REGIONS else default


def resolve_endpoint(region: str, workspace_id: str | None) -> DashScopeEndpoint:
    if region not in REGIONS:
        raise ValueError("DashScope region must be singapore or beijing")
    workspace_id = (workspace_id or "").strip()
    if workspace_id:
        return DashScopeEndpoint(region, f"{workspace_id}.{_WORKSPACE_HOSTS[region]}", True)
    return DashScopeEndpoint(region, _CLASSIC_HOSTS[region], False)


def other_region_hint(endpoint: DashScopeEndpoint) -> str:
    other = "beijing" if endpoint.region == "singapore" else "singapore"
    return (
        f"The key was sent to the {endpoint.region_name} region ({endpoint.host}). "
        f"Keys are region-specific: a key created in the {_REGION_NAMES[other]} console "
        f"only works with the {_REGION_NAMES[other]} region setting."
    )


def unauthorized_message(product: str, endpoint: DashScopeEndpoint) -> str:
    return (
        f"{product} rejected the API key (HTTP 401). {other_region_hint(endpoint)} "
        "Also verify that the key is active."
    )


def forbidden_message(product: str, endpoint: DashScopeEndpoint) -> str:
    if endpoint.workspace_scoped:
        return (
            f"{product} access was denied (HTTP 403). Verify that the API key belongs to "
            f"workspace host {endpoint.host} in the {endpoint.region_name} region, and that "
            f"the model is enabled for that workspace. Leaving the workspace ID empty uses "
            "the key's default workspace."
        )
    return (
        f"{product} access was denied (HTTP 403). Verify that the model is enabled in the "
        f"{endpoint.region_name} region and that the key's default workspace has access."
    )
