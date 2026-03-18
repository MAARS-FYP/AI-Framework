from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class AgentOutput:
    value: Any
    unit: str
    label: str
    status: str = "ok"


@dataclass
class InferenceOutput:
    lna: AgentOutput
    filter: AgentOutput
    mixer_power: AgentOutput
    mixer_center_freq: AgentOutput
    if_amp: AgentOutput
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lna": asdict(self.lna),
            "filter": asdict(self.filter),
            "mixer_power": asdict(self.mixer_power),
            "mixer_center_freq": asdict(self.mixer_center_freq),
            "if_amp": asdict(self.if_amp),
            "metadata": self.metadata,
        }
