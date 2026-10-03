"""Append-only shared message pool with per-agent visibility (MetaGPT-style)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from llm_agents.protocol import Message


class Blackboard:
    def __init__(self, path: Optional[Path] = None):
        self.messages: List[Message] = []
        self.fh = None
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.fh = path.open("a", encoding="utf-8")

    def post(self, message: Message) -> Message:
        self.messages.append(message)
        if self.fh is not None:
            self.fh.write(message.to_json() + "\n")
            self.fh.flush()
        return message

    def feed(self, agent: str, t: Optional[int] = None,
             kinds: Optional[List[str]] = None, limit: int = 24) -> List[Message]:
        out = [m for m in self.messages
               if m.visible_to(agent)
               and (t is None or m.t == t)
               and (kinds is None or m.kind in kinds)]
        return out[-limit:]

    def render_feed(self, agent: str, t: Optional[int] = None,
                    kinds: Optional[List[str]] = None, limit: int = 24) -> str:
        msgs = self.feed(agent, t=t, kinds=kinds, limit=limit)
        return "\n".join(m.render() for m in msgs) if msgs else "(no messages yet)"

    def step_transcript(self, t: int) -> str:
        return "\n".join(m.render() for m in self.messages if m.t == t)

    def close(self):
        if self.fh is not None:
            self.fh.close()
            self.fh = None