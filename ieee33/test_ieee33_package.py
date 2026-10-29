"""Smoke checks for the shareable IEEE33 V23 simulator package."""

from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path

import ieee33_impedance_simulator_web as case


HERE = Path(__file__).resolve().parent


class EmptyRequest:
    async def json(self) -> dict:
        return {}


class IEEE33PackageTest(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        case.v23.backend.load_map(render_output=False)

    def test_case_inputs_are_complete(self) -> None:
        metadata = json.loads((HERE / "scenario.json").read_text())
        self.assertEqual(metadata["scale"]["buses"], 33)
        self.assertEqual(metadata["scale"]["zones"], 8)
        self.assertEqual(len(case.v23.backend.nodes), 33)
        self.assertEqual(len(case.v23.backend.electrical_lines), 37)
        self.assertEqual(len(case.v23.backend.switches), 13)
        self.assertEqual(len(case.v23.backend.areas), 8)
        self.assertEqual(sum(z["p_kw"] for z in metadata["zone_loads"].values()), 3715)

    def test_dashboard_and_public_routes(self) -> None:
        html = case.INDEX_PATH.read_text(encoding="utf-8")
        for label in (
            "IEEE 33-Bus Impedance-Length Simulator",
            "Simulator Status",
            "Current-Step Commands",
            "Road Network",
            "Electrical Grid",
            "Combined",
        ):
            self.assertIn(label, html)
        paths = {route.path for route in case.app.routes}
        for path in ("/", "/text", "/api/state", "/api/text/state", "/api/execute", "/api/events"):
            self.assertIn(path, paths)
        self.assertFalse(any(path.startswith("/replay/") for path in paths))
        self.assertFalse(any(path.startswith("/api/controller/") for path in paths))

    def test_fault_reset_and_five_minute_step(self) -> None:
        backend = case.v23.backend
        backend.load_map(render_output=False)
        initial = json.loads(case.v23.get_state().body)
        self.assertEqual(initial["summary"]["energized_active_power"], "3715")
        backend.load_faults_and_unknown(render_output=False)
        reset = json.loads(case.v23.get_state().body)
        self.assertEqual(len(backend.faults), 3)
        self.assertEqual(reset["summary"]["known_faults"], 0)
        self.assertGreater(reset["inspection"]["unknown_pixel_count"], 0)

        result = json.loads(
            asyncio.run(case.v23.execute_current_step(EmptyRequest())).body
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["clock"]["time_step"], 1)
        self.assertEqual(result["clock"]["time_min"], 5.0)
        self.assertEqual(len(result["animation_frames"]), 4)


if __name__ == "__main__":
    unittest.main()
