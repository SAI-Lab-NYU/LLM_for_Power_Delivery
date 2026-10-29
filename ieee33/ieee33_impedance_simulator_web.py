"""Run the checked-in IEEE33 impedance-length case on DSR Web V23."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from simulator_core_v23 import DSRDashboardBackend
import simulator_web_v23 as v23


PROJECT = Path(__file__).resolve().parent
SCENARIO_DIR = PROJECT
RUNTIME_DIR = PROJECT / "runtime"
INDEX_PATH = RUNTIME_DIR / "ieee33_impedance_simulator.html"


def build_dashboard_html() -> Path:
    source = (PROJECT / "index_v23.html").read_text(encoding="utf-8")
    scenario = json.loads(
        (SCENARIO_DIR / "scenario.json").read_text(encoding="utf-8")
    )
    node_rows = json.loads(
        (SCENARIO_DIR / "Nodes_v5.json").read_text(encoding="utf-8")
    )["Nodes"]
    bus_positions = {
        row["name"]: row["position"] for row in node_rows
    }
    # Rounded display limits cover both folded feeders and their outer tie
    # corridors without clipping the lower service road.
    initial_map_y_range = [-32, 40]
    tie_8_21_lane = bus_positions["B8"][1] + 10
    tie_12_22_lane = max(
        bus_positions["B12"][1], bus_positions["B22"][1]
    ) + 3
    tie_18_33_lane = min(
        bus_positions["B18"][1], bus_positions["B33"][1]
    ) - 2
    tie_18_33_outer_x = bus_positions["B18"][0] + 2
    tie_25_29_lane = min(
        bus_positions["B25"][1], bus_positions["B29"][1]
    ) - 2
    compact_tie_display_routes = {
        # Compact orthogonal doglegs keep the single-line diagram readable
        # without implying that an outer rectangular corridor is physical.
        "TIE_8_21": [
            bus_positions["B8"],
            [bus_positions["B8"][0], tie_8_21_lane],
            [bus_positions["B21"][0], tie_8_21_lane],
            bus_positions["B21"],
        ],
        "TIE_9_15": [
            bus_positions["B9"],
            [bus_positions["B9"][0] + 2, bus_positions["B9"][1]],
            [bus_positions["B9"][0] + 2, bus_positions["B15"][1]],
            bus_positions["B15"],
        ],
        "TIE_12_22": [
            bus_positions["B12"],
            [bus_positions["B12"][0], tie_12_22_lane],
            [bus_positions["B22"][0], tie_12_22_lane],
            bus_positions["B22"],
        ],
        "TIE_18_33": [
            bus_positions["B18"],
            [tie_18_33_outer_x, bus_positions["B18"][1]],
            [tie_18_33_outer_x, tie_18_33_lane],
            [bus_positions["B33"][0], tie_18_33_lane],
            bus_positions["B33"],
        ],
        "TIE_25_29": [
            bus_positions["B25"],
            [bus_positions["B25"][0], tie_25_29_lane],
            [bus_positions["B29"][0], tie_25_29_lane],
            bus_positions["B29"],
        ],
    }
    compact_tie_display_json = json.dumps(
        compact_tie_display_routes, separators=(",", ":")
    )
    initial_map_y_range_json = json.dumps(
        initial_map_y_range, separators=(",", ":")
    )
    line_rows = json.loads(
        (SCENARIO_DIR / "ElectricalLines_v4.json").read_text(
            encoding="utf-8"
        )
    )["ElectricalLines"]
    lines_by_name = {row["name"]: row for row in line_rows}
    switch_rows = json.loads(
        (SCENARIO_DIR / "Switch_v5.json").read_text(encoding="utf-8")
    )["Switches"]
    switch_display_positions = {}
    for switch in switch_rows:
        line = lines_by_name[switch["line"]]
        route = compact_tie_display_routes.get(
            line["name"], line["position"]
        )
        node_name = switch["node"]
        node_position = bus_positions[node_name]
        adjacent = route[1] if node_name == line["from"] else route[-2]
        dx = (adjacent[0] > node_position[0]) - (
            adjacent[0] < node_position[0]
        )
        dy = (adjacent[1] > node_position[1]) - (
            adjacent[1] < node_position[1]
        )
        switch_display_positions[switch["name"]] = [
            node_position[0] + dx,
            node_position[1] + dy,
        ]
    switch_display_json = json.dumps(
        switch_display_positions, separators=(",", ":")
    )
    scale = scenario["scale"]
    scenario_css = """
    .scenario-identity {
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      align-items: center;
      padding: 8px 14px;
      border-bottom: 1px solid var(--border);
      background: #eef5f2;
      color: #29443a;
      font-size: 12px;
    }
    .scenario-identity strong { color: #173c31; }
    .map-layer-toolbar {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px;
      min-height: 52px;
      padding: 8px 12px;
      border-bottom: 1px solid var(--border);
      background: #fbfcfd;
    }
    .map-layer-toolbar-label {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }
    .map-layer-segmented {
      display: grid;
      grid-template-columns: repeat(3, minmax(92px, 1fr));
      gap: 4px;
      padding: 3px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #eef1f4;
    }
    .map-layer-segmented button {
      min-height: 32px;
      padding: 5px 10px;
      border-color: transparent;
      border-radius: 6px;
      background: transparent;
      font-size: 12px;
    }
    .map-layer-segmented button.active {
      border-color: #b8cce2;
      background: #fff;
      color: var(--accent-dark);
      font-weight: 700;
      box-shadow: 0 1px 2px rgba(30, 50, 70, 0.08);
    }
    #map-layer-description {
      min-width: 220px;
      color: var(--muted);
      font-size: 12px;
    }
    .map-workspace {
      min-height: 560px;
    }
    #grid-map {
      height: 560px;
      min-height: 560px;
    }
    @media (max-width: 680px) {
      .map-layer-toolbar { align-items: stretch; }
      .map-layer-toolbar-label { width: 100%; }
      .map-layer-segmented { width: 100%; }
      #map-layer-description { width: 100%; }
    }
    """
    banner = (
        '<div class="scenario-identity">'
        '<strong>IEEE33 V1 · Impedance-Length Layout</strong>'
        f'<span>{scale["buses"]} buses</span>'
        f'<span>{scale["zones"]} balanced zones</span>'
        f'<span>{scale["switches"]} switches</span>'
        '<span>|Z| mapped to 2–20 Pixel traversal length</span>'
        '<span>1:1 coordinate scale</span>'
        '<span>Topology simulator; power flow disabled</span>'
        '</div>'
    )
    html = source.replace(
        "<title>DSR Grid Map v23</title>",
        "<title>IEEE 33-Bus Impedance-Length Simulator</title>",
    ).replace(
        '<body>',
        '<body data-scenario="ieee33_impedance_layout">',
    ).replace(
        "<h1>DSR Grid Map v23</h1>",
        "<h1>IEEE 33-Bus Impedance-Length Simulator</h1>",
    ).replace(
        "  </style>", scenario_css + "\n  </style>", 1
    ).replace(
        '  <div id="message">Loading simulator state...</div>',
        '  <div id="message">Loading simulator state...</div>\n  ' + banner,
    ).replace(
        '    <section id="map-panel" class="panel">\n'
        '      <div class="map-workspace">',
        '    <section id="map-panel" class="panel">\n'
        '      <div class="map-layer-toolbar" aria-label="Map layer controls">\n'
        '        <span class="map-layer-toolbar-label">Map layers</span>\n'
        '        <div class="map-layer-segmented" role="group" aria-label="Select map layers">\n'
        '          <button type="button" data-map-layer="road">Road Network</button>\n'
        '          <button type="button" data-map-layer="electrical">Electrical Grid</button>\n'
        '          <button type="button" data-map-layer="combined" class="active">Combined</button>\n'
        '        </div>\n'
        '        <span id="map-layer-description">Road and electrical layers are visible.</span>\n'
        '      </div>\n'
        '      <div class="map-workspace">',
    ).replace(
        "    let latestState = null;",
        "    let latestState = null;\n"
        "    let impedanceMapLayer = 'combined';\n"
        f"    const compactTieDisplayRoutes = {compact_tie_display_json};\n"
        f"    const initialMapYRange = {initial_map_y_range_json};\n"
        f"    const switchDisplayPositions = {switch_display_json};\n"
        "    const impedanceLayerDescriptions = {\n"
        "      road: 'Road network only; electrical assets are hidden.',\n"
        "      electrical: 'Electrical grid only; road traces are hidden.',\n"
        "      combined: 'Road and electrical layers are visible.',\n"
        "    };\n"
        "    function impedanceTraceLayer(trace) {\n"
        "      const name = String(trace?.name || '');\n"
        "      const legendGroup = String(trace?.legendgroup || '');\n"
        "      if (legendGroup.startsWith('road-') || name.startsWith('Level ')) return 'road';\n"
        "      if (name === 'Pixels') return 'context';\n"
        "      return 'electrical';\n"
        "    }\n"
        "    function compactTieName(trace) {\n"
        "      const name = String(trace?.name || '');\n"
        "      if (Object.hasOwn(compactTieDisplayRoutes, name)) return name;\n"
        "      const hover = String(trace?.hovertemplate || '');\n"
        "      return Object.keys(compactTieDisplayRoutes).find(tie => hover.includes(tie)) || null;\n"
        "    }\n"
        "    function applyCompactTieDisplay(trace) {\n"
        "      if (!String(trace?.mode || '').includes('lines')) return trace;\n"
        "      const tie = compactTieName(trace);\n"
        "      if (!tie) return trace;\n"
        "      const points = compactTieDisplayRoutes[tie];\n"
        "      return {\n"
        "        ...trace,\n"
        "        x: points.map(point => point[0]),\n"
        "        y: points.map(point => point[1]),\n"
        "      };\n"
        "    }\n"
        "    function applySwitchDisplayPosition(trace) {\n"
        "      const position = switchDisplayPositions[String(trace?.name || '')];\n"
        "      if (!position || !String(trace?.mode || '').includes('markers')) return trace;\n"
        "      return {...trace, x: [position[0]], y: [position[1]]};\n"
        "    }\n"
        "    function applyCompactAssetMarker(trace) {\n"
        "      const name = String(trace?.name || '');\n"
        "      const isBus = /^B\\d+$/.test(name);\n"
        "      const isSwitch = name === 'S_SOURCE'\n"
        "        || name.startsWith('RCS_')\n"
        "        || name.startsWith('MS_')\n"
        "        || name.startsWith('SW_TIE_');\n"
        "      if (!trace?.marker || (!isBus && !isSwitch)) return trace;\n"
        "      return {\n"
        "        ...trace,\n"
        "        marker: {\n"
        "          ...trace.marker,\n"
        "          size: isBus ? 9 : 7,\n"
        "          line: {\n"
        "            ...(trace.marker.line || {}),\n"
        "            width: isBus ? 1.4 : 1.1,\n"
        "          },\n"
        "        },\n"
        "      };\n"
        "    }\n"
        "    function impedanceShapeVisible(shape) {\n"
        "      if (impedanceMapLayer === 'combined') return true;\n"
        "      if (shape?.type === 'line') return true;\n"
        "      if (impedanceMapLayer === 'road') return false;\n"
        "      return !String(shape?.name || '').startsWith('inspection-range:');\n"
        "    }\n"
        "    function updateImpedanceLayerControls() {\n"
        "      document.querySelectorAll('[data-map-layer]').forEach(button => {\n"
        "        const active = button.dataset.mapLayer === impedanceMapLayer;\n"
        "        button.classList.toggle('active', active);\n"
        "        button.setAttribute('aria-pressed', String(active));\n"
        "      });\n"
        "      const description = document.getElementById('map-layer-description');\n"
        "      if (description) description.textContent = impedanceLayerDescriptions[impedanceMapLayer];\n"
        "    }\n"
        "    function setImpedanceMapLayer(layer) {\n"
        "      if (!Object.hasOwn(impedanceLayerDescriptions, layer)) return;\n"
        "      impedanceMapLayer = layer;\n"
        "      updateImpedanceLayerControls();\n"
        "      if (latestState?.figure) renderMap(latestState.figure);\n"
        "    }\n"
        "    document.querySelectorAll('[data-map-layer]').forEach(button => {\n"
        "      button.addEventListener('click', () => setImpedanceMapLayer(button.dataset.mapLayer));\n"
        "    });\n"
        "    updateImpedanceLayerControls();\n"
        "    function applyImpedanceLayoutAspect(figure) {\n"
        "      if (!figure?.layout?.yaxis) return figure;\n"
        "      const mobileTextPositions = {\n"
        "        RC1: 'bottom right',\n"
        "        RC2: 'top right',\n"
        "        Scott1: 'top left',\n"
        "        MPS1: 'bottom left',\n"
        "      };\n"
        "      const data = (figure.data || []).filter(trace => {\n"
        "        if (impedanceMapLayer === 'combined') return true;\n"
        "        const layer = impedanceTraceLayer(trace);\n"
        "        if (layer === 'context') return false;\n"
        "        return layer === impedanceMapLayer;\n"
        "      }).map(applyCompactTieDisplay).map(applySwitchDisplayPosition)\n"
        "        .map(applyCompactAssetMarker).map(trace => {\n"
        "        const name = String(trace.name || '');\n"
        "        const isSwitch = name === 'S_SOURCE'\n"
        "          || name.startsWith('RCS_')\n"
        "          || name.startsWith('MS_')\n"
        "          || name.startsWith('SW_TIE_');\n"
        "        if (isSwitch) return {...trace, mode: 'markers'};\n"
        "        if (mobileTextPositions[name] && String(trace.mode || '').includes('text')) {\n"
        "          return {...trace, textposition: mobileTextPositions[name]};\n"
        "        }\n"
        "        return trace;\n"
        "      });\n"
        "      return {\n"
        "        ...figure,\n"
        "        data,\n"
        "        layout: {\n"
        "          ...figure.layout,\n"
        "          annotations: [],\n"
        "          shapes: (figure.layout.shapes || []).filter(impedanceShapeVisible),\n"
        "          xaxis: {\n"
        "            ...figure.layout.xaxis,\n"
        "            dtick: 10,\n"
        "          },\n"
        "          yaxis: {\n"
        "            ...figure.layout.yaxis,\n"
        "            range: initialMapYRange,\n"
        "            scaleratio: 1.0,\n"
        "            constrain: 'domain',\n"
        "            dtick: 4,\n"
        "          },\n"
        "          uirevision: 'ieee33-impedance-layout-v2',\n"
        "        },\n"
        "      };\n"
        "    }",
    ).replace(
        "      await Plotly.react(map, figure.data, figure.layout, plotConfig);",
        "      const displayFigure = applyImpedanceLayoutAspect(figure);\n"
        "      await Plotly.react(map, displayFigure.data, displayFigure.layout, plotConfig);",
        1,
    ).replace(
        "          await Plotly.react(map, frame.figure.data, frame.figure.layout, plotConfig);",
        "          const displayFrame = applyImpedanceLayoutAspect(frame.figure);\n"
        "          await Plotly.react(map, displayFrame.data, displayFrame.layout, plotConfig);",
        1,
    )
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(html, encoding="utf-8")
    return INDEX_PATH


def configure_app() -> None:
    dashboard_path = build_dashboard_html()
    backend = DSRDashboardBackend(
        data_dir=SCENARIO_DIR,
        output_dir=RUNTIME_DIR,
    )
    backend.display_title = "IEEE 33-Bus Impedance-Length Map"
    backend.load_map(render_output=False)
    v23.backend = backend
    v23.RUNTIME_DIR = RUNTIME_DIR
    v23.INDEX_PATH = dashboard_path
    v23.app.title = "IEEE 33-Bus Impedance-Length Simulator"
    excluded = ("/replay/", "/api/controller/", "/api/planning/", "/docs/")
    v23.app.router.routes[:] = [
        route for route in v23.app.router.routes
        if not str(getattr(route, "path", "")).startswith(excluded)
    ]


configure_app()
app = v23.app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the IEEE33 V1-compatible impedance-length layout."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8067, type=int)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    v23.run_server(
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
