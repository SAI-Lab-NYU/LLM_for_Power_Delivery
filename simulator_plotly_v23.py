from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import plotly.graph_objects as go

from simulator_core_v23 import DSRDashboardBackend, Position


NODE_TEXT_POSITIONS = {
    1: "top left",
    2: "top center",
    3: "top right",
    4: "middle left",
    5: "middle center",
    6: "middle right",
    7: "bottom left",
    8: "bottom center",
    9: "bottom right",
}


def _add_polyline_trace(
    figure: go.Figure,
    positions: Iterable[Position],
    *,
    color: str,
    width: float,
    dash: str = "solid",
    hover_text: str = "",
    name: str = "",
    legendgroup: str | None = None,
    showlegend: bool = False,
    opacity: float = 1.0,
) -> None:
    points = list(positions)
    if not points:
        return

    figure.add_trace(
        go.Scatter(
            x=[point[0] for point in points],
            y=[point[1] for point in points],
            mode="lines",
            line={"color": color, "width": width, "dash": dash},
            hovertemplate=(hover_text + "<extra></extra>") if hover_text else None,
            hoverinfo="text" if hover_text else "skip",
            name=name,
            legendgroup=legendgroup,
            showlegend=showlegend,
            opacity=opacity,
        )
    )


def _mobile_vertices(unit) -> Tuple[Tuple[float, float], ...]:
    x, y = unit.position
    heading_x, heading_y = unit.heading
    heading_norm = float(np.hypot(heading_x, heading_y))
    if heading_norm == 0:
        heading_x, heading_y, heading_norm = 1.0, 0.0, 1.0

    forward_x = heading_x / heading_norm
    forward_y = heading_y / heading_norm
    side_x = -forward_y
    side_y = forward_x

    if unit.unit_type == "MPS":
        # Use an acute isosceles triangle whose tip indicates heading.
        local_vertices = (
            (0.72, 0.0),
            (-0.42, 0.34),
            (-0.42, -0.34),
        )
    else:
        body_length = 0.62
        body_width = 0.42
        half_width = body_width / 2
        local_vertices = (
            (-body_length * 0.42, half_width),
            (body_length * 0.18, half_width),
            (body_length * 0.58, 0.0),
            (body_length * 0.18, -half_width),
            (-body_length * 0.42, -half_width),
        )
    return tuple(
        (
            x + forward_x * longitudinal + side_x * lateral,
            y + forward_y * longitudinal + side_y * lateral,
        )
        for longitudinal, lateral in local_vertices
    )


def build_grid_figure(backend: DSRDashboardBackend) -> go.Figure:
    """Build a layered Plotly figure directly from simulator state."""

    if backend.graph is None:
        figure = go.Figure()
        figure.update_layout(
            title="DSR Grid Map",
            annotations=[{
                "text": "Map data are not loaded.",
                "x": 0.5,
                "y": 0.5,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 18},
            }],
        )
        return figure

    graph = backend.graph
    figure = go.Figure()

    # Layer 1: one matrix-backed unknown-Pixel background.
    x_coordinates = list(range(graph.x_min, graph.x_max + 1))
    y_coordinates = list(range(graph.y_min, graph.y_max + 1))
    unknown_matrix = [
        [
            0 if graph.get_pixel((x, y)).is_inspected(backend.event_id) else 1
            for x in x_coordinates
        ]
        for y in y_coordinates
    ]
    pixel_knowledge = [
        [
            "Known" if graph.get_pixel((x, y)).is_inspected(backend.event_id) else "Unknown"
            for x in x_coordinates
        ]
        for y in y_coordinates
    ]
    figure.add_trace(
        go.Heatmap(
            x=x_coordinates,
            y=y_coordinates,
            z=unknown_matrix,
            customdata=pixel_knowledge,
            zmin=0,
            zmax=1,
            colorscale=[
                [0.0, "rgba(255,255,255,0.0)"],
                [0.499, "rgba(255,255,255,0.0)"],
                [0.5, "rgba(242,243,245,0.88)"],
                [1.0, "rgba(242,243,245,0.88)"],
            ],
            showscale=False,
            hovertemplate=(
                "Pixel: (%{x}, %{y})<br>Knowledge: %{customdata}"
                "<extra></extra>"
            ),
            xgap=1,
            ygap=1,
            name="Pixels",
        )
    )

    # Subtle square grid at Pixel boundaries.
    shapes = []
    for x in np.arange(graph.x_min - 0.5, graph.x_max + 1.0, 1.0):
        shapes.append({
            "type": "line",
            "x0": x,
            "x1": x,
            "y0": graph.y_min - 0.5,
            "y1": graph.y_max + 0.5,
            "line": {"color": "rgba(160,165,170,0.28)", "width": 0.6, "dash": "dot"},
            "layer": "below",
        })
    for y in np.arange(graph.y_min - 0.5, graph.y_max + 1.0, 1.0):
        shapes.append({
            "type": "line",
            "x0": graph.x_min - 0.5,
            "x1": graph.x_max + 0.5,
            "y0": y,
            "y1": y,
            "line": {"color": "rgba(160,165,170,0.28)", "width": 0.6, "dash": "dot"},
            "layer": "below",
        })

    # Layer 2: roads. Electrical-network Pixels are implicit Level 1 roads.
    for line in backend.electrical_lines.values():
        _add_polyline_trace(
            figure,
            line.position,
            color="#D5C8B5",
            width=7.0,
            hover_text=f"Implicit Level 1 road under {line.name}",
            name="Level 1 Road",
            legendgroup="road-level-1",
        )

    for road in sorted(backend.roads.values(), key=lambda item: item.level):
        if road.level == 1:
            _add_polyline_trace(
                figure,
                road.position,
                color="#D5C8B5",
                width=7.0,
                hover_text=f"{road.name}<br>Road Level 1",
                name="Level 1 Road",
                legendgroup="road-level-1",
            )
        else:
            _add_polyline_trace(
                figure,
                road.position,
                color="#BFAE95",
                width=11.0,
                hover_text=f"{road.name}<br>Road Level 2",
                name="Level 2 Road",
                legendgroup="road-level-2",
            )
            _add_polyline_trace(
                figure,
                road.position,
                color="#E1C56F",
                width=1.8,
                dash="dash",
                hover_text=f"{road.name}<br>Road Level 2",
                name="Level 2 Centerline",
                legendgroup="road-level-2",
            )

    # Static unknown-region boundaries remain visible while Pixel fills disappear.
    if backend.unknown_state is not None:
        for region in backend.unknown_state.unknown_regions:
            x_min, y_min = region.lower_left
            x_max, y_max = region.upper_right
            shapes.append({
                "type": "rect",
                "x0": x_min - 0.5,
                "x1": x_max + 0.5,
                "y0": y_min - 0.5,
                "y1": y_max + 0.5,
                "fillcolor": "rgba(0,0,0,0)",
                "line": {"color": "rgba(150,155,160,0.75)", "width": 1.0},
                "layer": "below",
            })

    # Layer 3: electrical lines and labels.
    annotations = []
    for line in backend.electrical_lines.values():
        knowledge = line.knowledge_state(backend.event_id)
        hover_text = (
            f"{line.name}<br>Knowledge: {knowledge}"
            f"<br>State: {line.state}<br>Power: "
            f"{'Energized' if line.energized else 'De-energized'}"
            f"<br>Area: {line.area_name or '-'}"
        )
        if knowledge == "known":
            _add_polyline_trace(
                figure,
                line.position,
                color="#188A2E" if line.energized else "#E02020",
                width=3.2,
                dash="solid" if line.state == "closed" else "dash",
                hover_text=hover_text,
                name=line.name,
            )
        else:
            for first_pixel, second_pixel in zip(
                line.physical_pixels[:-1], line.physical_pixels[1:]
            ):
                segment_known = (
                    first_pixel.is_inspected(backend.event_id)
                    and second_pixel.is_inspected(backend.event_id)
                )
                _add_polyline_trace(
                    figure,
                    [first_pixel.position, second_pixel.position],
                    color="#E02020" if segment_known else "#777C82",
                    width=3.0,
                    dash="dash",
                    hover_text=hover_text,
                    name=line.name,
                )

        label_x, label_y, _, _ = line._get_label_location(label_gap=0.30)
        annotations.append({
            "x": label_x,
            "y": label_y,
            "text": f"<i>{line.name}</i>",
            "showarrow": False,
            "font": {"color": "#3D91E8", "size": 12, "family": "Georgia, serif"},
        })

    # Active route layer for units that are already Moving.
    for unit in backend.mobile_units.values():
        if unit.state == "Moving" and unit.planned_path:
            _add_polyline_trace(
                figure,
                [unit.position, *unit.planned_path],
                color=unit.label_color,
                width=2.5,
                dash="dash",
                hover_text=f"{unit.name} active route",
                name=f"{unit.name} route",
            )

    # Layer 4: nodes.
    for node in backend.nodes.values():
        knowledge = node.knowledge_state(backend.event_id)
        marker_color = (
            "#80858A"
            if knowledge != "known"
            else ("#188A2E" if node.energized else "#E02020")
        )
        load_text = "No load"
        if node.has_load:
            load_text = (
                f"Load: P={node.active_power:g}, Q={node.reactive_power:g}, "
                f"weight={node.load_weight:g}"
            )
        figure.add_trace(
            go.Scatter(
                x=[node.position[0]],
                y=[node.position[1]],
                mode="markers+text",
                marker={
                    "symbol": "circle",
                    "size": 13,
                    "color": marker_color,
                    "line": {"color": "#111111", "width": 2},
                },
                text=[node.name],
                textposition=NODE_TEXT_POSITIONS.get(node.label_position, "top center"),
                textfont={"color": "#E67E22", "size": 12},
                hovertemplate=(
                    f"{node.name}<br>Position: {node.position}"
                    f"<br>Knowledge: {knowledge}"
                    f"<br>Power: {'Energized' if node.energized else 'De-energized'}"
                    f"<br>{load_text}<br>Area: {node.area_name or 'Transmission'}"
                    "<extra></extra>"
                ),
                showlegend=False,
                name=node.name,
            )
        )

    # Switches are rectangles in the conceptual model; square markers remain crisp.
    for switch in backend.switches.values():
        center_x, center_y = switch.get_center_position(backend.nodes)
        switch_color = {
            "closed": "#188A2E",
            "opened": "#E02020",
            "unknown": "#80858A",
        }[switch.state]
        figure.add_trace(
            go.Scatter(
                x=[center_x],
                y=[center_y],
                mode="markers+text",
                marker={
                    "symbol": "square",
                    "size": 11,
                    "color": switch_color,
                    "line": {"color": "#111111", "width": 1.5},
                },
                text=[switch.name],
                textposition=NODE_TEXT_POSITIONS.get(switch.label_position, "top center"),
                textfont={"color": "#111111", "size": 10},
                hovertemplate=(
                    f"{switch.name}<br>Current state: {switch.state}"
                    f"<br>Node: {switch.node_name}<br>Line: {switch.line_name}"
                    "<extra></extra>"
                ),
                showlegend=False,
                name=switch.name,
            )
        )

    # Layer 5: known active faults.
    for fault in backend.faults.values():
        if not fault.active or not fault.is_known(backend.event_id):
            continue
        public_required = fault.public_repair_resource
        public_remaining = fault.public_remaining_repair_resource
        analysis_status = "Complete" if fault.analyzed else "Pending"
        required_text = (
            str(public_required) if public_required is not None else "Hidden"
        )
        remaining_text = (
            str(public_remaining) if public_remaining is not None else "Hidden"
        )
        fault_hover = (
            f"Fault {fault.name}<br>Position: {fault.position}"
            f"<br>Status: Active"
            f"<br>Analysis: {analysis_status}"
            f"<br>Required repair resource: {required_text}"
            f"<br>Remaining repair resource: {remaining_text}"
            f"<br>Area: {', '.join(fault.area_names)}"
            "<extra></extra>"
        )

        figure.add_trace(
            go.Scatter(
                x=[fault.position[0]],
                y=[fault.position[1]],
                mode="markers+text",
                marker={
                    "symbol": "circle-x-open",
                    "size": 20,
                    "color": "#7F0000",
                    "line": {"color": "#7F0000", "width": 2},
                },
                text=[fault.name],
                textposition="top right",
                textfont={"color": "#7F0000", "size": 12},
                hovertemplate=fault_hover,
                showlegend=False,
                name=f"Fault {fault.name}",
            )
        )

    # Transmission-source outage marker.
    source = getattr(backend, "transmission_source", None)
    if source is not None and source.fault_active:
        source_node = backend.nodes[source.node_name]
        figure.add_trace(
            go.Scatter(
                x=[source_node.position[0]],
                y=[source_node.position[1]],
                mode="markers+text",
                marker={
                    "symbol": "x-open",
                    "size": 24,
                    "color": "#8B0000",
                    "line": {"color": "#8B0000", "width": 2.5},
                },
                text=["GRID OUT"],
                textposition="top left",
                textfont={"color": "#8B0000", "size": 11},
                hovertemplate=(
                    f"{source.name}<br>Status: Outage"
                    f"<br>Restoration time step: {source.restoration_time_step}"
                    f"<br>Isolated switches: {', '.join(source.isolated_switch_names) or '-'}"
                    "<extra></extra>"
                ),
                showlegend=False,
                name="Transmission Source Outage",
            )
        )

    # Layer 6/7: mobile units and square L-infinity inspection ranges.
    for unit in backend.mobile_units.values():
        x, y = unit.position
        r = unit.inspection_range
        shapes.append({
            "type": "rect",
            "x0": x - r - 0.5,
            "x1": x + r + 0.5,
            "y0": y - r - 0.5,
            "y1": y + r + 0.5,
            "fillcolor": "rgba(0,0,0,0)",
            "line": {"color": unit.range_color, "width": 1.6, "dash": "dashdot"},
            "layer": "above",
        })
        vertices = _mobile_vertices(unit)
        closed_vertices = [*vertices, vertices[0]]
        figure.add_trace(
            go.Scatter(
                x=[point[0] for point in closed_vertices],
                y=[point[1] for point in closed_vertices],
                mode="lines",
                fill="toself",
                fillcolor=unit.body_color,
                line={"color": unit.edge_color, "width": 1.8},
                hoverinfo="skip",
                showlegend=False,
                name=unit.name,
            )
        )
        if unit.unit_type == "RC":
            resource_text = f"{unit.remaining_resources}/{unit.total_resources}"
            extra_text = f"<br>Repair resources: {resource_text}"
        elif unit.unit_type == "MPS":
            resource_text = f"{unit.energy:g}/{unit.energy_capacity:g} kW.step"
            extra_text = (
                f"<br>Energy: {resource_text}"
                f"<br>Output: P={unit.current_p:g} kW, Q={unit.current_q:g} kVar"
                f"<br>Limits: P={unit.p_limit:g} kW, S={unit.s_limit:g} kVA"
                f"<br>Connected Areas: {', '.join(unit.connected_area_names) or '-'}"
            )
        else:
            resource_text = "N/A"
            extra_text = ""
        figure.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers+text",
                marker={"size": 22, "opacity": 0},
                text=[unit.name],
                textposition="top center",
                textfont={"color": unit.label_color, "size": 12},
                hovertemplate=(
                    f"{unit.name}<br>Type: {unit.unit_type}"
                    f"<br>State: {unit.state}<br>Position: {unit.position}"
                    f"<br>Move speed: {unit.move_speed}"
                    f"<br>Inspection range: {unit.inspection_range}"
                    f"{extra_text}<extra></extra>"
                ),
                showlegend=False,
                name=unit.name,
            )
        )

    figure.update_layout(
        title={"text": "DSR Grid Map", "x": 0.5, "xanchor": "center"},
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin={"l": 45, "r": 20, "t": 55, "b": 45},
        hovermode="closest",
        showlegend=False,
        shapes=shapes,
        annotations=annotations,
        uirevision="dsr-grid-v23",
        dragmode="pan",
        xaxis={
            "range": [graph.x_min - 0.5, graph.x_max + 0.5],
            "tickmode": "linear",
            "tick0": graph.x_min + (graph.x_min % 2),
            "dtick": 2,
            "title": "X",
            "showgrid": False,
            "zeroline": False,
            "constrain": "domain",
            "constraintoward": "left",
        },
        yaxis={
            "range": [graph.y_min - 0.5, graph.y_max + 0.5],
            "tickmode": "linear",
            "tick0": graph.y_min + (graph.y_min % 2),
            "dtick": 2,
            "title": "Y",
            "showgrid": False,
            "zeroline": False,
            "scaleanchor": "x",
            "scaleratio": 1,
        },
    )
    return figure
