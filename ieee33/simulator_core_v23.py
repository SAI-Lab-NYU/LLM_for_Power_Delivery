from dataclasses import dataclass, field
import copy
from pathlib import Path
from typing import Dict, Tuple, Optional, Iterable

import heapq
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage

from matplotlib.axes import Axes
from matplotlib.patches import Circle, Polygon, Rectangle


Position = Tuple[int, int]


@dataclass
class Pixel:
    """Represent one unique integer coordinate managed by the graph."""

    position: Position
    last_inspected_event: Optional[int] = None
    active_fault_ids: set[str] = field(default_factory=set)
    always_inspected: bool = False
    fault_protected: bool = False
    node_names: set[str] = field(default_factory=set)
    line_names: set[str] = field(default_factory=set)
    road_names: set[str] = field(default_factory=set)
    road_level: int = 0

    def mark_inspected(self, event_id: int = 0) -> None:
        """Mark this pixel as inspected for the specified event."""

        self.last_inspected_event = event_id

    def is_inspected(self, event_id: int = 0) -> bool:
        """Return whether this pixel was inspected for the specified event."""

        return (
            self.always_inspected
            or self.last_inspected_event == event_id
        )

    def add_fault(self, fault_id: str) -> None:
        """Attach an active fault identifier to this pixel."""

        if self.fault_protected:
            raise ValueError(
                f"Pixel {self.position} is protected from faults."
            )

        self.active_fault_ids.add(fault_id)

    def remove_fault(self, fault_id: str) -> None:
        """Remove an active fault identifier from this pixel."""

        self.active_fault_ids.discard(fault_id)

    @property
    def has_active_fault(self) -> bool:
        """Return whether at least one active fault occupies this pixel."""

        return bool(self.active_fault_ids)


@dataclass
class Graph:
    """Represent the two-dimensional integer grid and own all Pixel objects."""

    name: str
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    pixels: Dict[Position, Pixel] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.x_min > self.x_max:
            raise ValueError("x_min cannot be greater than x_max.")

        if self.y_min > self.y_max:
            raise ValueError("y_min cannot be greater than y_max.")

        # Create exactly one Pixel object for every integer coordinate.
        self.pixels = {
            (x, y): Pixel(position=(x, y))
            for x in range(self.x_min, self.x_max + 1)
            for y in range(self.y_min, self.y_max + 1)
        }

    @classmethod
    def from_json(cls, json_path: Path) -> "Graph":
        """Load graph bounds from a JSON file."""

        if not json_path.exists():
            raise FileNotFoundError(
                f"Cannot find graph file: {json_path}"
            )

        with json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        # Preserve compatibility with the earlier nested Graph.json format.
        if "Map" in data:
            data = data["Map"]

        return cls(
            name=data.get("name", "Basic DSR Grid Map"),
            x_min=data["x_min"],
            x_max=data["x_max"],
            y_min=data["y_min"],
            y_max=data["y_max"],
        )

    def contains(self, position: Position) -> bool:
        """Return whether an integer coordinate is inside the graph."""

        x, y = position

        return (
            self.x_min <= x <= self.x_max
            and self.y_min <= y <= self.y_max
        )

    def get_pixel(self, position: Position) -> Pixel:
        """Return the unique Pixel object for a coordinate."""

        normalized_position = tuple(position)

        if normalized_position not in self.pixels:
            raise ValueError(
                f"Position is outside the graph: {normalized_position}"
            )

        return self.pixels[normalized_position]

    def draw(
        self,
        figsize: Tuple[int, int] = (10, 13),
        coordinate_label_step: int = 2,
    ):
        """Draw the base integer grid."""

        fig, ax = plt.subplots(figsize=figsize)

        # Add 0.5 around the map so integer coordinates remain cell centers.
        ax.set_xlim(
            self.x_min - 0.5,
            self.x_max + 0.5,
        )

        ax.set_ylim(
            self.y_min - 0.5,
            self.y_max + 0.5,
        )

        # Keep every grid cell square.
        ax.set_aspect("equal", adjustable="box")

        # Major ticks display coordinate labels.
        ax.set_xticks(
            np.arange(
                self.x_min,
                self.x_max + 1,
                coordinate_label_step,
            )
        )

        ax.set_yticks(
            np.arange(
                self.y_min,
                self.y_max + 1,
                coordinate_label_step,
            )
        )

        # Minor ticks define cell boundaries.
        ax.set_xticks(
            np.arange(
                self.x_min - 0.5,
                self.x_max + 1.0,
                1,
            ),
            minor=True,
        )

        ax.set_yticks(
            np.arange(
                self.y_min - 0.5,
                self.y_max + 1.0,
                1,
            ),
            minor=True,
        )

        ax.grid(
            which="minor",
            linestyle="--",
            linewidth=0.5,
            alpha=0.6,
        )

        ax.grid(
            which="major",
            visible=False,
        )

        ax.tick_params(
            which="minor",
            length=0,
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(self.name)

        return fig, ax


@dataclass
class Road:
    """Represent one static road polyline used by repair crews."""

    name: str
    level: int
    position: Tuple[Position, ...]
    pixels: Tuple[Pixel, ...] = field(
        default_factory=tuple,
        init=False,
        repr=False,
    )

    @classmethod
    def load_all(
        cls,
        json_path: Path,
        graph: Graph,
    ) -> Dict[str, "Road"]:
        """Load explicit roads and register their Pixel road levels."""

        if not json_path.exists():
            raise FileNotFoundError(
                f"Cannot find road file: {json_path}"
            )

        with json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        roads: Dict[str, Road] = {}

        for road_data in data.get("Roads", []):
            road = cls(
                name=str(road_data["name"]),
                level=int(road_data["level"]),
                position=tuple(
                    tuple(point)
                    for point in road_data["position"]
                ),
            )
            road.validate(graph)
            road.bind_pixels(graph)

            if road.name in roads:
                raise ValueError(f"Duplicate road name: {road.name}")

            roads[road.name] = road

        return roads

    def validate(self, graph: Graph) -> None:
        """Validate the road level and orthogonal integer geometry."""

        if self.level not in {1, 2}:
            raise ValueError(
                f"Road {self.name} has unsupported level {self.level}. "
                "Supported road levels are 1 and 2."
            )

        if len(self.position) < 2:
            raise ValueError(
                f"Road {self.name} must contain at least two positions."
            )

        for point in self.position:
            if len(point) != 2:
                raise ValueError(
                    f"Road {self.name} has an invalid coordinate: {point}"
                )

            if not all(isinstance(value, int) for value in point):
                raise TypeError(
                    f"Road {self.name} coordinates must be integers: {point}"
                )

            if not graph.contains(point):
                raise ValueError(
                    f"Road {self.name} is outside the graph at {point}."
                )

        for first, second in zip(
            self.position[:-1],
            self.position[1:],
        ):
            if first[0] != second[0] and first[1] != second[1]:
                raise ValueError(
                    f"Road {self.name} must use orthogonal segments: "
                    f"{first} to {second}."
                )

            if first == second:
                raise ValueError(
                    f"Road {self.name} contains a zero-length segment at {first}."
                )

    @staticmethod
    def _expand_polyline(
        vertices: Tuple[Position, ...],
    ) -> Tuple[Position, ...]:
        """Expand sparse orthogonal vertices into integer Pixels."""

        expanded: list[Position] = [vertices[0]]

        for first, second in zip(vertices[:-1], vertices[1:]):
            delta_x = second[0] - first[0]
            delta_y = second[1] - first[1]
            step_x = 0 if delta_x == 0 else (1 if delta_x > 0 else -1)
            step_y = 0 if delta_y == 0 else (1 if delta_y > 0 else -1)
            segment_length = abs(delta_x) + abs(delta_y)

            for step in range(1, segment_length + 1):
                expanded.append((
                    first[0] + step * step_x,
                    first[1] + step * step_y,
                ))

        return tuple(expanded)

    def bind_pixels(self, graph: Graph) -> None:
        """Bind the road to canonical Pixels with level override."""

        expanded_positions = self._expand_polyline(self.position)
        self.pixels = tuple(
            graph.get_pixel(position)
            for position in expanded_positions
        )

        for pixel in self.pixels:
            pixel.road_names.add(self.name)
            pixel.road_level = max(pixel.road_level, self.level)

    @staticmethod
    def register_electrical_network_as_level_one(
        nodes: Dict[str, object],
        electrical_lines: Dict[str, object],
    ) -> None:
        """Treat every Node and ElectricalLine Pixel as a Level 1 road."""

        for node in nodes.values():
            node.pixel.road_names.add("__electrical_network__")
            node.pixel.road_level = max(node.pixel.road_level, 1)

        for line in electrical_lines.values():
            for pixel in line.physical_pixels:
                pixel.road_names.add("__electrical_network__")
                pixel.road_level = max(pixel.road_level, 1)

    @staticmethod
    def speed_multiplier_for_level(level: int) -> float:
        """Return the RC speed multiplier for a road level."""

        if level >= 2:
            return 1.5
        if level >= 1:
            return 1.0
        return 0.0

    def knowledge_state(self, event_id: int = 0) -> str:
        """Return unknown, partially_known, or known from Road Pixels."""

        inspected_count = sum(
            pixel.is_inspected(event_id)
            for pixel in self.pixels
        )

        if inspected_count == 0:
            return "unknown"
        if inspected_count == len(self.pixels):
            return "known"
        return "partially_known"

    def draw(self, ax: Axes) -> None:
        """Draw the road below electrical-system elements."""

        x_values = [point[0] for point in self.position]
        y_values = [point[1] for point in self.position]

        if self.level == 1:
            ax.plot(
                x_values,
                y_values,
                color="#D5C8B5",
                linewidth=5.2,
                linestyle="-",
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=1.05,
            )
            return

        ax.plot(
            x_values,
            y_values,
            color="#BFAE95",
            linewidth=8.0,
            linestyle="-",
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=1.10,
        )
        ax.plot(
            x_values,
            y_values,
            color="#E1C56F",
            linewidth=1.2,
            linestyle=(0, (6, 4)),
            dash_capstyle="round",
            zorder=1.20,
        )

    @staticmethod
    def draw_electrical_network_level_one(
        ax: Axes,
        electrical_lines: Dict[str, object],
    ) -> None:
        """Draw implicit Level 1 roads beneath all electrical lines."""

        for line in electrical_lines.values():
            x_values = [point[0] for point in line.position]
            y_values = [point[1] for point in line.position]
            ax.plot(
                x_values,
                y_values,
                color="#D5C8B5",
                linewidth=5.2,
                linestyle="-",
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=1.00,
            )


@dataclass
class Node:
    name: str
    position: Position
    label_position: int
    node_type: str = "distribution"
    always_inspected: bool = False
    fault_protected: bool = False
    load: Optional[Tuple[float, float, float]] = None
    area_name: Optional[str] = field(
        default=None,
        init=False,
    )
    energized: bool = field(
        default=False,
        init=False,
    )
    pixel: Pixel = field(init=False, repr=False)
    terminals: Dict[str, "Terminal"] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    @property
    def has_load(self) -> bool:
        """Return whether this node carries a load tuple."""

        return self.load is not None

    @property
    def active_power(self) -> float:
        """Return this node's active load power, or zero when absent."""

        return 0.0 if self.load is None else self.load[0]

    @property
    def reactive_power(self) -> float:
        """Return this node's reactive load power, or zero when absent."""

        return 0.0 if self.load is None else self.load[1]

    @property
    def load_weight(self) -> float:
        """Return this node's load weight, or zero when absent."""

        return 0.0 if self.load is None else self.load[2]

    @classmethod
    def load_all(
        cls,
        json_path: Path,
        graph: Graph,
    ) -> Dict[str, "Node"]:
        """Load all nodes and bind each node to one graph-owned Pixel."""

        if not json_path.exists():
            raise FileNotFoundError(
                f"Cannot find node file: {json_path}"
            )

        with json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        nodes: Dict[str, Node] = {}
        occupied_positions: Dict[Position, str] = {}

        for node_data in data["Nodes"]:
            node_type = node_data.get(
                "node_type",
                "distribution",
            ).lower()

            if node_type not in {"distribution", "transmission"}:
                raise ValueError(
                    f"Node {node_data['name']} has invalid "
                    f"node_type: {node_type}."
                )

            protected_by_default = node_type == "transmission"

            raw_load = node_data.get("load")
            normalized_load = None

            if raw_load is not None:
                if (
                    not isinstance(raw_load, (list, tuple))
                    or len(raw_load) != 3
                    or not all(
                        isinstance(value, (int, float, np.integer, np.floating))
                        for value in raw_load
                    )
                ):
                    raise ValueError(
                        f"Node {node_data['name']} has an invalid load. "
                        "A load must be [active_power, reactive_power, weight]."
                    )

                active_power = float(raw_load[0])
                reactive_power = float(raw_load[1])
                weight = float(raw_load[2])

                if weight < 0:
                    raise ValueError(
                        f"Node {node_data['name']} has a negative load weight."
                    )

                normalized_load = (
                    active_power,
                    reactive_power,
                    weight,
                )

            node = cls(
                name=node_data["name"],
                position=tuple(node_data["position"]),
                label_position=node_data["label_position"],
                node_type=node_type,
                always_inspected=node_data.get(
                    "always_inspected",
                    protected_by_default,
                ),
                fault_protected=node_data.get(
                    "fault_protected",
                    protected_by_default,
                ),
                load=normalized_load,
            )

            if node.name in nodes:
                raise ValueError(
                    f"Duplicate node name: {node.name}"
                )

            if not graph.contains(node.position):
                raise ValueError(
                    f"Node {node.name} is outside the graph: "
                    f"{node.position}"
                )

            if node.position in occupied_positions:
                other_name = occupied_positions[node.position]

                raise ValueError(
                    f"Nodes {node.name} and {other_name} "
                    f"have the same position: {node.position}"
                )

            if node.label_position not in range(1, 10):
                raise ValueError(
                    f"Node {node.name} has invalid label_position: "
                    f"{node.label_position}. It must be 1-9."
                )

            node.pixel = graph.get_pixel(node.position)
            node.pixel.node_names.add(node.name)
            node.pixel.always_inspected = (
                node.pixel.always_inspected
                or node.always_inspected
            )
            node.pixel.fault_protected = (
                node.pixel.fault_protected
                or node.fault_protected
            )

            nodes[node.name] = node
            occupied_positions[node.position] = node.name

        return nodes

    @property
    def is_transmission(self) -> bool:
        """Return whether this node represents the transmission grid."""

        return self.node_type == "transmission"

    @property
    def pixels(self) -> Tuple[Pixel, ...]:
        """Return the pixels occupied by this node."""

        return (self.pixel,)

    def register_terminal(self, terminal: "Terminal") -> None:
        """Register one line terminal connected to this node."""

        if terminal.line_name in self.terminals:
            raise ValueError(
                f"Node {self.name} already has a terminal for "
                f"line {terminal.line_name}."
            )

        self.terminals[terminal.line_name] = terminal

    def knowledge_state(self, event_id: int = 0) -> str:
        """Derive the node knowledge state from its Pixel."""

        return "known" if self.pixel.is_inspected(event_id) else "unknown"

    @property
    def has_direct_fault(self) -> bool:
        """Return whether the node Pixel contains an active fault."""

        return self.pixel.has_active_fault

    def _get_label_location(
        self,
        radius: float,
        label_gap: float,
    ):
        """Calculate label coordinates from the numeric label position."""

        x, y = self.position
        offset = radius + label_gap

        layouts = {
            1: (x - offset, y + offset, "right", "bottom"),
            2: (x,          y + offset, "center", "bottom"),
            3: (x + offset, y + offset, "left", "bottom"),
            4: (x - offset, y,          "right", "center"),
            5: (x,          y,          "center", "center"),
            6: (x + offset, y,          "left", "center"),
            7: (x - offset, y - offset, "right", "top"),
            8: (x,          y - offset, "center", "top"),
            9: (x + offset, y - offset, "left", "top"),
        }

        return layouts[self.label_position]

    def draw(
        self,
        ax: Axes,
        event_id: int = 0,
        radius: float = 0.2,
        label_gap: float = 0.16,
        show_label: bool = True,
    ) -> None:
        """Draw the node according to knowledge and energization states."""

        x, y = self.position
        is_known = self.knowledge_state(event_id) == "known"

        if not is_known:
            # Unknown has higher display priority than energization.
            face_color = "gray"
            edge_color = "black"
        elif self.energized:
            face_color = "green"
            edge_color = "black"
        else:
            face_color = "red"
            edge_color = "black"

        circle = Circle(
            xy=(x, y),
            radius=radius,
            facecolor=face_color,
            edgecolor=edge_color,
            linewidth=1.5,
            zorder=40,
        )

        ax.add_patch(circle)

        if show_label:
            label_x, label_y, horizontal, vertical = (
                self._get_label_location(
                    radius=radius,
                    label_gap=label_gap,
                )
            )

            ax.text(
                label_x,
                label_y,
                self.name,
                color="#E67E22",
                fontsize=9,
                horizontalalignment=horizontal,
                verticalalignment=vertical,
                zorder=90,
            )


@dataclass
class Terminal:
    """Represent one connection between an undirected line and one endpoint node."""

    line_name: str
    node_name: str
    switch: Optional["Switch"] = field(
        default=None,
        init=False,
        repr=False,
    )

    def attach_switch(self, switch: "Switch") -> None:
        """Attach the switch controlling this terminal."""

        if self.switch is not None:
            raise ValueError(
                f"Terminal {self.node_name}-{self.line_name} "
                f"already has switch {self.switch.name}."
            )

        self.switch = switch

    @property
    def state(self) -> str:
        """Return closed for an uncontrolled terminal or the switch state."""

        return "closed" if self.switch is None else self.switch.state

    @property
    def is_connected(self) -> bool:
        """Return whether this terminal currently connects the node and line."""

        return self.state == "closed"


@dataclass
class ElectricalLine:
    name: str
    node_a: str
    node_b: str
    position: Tuple[Position, ...]
    label_position: int
    area_name: Optional[str] = field(
        default=None,
        init=False,
    )

    state: str = field(
        default="closed",
        init=False,
    )

    energized: bool = field(
        default=False,
        init=False,
    )

    terminals: Dict[str, Terminal] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    switches: list["Switch"] = field(
        default_factory=list,
        init=False,
        repr=False,
    )

    physical_pixels: Tuple[Pixel, ...] = field(
        default_factory=tuple,
        init=False,
        repr=False,
    )

    body_pixels: Tuple[Pixel, ...] = field(
        default_factory=tuple,
        init=False,
        repr=False,
    )

    _endpoint_nodes: Dict[str, Node] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.terminals = {
            self.node_a: Terminal(
                line_name=self.name,
                node_name=self.node_a,
            ),
            self.node_b: Terminal(
                line_name=self.name,
                node_name=self.node_b,
            ),
        }

    @classmethod
    def load_all(
        cls,
        json_path: Path,
        graph: Graph,
        nodes: Dict[str, Node],
    ) -> Dict[str, "ElectricalLine"]:
        """Load undirected lines while preserving legacy from/to JSON fields."""

        if not json_path.exists():
            raise FileNotFoundError(
                f"Cannot find electrical-line file: {json_path}"
            )

        with json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        lines: Dict[str, ElectricalLine] = {}

        for line_data in data["ElectricalLines"]:
            if "nodes" in line_data:
                endpoint_names = line_data["nodes"]

                if len(endpoint_names) != 2:
                    raise ValueError(
                        f"Line {line_data['name']} must define exactly "
                        f"two endpoint nodes."
                    )

                node_a, node_b = endpoint_names
            else:
                # Read the earlier JSON format without treating the line as directed.
                node_a = line_data["from"]
                node_b = line_data["to"]

            line = cls(
                name=line_data["name"],
                node_a=node_a,
                node_b=node_b,
                position=tuple(
                    tuple(point)
                    for point in line_data["position"]
                ),
                label_position=line_data["label_position"],
            )

            if line.name in lines:
                raise ValueError(
                    f"Duplicate electrical-line name: {line.name}"
                )

            line.validate(
                graph=graph,
                nodes=nodes,
            )
            line.bind_pixels(graph)
            line.register_with_nodes(nodes)
            line.update_state()

            lines[line.name] = line

        return lines

    @property
    def end_nodes(self) -> Tuple[str, str]:
        """Return the two undirected endpoint node names."""

        return self.node_a, self.node_b

    @property
    def from_node(self) -> str:
        """Provide a compatibility alias for earlier notebook code."""

        return self.node_a

    @property
    def to_node(self) -> str:
        """Provide a compatibility alias for earlier notebook code."""

        return self.node_b

    def validate(
        self,
        graph: Graph,
        nodes: Dict[str, Node],
    ) -> None:
        """Validate endpoint nodes and orthogonal polyline geometry."""

        if self.node_a not in nodes:
            raise ValueError(
                f"Unknown endpoint node on {self.name}: {self.node_a}"
            )

        if self.node_b not in nodes:
            raise ValueError(
                f"Unknown endpoint node on {self.name}: {self.node_b}"
            )

        if self.node_a == self.node_b:
            raise ValueError(
                f"Line {self.name} cannot connect a node to itself."
            )

        if self.label_position not in {2, 4, 6, 8}:
            raise ValueError(
                f"Line {self.name} has invalid label_position: "
                f"{self.label_position}. It must be 2, 4, 6, or 8."
            )

        if len(self.position) < 2:
            raise ValueError(
                f"Line {self.name} must contain at least two positions."
            )

        expected_start = nodes[self.node_a].position
        expected_end = nodes[self.node_b].position

        if self.position[0] != expected_start:
            raise ValueError(
                f"Line {self.name} must start at endpoint "
                f"{self.node_a} {expected_start}."
            )

        if self.position[-1] != expected_end:
            raise ValueError(
                f"Line {self.name} must end at endpoint "
                f"{self.node_b} {expected_end}."
            )

        for point in self.position:
            if not graph.contains(point):
                raise ValueError(
                    f"Line {self.name} contains a point outside "
                    f"the graph: {point}"
                )

        for start, end in zip(
            self.position[:-1],
            self.position[1:],
        ):
            if start == end:
                raise ValueError(
                    f"Line {self.name} contains a zero-length "
                    f"segment at {start}."
                )

            x1, y1 = start
            x2, y2 = end

            if x1 != x2 and y1 != y2:
                raise ValueError(
                    f"Diagonal segment is not allowed in "
                    f"{self.name}: {start} -> {end}"
                )

    @staticmethod
    def _expand_polyline(
        vertices: Tuple[Position, ...],
    ) -> Tuple[Position, ...]:
        """Expand sparse orthogonal vertices into every integer coordinate."""

        expanded: list[Position] = [vertices[0]]

        for start, end in zip(vertices[:-1], vertices[1:]):
            x1, y1 = start
            x2, y2 = end

            dx = (x2 > x1) - (x2 < x1)
            dy = (y2 > y1) - (y2 < y1)
            step_count = max(abs(x2 - x1), abs(y2 - y1))

            for step in range(1, step_count + 1):
                expanded.append(
                    (x1 + dx * step, y1 + dy * step)
                )

        return tuple(expanded)

    def bind_pixels(self, graph: Graph) -> None:
        """Bind this line to graph-owned Pixel objects."""

        expanded_positions = self._expand_polyline(self.position)
        self.physical_pixels = tuple(
            graph.get_pixel(position)
            for position in expanded_positions
        )

        # Endpoint Pixels belong to nodes; body Pixels belong only to the line body.
        self.body_pixels = self.physical_pixels[1:-1]

        for pixel in self.physical_pixels:
            pixel.line_names.add(self.name)

    def register_with_nodes(
        self,
        nodes: Dict[str, Node],
    ) -> None:
        """Register both terminals with their endpoint Node objects."""

        self._endpoint_nodes = {
            self.node_a: nodes[self.node_a],
            self.node_b: nodes[self.node_b],
        }

        for node_name, terminal in self.terminals.items():
            nodes[node_name].register_terminal(terminal)

    def get_terminal(self, node_name: str) -> Terminal:
        """Return the terminal connecting this line to one endpoint node."""

        if node_name not in self.terminals:
            raise ValueError(
                f"Node {node_name} is not an endpoint of line {self.name}."
            )

        return self.terminals[node_name]

    def add_switch(self, switch: "Switch") -> None:
        """Attach a switch to the matching node-line terminal."""

        if any(
            existing_switch.name == switch.name
            for existing_switch in self.switches
        ):
            raise ValueError(
                f"Switch {switch.name} is already attached "
                f"to line {self.name}."
            )

        terminal = self.get_terminal(switch.node_name)
        terminal.attach_switch(switch)
        self.switches.append(switch)
        self.update_state()

    @property
    def effective_pixels(self) -> Tuple[Pixel, ...]:
        """Return body Pixels plus endpoint Pixels connected by closed terminals."""

        pixels: list[Pixel] = list(self.body_pixels)

        if self.terminals[self.node_a].is_connected:
            pixels.insert(0, self._endpoint_nodes[self.node_a].pixel)

        if self.terminals[self.node_b].is_connected:
            pixels.append(self._endpoint_nodes[self.node_b].pixel)

        return tuple(pixels)

    @property
    def has_direct_fault(self) -> bool:
        """Return whether an active fault occupies a line-body Pixel."""

        return (any(f.active for f in self.segment_faults) if hasattr(self, "segment_faults")
                else any(pixel.has_active_fault for pixel in self.body_pixels))

    @property
    def has_active_fault(self) -> bool:
        """Return whether any currently connected Pixel has an active fault."""

        if hasattr(self, "segment_faults"):
            return any(f.active for f in self.segment_faults)
        return any(
            pixel.has_active_fault
            for pixel in self.effective_pixels
        )

    def knowledge_state(self, event_id: int = 0) -> str:
        """Derive unknown, partially_known, or known from effective Pixels."""

        if hasattr(self, "segment_knowledge"):
            return self.segment_knowledge()
        pixels = self.effective_pixels

        if not pixels:
            return "unknown"

        inspected_count = sum(
            pixel.is_inspected(event_id)
            for pixel in pixels
        )

        if inspected_count == 0:
            return "unknown"

        if inspected_count == len(pixels):
            return "known"

        return "partially_known"

    def update_state(self) -> None:
        """Update line state from terminal connectivity and active faults."""

        all_terminals_connected = all(
            terminal.is_connected
            for terminal in self.terminals.values()
        )

        self.state = (
            "closed"
            if all_terminals_connected and not self.has_active_fault
            else "opened"
        )

    def refresh_state(self) -> None:
        """Preserve the earlier method name as a compatibility alias."""

        self.update_state()

    def _get_longest_segment_center(self):
        """Return the midpoint of the longest polyline segment."""

        longest_start = None
        longest_end = None
        longest_length = -1

        for start, end in zip(
            self.position[:-1],
            self.position[1:],
        ):
            x1, y1 = start
            x2, y2 = end

            segment_length = abs(x2 - x1) + abs(y2 - y1)

            if segment_length > longest_length:
                longest_length = segment_length
                longest_start = start
                longest_end = end

        if longest_start is None or longest_end is None:
            raise ValueError(
                f"Cannot find a valid segment for {self.name}."
            )

        x1, y1 = longest_start
        x2, y2 = longest_end

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        return center_x, center_y

    def _get_label_location(
        self,
        label_gap: float,
    ):
        """Calculate one of four label positions around the line midpoint."""

        center_x, center_y = self._get_longest_segment_center()

        layouts = {
            2: (center_x, center_y + label_gap, "center", "bottom"),
            4: (center_x - label_gap, center_y, "right", "center"),
            6: (center_x + label_gap, center_y, "left", "center"),
            8: (center_x, center_y - label_gap, "center", "top"),
        }

        return layouts[self.label_position]

    def draw(
        self,
        ax: Axes,
        event_id: int = 0,
        label_gap: float = 0.24,
        show_label: bool = True,
    ) -> None:
        """Draw known, unknown, energized, and disconnected line sections."""

        knowledge = self.knowledge_state(event_id)

        if knowledge == "known":
            x_values = [point[0] for point in self.position]
            y_values = [point[1] for point in self.position]
            line_color = "green" if self.energized else "red"
            line_style = "solid" if self.state == "closed" else "dashed"

            ax.plot(
                x_values,
                y_values,
                color=line_color,
                linewidth=1.8,
                linestyle=line_style,
                zorder=2,
            )
        else:
            # Unknown line sections are assumed disconnected.
            for first_pixel, second_pixel in zip(
                self.physical_pixels[:-1],
                self.physical_pixels[1:],
            ):
                segment_is_known = (
                    first_pixel.is_inspected(event_id)
                    and second_pixel.is_inspected(event_id)
                )
                segment_color = "red" if segment_is_known else "gray"

                ax.plot(
                    [first_pixel.position[0], second_pixel.position[0]],
                    [first_pixel.position[1], second_pixel.position[1]],
                    color=segment_color,
                    linewidth=1.8,
                    linestyle="dashed",
                    zorder=2,
                )

        if show_label:
            label_x, label_y, horizontal, vertical = (
                self._get_label_location(
                    label_gap=label_gap,
                )
            )

            ax.text(
                label_x,
                label_y,
                self.name,
                color="#4DA3FF",
                fontsize=9,
                fontfamily="DejaVu Serif",
                fontstyle="italic",
                horizontalalignment=horizontal,
                verticalalignment=vertical,
                zorder=70,
            )


@dataclass
class Switch:
    name: str
    line_name: str
    node_name: str
    state: str
    label_position: int
    operation_type: str = "remote"
    operation_time_min: float = 0.0
    true_state: str = field(
        init=False,
    )

    _line: Optional[ElectricalLine] = field(
        default=None,
        init=False,
        repr=False,
    )

    _terminal: Optional[Terminal] = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Initialize the hidden physical state used by the simulator."""

        normalized_state = self.state.lower()

        if normalized_state not in {"opened", "closed", "unknown"}:
            raise ValueError(
                f"Invalid state for switch {self.name}: {self.state}"
            )

        self.state = normalized_state
        self.operation_type = str(self.operation_type).strip().lower()
        if self.operation_type not in {"remote", "manual"}:
            raise ValueError(
                f"Invalid operation_type for switch {self.name}: "
                f"{self.operation_type}"
            )
        self.operation_time_min = float(self.operation_time_min)
        if self.operation_time_min < 0:
            raise ValueError(
                f"Switch {self.name} operation_time_min cannot be negative."
            )
        if self.operation_type == "manual" and self.operation_time_min <= 0:
            raise ValueError(
                f"Manual switch {self.name} requires positive operation time."
            )
        self.true_state = (
            normalized_state
            if normalized_state in {"opened", "closed"}
            else "opened"
        )

    @classmethod
    def load_all(
        cls,
        json_path: Path,
        nodes: Dict[str, Node],
        electrical_lines: Dict[str, ElectricalLine],
    ) -> Dict[str, "Switch"]:
        """Load switches and attach each one to a node-line terminal."""

        if not json_path.exists():
            raise FileNotFoundError(
                f"Cannot find switch file: {json_path}"
            )

        with json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        switches: Dict[str, Switch] = {}

        for switch_data in data["Switches"]:
            switch = cls(
                name=switch_data["name"],
                line_name=switch_data["line"],
                node_name=switch_data["node"],
                state=switch_data["state"].lower(),
                label_position=switch_data["label_position"],
                operation_type=switch_data.get("operation_type", "remote"),
                operation_time_min=switch_data.get("operation_time_min", 0.0),
            )

            if switch.name in switches:
                raise ValueError(
                    f"Duplicate switch name: {switch.name}"
                )

            switch.validate(
                nodes=nodes,
                electrical_lines=electrical_lines,
            )

            switch.attach_line(
                electrical_lines[switch.line_name]
            )

            switches[switch.name] = switch

        return switches

    def validate(
        self,
        nodes: Dict[str, Node],
        electrical_lines: Dict[str, ElectricalLine],
    ) -> None:
        """Validate switch state, label position, line, and endpoint node."""

        if self.state not in {"opened", "closed", "unknown"}:
            raise ValueError(
                f"Invalid state for switch {self.name}: {self.state}"
            )

        if self.label_position not in range(1, 10):
            raise ValueError(
                f"Switch {self.name} has invalid label_position: "
                f"{self.label_position}. It must be 1-9."
            )

        if self.line_name not in electrical_lines:
            raise ValueError(
                f"Switch {self.name} references unknown line: "
                f"{self.line_name}"
            )

        if self.node_name not in nodes:
            raise ValueError(
                f"Switch {self.name} references unknown node: "
                f"{self.node_name}"
            )

        line = electrical_lines[self.line_name]

        if self.node_name not in line.end_nodes:
            raise ValueError(
                f"Switch {self.name} is near {self.node_name}, "
                f"but this node is not an endpoint of "
                f"{self.line_name}."
            )

    def attach_line(
        self,
        line: ElectricalLine,
    ) -> None:
        """Attach this switch to the matching line terminal."""

        self._line = line
        self._terminal = line.get_terminal(self.node_name)
        line.add_switch(self)

    def open(self) -> None:
        """Open the physical switch while preserving hidden knowledge."""

        self.true_state = "opened"

        if self.state != "unknown":
            self.state = "opened"

        if self._line is not None:
            self._line.update_state()

    def close(self) -> None:
        """Close the physical switch while preserving hidden knowledge."""

        self.true_state = "closed"

        if self.state != "unknown":
            self.state = "closed"

        if self._line is not None:
            self._line.update_state()

    def hide_state(self) -> None:
        """Hide the switch state and assume it is disconnected for safety."""

        self.state = "unknown"

        if self._line is not None:
            self._line.update_state()

    def reveal_state(self) -> None:
        """Reveal the current physical switch state."""

        self.state = self.true_state

        if self._line is not None:
            self._line.update_state()

    def update_knowledge(
        self,
        nodes: Dict[str, Node],
        event_id: int = 0,
    ) -> None:
        """Reveal the switch only after its corresponding node is inspected."""

        node_is_known = (
            nodes[self.node_name].knowledge_state(event_id) == "known"
        )

        if node_is_known:
            self.reveal_state()
        else:
            self.hide_state()

    def get_center_position(
        self,
        nodes: Dict[str, Node],
        distance_from_node: float = 0.6,
    ) -> Tuple[float, float]:
        """Return the geometric center used to draw this switch."""

        if self._line is None:
            raise RuntimeError(
                f"Switch {self.name} is not attached to a line."
            )

        line = self._line
        node_x, node_y = nodes[self.node_name].position

        if self.node_name == line.node_a:
            adjacent_point = line.position[1]
        else:
            adjacent_point = line.position[-2]

        next_x, next_y = adjacent_point

        if next_x > node_x:
            dx, dy = 1, 0
        elif next_x < node_x:
            dx, dy = -1, 0
        elif next_y > node_y:
            dx, dy = 0, 1
        elif next_y < node_y:
            dx, dy = 0, -1
        else:
            raise ValueError(
                f"Cannot determine switch direction: {self.name}"
            )

        return (
            node_x + dx * distance_from_node,
            node_y + dy * distance_from_node,
        )

    def _get_label_location(
        self,
        center_x: float,
        center_y: float,
        rectangle_width: float,
        rectangle_height: float,
        label_gap: float,
    ):
        """Calculate label coordinates from the numeric label position."""

        x_offset = rectangle_width / 2 + label_gap
        y_offset = rectangle_height / 2 + label_gap

        layouts = {
            1: (center_x - x_offset, center_y + y_offset, "right", "bottom"),
            2: (center_x, center_y + y_offset, "center", "bottom"),
            3: (center_x + x_offset, center_y + y_offset, "left", "bottom"),
            4: (center_x - x_offset, center_y, "right", "center"),
            5: (center_x, center_y, "center", "center"),
            6: (center_x + x_offset, center_y, "left", "center"),
            7: (center_x - x_offset, center_y - y_offset, "right", "top"),
            8: (center_x, center_y - y_offset, "center", "top"),
            9: (center_x + x_offset, center_y - y_offset, "left", "top"),
        }

        return layouts[self.label_position]

    def draw(
        self,
        ax: Axes,
        nodes: Dict[str, Node],
        event_id: int = 0,
        distance_from_node: float = 0.6,
        short_side: float = 0.3,
        long_side: float = 0.4,
        label_gap: float = 0.12,
        show_label: bool = True,
    ) -> None:
        """Draw the switch near its endpoint node."""

        if self._line is None:
            raise RuntimeError(
                f"Switch {self.name} is not attached to a line."
            )

        line = self._line
        node_x, node_y = nodes[self.node_name].position
        center_x, center_y = self.get_center_position(
            nodes=nodes,
            distance_from_node=distance_from_node,
        )

        dx = int(np.sign(center_x - node_x))
        dy = int(np.sign(center_y - node_y))

        if dx != 0:
            rectangle_width = long_side
            rectangle_height = short_side
        else:
            rectangle_width = short_side
            rectangle_height = long_side

        lower_left_x = center_x - rectangle_width / 2
        lower_left_y = center_y - rectangle_height / 2

        if self.state == "unknown":
            fill_color = "gray"
        elif self.state == "closed":
            fill_color = "green"
        else:
            fill_color = "red"

        rectangle = Rectangle(
            xy=(lower_left_x, lower_left_y),
            width=rectangle_width,
            height=rectangle_height,
            facecolor=fill_color,
            edgecolor="black",
            linewidth=1.0,
            zorder=30,
        )

        ax.add_patch(rectangle)

        if show_label:
            label_x, label_y, horizontal, vertical = (
                self._get_label_location(
                    center_x=center_x,
                    center_y=center_y,
                    rectangle_width=rectangle_width,
                    rectangle_height=rectangle_height,
                    label_gap=label_gap,
                )
            )

            ax.text(
                label_x,
                label_y,
                self.name,
                color="black",
                fontsize=8,
                horizontalalignment=horizontal,
                verticalalignment=vertical,
                zorder=80,
            )


@dataclass
class Area:
    """Represent one static minimum isolatable unit of the network."""

    name: str
    node_names: Tuple[str, ...]
    line_names: Tuple[str, ...]
    boundary_switch_names: Tuple[str, ...]

    _nodes: Dict[str, Node] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    _electrical_lines: Dict[str, ElectricalLine] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    _switches: Dict[str, Switch] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    _area_lookup: Dict[str, "Area"] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    @staticmethod
    def _natural_name_key(name: str):
        """Return a stable key that places L2 before L10."""

        prefix = "".join(character for character in name if not character.isdigit())
        digits = "".join(character for character in name if character.isdigit())
        number = int(digits) if digits else -1

        return prefix, number, name

    @classmethod
    def generate_all(
        cls,
        nodes: Dict[str, Node],
        electrical_lines: Dict[str, ElectricalLine],
        switches: Dict[str, Switch],
    ) -> Dict[str, "Area"]:
        """
        Generate static areas once by removing every switch-controlled terminal.

        Switch states are intentionally ignored during area recognition. Later
        switch operations change only dynamic area properties and never merge
        or split the generated Area objects.
        """

        if any(
            node.area_name is not None
            for node in nodes.values()
            if not node.is_transmission
        ) or any(
            line.area_name is not None
            for line in electrical_lines.values()
        ):
            raise RuntimeError(
                "Area generation has already been completed for these objects."
            )

        Entity = Tuple[str, str]
        adjacency: Dict[Entity, set[Entity]] = {}

        def add_connection(entity_a: Entity, entity_b: Entity) -> None:
            adjacency.setdefault(entity_a, set()).add(entity_b)
            adjacency.setdefault(entity_b, set()).add(entity_a)

        # Only terminals without switches create permanent within-area links.
        for line in electrical_lines.values():
            line_entity = ("line", line.name)
            adjacency.setdefault(line_entity, set())

            for node_name in line.end_nodes:
                node = nodes[node_name]
                terminal = line.get_terminal(node_name)

                if node.is_transmission:
                    continue

                node_entity = ("node", node_name)
                adjacency.setdefault(node_entity, set())

                if terminal.switch is None:
                    add_connection(node_entity, line_entity)

        entities: list[Entity] = [
            ("node", node.name)
            for node in nodes.values()
            if not node.is_transmission
        ] + [
            ("line", line.name)
            for line in electrical_lines.values()
        ]

        visited: set[Entity] = set()
        components: list[set[Entity]] = []

        for start_entity in entities:
            if start_entity in visited:
                continue

            component: set[Entity] = set()
            stack = [start_entity]
            visited.add(start_entity)

            while stack:
                current = stack.pop()
                component.add(current)

                for neighbor in adjacency.get(current, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)

            components.append(component)

        def component_sort_key(component: set[Entity]):
            line_names = sorted(
                (
                    entity_name
                    for entity_type, entity_name in component
                    if entity_type == "line"
                ),
                key=cls._natural_name_key,
            )
            node_names = sorted(
                (
                    entity_name
                    for entity_type, entity_name in component
                    if entity_type == "node"
                ),
                key=cls._natural_name_key,
            )

            if line_names:
                return 0, cls._natural_name_key(line_names[0])

            return 1, cls._natural_name_key(node_names[0])

        components.sort(key=component_sort_key)

        memberships: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {}

        for area_index, component in enumerate(components, start=1):
            area_name = f"A{area_index}"
            area_node_names = tuple(
                sorted(
                    (
                        entity_name
                        for entity_type, entity_name in component
                        if entity_type == "node"
                    ),
                    key=cls._natural_name_key,
                )
            )
            area_line_names = tuple(
                sorted(
                    (
                        entity_name
                        for entity_type, entity_name in component
                        if entity_type == "line"
                    ),
                    key=cls._natural_name_key,
                )
            )

            memberships[area_name] = (
                area_node_names,
                area_line_names,
            )

            for node_name in area_node_names:
                nodes[node_name].area_name = area_name

            for line_name in area_line_names:
                electrical_lines[line_name].area_name = area_name

        boundary_switches: Dict[str, set[str]] = {
            area_name: set()
            for area_name in memberships
        }

        for switch in switches.values():
            node_area_name = nodes[switch.node_name].area_name
            line_area_name = electrical_lines[switch.line_name].area_name

            if node_area_name == line_area_name:
                continue

            if node_area_name is not None:
                boundary_switches[node_area_name].add(switch.name)

            if line_area_name is not None:
                boundary_switches[line_area_name].add(switch.name)

        areas: Dict[str, Area] = {}

        for area_name, (area_node_names, area_line_names) in memberships.items():
            area = cls(
                name=area_name,
                node_names=area_node_names,
                line_names=area_line_names,
                boundary_switch_names=tuple(
                    sorted(
                        boundary_switches[area_name],
                        key=cls._natural_name_key,
                    )
                ),
            )
            area._nodes = nodes
            area._electrical_lines = electrical_lines
            area._switches = switches
            areas[area_name] = area

        for area in areas.values():
            area._area_lookup = areas

        return areas

    @property
    def pixels(self) -> Tuple[Pixel, ...]:
        """Return the fixed union of all node and line Pixels in this area."""

        unique_pixels: Dict[Position, Pixel] = {}

        for node_name in self.node_names:
            node = self._nodes[node_name]
            unique_pixels[node.pixel.position] = node.pixel

        for line_name in self.line_names:
            line = self._electrical_lines[line_name]

            for pixel in line.physical_pixels:
                unique_pixels[pixel.position] = pixel

        return tuple(unique_pixels.values())

    def knowledge_state(self, event_id: int = 0) -> str:
        """Derive unknown, partially_known, or known from all area Pixels."""

        if hasattr(self, "segment_knowledge"):
            return self.segment_knowledge()
        area_pixels = self.pixels

        if not area_pixels:
            return "unknown"

        inspected_count = sum(
            pixel.is_inspected(event_id)
            for pixel in area_pixels
        )

        if inspected_count == 0:
            return "unknown"

        if inspected_count == len(area_pixels):
            return "known"

        return "partially_known"

    def is_inspected(self, event_id: int = 0) -> bool:
        """Return whether every Pixel in this area has been inspected."""

        return self.knowledge_state(event_id) == "known"

    @property
    def has_active_fault(self) -> bool:
        """Return whether any Pixel in this area contains an active fault."""

        return (any(f.active for f in self.segment_faults) if hasattr(self, "segment_faults")
                else any(pixel.has_active_fault for pixel in self.pixels))

    @property
    def is_fault_free(self) -> bool:
        """Return whether this area contains no active fault."""

        return not self.has_active_fault

    def current_connected_area_names(self) -> Tuple[str, ...]:
        """Return static areas currently linked through closed switches."""

        visited = {self.name}
        queue = [self.name]

        while queue:
            current_area_name = queue.pop(0)
            current_area = self._area_lookup[current_area_name]

            for switch_name in current_area.boundary_switch_names:
                switch = self._switches[switch_name]

                if switch.state != "closed":
                    continue

                node_area_name = self._nodes[switch.node_name].area_name
                line_area_name = self._electrical_lines[
                    switch.line_name
                ].area_name

                if current_area_name == node_area_name:
                    neighbor_area_name = line_area_name
                elif current_area_name == line_area_name:
                    neighbor_area_name = node_area_name
                else:
                    continue

                if (
                    neighbor_area_name is not None
                    and neighbor_area_name not in visited
                ):
                    visited.add(neighbor_area_name)
                    queue.append(neighbor_area_name)

        return tuple(
            sorted(
                visited,
                key=self._natural_name_key,
            )
        )

    def connected_transmission_node_names(self) -> Tuple[str, ...]:
        """Return transmission nodes connected to the current area component."""

        transmission_nodes: set[str] = set()

        for area_name in self.current_connected_area_names():
            area = self._area_lookup[area_name]

            for line_name in area.line_names:
                line = self._electrical_lines[line_name]

                for node_name in line.end_nodes:
                    node = self._nodes[node_name]

                    if (
                        node.is_transmission
                        and line.get_terminal(node_name).is_connected
                    ):
                        transmission_nodes.add(node_name)

        return tuple(
            sorted(
                transmission_nodes,
                key=self._natural_name_key,
            )
        )

    def connected_component_is_fault_free(self) -> bool:
        """Return whether this area and all currently connected areas are fault-free."""

        return all(
            self._area_lookup[area_name].is_fault_free
            for area_name in self.current_connected_area_names()
        )

    def is_safe_to_energize(self, event_id: int = 0) -> bool:
        """Return whether all connected areas are known and fault-free."""

        return all(
            self._area_lookup[area_name].is_inspected(event_id)
            and self._area_lookup[area_name].is_fault_free
            for area_name in self.current_connected_area_names()
        )

    def is_energizable(self, event_id: int = 0) -> bool:
        """Return whether this area is safe and connected to the transmission grid."""

        return (
            self.is_safe_to_energize(event_id)
            and bool(self.connected_transmission_node_names())
        )


@dataclass
class Fault:
    """Represent one active fault occupying a graph-owned Pixel."""

    name: str
    position: Position
    repair_resource: int
    active: bool = True
    remaining_repair_resource: int = field(init=False)
    analyzed: bool = field(default=False, init=False)
    analyzed_by: Optional[str] = field(default=None, init=False)
    analyzed_time_step: Optional[int] = field(default=None, init=False)
    pixel: Pixel = field(init=False, repr=False)
    area_names: Tuple[str, ...] = field(
        default_factory=tuple,
        init=False,
    )

    def __post_init__(self) -> None:
        """Initialize the remaining repair requirement."""

        if self.repair_resource < 0:
            raise ValueError(
                f"Fault {self.name} has negative repair_resource."
            )

        self.remaining_repair_resource = (
            self.repair_resource
            if self.active
            else 0
        )

    def mark_analyzed(self, unit_name: str, time_step: Optional[int] = None) -> None:
        """Reveal repair requirements after one automatic analysis step."""

        self.analyzed = True
        self.analyzed_by = str(unit_name)
        self.analyzed_time_step = time_step

    @classmethod
    def load_all(
        cls,
        json_path: Path,
        graph: Graph,
        nodes: Dict[str, Node],
        electrical_lines: Dict[str, ElectricalLine],
        areas: Dict[str, Area],
    ) -> Dict[str, "Fault"]:
        """Load faults, validate their locations, and attach them to Pixels."""

        if not json_path.exists():
            raise FileNotFoundError(
                f"Cannot find fault file: {json_path}"
            )

        with json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        faults: Dict[str, Fault] = {}

        for fault_data in data.get("Faults", []):
            fault = cls(
                name=fault_data["name"],
                position=tuple(fault_data["position"]),
                repair_resource=int(fault_data["repair_resource"]),
                active=bool(fault_data.get("active", True)),
            )

            if fault.name in faults:
                raise ValueError(
                    f"Duplicate fault name: {fault.name}"
                )

            fault.bind(
                graph=graph,
                nodes=nodes,
                electrical_lines=electrical_lines,
                areas=areas,
            )
            faults[fault.name] = fault

        return faults

    def bind(
        self,
        graph: Graph,
        nodes: Dict[str, Node],
        electrical_lines: Dict[str, ElectricalLine],
        areas: Dict[str, Area],
    ) -> None:
        """Bind the fault to one valid node or line Pixel."""

        if self.repair_resource < 0:
            raise ValueError(
                f"Fault {self.name} has negative repair_resource."
            )

        if not graph.contains(self.position):
            raise ValueError(
                f"Fault {self.name} is outside the graph: {self.position}"
            )

        pixel = graph.get_pixel(self.position)

        if not pixel.node_names and not pixel.line_names:
            raise ValueError(
                f"Fault {self.name} at {self.position} "
                "must be located on a node or electrical line."
            )

        if pixel.fault_protected:
            raise ValueError(
                f"Fault {self.name} cannot occupy protected Pixel "
                f"{self.position}."
            )

        self.pixel = pixel
        self.area_names = self._resolve_area_names(
            nodes=nodes,
            electrical_lines=electrical_lines,
            areas=areas,
        )

        if self.active:
            self.pixel.add_fault(self.name)

    def _resolve_area_names(
        self,
        nodes: Dict[str, Node],
        electrical_lines: Dict[str, ElectricalLine],
        areas: Dict[str, Area],
    ) -> Tuple[str, ...]:
        """Return the static Area or Areas directly containing this fault."""

        area_names: set[str] = set()

        # A node fault belongs to the node's static Area. Incident lines on
        # switch-separated sides are affected only while their terminals connect.
        if self.pixel.node_names:
            for node_name in self.pixel.node_names:
                area_name = nodes[node_name].area_name

                if area_name is not None:
                    area_names.add(area_name)
        else:
            for line_name in self.pixel.line_names:
                area_name = electrical_lines[line_name].area_name

                if area_name is not None:
                    area_names.add(area_name)

        unknown_area_names = area_names.difference(areas)

        if unknown_area_names:
            raise ValueError(
                f"Fault {self.name} references unknown Areas: "
                f"{sorted(unknown_area_names)}"
            )

        if not area_names:
            raise ValueError(
                f"Fault {self.name} at {self.position} "
                "cannot be assigned to a static Area."
            )

        return tuple(
            sorted(
                area_names,
                key=Area._natural_name_key,
            )
        )

    def is_known(self, event_id: int = 0) -> bool:
        """Return whether the fault Pixel has been inspected."""

        return self.pixel.is_inspected(event_id)

    def apply_repair(self, resource_amount: int) -> int:
        """Apply repair resources and return the amount actually used."""

        if resource_amount < 0:
            raise ValueError("Repair resource amount cannot be negative.")

        if not self.active or resource_amount == 0:
            return 0

        used_resource = min(
            int(resource_amount),
            self.remaining_repair_resource,
        )
        self.remaining_repair_resource -= used_resource

        if self.remaining_repair_resource == 0:
            self.repair()

        return used_resource

    def repair(self) -> None:
        """Complete this fault repair and remove it from the Pixel."""

        if not self.active:
            return

        self.remaining_repair_resource = 0
        self.pixel.remove_fault(self.name)
        self.active = False

    @staticmethod
    def isolate_affected_areas(
        faults: Dict[str, "Fault"],
        areas: Dict[str, Area],
        switches: Dict[str, Switch],
    ) -> Tuple[str, ...]:
        """Open every boundary switch of every Area containing an active fault."""

        affected_area_names = {
            area_name
            for fault in faults.values()
            if fault.active
            for area_name in fault.area_names
        }

        opened_switch_names: set[str] = set()

        for area_name in affected_area_names:
            area = areas[area_name]

            for switch_name in area.boundary_switch_names:
                switches[switch_name].open()
                opened_switch_names.add(switch_name)

        return tuple(
            sorted(
                opened_switch_names,
                key=Area._natural_name_key,
            )
        )

    def draw(
        self,
        ax: Axes,
        event_id: int = 0,
        radius: float = 0.28,
        show_label: bool = True,
    ) -> None:
        """Draw a known active fault as a circle containing an X."""

        if not self.active or not self.is_known(event_id):
            return

        x, y = self.position
        circle = Circle(
            xy=(x, y),
            radius=radius,
            facecolor="white",
            edgecolor="darkred",
            linewidth=1.8,
            zorder=65,
        )
        ax.add_patch(circle)

        diagonal = radius * 0.62
        ax.plot(
            [x - diagonal, x + diagonal],
            [y - diagonal, y + diagonal],
            color="darkred",
            linewidth=1.8,
            zorder=66,
        )
        ax.plot(
            [x - diagonal, x + diagonal],
            [y + diagonal, y - diagonal],
            color="darkred",
            linewidth=1.8,
            zorder=66,
        )

        if show_label:
            ax.text(
                x + radius + 0.10,
                y + radius + 0.10,
                self.name,
                color="darkred",
                fontsize=9,
                fontweight="bold",
                horizontalalignment="left",
                verticalalignment="bottom",
                zorder=91,
            )

@dataclass
class LegalityChecker:
    """Run all currently implemented simulator constraints."""

    nodes: Dict[str, Node]
    electrical_lines: Dict[str, ElectricalLine]

    def check_no_cycle(self) -> bool:
        """Check that currently closed lines form an acyclic graph."""

        parent = {
            node_name: node_name
            for node_name in self.nodes
        }

        rank = {
            node_name: 0
            for node_name in self.nodes
        }

        def find(node_name: str) -> str:
            if parent[node_name] != node_name:
                parent[node_name] = find(parent[node_name])

            return parent[node_name]

        def union(
            node_1: str,
            node_2: str,
        ) -> bool:
            root_1 = find(node_1)
            root_2 = find(node_2)

            # Connecting nodes already in one component creates a cycle.
            if root_1 == root_2:
                return False

            if rank[root_1] < rank[root_2]:
                parent[root_1] = root_2
            elif rank[root_1] > rank[root_2]:
                parent[root_2] = root_1
            else:
                parent[root_2] = root_1
                rank[root_1] += 1

            return True

        for line in self.electrical_lines.values():
            line.update_state()

            # Opened lines are excluded from the current electrical graph.
            if line.state != "closed":
                continue

            node_a, node_b = line.end_nodes
            successfully_joined = union(node_a, node_b)

            if not successfully_joined:
                raise ValueError(
                    f"Cycle detected when adding closed line "
                    f"{line.name}: {node_a} -- {node_b}"
                )

        return True

    def check_no_energized_cycle(self) -> bool:
        """Check that the currently energized network is radial."""

        parent = {
            node_name: node_name
            for node_name in self.nodes
        }
        rank = {
            node_name: 0
            for node_name in self.nodes
        }

        def find(node_name: str) -> str:
            if parent[node_name] != node_name:
                parent[node_name] = find(parent[node_name])
            return parent[node_name]

        def union(node_1: str, node_2: str) -> bool:
            root_1 = find(node_1)
            root_2 = find(node_2)

            if root_1 == root_2:
                return False

            if rank[root_1] < rank[root_2]:
                parent[root_1] = root_2
            elif rank[root_1] > rank[root_2]:
                parent[root_2] = root_1
            else:
                parent[root_2] = root_1
                rank[root_1] += 1

            return True

        for line in self.electrical_lines.values():
            if not line.energized:
                continue

            node_a, node_b = line.end_nodes
            if not union(node_a, node_b):
                raise ValueError(
                    "The proposed switch batch creates an energized loop "
                    f"when line {line.name} connects {node_a} and {node_b}."
                )

        return True

    def validate(self) -> bool:
        """Run every currently implemented legality check."""

        self.check_no_cycle()

        print(
            "Legality check passed: "
            "the closed-line graph contains no cycle."
        )

        return True


@dataclass(frozen=True)
class UnknownRegion:
    """Represent one rectangular region whose Pixels are initially unknown."""

    name: str
    lower_left: Position
    upper_right: Position

    def contains(self, position: Tuple[float, float]) -> bool:
        """Return whether a coordinate lies inside this region."""

        x, y = position
        return (
            self.lower_left[0] <= x <= self.upper_right[0]
            and self.lower_left[1] <= y <= self.upper_right[1]
        )

    def draw(self, ax: Axes) -> None:
        """Draw a subtle boundary around this unknown region."""

        x_min, y_min = self.lower_left
        x_max, y_max = self.upper_right

        patch = Rectangle(
            xy=(x_min - 0.5, y_min - 0.5),
            width=x_max - x_min + 1,
            height=y_max - y_min + 1,
            facecolor="none",
            edgecolor="#B8BDC2",
            linewidth=1.0,
            linestyle="solid",
            alpha=0.85,
            zorder=1.60,
        )
        ax.add_patch(patch)


@dataclass
class UnknownState:
    """Load and apply Pixel knowledge and related switch safety actions."""

    event_id: int
    default_inspected: bool
    unknown_regions: Tuple[UnknownRegion, ...]
    open_switches_in_unknown_regions: bool = True
    open_switches_adjacent_to_unknown_areas: bool = True

    @classmethod
    def from_json(cls, json_path: Path) -> "UnknownState":
        """Load one unknown-state definition from a JSON file."""

        if not json_path.exists():
            raise FileNotFoundError(
                f"Cannot find unknown-state file: {json_path}"
            )

        with json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        regions = tuple(
            UnknownRegion(
                name=region_data["name"],
                lower_left=tuple(region_data["lower_left"]),
                upper_right=tuple(region_data["upper_right"]),
            )
            for region_data in data.get("unknown_regions", [])
        )

        return cls(
            event_id=int(data.get("event_id", 0)),
            default_inspected=bool(
                data.get("default_inspected", True)
            ),
            unknown_regions=regions,
            open_switches_in_unknown_regions=bool(
                data.get(
                    "open_switches_in_unknown_regions",
                    True,
                )
            ),
            open_switches_adjacent_to_unknown_areas=bool(
                data.get(
                    "open_switches_adjacent_to_unknown_areas",
                    True,
                )
            ),
        )

    def _touches_unknown_area(
        self,
        switch: Switch,
        nodes: Dict[str, Node],
        electrical_lines: Dict[str, ElectricalLine],
        areas: Dict[str, Area],
    ) -> bool:
        """Return whether either static side of a switch is not fully known."""

        node_area_name = nodes[switch.node_name].area_name
        line_area_name = electrical_lines[switch.line_name].area_name

        adjacent_area_names = {
            area_name
            for area_name in (node_area_name, line_area_name)
            if area_name is not None
        }

        return any(
            areas[area_name].knowledge_state(self.event_id) != "known"
            for area_name in adjacent_area_names
        )

    def apply(
        self,
        graph: Graph,
        nodes: Dict[str, Node],
        electrical_lines: Dict[str, ElectricalLine],
        switches: Dict[str, Switch],
        areas: Dict[str, Area],
    ) -> Tuple[str, ...]:
        """Apply Pixel knowledge and open switches required for safety."""

        for pixel in graph.pixels.values():
            if self.default_inspected:
                pixel.mark_inspected(self.event_id)
            else:
                pixel.last_inspected_event = None

            if any(
                region.contains(pixel.position)
                for region in self.unknown_regions
            ) and not pixel.always_inspected:
                pixel.last_inspected_event = None

        safety_opened_switch_names: set[str] = set()

        if self.open_switches_in_unknown_regions:
            for switch in switches.values():
                center = switch.get_center_position(nodes)

                if any(
                    region.contains(center)
                    for region in self.unknown_regions
                ):
                    switch.open()
                    safety_opened_switch_names.add(switch.name)

        if self.open_switches_adjacent_to_unknown_areas:
            for switch in switches.values():
                if self._touches_unknown_area(
                    switch=switch,
                    nodes=nodes,
                    electrical_lines=electrical_lines,
                    areas=areas,
                ):
                    switch.open()
                    safety_opened_switch_names.add(switch.name)

        # A switch is visually unknown until its corresponding node is inspected.
        for switch in switches.values():
            switch.update_knowledge(
                nodes=nodes,
                event_id=self.event_id,
            )

        for line in electrical_lines.values():
            line.update_state()

        update_energization_state(
            nodes=nodes,
            electrical_lines=electrical_lines,
            event_id=self.event_id,
        )

        return tuple(
            sorted(
                safety_opened_switch_names,
                key=Area._natural_name_key,
            )
        )

    def draw_unknown_pixel_background(
        self,
        ax: Axes,
        graph: Graph,
    ) -> None:
        """Shade every currently unknown Pixel with a very light gray fill."""

        for pixel in graph.pixels.values():
            if pixel.is_inspected(self.event_id):
                continue

            x, y = pixel.position
            patch = Rectangle(
                xy=(x - 0.5, y - 0.5),
                width=1.0,
                height=1.0,
                facecolor="#F2F3F5",
                edgecolor="none",
                alpha=0.88,
                zorder=0.35,
            )
            ax.add_patch(patch)

    def draw_unknown_regions(self, ax: Axes) -> None:
        """Draw all unknown-region boundaries."""

        for region in self.unknown_regions:
            region.draw(ax)


# Preserve the previous class name for backward compatibility.
InitialState = UnknownState


def update_energization_state(
    nodes: Dict[str, Node],
    electrical_lines: Dict[str, ElectricalLine],
    event_id: int = 0,
) -> None:
    """Update energized nodes and lines from the transmission-grid sources."""

    for node in nodes.values():
        node.energized = False

    for line in electrical_lines.values():
        line.energized = False

    adjacency: Dict[str, list[Tuple[str, ElectricalLine]]] = {
        node_name: []
        for node_name in nodes
    }

    for line in electrical_lines.values():
        line_is_usable = (
            line.state == "closed"
            and line.knowledge_state(event_id) == "known"
            and not line.has_active_fault
        )

        if not line_is_usable:
            continue

        adjacency[line.node_a].append((line.node_b, line))
        adjacency[line.node_b].append((line.node_a, line))

    queue = [
        node.name
        for node in nodes.values()
        if node.is_transmission
        and node.knowledge_state(event_id) == "known"
        and not node.has_direct_fault
    ]
    visited = set(queue)

    while queue:
        current_name = queue.pop(0)
        nodes[current_name].energized = True

        for neighbor_name, line in adjacency[current_name]:
            neighbor = nodes[neighbor_name]

            if (
                neighbor.knowledge_state(event_id) != "known"
                or neighbor.has_direct_fault
            ):
                continue

            line.energized = True

            if neighbor_name not in visited:
                visited.add(neighbor_name)
                queue.append(neighbor_name)

    # A usable line is energized only when both endpoint nodes are energized.
    for line in electrical_lines.values():
        line.energized = (
            line.state == "closed"
            and line.knowledge_state(event_id) == "known"
            and nodes[line.node_a].energized
            and nodes[line.node_b].energized
            and not line.has_active_fault
        )

@dataclass
class RepairCrew:
    """Represent one repair crew moving, inspecting, and repairing on the grid."""

    name: str
    initial_position: Position
    inspection_range: int
    move_speed: int
    repair_speed: int
    total_resources: int
    graph: Graph = field(repr=False)
    minutes_per_step: float = 5.0
    unit_type: str = field(default="RC", init=False)
    body_color: str = field(default="#1976D2", init=False, repr=False)
    edge_color: str = field(default="#0D2A4A", init=False, repr=False)
    label_color: str = field(default="#0D47A1", init=False, repr=False)
    range_color: str = field(default="#64B5F6", init=False, repr=False)
    position: Position = field(init=False)
    remaining_resources: int = field(init=False)
    state: str = field(default="Idle", init=False)
    heading: Tuple[float, float] = field(
        default=(1.0, 0.0),
        init=False,
    )
    waypoints: list[Position] = field(
        default_factory=list,
        init=False,
        repr=False,
    )
    planned_path: list[Position] = field(
        default_factory=list,
        init=False,
        repr=False,
    )
    repair_fault_name: Optional[str] = field(
        default=None,
        init=False,
    )
    pending_repair_amount: int = field(default=0, init=False)
    repair_applied_this_step: int = field(default=0, init=False)
    movement_time_credit: float = field(default=0.0, init=False)
    last_command_type: str = field(default="idle", init=False)
    last_repair_fault_name: Optional[str] = field(default=None, init=False)
    last_repair_amount: int = field(default=0, init=False)
    fault_analysis_enabled: bool = field(default=True, init=False)
    analysis_fault_names: Tuple[str, ...] = field(default_factory=tuple, init=False)
    analysis_steps_remaining: int = field(default=0, init=False)
    analysis_started_this_step: bool = field(default=False, init=False)
    analysis_resume_state: str = field(default="Idle", init=False)
    switch_operation_name: Optional[str] = field(default=None, init=False)
    switch_target_state: Optional[str] = field(default=None, init=False)
    switch_operation_time_min: float = field(default=0.0, init=False)
    switch_operation_elapsed_min: float = field(default=0.0, init=False)
    active_patrol_task_id: Optional[str] = field(default=None, init=False)
    active_patrol_area: Optional[str] = field(default=None, init=False)
    active_patrol_required_lines: Tuple[str, ...] = field(
        default_factory=tuple,
        init=False,
    )
    cumulative_travel_min: float = field(default=0.0, init=False)
    cumulative_travel_pixels: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        """Validate parameters and initialize runtime state."""

        self.initial_position = tuple(self.initial_position)

        if not self.graph.contains(self.initial_position):
            raise ValueError(
                f"Repair crew {self.name} starts outside the graph: "
                f"{self.initial_position}"
            )

        if not isinstance(self.inspection_range, int):
            raise TypeError("inspection_range must be an integer.")

        if self.inspection_range < 0:
            raise ValueError("inspection_range cannot be negative.")

        if not isinstance(self.move_speed, int):
            raise TypeError("move_speed must be an integer.")

        if self.move_speed <= 1:
            raise ValueError("move_speed must be greater than 1.")

        if not isinstance(self.repair_speed, int):
            raise TypeError("repair_speed must be an integer.")

        if self.repair_speed <= 0:
            raise ValueError("repair_speed must be positive.")

        if not isinstance(self.total_resources, int):
            raise TypeError("total_resources must be an integer.")

        if self.total_resources < 0:
            raise ValueError("total_resources cannot be negative.")

        self.position = self.initial_position
        self.remaining_resources = self.total_resources

    @classmethod
    def load_all(
        cls,
        json_path: Path,
        graph: Graph,
    ) -> Dict[str, "RepairCrew"]:
        """Load all repair crews from a JSON file."""

        if not json_path.exists():
            raise FileNotFoundError(
                f"Cannot find repair-crew file: {json_path}"
            )

        with json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        repair_crews: Dict[str, RepairCrew] = {}

        for crew_data in data.get("RepairCrews", []):
            repair_crew = cls(
                name=crew_data["name"],
                initial_position=tuple(crew_data["initial_position"]),
                inspection_range=int(crew_data["inspection_range"]),
                move_speed=int(crew_data["move_speed"]),
                repair_speed=int(crew_data["repair_speed"]),
                total_resources=int(crew_data["total_resources"]),
                graph=graph,
            )

            if repair_crew.name in repair_crews:
                raise ValueError(
                    f"Duplicate repair-crew name: {repair_crew.name}"
                )

            repair_crews[repair_crew.name] = repair_crew

        return repair_crews

    def _normalize_waypoints(self, coordinates) -> list[Position]:
        """Normalize one coordinate or a sequence of coordinates."""

        if len(coordinates) == 1:
            candidate = coordinates[0]

            if (
                isinstance(candidate, (list, tuple))
                and len(candidate) == 2
                and all(
                    isinstance(value, (int, np.integer))
                    for value in candidate
                )
            ):
                raw_waypoints = [candidate]
            else:
                raw_waypoints = list(candidate)
        else:
            raw_waypoints = list(coordinates)

        if not raw_waypoints:
            raise ValueError(
                f"Repair crew {self.name} received no waypoints."
            )

        if len(raw_waypoints) > 5:
            raise ValueError("A Move command accepts at most five waypoints.")

        normalized_waypoints: list[Position] = []

        for raw_waypoint in raw_waypoints:
            if (
                not isinstance(raw_waypoint, (list, tuple))
                or len(raw_waypoint) != 2
                or not all(
                    isinstance(value, (int, np.integer))
                    for value in raw_waypoint
                )
            ):
                raise ValueError(
                    f"Invalid waypoint for {self.name}: {raw_waypoint}"
                )

            waypoint = (int(raw_waypoint[0]), int(raw_waypoint[1]))

            if not self.graph.contains(waypoint):
                raise ValueError(
                    f"Waypoint is outside the graph: {waypoint}"
                )

            normalized_waypoints.append(waypoint)

        return normalized_waypoints

    def _neighbor_positions(self, position: Position) -> Tuple[Position, ...]:
        """Return valid orthogonal neighboring Pixels."""

        x, y = position
        candidate_positions = (
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
        )

        return tuple(
            candidate
            for candidate in candidate_positions
            if self.graph.contains(candidate)
        )

    def movement_cost(
        self,
        first_position: Position,
        second_position: Position,
    ) -> float:
        """Return travel time for one orthogonal Pixel-to-Pixel step."""

        distance = (
            abs(first_position[0] - second_position[0])
            + abs(first_position[1] - second_position[1])
        )

        if distance != 1:
            raise ValueError(
                "Movement cost requires two orthogonally adjacent Pixels."
            )

        first_pixel = self.graph.get_pixel(first_position)
        second_pixel = self.graph.get_pixel(second_position)
        segment_road_level = min(
            first_pixel.road_level,
            second_pixel.road_level,
        )
        speed_multiplier = Road.speed_multiplier_for_level(
            segment_road_level
        )

        if speed_multiplier > 0.0:
            return 1.0 / (self.move_speed * speed_multiplier)

        return 1.0

    def _find_fastest_path(
        self,
        start: Position,
        destination: Position,
    ) -> list[Position]:
        """Find the minimum-time path using road-level travel speeds."""

        if start == destination:
            return [start]

        distances: Dict[Position, float] = {start: 0.0}
        previous: Dict[Position, Position] = {}
        queue = [(0.0, start)]

        while queue:
            current_distance, current = heapq.heappop(queue)

            if current_distance > distances[current] + 1e-12:
                continue

            if current == destination:
                break

            for neighbor in self._neighbor_positions(current):
                candidate_distance = (
                    current_distance
                    + self.movement_cost(current, neighbor)
                )

                if candidate_distance + 1e-12 < distances.get(
                    neighbor,
                    float("inf"),
                ):
                    distances[neighbor] = candidate_distance
                    previous[neighbor] = current
                    heapq.heappush(
                        queue,
                        (candidate_distance, neighbor),
                    )

        if destination not in distances:
            raise RuntimeError(
                f"Repair crew {self.name} cannot reach {destination}."
            )

        reverse_path = [destination]
        current = destination

        while current != start:
            current = previous[current]
            reverse_path.append(current)

        reverse_path.reverse()
        return reverse_path

    def _build_complete_path(
        self,
        waypoints: list[Position],
    ) -> list[Position]:
        """Build the ordered minimum-time route through all waypoints."""

        complete_path: list[Position] = []
        path_start = self.position

        for waypoint in waypoints:
            path_segment = self._find_fastest_path(
                start=path_start,
                destination=waypoint,
            )

            if len(path_segment) > 1:
                complete_path.extend(path_segment[1:])

            path_start = waypoint

        return complete_path

    def preview_move(self, *coordinates) -> Tuple[Position, ...]:
        """Return a Move route without changing the current command."""

        normalized_waypoints = self._normalize_waypoints(coordinates)
        return tuple(self._build_complete_path(normalized_waypoints))

    def set_patrol_context(
        self,
        task_id: Optional[str],
        area: Optional[str],
        required_lines: Iterable[str] = (),
    ) -> None:
        """Attach public patrol continuity metadata to the current move."""

        self.active_patrol_task_id = str(task_id) if task_id else None
        self.active_patrol_area = str(area) if area else None
        self.active_patrol_required_lines = tuple(sorted({
            str(line_name) for line_name in required_lines if line_name
        }))

    def clear_patrol_context(self) -> None:
        """Clear patrol metadata when the unit starts a different task."""

        self.active_patrol_task_id = None
        self.active_patrol_area = None
        self.active_patrol_required_lines = ()

    def move(self, *coordinates) -> None:
        """Replace any current command with an ordered Move command."""

        normalized_waypoints = self._normalize_waypoints(coordinates)
        complete_path = self._build_complete_path(normalized_waypoints)

        self.waypoints = normalized_waypoints
        self.planned_path = complete_path
        self.repair_fault_name = None
        self.pending_repair_amount = 0
        self.repair_applied_this_step = 0
        self.movement_time_credit = 0.0
        self.state = "Moving" if self.planned_path else "Idle"
        self.last_command_type = "move" if self.planned_path else "idle"
        self.last_repair_fault_name = None
        self.last_repair_amount = 0

    def Move(self, *coordinates) -> None:
        """Preserve the command-style Move name."""

        self.move(*coordinates)

    def schedule_repair(
        self,
        fault_name: str,
        resource_amount: int,
        faults: Dict[str, Fault],
        event_id: int = 0,
    ) -> None:
        """Schedule an explicit repair amount for the next time step."""

        if fault_name not in faults:
            raise KeyError(f"Unknown fault: {fault_name}")

        if not isinstance(resource_amount, (int, np.integer)):
            raise TypeError("Repair resource amount must be an integer.")

        resource_amount = int(resource_amount)
        fault = faults[fault_name]

        if self.position != fault.position:
            raise ValueError(
                f"Repair crew {self.name} must be at {fault.position} "
                f"to repair fault {fault_name}."
            )

        if not fault.active:
            raise ValueError(f"Fault {fault_name} is already repaired.")

        if not fault.is_known(event_id):
            raise ValueError(f"Fault {fault_name} is still unknown.")

        if not fault.analyzed:
            raise ValueError(
                f"Fault {fault_name} has not completed the required analysis step."
            )

        if self.analysis_locked:
            raise ValueError(
                f"Repair crew {self.name} is locked for automatic fault analysis."
            )

        if self.remaining_resources <= 0:
            raise ValueError(
                f"Repair crew {self.name} has no repair resources."
            )

        if resource_amount <= 0:
            raise ValueError("Repair resource amount must be positive.")

        maximum_amount = min(
            self.repair_speed,
            self.remaining_resources,
            fault.remaining_repair_resource,
        )

        if resource_amount > maximum_amount:
            raise ValueError(
                f"Illegal repair amount {resource_amount}. "
                f"The maximum allowed amount is {maximum_amount}."
            )

        self.waypoints.clear()
        self.planned_path.clear()
        self.clear_patrol_context()
        self.movement_time_credit = 0.0
        self.repair_fault_name = fault_name
        self.pending_repair_amount = resource_amount
        self.repair_applied_this_step = 0
        self.state = "Repairing"
        self.last_command_type = "repair"
        self.last_repair_fault_name = fault_name
        self.last_repair_amount = resource_amount

    def repair(
        self,
        fault_name: str,
        faults: Dict[str, Fault],
        event_id: int = 0,
    ) -> None:
        """Schedule the maximum legal one-step repair amount."""

        if fault_name not in faults:
            raise KeyError(f"Unknown fault: {fault_name}")

        fault = faults[fault_name]
        maximum_amount = min(
            self.repair_speed,
            self.remaining_resources,
            fault.remaining_repair_resource,
        )
        self.schedule_repair(
            fault_name=fault_name,
            resource_amount=maximum_amount,
            faults=faults,
            event_id=event_id,
        )

    def Repaire(
        self,
        fault_name: str,
        faults: Dict[str, Fault],
        event_id: int = 0,
    ) -> None:
        """Preserve the requested Repaire command spelling."""

        self.repair(
            fault_name=fault_name,
            faults=faults,
            event_id=event_id,
        )

    def idle(self) -> None:
        """Stop the current command and enter the Idle state."""

        self.waypoints.clear()
        self.planned_path.clear()
        self.clear_patrol_context()
        self.repair_fault_name = None
        self.pending_repair_amount = 0
        self.repair_applied_this_step = 0
        self.movement_time_credit = 0.0
        self.state = "Idle"
        self.last_command_type = "idle"
        self.last_repair_fault_name = None
        self.last_repair_amount = 0

    def Stay(self) -> None:
        """Preserve the requested command-style Stay name."""

        self.idle()

    def inspect(self, event_id: int = 0) -> Tuple[Position, ...]:
        """Inspect every Pixel inside the square inspection range."""

        center_x, center_y = self.position
        newly_inspected_positions: list[Position] = []

        for x in range(
            max(self.graph.x_min, center_x - self.inspection_range),
            min(self.graph.x_max, center_x + self.inspection_range) + 1,
        ):
            for y in range(
                max(self.graph.y_min, center_y - self.inspection_range),
                min(self.graph.y_max, center_y + self.inspection_range) + 1,
            ):
                chebyshev_distance = max(
                    abs(x - center_x),
                    abs(y - center_y),
                )

                if chebyshev_distance > self.inspection_range:
                    continue

                pixel = self.graph.get_pixel((x, y))

                if not pixel.is_inspected(event_id):
                    newly_inspected_positions.append((x, y))

                pixel.mark_inspected(event_id)

        return tuple(newly_inspected_positions)

    def inspect_and_refresh(
        self,
        event_id: int,
        nodes: Dict[str, Node],
        electrical_lines: Dict[str, ElectricalLine],
        switches: Dict[str, Switch],
    ) -> Tuple[Position, ...]:
        """Inspect nearby Pixels and refresh all derived display states."""

        newly_inspected_positions = self.inspect(event_id)

        for switch in switches.values():
            switch.update_knowledge(nodes=nodes, event_id=event_id)

        for line in electrical_lines.values():
            line.update_state()

        update_energization_state(
            nodes=nodes,
            electrical_lines=electrical_lines,
            event_id=event_id,
        )

        return newly_inspected_positions

    @property
    def analysis_locked(self) -> bool:
        """Return whether this unit must spend the next step analyzing faults."""

        return self.state == "Analyzing" and self.analysis_steps_remaining > 0

    def _known_unanalyzed_fault_names_at_position(
        self,
        faults: Dict[str, Fault],
        event_id: int,
    ) -> Tuple[str, ...]:
        """Return known active unanalyzed faults at the unit's exact Pixel."""

        return tuple(sorted(
            fault.name
            for fault in faults.values()
            if (
                fault.active
                and not fault.analyzed
                and fault.position == self.position
                and fault.is_known(event_id)
            )
        ))

    def begin_fault_analysis(
        self,
        fault_names: Iterable[str],
        faults: Dict[str, Fault],
    ) -> bool:
        """Pause the current operation and schedule one automatic analysis step."""

        if not self.fault_analysis_enabled:
            return False

        eligible = tuple(sorted({
            str(name)
            for name in fault_names
            if (
                str(name) in faults
                and faults[str(name)].active
                and not faults[str(name)].analyzed
            )
        }))
        if not eligible:
            return False

        if self.analysis_locked:
            self.analysis_fault_names = tuple(sorted(
                set(self.analysis_fault_names).union(eligible)
            ))
            return True

        self.analysis_resume_state = self.state
        self.analysis_fault_names = eligible
        self.analysis_steps_remaining = 1
        self.analysis_started_this_step = True
        self.movement_time_credit = 0.0
        self.state = "Analyzing"
        return True

    def _complete_fault_analysis(
        self,
        faults: Dict[str, Fault],
    ) -> None:
        """Reveal fault resources and restore the paused operation."""

        for fault_name in self.analysis_fault_names:
            fault = faults.get(fault_name)
            if fault is not None and fault.active:
                fault.mark_analyzed(self.name)

        self.analysis_fault_names = tuple()
        self.analysis_steps_remaining = 0
        self.analysis_started_this_step = False

        resume_state = self.analysis_resume_state
        self.analysis_resume_state = "Idle"
        if resume_state == "Moving" and self.planned_path:
            self.state = "Moving"
        elif (
            resume_state == "Repairing"
            and self.repair_fault_name in faults
            and faults[self.repair_fault_name].active
            and self.remaining_resources > 0
        ):
            self.state = "Repairing"
        else:
            self.state = "Idle"

    def prepare_time_step(self) -> None:
        """Reset per-time-step counters without changing the current command."""

        self.repair_applied_this_step = 0
        if self.analysis_locked:
            self.analysis_started_this_step = False

    def _newly_discovered_fault_names(
        self,
        newly_inspected_positions: Tuple[Position, ...],
        faults: Dict[str, Fault],
    ) -> Tuple[str, ...]:
        """Return active faults revealed by this inspection operation."""

        newly_inspected_set = set(newly_inspected_positions)
        return tuple(
            sorted(
                fault.name
                for fault in faults.values()
                if fault.active and fault.position in newly_inspected_set
            )
        )

    def _advance_movement_frame(
        self,
        frame_duration: float,
        event_id: int,
        nodes: Dict[str, Node],
        electrical_lines: Dict[str, ElectricalLine],
        switches: Dict[str, Switch],
        faults: Dict[str, Fault],
    ) -> Dict[str, object]:
        """Advance movement for one animation frame."""

        self.movement_time_credit += frame_duration
        visited_positions: list[Position] = []
        discovered_fault_names: list[str] = []
        stopped_for_fault = False

        while self.planned_path:
            next_position = self.planned_path[0]
            step_cost = self.movement_cost(self.position, next_position)

            if step_cost > self.movement_time_credit + 1e-12:
                break

            previous_position = self.position
            self.position = next_position
            self.planned_path.pop(0)
            self.movement_time_credit -= step_cost
            self.cumulative_travel_min += (
                step_cost * float(self.minutes_per_step)
            )
            self.cumulative_travel_pixels += 1
            visited_positions.append(self.position)

            direction_x = self.position[0] - previous_position[0]
            direction_y = self.position[1] - previous_position[1]
            direction_norm = np.hypot(direction_x, direction_y)

            if direction_norm > 0:
                self.heading = (
                    direction_x / direction_norm,
                    direction_y / direction_norm,
                )

            newly_inspected_positions = self.inspect_and_refresh(
                event_id=event_id,
                nodes=nodes,
                electrical_lines=electrical_lines,
                switches=switches,
            )
            newly_discovered = self._newly_discovered_fault_names(
                newly_inspected_positions=newly_inspected_positions,
                faults=faults,
            )

            if newly_discovered:
                discovered_fault_names.extend(newly_discovered)

            analysis_targets = tuple(dict.fromkeys(
                list(newly_discovered)
                + list(self._known_unanalyzed_fault_names_at_position(
                    faults=faults,
                    event_id=event_id,
                ))
            ))
            if self.begin_fault_analysis(analysis_targets, faults):
                stopped_for_fault = True
                break

        if self.state == "Moving" and not self.planned_path:
            patrol_context = (
                self.active_patrol_task_id,
                self.active_patrol_area,
                self.active_patrol_required_lines,
            )
            self.idle()
            if patrol_context[0] is not None:
                self.set_patrol_context(*patrol_context)

        if not visited_positions:
            newly_inspected_positions = self.inspect_and_refresh(
                event_id=event_id,
                nodes=nodes,
                electrical_lines=electrical_lines,
                switches=switches,
            )
            newly_discovered = self._newly_discovered_fault_names(
                newly_inspected_positions=newly_inspected_positions,
                faults=faults,
            )
            discovered_fault_names.extend(newly_discovered)

            analysis_targets = tuple(dict.fromkeys(
                list(newly_discovered)
                + list(self._known_unanalyzed_fault_names_at_position(
                    faults=faults,
                    event_id=event_id,
                ))
            ))
            if self.begin_fault_analysis(analysis_targets, faults):
                stopped_for_fault = True

        return {
            "visited_positions": tuple(visited_positions),
            "discovered_fault_names": tuple(dict.fromkeys(discovered_fault_names)),
            "stopped_for_fault": stopped_for_fault,
        }

    def _advance_repair_frame(
        self,
        frame_number: int,
        frames_per_step: int,
        faults: Dict[str, Fault],
    ) -> int:
        """Apply the scheduled repair amount gradually across one time step."""

        if self.repair_fault_name is None:
            raise RuntimeError(
                f"Repair crew {self.name} has no repair target."
            )

        fault = faults[self.repair_fault_name]

        if self.position != fault.position:
            raise RuntimeError(
                f"Repair crew {self.name} left fault {fault.name} "
                "while repairing."
            )

        cumulative_target = (
            self.pending_repair_amount * frame_number
        ) // frames_per_step
        frame_amount = cumulative_target - self.repair_applied_this_step
        used_resource = fault.apply_repair(frame_amount)
        self.repair_applied_this_step += used_resource
        self.remaining_resources -= used_resource

        if (
            frame_number == frames_per_step
            or not fault.active
            or self.remaining_resources == 0
        ):
            repair_can_continue = (
                fault.active
                and self.remaining_resources > 0
                and fault.remaining_repair_resource > 0
            )
            self.repair_applied_this_step = 0

            if repair_can_continue:
                next_amount = min(
                    self.last_repair_amount,
                    self.repair_speed,
                    self.remaining_resources,
                    fault.remaining_repair_resource,
                )
                self.repair_fault_name = fault.name
                self.pending_repair_amount = next_amount
                self.state = "Repairing"
            else:
                self.repair_fault_name = None
                self.pending_repair_amount = 0
                self.state = "Idle"
                self.last_command_type = "idle"
                self.last_repair_fault_name = None
                self.last_repair_amount = 0

        return used_resource

    def schedule_switch_operation(
        self,
        switch: Switch,
        target_state: str,
        nodes: Dict[str, Node],
    ) -> None:
        """Lock this RC at a manual switch until its operation completes."""

        if switch.operation_type != "manual":
            raise ValueError(
                f"Switch {switch.name} is remote and needs no RC operation."
            )
        if switch.state == "unknown":
            raise ValueError(f"Switch {switch.name} is unknown.")
        target_state = str(target_state).strip().lower()
        if target_state not in {"opened", "closed"}:
            raise ValueError("target_state must be 'opened' or 'closed'.")
        if switch.true_state == target_state:
            raise ValueError(
                f"Switch {switch.name} is already {target_state}."
            )
        terminal_position = nodes[switch.node_name].position
        if self.position != terminal_position:
            raise ValueError(
                f"{self.name} must be at terminal Node {switch.node_name} "
                f"{terminal_position} to operate {switch.name}."
            )
        self.waypoints.clear()
        self.planned_path.clear()
        self.clear_patrol_context()
        self.movement_time_credit = 0.0
        self.switch_operation_name = switch.name
        self.switch_target_state = target_state
        self.switch_operation_time_min = switch.operation_time_min
        self.switch_operation_elapsed_min = 0.0
        self.state = "OperatingSwitch"
        self.last_command_type = "operate_switch"

    def _advance_switch_operation_frame(
        self,
        frame_duration_min: float,
        switches: Dict[str, Switch],
    ) -> bool:
        """Advance a physical MS operation and change state only at completion."""

        if self.switch_operation_name is None or self.switch_target_state is None:
            raise RuntimeError(f"{self.name} has incomplete switch-operation state.")
        self.switch_operation_elapsed_min += frame_duration_min
        completed = (
            self.switch_operation_elapsed_min + 1e-9
            >= self.switch_operation_time_min
        )
        if completed:
            switch = switches[self.switch_operation_name]
            if self.switch_target_state == "closed":
                switch.close()
            else:
                switch.open()
            self.switch_operation_name = None
            self.switch_target_state = None
            self.switch_operation_time_min = 0.0
            self.switch_operation_elapsed_min = 0.0
            self.state = "Idle"
            self.last_command_type = "idle"
        return completed

    def advance_frame(
        self,
        frame_duration: float,
        frame_number: int,
        frames_per_step: int,
        event_id: int,
        nodes: Dict[str, Node],
        electrical_lines: Dict[str, ElectricalLine],
        switches: Dict[str, Switch],
        faults: Dict[str, Fault],
    ) -> Dict[str, object]:
        """Advance this repair crew by one animation frame."""

        previous_position = self.position
        previous_state = self.state
        visited_positions: Tuple[Position, ...] = tuple()
        discovered_fault_names: Tuple[str, ...] = tuple()
        stopped_for_fault = False
        used_repair_resource = 0
        switch_operation_completed = False

        if self.state == "Moving":
            movement_result = self._advance_movement_frame(
                frame_duration=frame_duration,
                event_id=event_id,
                nodes=nodes,
                electrical_lines=electrical_lines,
                switches=switches,
                faults=faults,
            )
            visited_positions = movement_result["visited_positions"]
            discovered_fault_names = movement_result["discovered_fault_names"]
            stopped_for_fault = movement_result["stopped_for_fault"]
        elif self.state == "Repairing":
            used_repair_resource = self._advance_repair_frame(
                frame_number=frame_number,
                frames_per_step=frames_per_step,
                faults=faults,
            )
            newly_inspected = self.inspect_and_refresh(
                event_id=event_id,
                nodes=nodes,
                electrical_lines=electrical_lines,
                switches=switches,
            )
            newly_discovered = self._newly_discovered_fault_names(
                newly_inspected_positions=newly_inspected,
                faults=faults,
            )
            discovered_fault_names = newly_discovered
            if self.begin_fault_analysis(newly_discovered, faults):
                stopped_for_fault = True
        elif self.state == "Idle":
            newly_inspected = self.inspect_and_refresh(
                event_id=event_id,
                nodes=nodes,
                electrical_lines=electrical_lines,
                switches=switches,
            )
            newly_discovered = self._newly_discovered_fault_names(
                newly_inspected_positions=newly_inspected,
                faults=faults,
            )
            discovered_fault_names = newly_discovered
            analysis_targets = tuple(dict.fromkeys(
                list(newly_discovered)
                + list(self._known_unanalyzed_fault_names_at_position(
                    faults=faults,
                    event_id=event_id,
                ))
            ))
            if self.begin_fault_analysis(analysis_targets, faults):
                stopped_for_fault = True
        elif self.state == "Analyzing":
            newly_inspected = self.inspect_and_refresh(
                event_id=event_id,
                nodes=nodes,
                electrical_lines=electrical_lines,
                switches=switches,
            )
            newly_discovered = self._newly_discovered_fault_names(
                newly_inspected_positions=newly_inspected,
                faults=faults,
            )
            discovered_fault_names = newly_discovered
            self.begin_fault_analysis(newly_discovered, faults)
            if (
                frame_number == frames_per_step
                and not self.analysis_started_this_step
            ):
                self._complete_fault_analysis(faults)
        elif self.state == "OperatingSwitch":
            switch_operation_completed = self._advance_switch_operation_frame(
                frame_duration_min=frame_duration * self.minutes_per_step,
                switches=switches,
            )
        else:
            raise ValueError(
                f"Unsupported repair-crew state: {self.state}"
            )

        return {
            "name": self.name,
            "previous_state": previous_state,
            "state": self.state,
            "previous_position": previous_position,
            "position": self.position,
            "visited_positions": visited_positions,
            "discovered_fault_names": discovered_fault_names,
            "stopped_for_fault": stopped_for_fault,
            "used_repair_resource": used_repair_resource,
            "remaining_resources": self.remaining_resources,
            "analysis_locked": self.analysis_locked,
            "analysis_fault_names": self.analysis_fault_names,
            "switch_operation_completed": switch_operation_completed,
        }

    def step(
        self,
        event_id: int,
        nodes: Dict[str, Node],
        electrical_lines: Dict[str, ElectricalLine],
        switches: Dict[str, Switch],
        faults: Dict[str, Fault],
        frames_per_step: int = 4,
    ) -> Dict[str, object]:
        """Advance one complete time step using four default frames."""

        self.prepare_time_step()
        frame_results = []

        for frame_number in range(1, frames_per_step + 1):
            frame_results.append(
                self.advance_frame(
                    frame_duration=1.0 / frames_per_step,
                    frame_number=frame_number,
                    frames_per_step=frames_per_step,
                    event_id=event_id,
                    nodes=nodes,
                    electrical_lines=electrical_lines,
                    switches=switches,
                    faults=faults,
                )
            )

        return {
            "name": self.name,
            "state": self.state,
            "position": self.position,
            "remaining_resources": self.remaining_resources,
            "frames": frame_results,
        }

    def draw(
        self,
        ax: Axes,
        body_length: float = 0.62,
        body_width: float = 0.42,
        show_label: bool = True,
        show_inspection_range: bool = False,
    ) -> None:
        """Draw a blue five-sided vehicle pointing in its heading direction."""

        x, y = self.position
        heading_x, heading_y = self.heading
        heading_norm = np.hypot(heading_x, heading_y)

        if heading_norm == 0:
            heading_x, heading_y = 1.0, 0.0
            heading_norm = 1.0

        forward_x = heading_x / heading_norm
        forward_y = heading_y / heading_norm
        side_x = -forward_y
        side_y = forward_x

        half_width = body_width / 2
        back_offset = -body_length * 0.42
        shoulder_offset = body_length * 0.18
        tip_offset = body_length * 0.58

        local_vertices = (
            (back_offset, half_width),
            (shoulder_offset, half_width),
            (tip_offset, 0.0),
            (shoulder_offset, -half_width),
            (back_offset, -half_width),
        )
        vertices = [
            (
                x + forward_x * longitudinal + side_x * lateral,
                y + forward_y * longitudinal + side_y * lateral,
            )
            for longitudinal, lateral in local_vertices
        ]

        patch = Polygon(
            vertices,
            closed=True,
            facecolor=self.body_color,
            edgecolor=self.edge_color,
            linewidth=1.2,
            zorder=75,
        )
        ax.add_patch(patch)

        if show_inspection_range:
            inspection_square = Rectangle(
                (
                    x - self.inspection_range - 0.5,
                    y - self.inspection_range - 0.5,
                ),
                width=2 * self.inspection_range + 1,
                height=2 * self.inspection_range + 1,
                facecolor="none",
                edgecolor=self.range_color,
                linewidth=1.1,
                linestyle="dashdot",
                alpha=0.80,
                zorder=3,
            )
            ax.add_patch(inspection_square)

        if show_label:
            ax.text(
                x,
                y + half_width + 0.18,
                self.name,
                color=self.label_color,
                fontsize=8,
                fontweight="bold",
                horizontalalignment="center",
                verticalalignment="bottom",
                zorder=92,
            )


@dataclass
class Scott(RepairCrew):
    """Represent a fast inspection-only mobile unit without repair ability."""

    repair_speed: int = field(default=0, init=False)
    total_resources: int = field(default=0, init=False)
    unit_type: str = field(default="Scott", init=False)
    body_color: str = field(default="#00897B", init=False, repr=False)
    edge_color: str = field(default="#004D40", init=False, repr=False)
    label_color: str = field(default="#00695C", init=False, repr=False)
    range_color: str = field(default="#4DB6AC", init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate movement and inspection parameters for a Scott unit."""

        self.initial_position = tuple(self.initial_position)

        if not self.graph.contains(self.initial_position):
            raise ValueError(
                f"Scott {self.name} starts outside the graph: "
                f"{self.initial_position}"
            )

        if not isinstance(self.inspection_range, int):
            raise TypeError("inspection_range must be an integer.")

        if self.inspection_range < 0:
            raise ValueError("inspection_range cannot be negative.")

        if not isinstance(self.move_speed, int):
            raise TypeError("move_speed must be an integer.")

        if self.move_speed <= 1:
            raise ValueError("move_speed must be greater than 1.")

        self.position = self.initial_position
        self.remaining_resources = 0

    @classmethod
    def load_all(
        cls,
        json_path: Path,
        graph: Graph,
        nodes: Dict[str, Node],
    ) -> Dict[str, "Scott"]:
        """Load all Scott units and resolve optional node-based positions."""

        if not json_path.exists():
            raise FileNotFoundError(
                f"Cannot find Scott file: {json_path}"
            )

        with json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        scotts: Dict[str, Scott] = {}

        for unit_data in data.get("Scotts", []):
            if "initial_node" in unit_data:
                node_name = unit_data["initial_node"]
                if node_name not in nodes:
                    raise KeyError(
                        f"Scott {unit_data['name']} references unknown "
                        f"node {node_name}."
                    )
                initial_position = nodes[node_name].position
            else:
                initial_position = tuple(unit_data["initial_position"])

            scott = cls(
                name=unit_data["name"],
                initial_position=initial_position,
                inspection_range=int(unit_data["inspection_range"]),
                move_speed=int(unit_data["move_speed"]),
                graph=graph,
            )

            if scott.name in scotts:
                raise ValueError(f"Duplicate Scott name: {scott.name}")

            scotts[scott.name] = scott

        return scotts

    def schedule_repair(self, *args, **kwargs) -> None:
        """Reject repair commands because Scott has no repair function."""

        raise RuntimeError(
            f"Scott {self.name} does not have repair capability."
        )

    def repair(self, *args, **kwargs) -> None:
        """Reject repair commands because Scott has no repair function."""

        self.schedule_repair(*args, **kwargs)

    def Repaire(self, *args, **kwargs) -> None:
        """Reject the command-style Repaire command."""

        self.schedule_repair(*args, **kwargs)


# Preserve the short class name requested by the simulator design.
RC = RepairCrew


@dataclass
class DSRDashboardBackend:
    """Manage staged simulator loading, commands, and dashboard snapshots."""

    data_dir: Path
    output_dir: Path
    graph: Optional[Graph] = field(default=None, init=False)
    nodes: Dict[str, Node] = field(default_factory=dict, init=False)
    electrical_lines: Dict[str, ElectricalLine] = field(default_factory=dict, init=False)
    roads: Dict[str, Road] = field(default_factory=dict, init=False)
    switches: Dict[str, Switch] = field(default_factory=dict, init=False)
    areas: Dict[str, Area] = field(default_factory=dict, init=False)
    faults: Dict[str, Fault] = field(default_factory=dict, init=False)
    repair_crews: Dict[str, RepairCrew] = field(default_factory=dict, init=False)
    scotts: Dict[str, Scott] = field(default_factory=dict, init=False)
    unknown_state: Optional[UnknownState] = field(default=None, init=False)
    event_id: int = field(default=0, init=False)
    time_step: int = field(default=0, init=False)
    current_frame: int = field(default=0, init=False)
    stage: str = field(default="Not loaded", init=False)
    fault_isolation_switch_names: Tuple[str, ...] = field(default_factory=tuple, init=False)
    unknown_safety_switch_names: Tuple[str, ...] = field(default_factory=tuple, init=False)
    preview_paths: Dict[str, Tuple[Position, ...]] = field(default_factory=dict, init=False)
    last_switch_preview: Tuple[Tuple[str, str], ...] = field(
        default_factory=tuple,
        init=False,
    )
    render_revision: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def mobile_units(self) -> Dict[str, RepairCrew]:
        """Return all commandable RC and Scott units by name."""

        return {**self.repair_crews, **self.scotts}

    def _path(self, file_name: str) -> Path:
        path = self.data_dir / file_name
        if not path.exists():
            raise FileNotFoundError(
                f"Cannot find dashboard data file: {path}"
            )
        return path

    def _require_loaded(self) -> None:
        """Raise an error when commands are issued before initialization."""

        if self.graph is None or not self.mobile_units:
            raise RuntimeError("Load the map before issuing commands.")

    def _load_topology(self) -> None:
        """Load a fresh topology and generate static Areas exactly once."""

        self.graph = Graph.from_json(self._path("GRAPH.json"))
        self.nodes = Node.load_all(
            json_path=self._path("Nodes_v5.json"),
            graph=self.graph,
        )
        self.electrical_lines = ElectricalLine.load_all(
            json_path=self._path("ElectricalLines_v4.json"),
            graph=self.graph,
            nodes=self.nodes,
        )
        self.roads = Road.load_all(
            json_path=self._path("Roads_v1.json"),
            graph=self.graph,
        )
        Road.register_electrical_network_as_level_one(
            nodes=self.nodes,
            electrical_lines=self.electrical_lines,
        )
        self.switches = Switch.load_all(
            json_path=self._path("Switch_v5.json"),
            nodes=self.nodes,
            electrical_lines=self.electrical_lines,
        )
        self.areas = Area.generate_all(
            nodes=self.nodes,
            electrical_lines=self.electrical_lines,
            switches=self.switches,
        )
        self.faults = {}
        self.unknown_state = None
        self.fault_isolation_switch_names = tuple()
        self.unknown_safety_switch_names = tuple()
        self.preview_paths = {}
        self.last_switch_preview = tuple()
        self.time_step = 0
        self.current_frame = 0

    def _load_mobile_units(self) -> None:
        """Load fresh RC and Scott runtime objects."""

        if self.graph is None:
            raise RuntimeError("Load the topology before mobile units.")

        self.repair_crews = RepairCrew.load_all(
            json_path=self._path("RepairCrews_v1.json"),
            graph=self.graph,
        )
        self.scotts = Scott.load_all(
            json_path=self._path("Scotts_v1.json"),
            graph=self.graph,
            nodes=self.nodes,
        )

        duplicate_names = set(self.repair_crews) & set(self.scotts)
        if duplicate_names:
            raise ValueError(
                "RC and Scott names must be unique: "
                + ", ".join(sorted(duplicate_names))
            )

    def _refresh_derived_states(self) -> None:
        """Refresh switch knowledge, line states, and energization."""

        for switch in self.switches.values():
            switch.update_knowledge(
                nodes=self.nodes,
                event_id=self.event_id,
            )

        for line in self.electrical_lines.values():
            line.update_state()

        update_energization_state(
            nodes=self.nodes,
            electrical_lines=self.electrical_lines,
            event_id=self.event_id,
        )

    def load_map(self, render_output: bool = True) -> str:
        """Load the known base map, static Areas, switches, and mobile units."""

        self._load_topology()
        self.event_id = 0

        if self.graph is None:
            raise RuntimeError("Graph loading failed.")

        for pixel in self.graph.pixels.values():
            pixel.mark_inspected(self.event_id)

        self._load_mobile_units()
        self._refresh_derived_states()
        self.stage = "Base map loaded"
        return self.render("base_map.png") if render_output else ""

    def load_faults_and_unknown(self, render_output: bool = True) -> str:
        """Load faults first, isolate them, then apply unknown-state rules."""

        self._load_topology()

        if self.graph is None:
            raise RuntimeError("Graph loading failed.")

        self.faults = Fault.load_all(
            json_path=self._path("Faults_v1.json"),
            graph=self.graph,
            nodes=self.nodes,
            electrical_lines=self.electrical_lines,
            areas=self.areas,
        )

        for line in self.electrical_lines.values():
            line.update_state()

        self.fault_isolation_switch_names = Fault.isolate_affected_areas(
            faults=self.faults,
            areas=self.areas,
            switches=self.switches,
        )

        for line in self.electrical_lines.values():
            line.update_state()

        self.unknown_state = UnknownState.from_json(
            self._path("UnknownState_v1.json")
        )
        self.event_id = self.unknown_state.event_id
        self.unknown_safety_switch_names = self.unknown_state.apply(
            graph=self.graph,
            nodes=self.nodes,
            electrical_lines=self.electrical_lines,
            switches=self.switches,
            areas=self.areas,
        )

        self._load_mobile_units()
        for mobile_unit in self.mobile_units.values():
            mobile_unit.inspect_and_refresh(
                event_id=self.event_id,
                nodes=self.nodes,
                electrical_lines=self.electrical_lines,
                switches=self.switches,
            )

        self._refresh_derived_states()
        self.stage = "Faults and unknown state loaded"
        return self.render("fault_unknown_map.png") if render_output else ""

    def _draw_preview_paths(self, ax: Axes) -> None:
        """Draw unconfirmed RC routes over the current state map."""

        for crew_name, route in self.preview_paths.items():
            if crew_name not in self.mobile_units or not route:
                continue

            crew = self.mobile_units[crew_name]
            points = [crew.position, *route]
            x_values = [point[0] for point in points]
            y_values = [point[1] for point in points]
            ax.plot(
                x_values,
                y_values,
                color="#1565C0",
                linewidth=2.0,
                linestyle=(0, (5, 3)),
                alpha=0.85,
                zorder=70,
            )
            ax.scatter(
                [route[-1][0]],
                [route[-1][1]],
                marker="*",
                s=90,
                facecolor="white",
                edgecolor="#1565C0",
                linewidth=1.4,
                zorder=72,
            )

    def _draw_active_paths(self, ax: Axes) -> None:
        """Draw the remaining confirmed route for every Moving RC."""

        for crew in self.mobile_units.values():
            if crew.state != "Moving" or not crew.planned_path:
                continue

            points = [crew.position, *crew.planned_path]
            x_values = [point[0] for point in points]
            y_values = [point[1] for point in points]
            ax.plot(
                x_values,
                y_values,
                color="#0D47A1",
                linewidth=2.2,
                linestyle=(0, (7, 2)),
                alpha=0.90,
                zorder=69,
            )
            ax.scatter(
                [crew.planned_path[-1][0]],
                [crew.planned_path[-1][1]],
                marker="*",
                s=88,
                facecolor="#BBDEFB",
                edgecolor="#0D47A1",
                linewidth=1.4,
                zorder=71,
            )

    def render(self, file_name: str = "dashboard_map.png") -> str:
        """Render the current simulator state to a unique image file."""

        if self.graph is None:
            raise RuntimeError("Load the map before rendering.")

        fig, ax = self.graph.draw()
        ax.set_title("DSR Grid Map")

        if self.unknown_state is not None:
            self.unknown_state.draw_unknown_pixel_background(
                ax=ax,
                graph=self.graph,
            )

        Road.draw_electrical_network_level_one(
            ax=ax,
            electrical_lines=self.electrical_lines,
        )
        for road in sorted(
            self.roads.values(),
            key=lambda item: item.level,
        ):
            road.draw(ax)

        if self.unknown_state is not None:
            self.unknown_state.draw_unknown_regions(ax)

        for line in self.electrical_lines.values():
            line.draw(
                ax=ax,
                event_id=self.event_id,
                show_label=True,
            )

        self._draw_active_paths(ax)
        self._draw_preview_paths(ax)

        for switch in self.switches.values():
            switch.draw(
                ax=ax,
                nodes=self.nodes,
                event_id=self.event_id,
                show_label=True,
            )

        for node in self.nodes.values():
            node.draw(
                ax=ax,
                event_id=self.event_id,
                radius=0.2,
                show_label=True,
            )

        for fault in self.faults.values():
            fault.draw(
                ax=ax,
                event_id=self.event_id,
                show_label=True,
            )

        for mobile_unit in self.mobile_units.values():
            mobile_unit.draw(
                ax=ax,
                show_label=True,
                show_inspection_range=True,
            )

        requested_path = Path(file_name)
        self.render_revision += 1
        output_path = self.output_dir / (
            f"{requested_path.stem}_{self.render_revision:05d}"
            f"{requested_path.suffix or '.png'}"
        )
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return str(output_path)

    def preview_rc_move(
        self,
        crew_name: str,
        waypoints: list[Position],
    ) -> str:
        """Render an unconfirmed ordered route for one RC."""

        self._require_loaded()

        if crew_name not in self.mobile_units:
            raise KeyError(f"Unknown mobile unit: {crew_name}")

        route = self.mobile_units[crew_name].preview_move(waypoints)
        self.preview_paths = {crew_name: route}
        self.stage = f"Previewing {crew_name} Move route"
        return self.render(f"preview_{crew_name}.png")

    def confirm_rc_move(
        self,
        crew_name: str,
        waypoints: list[Position],
    ) -> str:
        """Replace any current RC command with a confirmed Move route."""

        self._require_loaded()

        if crew_name not in self.mobile_units:
            raise KeyError(f"Unknown mobile unit: {crew_name}")

        self.mobile_units[crew_name].move(waypoints)
        self.preview_paths = {}
        self.stage = f"{crew_name} Move command confirmed"
        return self.render(f"move_confirmed_{crew_name}.png")

    def command_rc_stay(self, crew_name: str) -> str:
        """Immediately stop one RC and place it in the Idle state."""

        self._require_loaded()

        if crew_name not in self.mobile_units:
            raise KeyError(f"Unknown mobile unit: {crew_name}")

        self.mobile_units[crew_name].idle()
        self.preview_paths = {}
        self.stage = f"{crew_name} Stay command confirmed"
        return self.render(f"stay_{crew_name}.png")

    def repairable_fault_names(self, crew_name: str) -> list[str]:
        """Return known active faults repairable at the RC's current Pixel."""

        if crew_name not in self.repair_crews:
            return []

        crew = self.repair_crews[crew_name]

        if crew.remaining_resources <= 0 or crew.analysis_locked:
            return []

        return [
            fault.name
            for fault in self.faults.values()
            if (
                fault.active
                and fault.is_known(self.event_id)
                and fault.analyzed
                and fault.position == crew.position
                and fault.remaining_repair_resource > 0
            )
        ]

    def maximum_repair_amount(
        self,
        crew_name: str,
        fault_name: str,
    ) -> int:
        """Return the maximum legal one-step repair amount."""

        if (
            crew_name not in self.repair_crews
            or fault_name not in self.faults
        ):
            return 0

        crew = self.repair_crews[crew_name]
        fault = self.faults[fault_name]
        if crew.analysis_locked or not fault.analyzed:
            return 0
        return min(
            crew.repair_speed,
            crew.remaining_resources,
            fault.remaining_repair_resource,
        )

    def command_rc_repair(
        self,
        crew_name: str,
        fault_name: str,
        resource_amount: int,
    ) -> str:
        """Schedule a validated one-step repair command."""

        self._require_loaded()

        if crew_name not in self.repair_crews:
            raise KeyError(f"Unknown repair crew: {crew_name}")

        self.repair_crews[crew_name].schedule_repair(
            fault_name=fault_name,
            resource_amount=resource_amount,
            faults=self.faults,
            event_id=self.event_id,
        )
        self.preview_paths = {}
        self.stage = (
            f"{crew_name} scheduled {resource_amount} resource units "
            f"for Fault {fault_name}"
        )
        return self.render(f"repair_confirmed_{crew_name}.png")

    @staticmethod
    def _normalize_switch_operations(
        operations: list[tuple[str, str]] | list[list[str]],
    ) -> Tuple[Tuple[str, str], ...]:
        """Validate and normalize one simultaneous switch-operation batch."""

        if not operations:
            raise ValueError("Add at least one switch operation.")

        normalized: list[Tuple[str, str]] = []
        seen_switches: set[str] = set()

        for raw_switch_name, raw_action in operations:
            switch_name = str(raw_switch_name)
            action = str(raw_action).strip().lower()

            action_aliases = {
                "open": "opened",
                "opened": "opened",
                "close": "closed",
                "closed": "closed",
            }
            if action not in action_aliases:
                raise ValueError(
                    f"Unsupported switch action for {switch_name}: {raw_action}"
                )

            if switch_name in seen_switches:
                raise ValueError(
                    f"Switch {switch_name} appears more than once in one batch."
                )

            seen_switches.add(switch_name)
            normalized.append((switch_name, action_aliases[action]))

        return tuple(normalized)

    def _snapshot_switch_states(self) -> Dict[str, Tuple[str, str]]:
        """Return displayed and physical states for every switch."""

        return {
            switch_name: (switch.state, switch.true_state)
            for switch_name, switch in self.switches.items()
        }

    def _restore_switch_states(
        self,
        snapshot: Dict[str, Tuple[str, str]],
    ) -> None:
        """Restore one complete switch-state snapshot."""

        for switch_name, (state, true_state) in snapshot.items():
            switch = self.switches[switch_name]
            switch.state = state
            switch.true_state = true_state

        self._refresh_derived_states()

    def _apply_switch_operations(
        self,
        operations: Tuple[Tuple[str, str], ...],
    ) -> None:
        """Apply all switch targets before evaluating the final topology."""

        for switch_name, action in operations:
            if switch_name not in self.switches:
                raise KeyError(f"Unknown switch: {switch_name}")

            switch = self.switches[switch_name]
            if switch.operation_type != "remote":
                raise ValueError(
                    f"Switch {switch_name} is manual; assign an RC "
                    "operate_switch command instead."
                )
            if switch.state == "unknown":
                raise ValueError(
                    f"Switch {switch_name} is unknown and cannot be operated."
                )

            if action == "closed":
                switch.close()
            else:
                switch.open()

        self._refresh_derived_states()

    def _unsafe_area_names_for_switch(
        self,
        switch: Switch,
    ) -> Tuple[str, ...]:
        """Return unsafe Areas connected through the switch's line side."""

        line_area_name = self.electrical_lines[switch.line_name].area_name
        if line_area_name is None:
            return tuple()

        connected_area_names = self.areas[
            line_area_name
        ].current_connected_area_names()

        unsafe_area_names = [
            area_name
            for area_name in connected_area_names
            if (
                self.areas[area_name].knowledge_state(self.event_id) != "known"
                or self.areas[area_name].has_active_fault
            )
        ]

        return tuple(
            sorted(
                unsafe_area_names,
                key=Area._natural_name_key,
            )
        )

    def _validate_applied_switch_batch(
        self,
        operations: Tuple[Tuple[str, str], ...],
        previously_energized_nodes: Dict[str, bool],
    ) -> None:
        """Validate safety and radiality after all requested actions are applied."""

        for switch_name, action in operations:
            if action != "closed":
                continue

            switch = self.switches[switch_name]
            node = self.nodes[switch.node_name]
            node_is_or_was_energized = (
                previously_energized_nodes.get(node.name, False)
                or node.energized
            )

            if not node_is_or_was_energized:
                continue

            unsafe_area_names = self._unsafe_area_names_for_switch(switch)
            if unsafe_area_names:
                details = []
                for area_name in unsafe_area_names:
                    area = self.areas[area_name]
                    reasons = []
                    if area.knowledge_state(self.event_id) != "known":
                        reasons.append(
                            f"knowledge={area.knowledge_state(self.event_id)}"
                        )
                    if area.has_active_fault:
                        reasons.append("active fault")
                    details.append(
                        f"{area_name} ({', '.join(reasons)})"
                    )

                raise ValueError(
                    f"Closing {switch_name} from energized node "
                    f"{switch.node_name} would connect unsafe Area(s): "
                    + "; ".join(details)
                )

        LegalityChecker(
            nodes=self.nodes,
            electrical_lines=self.electrical_lines,
        ).check_no_energized_cycle()

    def preview_switch_operations(
        self,
        operations: list[tuple[str, str]] | list[list[str]],
    ) -> Dict[str, object]:
        """Validate and render a simultaneous switch batch without committing it."""

        self._require_loaded()
        normalized = self._normalize_switch_operations(operations)
        snapshot = self._snapshot_switch_states()
        previously_energized_nodes = {
            node_name: node.energized
            for node_name, node in self.nodes.items()
        }

        try:
            self._apply_switch_operations(normalized)
            self._validate_applied_switch_batch(
                operations=normalized,
                previously_energized_nodes=previously_energized_nodes,
            )
            self.last_switch_preview = normalized
            self.stage = "Previewing simultaneous switch operations"
            image_path = self.render("switch_batch_preview.png")
            candidate_energized_loads = self.energized_load_rows()
            candidate_switch_rows = self.switch_rows()
        finally:
            self._restore_switch_states(snapshot)

        self.stage = "Switch batch preview ready"
        return {
            "image_path": image_path,
            "operations": normalized,
            "candidate_energized_load_rows": candidate_energized_loads,
            "candidate_switch_rows": candidate_switch_rows,
        }

    def execute_switch_operations(
        self,
        operations: list[tuple[str, str]] | list[list[str]],
    ) -> str:
        """Atomically validate and immediately execute one switch batch."""

        self._require_loaded()
        normalized = self._normalize_switch_operations(operations)
        snapshot = self._snapshot_switch_states()
        previously_energized_nodes = {
            node_name: node.energized
            for node_name, node in self.nodes.items()
        }

        try:
            self._apply_switch_operations(normalized)
            self._validate_applied_switch_batch(
                operations=normalized,
                previously_energized_nodes=previously_energized_nodes,
            )
        except Exception:
            self._restore_switch_states(snapshot)
            raise

        self.last_switch_preview = tuple()
        operation_text = ", ".join(
            f"{switch_name}={action}"
            for switch_name, action in normalized
        )
        self.stage = f"Switch batch executed immediately: {operation_text}"
        return self.render("switch_batch_executed.png")

    def switch_rows(self) -> list[list[str]]:
        """Return current switch endpoint and physical-state information."""

        rows: list[list[str]] = []
        for switch in self.switches.values():
            node_area_name = self.nodes[switch.node_name].area_name
            line_area_name = self.electrical_lines[
                switch.line_name
            ].area_name
            rows.append([
                switch.name,
                switch.node_name,
                switch.line_name,
                switch.state,
                "Yes" if self.nodes[switch.node_name].energized else "No",
                node_area_name or "Transmission",
                line_area_name or "Transmission",
                switch.operation_type.title(),
                f"{switch.operation_time_min:g}",
            ])
        return rows

    def iterate_time_step_frames(
        self,
        frames_per_step: int = 4,
    ):
        """Yield one live simulator snapshot for each animation frame."""

        self._require_loaded()

        if frames_per_step <= 0:
            raise ValueError("frames_per_step must be positive.")

        self.preview_paths = {}
        next_time_step = self.time_step + 1

        for mobile_unit in self.mobile_units.values():
            mobile_unit.prepare_time_step()

        for frame_number in range(1, frames_per_step + 1):
            crew_results = []

            for mobile_unit in self.mobile_units.values():
                crew_results.append(
                    mobile_unit.advance_frame(
                        frame_duration=1.0 / frames_per_step,
                        frame_number=frame_number,
                        frames_per_step=frames_per_step,
                        event_id=self.event_id,
                        nodes=self.nodes,
                        electrical_lines=self.electrical_lines,
                        switches=self.switches,
                        faults=self.faults,
                    )
                )

            self.current_frame = frame_number
            self._refresh_derived_states()

            if frame_number == frames_per_step:
                self.time_step = next_time_step
                self.current_frame = 0
                self.stage = f"Time step {self.time_step} completed"
            else:
                self.stage = (
                    f"Time step {next_time_step}, "
                    f"frame {frame_number}/{frames_per_step}"
                )

            image_path = self.render(
                f"time_step_{next_time_step:03d}_"
                f"frame_{frame_number}.png"
            )
            yield {
                "time_step": next_time_step,
                "frame": frame_number,
                "image_path": image_path,
                "crew_results": crew_results,
            }

    def advance_time_step_animation(
        self,
        frames_per_step: int = 4,
        frame_duration_ms: int = 1000,
    ) -> Dict[str, object]:
        """Advance one time step and save its snapshots as one GIF."""

        if frame_duration_ms <= 0:
            raise ValueError("frame_duration_ms must be positive.")

        frame_results = list(
            self.iterate_time_step_frames(
                frames_per_step=frames_per_step,
            )
        )
        if not frame_results:
            raise RuntimeError("The time step produced no animation frames.")

        frames = []
        for frame_result in frame_results:
            with PILImage.open(frame_result["image_path"]) as image:
                frames.append(image.convert("RGB").copy())

        first_size = frames[0].size
        frames = [
            frame if frame.size == first_size else frame.resize(first_size)
            for frame in frames
        ]

        animation_path = self.output_dir / (
            f"time_step_{frame_results[-1]['time_step']:03d}_animation.gif"
        )
        frames[0].save(
            animation_path,
            save_all=True,
            append_images=frames[1:],
            duration=[frame_duration_ms] * len(frames),
            disposal=2,
            optimize=False,
        )

        return {
            "animation_path": str(animation_path),
            "frames": frame_results,
            "final_frame": frame_results[-1],
        }

    def advance_time_step_frames(
        self,
        frames_per_step: int = 4,
    ) -> list[Dict[str, object]]:
        """Advance one time step and collect all frame snapshots."""

        return list(self.iterate_time_step_frames(frames_per_step))


    def advance_time_step_once(self) -> Dict[str, object]:
        """Advance one complete simulator time step without image generation."""

        self._require_loaded()
        next_time_step = self.time_step + 1
        unit_results = []

        for mobile_unit in self.mobile_units.values():
            mobile_unit.prepare_time_step()

        for mobile_unit in self.mobile_units.values():
            unit_results.append(
                mobile_unit.advance_frame(
                    frame_duration=1.0,
                    frame_number=1,
                    frames_per_step=1,
                    event_id=self.event_id,
                    nodes=self.nodes,
                    electrical_lines=self.electrical_lines,
                    switches=self.switches,
                    faults=self.faults,
                )
            )

        self._refresh_derived_states()
        self.time_step = next_time_step
        self.current_frame = 0
        self.stage = f"Time step {self.time_step} completed"

        return {
            "time_step": self.time_step,
            "unit_results": unit_results,
        }

    def execute_current_step(
        self,
        mobile_commands: Dict[str, Dict[str, object]],
        switch_operations: list[tuple[str, str]] | list[list[str]],
        frames_per_step: int = 1,
        collect_frame_snapshots: bool = False,
    ) -> Dict[str, object]:
        """Atomically validate and execute one complete command step.

        Switch operations are applied instantaneously first. Mobile-unit
        commands then replace or preserve existing commands. When multiple
        frames are requested, all intermediate simulator states are computed
        on the candidate copy and may be returned for browser-side animation.
        """

        self._require_loaded()
        if frames_per_step <= 0:
            raise ValueError("frames_per_step must be positive.")

        candidate = copy.deepcopy(self)
        normalized_switches: Tuple[Tuple[str, str], ...] = tuple()

        if switch_operations:
            normalized_switches = candidate._normalize_switch_operations(
                switch_operations
            )
            previously_energized_nodes = {
                node_name: node.energized
                for node_name, node in candidate.nodes.items()
            }
            candidate._apply_switch_operations(normalized_switches)
            candidate._validate_applied_switch_batch(
                operations=normalized_switches,
                previously_energized_nodes=previously_energized_nodes,
            )

        applied_mobile_commands: Dict[str, str] = {}
        mobile_commands = mobile_commands or {}

        unknown_unit_names = set(mobile_commands).difference(
            candidate.mobile_units
        )
        if unknown_unit_names:
            raise KeyError(
                "Unknown mobile unit(s): "
                + ", ".join(sorted(unknown_unit_names))
            )

        for unit_name, command in mobile_commands.items():
            mobile_unit = candidate.mobile_units[unit_name]
            command_type = str(
                (command or {}).get("type", "continue")
            ).strip().lower()

            if command_type in {"", "continue", "keep"}:
                if (
                    unit_name in candidate.repair_crews
                    and mobile_unit.last_command_type == "repair"
                    and mobile_unit.last_repair_fault_name
                ):
                    fault_name = mobile_unit.last_repair_fault_name
                    fault = candidate.faults.get(fault_name)
                    maximum_amount = (
                        candidate.maximum_repair_amount(unit_name, fault_name)
                        if fault is not None
                        else 0
                    )
                    if (
                        fault is not None
                        and fault.active
                        and fault.is_known(candidate.event_id)
                        and fault.position == mobile_unit.position
                        and maximum_amount > 0
                    ):
                        amount = min(
                            mobile_unit.last_repair_amount,
                            maximum_amount,
                        )
                        candidate.repair_crews[unit_name].schedule_repair(
                            fault_name=fault_name,
                            resource_amount=amount,
                            faults=candidate.faults,
                            event_id=candidate.event_id,
                        )
                        applied_mobile_commands[unit_name] = (
                            f"Continue Repaire Fault {fault_name} with {amount}"
                        )
                    else:
                        mobile_unit.last_command_type = "idle"
                        mobile_unit.last_repair_fault_name = None
                        mobile_unit.last_repair_amount = 0
                        mobile_unit.state = "Idle"
                        applied_mobile_commands[unit_name] = (
                            "Continue; previous repair is complete or unavailable"
                        )
                else:
                    applied_mobile_commands[unit_name] = "Continue"
                continue

            if command_type == "move":
                raw_waypoints = (command or {}).get("waypoints", [])
                waypoints = [tuple(point) for point in raw_waypoints]
                mobile_unit.move(waypoints)
                mobile_unit.set_patrol_context(
                    (command or {}).get("patrol_task_id"),
                    (command or {}).get("patrol_area"),
                    (command or {}).get("patrol_required_lines", ()),
                )
                applied_mobile_commands[unit_name] = (
                    f"Move through {waypoints}"
                )
                continue

            if command_type == "stay":
                mobile_unit.idle()
                applied_mobile_commands[unit_name] = "Stay"
                continue

            if command_type in {"repair", "repaire"}:
                if unit_name not in candidate.repair_crews:
                    raise ValueError(
                        f"{unit_name} is inspection-only and cannot repair."
                    )

                fault_name = str((command or {}).get("fault", "")).strip()
                raw_amount = (command or {}).get("amount")
                if not fault_name:
                    raise ValueError(
                        f"Select one fault for {unit_name} Repaire."
                    )
                if isinstance(raw_amount, bool):
                    raise ValueError("Repair resource must be an integer.")
                try:
                    amount = int(raw_amount)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        "Repair resource must be an integer."
                    ) from error

                candidate.repair_crews[unit_name].schedule_repair(
                    fault_name=fault_name,
                    resource_amount=amount,
                    faults=candidate.faults,
                    event_id=candidate.event_id,
                )
                applied_mobile_commands[unit_name] = (
                    f"Repaire Fault {fault_name} with {amount}"
                )
                continue

            raise ValueError(
                f"Unsupported command for {unit_name}: {command_type}"
            )

        frame_results: list[Dict[str, object]] = []
        frame_snapshots: list["DSRDashboardBackend"] = []
        next_time_step = candidate.time_step + 1

        for mobile_unit in candidate.mobile_units.values():
            mobile_unit.prepare_time_step()

        for frame_number in range(1, frames_per_step + 1):
            unit_results = []
            for mobile_unit in candidate.mobile_units.values():
                unit_results.append(
                    mobile_unit.advance_frame(
                        frame_duration=1.0 / frames_per_step,
                        frame_number=frame_number,
                        frames_per_step=frames_per_step,
                        event_id=candidate.event_id,
                        nodes=candidate.nodes,
                        electrical_lines=candidate.electrical_lines,
                        switches=candidate.switches,
                        faults=candidate.faults,
                    )
                )

            candidate.current_frame = frame_number
            candidate._refresh_derived_states()

            if frame_number == frames_per_step:
                candidate.time_step = next_time_step
                candidate.current_frame = 0
                candidate.stage = f"Time step {candidate.time_step} completed"
            else:
                candidate.stage = (
                    f"Time step {next_time_step}, "
                    f"frame {frame_number}/{frames_per_step}"
                )

            frame_results.append({
                "frame": frame_number,
                "unit_results": unit_results,
            })
            if collect_frame_snapshots:
                frame_snapshots.append(copy.deepcopy(candidate))

        candidate.last_switch_preview = tuple()
        switch_text = [
            f"{switch_name}={action}"
            for switch_name, action in normalized_switches
        ]
        candidate.stage = (
            f"Command step {candidate.time_step} completed"
            + (f"; switches: {', '.join(switch_text)}" if switch_text else "")
        )

        self.__dict__.clear()
        self.__dict__.update(candidate.__dict__)

        return {
            "time_step": self.time_step,
            "switch_operations": normalized_switches,
            "mobile_commands": applied_mobile_commands,
            "unit_results": frame_results[-1]["unit_results"],
            "frame_results": frame_results,
            "frame_snapshots": frame_snapshots,
        }

    def summary_markdown(self) -> str:
        """Return a compact dashboard stage summary."""

        if self.graph is None:
            return "Map data have not been loaded."

        known_faults = sum(
            fault.active and fault.is_known(self.event_id)
            for fault in self.faults.values()
        )
        return (
            f"**Stage:** {self.stage}  \n"
            f"**Completed time steps:** {self.time_step}  \n"
            f"**Event ID:** {self.event_id}  \n"
            f"**Static Areas:** {len(self.areas)}  \n"
            f"**Explicit Roads:** {len(self.roads)}  \n"
            f"**Repair Crews:** {len(self.repair_crews)}  \n"
            f"**Scott units:** {len(self.scotts)}  \n"
            f"**Discovered active faults:** {known_faults}"
        )

    def area_rows(self) -> list[list[str]]:
        """Return static Area information for the dashboard table."""

        rows: list[list[str]] = []
        for area in self.areas.values():
            rows.append([
                area.name,
                ", ".join(area.node_names),
                ", ".join(area.line_names),
                ", ".join(area.boundary_switch_names),
                area.knowledge_state(self.event_id),
                "Yes" if area.is_energizable(self.event_id) else "No",
            ])
        return rows

    def mobile_unit_rows(self) -> list[list[str]]:
        """Return current RC and Scott state for the dashboard table."""

        rows: list[list[str]] = []
        for mobile_unit in self.mobile_units.values():
            command_detail = "-"
            if mobile_unit.state == "Moving":
                command_detail = (
                    f"{len(mobile_unit.planned_path)} Pixels remaining"
                )
            elif mobile_unit.state == "Repairing":
                command_detail = (
                    f"Fault {mobile_unit.repair_fault_name}: "
                    f"{mobile_unit.pending_repair_amount} scheduled"
                )
            elif (
                mobile_unit.unit_type == "RC"
                and mobile_unit.last_command_type == "repair"
                and mobile_unit.last_repair_fault_name
            ):
                command_detail = (
                    f"Continue repair Fault {mobile_unit.last_repair_fault_name}: "
                    f"up to {mobile_unit.last_repair_amount}/step"
                )

            if mobile_unit.unit_type == "RC":
                repair_speed = str(mobile_unit.repair_speed)
                resources = (
                    f"{mobile_unit.remaining_resources}/"
                    f"{mobile_unit.total_resources}"
                )
            else:
                repair_speed = "-"
                resources = "-"

            rows.append([
                mobile_unit.name,
                mobile_unit.unit_type,
                mobile_unit.state,
                str(mobile_unit.position),
                str(mobile_unit.inspection_range),
                str(mobile_unit.move_speed),
                repair_speed,
                resources,
                command_detail,
            ])
        return rows

    def repair_crew_rows(self) -> list[list[str]]:
        """Preserve the previous dashboard method name."""

        return self.mobile_unit_rows()

    @staticmethod
    def _format_load_value(value: float) -> str:
        """Format an integer-valued load without a decimal suffix."""

        return str(int(value)) if float(value).is_integer() else str(value)

    def energized_load_rows(self) -> list[list[str]]:
        """Return only loads whose host nodes are currently energized."""

        rows: list[list[str]] = []
        for node in self.nodes.values():
            if not node.has_load or not node.energized:
                continue

            rows.append([
                node.name,
                self._format_load_value(node.active_power),
                self._format_load_value(node.reactive_power),
                self._format_load_value(node.load_weight),
            ])

        return rows

    def fault_rows(self, known_only: bool = True) -> list[list[str]]:
        """Return fault parameters, hiding unknown faults by default."""

        rows: list[list[str]] = []
        for fault in self.faults.values():
            knowledge = (
                "Known" if fault.is_known(self.event_id) else "Unknown"
            )

            if known_only and knowledge == "Unknown":
                continue

            rows.append([
                fault.name,
                str(fault.position),
                knowledge,
                "Active" if fault.active else "Repaired",
                str(fault.repair_resource) if fault.analyzed else "Hidden",
                (
                    str(fault.remaining_repair_resource)
                    if fault.analyzed
                    else "Hidden"
                ),
                ", ".join(fault.area_names),
            ])
        return rows

    def element_rows(self) -> list[list[str]]:
        """Return road, node, line, switch, and fault dashboard states."""

        rows: list[list[str]] = []

        for road in self.roads.values():
            rows.append([
                "Road",
                road.name,
                road.knowledge_state(self.event_id),
                "-",
                f"Level {road.level}",
                "-",
            ])

        for node in self.nodes.values():
            knowledge = node.knowledge_state(self.event_id)
            electrical = (
                "Unknown"
                if knowledge != "known"
                else ("Energized" if node.energized else "De-energized")
            )
            rows.append([
                "Node",
                node.name,
                knowledge,
                electrical,
                "-",
                node.area_name or "Transmission",
            ])

        for line in self.electrical_lines.values():
            knowledge = line.knowledge_state(self.event_id)
            electrical = (
                "Unknown/partial"
                if knowledge != "known"
                else ("Energized" if line.energized else "De-energized")
            )
            rows.append([
                "Line",
                line.name,
                knowledge,
                electrical,
                line.state,
                line.area_name or "-",
            ])

        for switch in self.switches.values():
            rows.append([
                "Switch",
                switch.name,
                "unknown" if switch.state == "unknown" else "known",
                "-",
                switch.state,
                self.electrical_lines[switch.line_name].area_name or "-",
            ])

        for fault in self.faults.values():
            if not fault.is_known(self.event_id):
                continue
            rows.append([
                "Fault",
                fault.name,
                "known",
                "-",
                "active" if fault.active else "repaired",
                ", ".join(fault.area_names),
            ])

        return rows


# ---------------------------------------------------------------------------
# V23 extensions: Mobile Power Sources and a restorable transmission source.
# ---------------------------------------------------------------------------


def update_energization_state(
    nodes: Dict[str, Node],
    electrical_lines: Dict[str, ElectricalLine],
    event_id: int = 0,
    source_node_names: Optional[set[str]] = None,
    areas: Optional[Dict[str, Area]] = None,
) -> None:
    """Update energization from sources with atomic Area restoration."""

    for node in nodes.values():
        node.energized = False

    for line in electrical_lines.values():
        line.energized = False

    adjacency: Dict[str, list[Tuple[str, ElectricalLine]]] = {
        node_name: []
        for node_name in nodes
    }

    area_ready = {
        name: area.is_inspected(event_id) and not area.has_active_fault
        for name, area in (areas or {}).items()
    }

    def area_is_ready(area_name: Optional[str]) -> bool:
        return area_name is None or area_ready.get(area_name, False)

    for line in electrical_lines.values():
        line_is_usable = (
            line.state == "closed"
            and line.knowledge_state(event_id) == "known"
            and not line.has_active_fault
            and area_is_ready(line.area_name)
        )
        if not line_is_usable:
            continue
        adjacency[line.node_a].append((line.node_b, line))
        adjacency[line.node_b].append((line.node_a, line))

    if source_node_names is None:
        source_node_names = {
            node.name
            for node in nodes.values()
            if (
                (node.is_transmission and getattr(node, "source_enabled", True))
                or getattr(node, "mps_source_active", False)
            )
        }

    queue = [
        node_name
        for node_name in source_node_names
        if (
            node_name in nodes
            and nodes[node_name].knowledge_state(event_id) == "known"
            and not nodes[node_name].has_direct_fault
            and area_is_ready(nodes[node_name].area_name)
        )
    ]
    visited = set(queue)

    while queue:
        current_name = queue.pop(0)
        nodes[current_name].energized = True

        for neighbor_name, line in adjacency[current_name]:
            neighbor = nodes[neighbor_name]
            if (
                neighbor.knowledge_state(event_id) != "known"
                or neighbor.has_direct_fault
                or not area_is_ready(neighbor.area_name)
            ):
                continue
            line.energized = True
            if neighbor_name not in visited:
                visited.add(neighbor_name)
                queue.append(neighbor_name)

    for line in electrical_lines.values():
        line.energized = (
            line.state == "closed"
            and line.knowledge_state(event_id) == "known"
            and nodes[line.node_a].energized
            and nodes[line.node_b].energized
            and not line.has_active_fault
            and area_is_ready(line.area_name)
        )


@dataclass
class TransmissionSource:
    """Represent the transmission grid and its timed restoration fault."""

    name: str
    node_name: str
    restoration_time_step: int
    scenario_fault_enabled: bool = False
    available: bool = True
    active_power: float = 0.0
    reactive_power: float = 0.0
    isolated_switch_names: Tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_json(cls, json_path: Path) -> "TransmissionSource":
        """Load the transmission-source configuration."""

        if not json_path.exists():
            raise FileNotFoundError(
                f"Cannot find transmission-source file: {json_path}"
            )
        with json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)["TransmissionSource"]
        restoration_time_step = int(data.get("restoration_time_step", 0))
        if restoration_time_step < 0:
            raise ValueError("restoration_time_step cannot be negative.")
        return cls(
            name=str(data.get("name", "Transmission Grid")),
            node_name=str(data["node"]),
            restoration_time_step=restoration_time_step,
        )

    @property
    def fault_active(self) -> bool:
        """Return whether the source outage is still active."""

        return self.scenario_fault_enabled and not self.available

    def update_availability(self, time_step: int) -> None:
        """Update source availability from the current completed time step."""

        self.available = (
            not self.scenario_fault_enabled
            or self.restoration_time_step == 0
            or time_step >= self.restoration_time_step
        )
        if not self.available:
            self.active_power = 0.0
            self.reactive_power = 0.0


@dataclass
class MPS(Scott):
    """Represent a mobile power source with movement and supply functions."""

    energy_capacity: float = 0.0
    p_limit: float = 0.0
    s_limit: float = 0.0
    unit_type: str = field(default="MPS", init=False)
    fault_analysis_enabled: bool = field(default=False, init=False)
    body_color: str = field(default="#8E44AD", init=False, repr=False)
    edge_color: str = field(default="#4A235A", init=False, repr=False)
    label_color: str = field(default="#6C3483", init=False, repr=False)
    range_color: str = field(default="#BB8FCE", init=False, repr=False)
    energy: float = field(default=0.0, init=False)
    current_p: float = field(default=0.0, init=False)
    current_q: float = field(default=0.0, init=False)
    last_supply_p: float = field(default=0.0, init=False)
    last_supply_q: float = field(default=0.0, init=False)
    energy_used_this_step: float = field(default=0.0, init=False)
    connected_node_name: Optional[str] = field(default=None, init=False)
    connected_area_names: Tuple[str, ...] = field(default_factory=tuple, init=False)
    connected_load_p: float = field(default=0.0, init=False)
    connected_load_q: float = field(default=0.0, init=False)
    grid_connected: bool = field(default=False, init=False)
    minutes_per_step: float = 5.0

    def __post_init__(self) -> None:
        """Validate MPS parameters and initialize its stored energy."""

        super().__post_init__()
        for parameter_name, value in (
            ("energy", self.energy_capacity),
            ("P_limit", self.p_limit),
            ("S_limit", self.s_limit),
        ):
            if not isinstance(value, (int, float, np.integer, np.floating)):
                raise TypeError(f"{parameter_name} must be numeric.")
            if float(value) < 0:
                raise ValueError(f"{parameter_name} cannot be negative.")
        if self.p_limit <= 0:
            raise ValueError("P_limit must be positive.")
        if self.s_limit <= 0:
            raise ValueError("S_limit must be positive.")
        self.energy_capacity = float(self.energy_capacity)
        self.p_limit = float(self.p_limit)
        self.s_limit = float(self.s_limit)
        self.energy = self.energy_capacity

    @classmethod
    def load_all(
        cls,
        json_path: Path,
        graph: Graph,
        nodes: Dict[str, Node],
        minutes_per_step: float = 5.0,
    ) -> Dict[str, "MPS"]:
        """Load all MPS units and resolve node-based initial positions."""

        if not json_path.exists():
            raise FileNotFoundError(f"Cannot find MPS file: {json_path}")
        with json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        units: Dict[str, MPS] = {}
        for unit_data in data.get("MPSUnits", []):
            if "initial_node" in unit_data:
                node_name = str(unit_data["initial_node"])
                if node_name not in nodes:
                    raise KeyError(
                        f"MPS {unit_data['name']} references unknown node {node_name}."
                    )
                initial_position = nodes[node_name].position
            else:
                initial_position = tuple(unit_data["initial_position"])

            unit = cls(
                name=str(unit_data["name"]),
                initial_position=initial_position,
                inspection_range=int(unit_data["inspection_range"]),
                move_speed=int(unit_data["move_speed"]),
                graph=graph,
                energy_capacity=float(
                    unit_data.get(
                        "energy_kwh",
                        float(unit_data.get("energy", 0.0))
                        * minutes_per_step / 60.0,
                    )
                ),
                p_limit=float(unit_data["P_limit"]),
                s_limit=float(unit_data["S_limit"]),
                minutes_per_step=float(minutes_per_step),
            )
            if unit.name in units:
                raise ValueError(f"Duplicate MPS name: {unit.name}")
            units[unit.name] = unit
        return units

    def disconnect(self) -> None:
        """Disconnect this MPS and enter Idle without changing its position."""

        super().idle()
        self.current_p = 0.0
        self.current_q = 0.0
        self.last_supply_p = 0.0
        self.last_supply_q = 0.0
        self.energy_used_this_step = 0.0

    def idle(self) -> None:
        """Disconnect and enter Idle."""

        self.disconnect()

    def move(self, *coordinates) -> None:
        """Disconnect before replacing the current command with Move."""

        self.current_p = 0.0
        self.current_q = 0.0
        self.last_supply_p = 0.0
        self.last_supply_q = 0.0
        super().move(*coordinates)

    def schedule_supply(
        self,
        active_power: float,
        reactive_power: float,
        nodes: Dict[str, Node],
        event_id: int = 0,
    ) -> None:
        """Schedule a constant P/Q output for the next time step."""

        try:
            active_power = float(active_power)
            reactive_power = float(reactive_power)
        except (TypeError, ValueError) as error:
            raise ValueError("MPS P and Q outputs must be numeric.") from error

        if active_power < 0:
            raise ValueError("MPS active-power output cannot be negative.")
        if active_power > self.p_limit + 1e-9:
            raise ValueError(
                f"MPS {self.name} P={active_power:g} exceeds P_limit={self.p_limit:g}."
            )
        apparent_power = float(np.hypot(active_power, reactive_power))
        if apparent_power > self.s_limit + 1e-9:
            raise ValueError(
                f"MPS {self.name} apparent power {apparent_power:g} "
                f"exceeds S_limit={self.s_limit:g}."
            )

        node_names = [
            node.name
            for node in nodes.values()
            if node.position == self.position
        ]
        if not node_names:
            raise ValueError(
                f"MPS {self.name} can supply only while located at a Node."
            )
        node = nodes[node_names[0]]
        if node.knowledge_state(event_id) != "known" or node.has_direct_fault:
            raise ValueError(
                f"MPS {self.name} cannot supply from unsafe Node {node.name}."
            )

        self.waypoints.clear()
        self.planned_path.clear()
        self.movement_time_credit = 0.0
        self.current_p = active_power
        self.current_q = reactive_power
        self.last_supply_p = active_power
        self.last_supply_q = reactive_power
        self.energy_used_this_step = 0.0
        self.connected_node_name = node.name
        self.state = "Supplying"
        self.last_command_type = "supply"

    def prepare_time_step(self) -> None:
        """Reset per-step MPS energy accounting."""

        super().prepare_time_step()
        self.energy_used_this_step = 0.0

    def advance_frame(
        self,
        frame_duration: float,
        frame_number: int,
        frames_per_step: int,
        event_id: int,
        nodes: Dict[str, Node],
        electrical_lines: Dict[str, ElectricalLine],
        switches: Dict[str, Switch],
        faults: Dict[str, Fault],
    ) -> Dict[str, object]:
        """Advance movement or consume energy while supplying."""

        if self.state != "Supplying":
            return super().advance_frame(
                frame_duration=frame_duration,
                frame_number=frame_number,
                frames_per_step=frames_per_step,
                event_id=event_id,
                nodes=nodes,
                electrical_lines=electrical_lines,
                switches=switches,
                faults=faults,
            )

        previous_energy = self.energy
        cumulative_target = (
            self.current_p
            * self.minutes_per_step
            / 60.0
            * frame_number
            / frames_per_step
        )
        frame_energy = max(0.0, cumulative_target - self.energy_used_this_step)
        frame_energy = min(frame_energy, self.energy)
        self.energy -= frame_energy
        self.energy_used_this_step += frame_energy
        self.inspect_and_refresh(
            event_id=event_id,
            nodes=nodes,
            electrical_lines=electrical_lines,
            switches=switches,
        )

        disconnected = False
        next_step_energy = self.current_p * self.minutes_per_step / 60.0
        if frame_number == frames_per_step and self.energy + 1e-9 < next_step_energy:
            self.disconnect()
            disconnected = True

        return {
            "name": self.name,
            "previous_state": "Supplying",
            "state": self.state,
            "previous_position": self.position,
            "position": self.position,
            "visited_positions": tuple(),
            "discovered_fault_names": tuple(),
            "stopped_for_fault": False,
            "used_repair_resource": 0,
            "previous_energy": previous_energy,
            "energy": self.energy,
            "energy_used": frame_energy,
            "active_power": self.current_p,
            "reactive_power": self.current_q,
            "auto_disconnected": disconnected,
        }


# Save V19 methods before replacing selected backend behavior.
_v19_backend_post_init = DSRDashboardBackend.__post_init__
_v19_load_topology = DSRDashboardBackend._load_topology
_v19_load_map = DSRDashboardBackend.load_map
_v19_load_faults_and_unknown = DSRDashboardBackend.load_faults_and_unknown


def _v23_backend_post_init(self: DSRDashboardBackend) -> None:
    _v19_backend_post_init(self)
    self.minutes_per_step = 5.0
    self.physics_frames_per_step = 4
    self.event_history: list[Dict[str, object]] = []
    self.last_step_events: list[Dict[str, object]] = []
    self.next_event_sequence = 1
    self.mps_units: Dict[str, MPS] = {}
    self.transmission_source: Optional[TransmissionSource] = None
    self.source_fault_switch_names: Tuple[str, ...] = tuple()
    self.power_cluster_rows_cache: list[list[str]] = []


def _v23_mobile_units(self: DSRDashboardBackend) -> Dict[str, RepairCrew]:
    return {**self.repair_crews, **self.scotts, **self.mps_units}


def _v23_load_topology(self: DSRDashboardBackend) -> None:
    _v19_load_topology(self)
    self.transmission_source = TransmissionSource.from_json(
        self._path("TransmissionSource_v1.json")
    )
    if self.transmission_source.node_name not in self.nodes:
        raise KeyError(
            f"Transmission source references unknown Node "
            f"{self.transmission_source.node_name}."
        )
    self.transmission_source.scenario_fault_enabled = False
    self.transmission_source.update_availability(self.time_step)
    self.source_fault_switch_names = tuple()


def _v23_load_mobile_units(self: DSRDashboardBackend) -> None:
    if self.graph is None:
        raise RuntimeError("Load the topology before mobile units.")
    self.repair_crews = RepairCrew.load_all(
        json_path=self._path("RepairCrews_v1.json"),
        graph=self.graph,
    )
    self.scotts = Scott.load_all(
        json_path=self._path("Scotts_v1.json"),
        graph=self.graph,
        nodes=self.nodes,
    )
    self.mps_units = MPS.load_all(
        json_path=self._path("MPS_v1.json"),
        graph=self.graph,
        nodes=self.nodes,
        minutes_per_step=self.minutes_per_step,
    )
    for unit in [*self.repair_crews.values(), *self.scotts.values()]:
        unit.minutes_per_step = self.minutes_per_step
    all_names = [
        *self.repair_crews,
        *self.scotts,
        *self.mps_units,
    ]
    if len(all_names) != len(set(all_names)):
        raise ValueError("RC, Scott, and MPS names must be unique.")


def _v23_node_components(self: DSRDashboardBackend) -> list[set[str]]:
    adjacency: Dict[str, set[str]] = {
        node_name: set()
        for node_name in self.nodes
    }
    for line in self.electrical_lines.values():
        if line.state != "closed":
            continue
        adjacency[line.node_a].add(line.node_b)
        adjacency[line.node_b].add(line.node_a)

    components: list[set[str]] = []
    visited: set[str] = set()
    for start in self.nodes:
        if start in visited:
            continue
        component: set[str] = set()
        queue = [start]
        visited.add(start)
        while queue:
            current = queue.pop(0)
            component.add(current)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return components


def _v23_component_details(
    self: DSRDashboardBackend,
    node_names: set[str],
) -> Dict[str, object]:
    line_names = {
        line.name
        for line in self.electrical_lines.values()
        if (
            line.state == "closed"
            and line.node_a in node_names
            and line.node_b in node_names
        )
    }
    area_names = {
        area_name
        for area_name in (
            [self.nodes[name].area_name for name in node_names]
            + [self.electrical_lines[name].area_name for name in line_names]
        )
        if area_name is not None
    }
    load_p = sum(self.nodes[name].active_power for name in node_names)
    load_q = sum(self.nodes[name].reactive_power for name in node_names)
    unsafe_items: list[str] = []
    for node_name in node_names:
        node = self.nodes[node_name]
        if node.knowledge_state(self.event_id) != "known":
            unsafe_items.append(f"Node {node_name} is unknown")
        if node.has_direct_fault:
            unsafe_items.append(f"Node {node_name} has an active fault")
    for line_name in line_names:
        line = self.electrical_lines[line_name]
        if line.knowledge_state(self.event_id) != "known":
            unsafe_items.append(f"Line {line_name} is unknown/partial")
        if line.has_active_fault:
            unsafe_items.append(f"Line {line_name} has an active fault")
    return {
        "nodes": tuple(sorted(node_names, key=Area._natural_name_key)),
        "lines": tuple(sorted(line_names, key=Area._natural_name_key)),
        "areas": tuple(sorted(area_names, key=Area._natural_name_key)),
        "load_p": float(load_p),
        "load_q": float(load_q),
        "safe": not unsafe_items,
        "unsafe_items": tuple(unsafe_items),
    }


def _v23_node_at_position(
    self: DSRDashboardBackend,
    position: Position,
) -> Optional[str]:
    for node in self.nodes.values():
        if node.position == position:
            return node.name
    return None


def _v23_evaluate_power_balance(
    self: DSRDashboardBackend,
    strict: bool = False,
) -> set[str]:
    """Evaluate connected source/load clusters and return active source nodes."""

    tolerance = 1e-6
    if self.transmission_source is None:
        return set()
    self.transmission_source.update_availability(self.time_step)
    self.transmission_source.active_power = 0.0
    self.transmission_source.reactive_power = 0.0
    active_source_nodes: set[str] = set()
    cluster_rows: list[list[str]] = []

    for mps in self.mps_units.values():
        mps.connected_node_name = self._node_at_position(mps.position)
        mps.connected_area_names = tuple()
        mps.connected_load_p = 0.0
        mps.connected_load_q = 0.0
        mps.grid_connected = False

    for component_nodes in self._node_components():
        details = self._component_details(component_nodes)
        supplying_mps = [
            mps
            for mps in self.mps_units.values()
            if (
                mps.state == "Supplying"
                and mps.connected_node_name in component_nodes
                and mps.current_p > -tolerance
            )
        ]
        grid_connected = (
            self.transmission_source.available
            and self.transmission_source.node_name in component_nodes
        )
        mps_p = sum(mps.current_p for mps in supplying_mps)
        mps_q = sum(mps.current_q for mps in supplying_mps)
        load_p = float(details["load_p"])
        load_q = float(details["load_q"])
        safe = bool(details["safe"])

        for mps in self.mps_units.values():
            if mps.connected_node_name in component_nodes:
                mps.connected_area_names = details["areas"]
                mps.connected_load_p = load_p
                mps.connected_load_q = load_q
                mps.grid_connected = grid_connected

        valid_supply = False
        grid_p = 0.0
        grid_q = 0.0
        error_message = ""

        if (grid_connected or supplying_mps) and not safe:
            error_message = (
                "Power cannot be applied to an unsafe electrical cluster: "
                + "; ".join(details["unsafe_items"][:6])
            )
        elif grid_connected:
            grid_p = load_p - mps_p
            grid_q = load_q - mps_q
            if grid_p < -tolerance:
                error_message = (
                    f"MPS active output {mps_p:g} kW exceeds connected "
                    f"load {load_p:g} kW while the transmission grid is connected."
                )
            else:
                valid_supply = True
                grid_p = max(0.0, grid_p)
        elif supplying_mps:
            if (
                abs(mps_p - load_p) > tolerance
                or abs(mps_q - load_q) > tolerance
            ):
                error_message = (
                    f"Islanded cluster requires P={load_p:g} kW and "
                    f"Q={load_q:g} kVar, but its MPS units provide "
                    f"P={mps_p:g} kW and Q={mps_q:g} kVar."
                )
            else:
                valid_supply = True

        if strict and error_message:
            raise ValueError(error_message)

        if valid_supply:
            if grid_connected:
                active_source_nodes.add(self.transmission_source.node_name)
                self.transmission_source.active_power += grid_p
                self.transmission_source.reactive_power += grid_q
            for mps in supplying_mps:
                active_source_nodes.add(mps.connected_node_name)

        cluster_rows.append([
            ", ".join(details["areas"]) or "Transmission only",
            ", ".join(details["nodes"]),
            self._format_load_value(load_p),
            self._format_load_value(load_q),
            self._format_load_value(mps_p),
            self._format_load_value(mps_q),
            self._format_load_value(grid_p if valid_supply else 0.0),
            self._format_load_value(grid_q if valid_supply else 0.0),
            "Energized" if valid_supply else "De-energized",
        ])

    self.power_cluster_rows_cache = cluster_rows
    return active_source_nodes


def _v23_refresh_derived_states(self: DSRDashboardBackend) -> None:
    for switch in self.switches.values():
        switch.update_knowledge(nodes=self.nodes, event_id=self.event_id)
    for line in self.electrical_lines.values():
        line.update_state()

    source_nodes = self._evaluate_power_balance(strict=False)
    for node in self.nodes.values():
        node.source_enabled = (
            self.transmission_source is not None
            and node.name == self.transmission_source.node_name
            and self.transmission_source.available
            and node.name in source_nodes
        )
        node.mps_source_active = node.name in source_nodes and not node.is_transmission

    update_energization_state(
        nodes=self.nodes,
        electrical_lines=self.electrical_lines,
        event_id=self.event_id,
        source_node_names=source_nodes,
        areas=self.areas,
    )


def _v23_isolate_source_fault(self: DSRDashboardBackend) -> None:
    if self.transmission_source is None:
        return
    opened: list[str] = []
    source_node = self.transmission_source.node_name
    for switch in self.switches.values():
        if switch.node_name == source_node:
            switch.open()
            opened.append(switch.name)
    self.source_fault_switch_names = tuple(
        sorted(opened, key=Area._natural_name_key)
    )
    self.transmission_source.isolated_switch_names = self.source_fault_switch_names


def _v23_load_map(self: DSRDashboardBackend, render_output: bool = True) -> str:
    result = _v19_load_map(self, render_output=False)
    if self.transmission_source is not None:
        self.transmission_source.scenario_fault_enabled = False
        self.transmission_source.update_availability(self.time_step)
    self._refresh_derived_states()
    self.stage = "Base map loaded"
    return self.render("base_map.png") if render_output else result


def _v23_load_faults_and_unknown(
    self: DSRDashboardBackend,
    render_output: bool = True,
) -> str:
    result = _v19_load_faults_and_unknown(self, render_output=False)
    if self.transmission_source is None:
        raise RuntimeError("Transmission source configuration was not loaded.")
    self.transmission_source.scenario_fault_enabled = True
    self.transmission_source.update_availability(self.time_step)
    if self.transmission_source.fault_active:
        self._isolate_source_fault()
    else:
        self.source_fault_switch_names = ()
        self.transmission_source.isolated_switch_names = ()
    self._refresh_derived_states()
    self.stage = (
        "Faults, unknown state, and transmission-source outage loaded"
    )
    return self.render("fault_unknown_source_map.png") if render_output else result


def _v23_validate_restoration_topology(self: DSRDashboardBackend) -> None:
    """Block the step into restoration until the source network is safe."""

    source = self.transmission_source
    if (
        source is None
        or not source.scenario_fault_enabled
        or source.restoration_time_step == 0
        or self.time_step + 1 != source.restoration_time_step
    ):
        return

    source_component = next(
        (
            component
            for component in self._node_components()
            if source.node_name in component
        ),
        {source.node_name},
    )
    details = self._component_details(source_component)
    if not details["safe"]:
        raise ValueError(
            f"The transmission grid restores at time step "
            f"{source.restoration_time_step}, but its connected network is unsafe: "
            + "; ".join(details["unsafe_items"][:8])
        )

    parent = {name: name for name in source_component}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    for line_name in details["lines"]:
        line = self.electrical_lines[line_name]
        root_a = find(line.node_a)
        root_b = find(line.node_b)
        if root_a == root_b:
            raise ValueError(
                f"The transmission grid restores at time step "
                f"{source.restoration_time_step}, but its connected network "
                f"contains a loop through line {line.name}."
            )
        parent[root_b] = root_a


def _v23_mps_supply_context(
    self: DSRDashboardBackend,
    mps_name: str,
) -> Dict[str, object]:
    if mps_name not in self.mps_units:
        raise KeyError(f"Unknown MPS: {mps_name}")
    mps = self.mps_units[mps_name]
    node_name = self._node_at_position(mps.position)
    if node_name is None:
        return {
            "available": False,
            "node": None,
            "nodes": [],
            "areas": [],
            "load_p": 0.0,
            "load_q": 0.0,
            "weighted_load": 0.0,
            "grid_connected": False,
            "message": "Supply is available only at a Node.",
        }
    component = next(
        component
        for component in self._node_components()
        if node_name in component
    )
    details = self._component_details(component)
    source = self.transmission_source
    grid_connected = bool(
        source is not None
        and source.available
        and source.node_name in component
    )
    return {
        "available": bool(details["safe"]),
        "node": node_name,
        "nodes": list(details["nodes"]),
        "areas": list(details["areas"]),
        "load_p": details["load_p"],
        "load_q": details["load_q"],
        "weighted_load": float(sum(
            self.nodes[name].active_power * self.nodes[name].load_weight
            for name in details["nodes"]
            if self.nodes[name].has_load
        )),
        "grid_connected": grid_connected,
        "message": (
            "Connected cluster is known and fault-free."
            if details["safe"]
            else "; ".join(details["unsafe_items"][:5])
        ),
    }


def _v23_clock_payload(
    self: DSRDashboardBackend,
    time_step: Optional[int] = None,
    frame: Optional[int] = None,
) -> Dict[str, object]:
    """Return the unified step/frame/minute clock."""

    step = self.time_step if time_step is None else int(time_step)
    current_frame = self.current_frame if frame is None else int(frame)
    time_min = self.minutes_per_step * (
        step + current_frame / self.physics_frames_per_step
    )
    return {
        "time_step": step,
        "frame": current_frame,
        "time_min": round(time_min, 6),
        "minutes_per_step": self.minutes_per_step,
        "physics_frames_per_step": self.physics_frames_per_step,
        "minutes_per_frame": (
            self.minutes_per_step / self.physics_frames_per_step
        ),
    }


def _v23_emit_event(
    self: DSRDashboardBackend,
    event_type: str,
    *,
    time_step: int,
    frame: int,
    unit: Optional[str] = None,
    position: Optional[Position] = None,
    fault: Optional[str] = None,
    switch: Optional[str] = None,
    load: Optional[str] = None,
    details: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Append one public event to this candidate environment."""

    clock = self.clock_payload(time_step=time_step, frame=frame)
    event: Dict[str, object] = {
        "sequence": self.next_event_sequence,
        "event_type": event_type,
        "time_step": clock["time_step"],
        "frame": clock["frame"],
        "time_min": clock["time_min"],
        "unit": unit,
        "position": list(position) if position is not None else None,
        "fault": fault,
        "switch": switch,
        "load": load,
        "details": details or {},
    }
    self.next_event_sequence += 1
    self.event_history.append(event)
    self.last_step_events.append(event)
    return event


def _v23_runtime_snapshot(self: DSRDashboardBackend) -> Dict[str, object]:
    """Capture public runtime values used to derive structured events."""

    return {
        "faults": {
            name: {
                "known": fault.is_known(self.event_id),
                "analyzed": fault.analyzed,
                "active": fault.active,
                "remaining": (
                    fault.remaining_repair_resource if fault.analyzed else None
                ),
            }
            for name, fault in self.faults.items()
        },
        "units": {
            name: {
                "state": unit.state,
                "position": unit.position,
                "switch": getattr(unit, "switch_operation_name", None),
                "target": getattr(unit, "switch_target_state", None),
            }
            for name, unit in self.mobile_units.items()
        },
        "switches": {
            name: switch.true_state
            for name, switch in self.switches.items()
        },
        "source_available": bool(
            self.transmission_source is not None
            and self.transmission_source.available
        ),
        "loads": {
            name: node.energized
            for name, node in self.nodes.items()
            if node.has_load
        },
    }


def _v23_emit_transition_events(
    self: DSRDashboardBackend,
    before: Dict[str, object],
    *,
    time_step: int,
    frame: int,
    unit_results: list[Dict[str, object]],
) -> Dict[str, object]:
    """Convert one physics-frame state transition into public events."""

    for result in unit_results:
        unit_name = str(result["name"])
        position = tuple(result["position"])
        for fault_name in result.get("discovered_fault_names", ()):
            fault_obj = self.faults.get(str(fault_name))
            if fault_obj is not None and fault_obj.is_known(self.event_id):
                self._emit_event(
                    "FAULT_DISCOVERED",
                    time_step=time_step,
                    frame=frame,
                    unit=unit_name,
                    position=position,
                    fault=fault_obj.name,
                )

        previous_unit = before["units"][unit_name]
        current_unit = self.mobile_units[unit_name]
        if (
            previous_unit["state"] != "Analyzing"
            and current_unit.state == "Analyzing"
        ):
            for fault_name in current_unit.analysis_fault_names:
                self._emit_event(
                    "FAULT_ANALYSIS_STARTED",
                    time_step=time_step,
                    frame=frame,
                    unit=unit_name,
                    position=current_unit.position,
                    fault=fault_name,
                )
        if (
            previous_unit["state"] == "Moving"
            and current_unit.state != "Moving"
        ):
            self._emit_event(
                "MOBILE_ARRIVED",
                time_step=time_step,
                frame=frame,
                unit=unit_name,
                position=current_unit.position,
            )

    for unit_name, current_unit in self.mps_units.items():
        previous_unit = before["units"][unit_name]
        if (
            previous_unit["state"] != "Supplying"
            and current_unit.state == "Supplying"
        ):
            self._emit_event(
                "MPS_CONNECTED",
                time_step=time_step,
                frame=frame,
                unit=unit_name,
                position=current_unit.position,
                details={
                    "p_kw": current_unit.current_p,
                    "q_kvar": current_unit.current_q,
                },
            )
        elif (
            previous_unit["state"] == "Supplying"
            and current_unit.state != "Supplying"
        ):
            self._emit_event(
                "MPS_DISCONNECTED",
                time_step=time_step,
                frame=frame,
                unit=unit_name,
                position=current_unit.position,
            )

    for fault_name, previous in before["faults"].items():
        fault_obj = self.faults[fault_name]
        if not fault_obj.is_known(self.event_id):
            continue
        if not previous["analyzed"] and fault_obj.analyzed:
            self._emit_event(
                "FAULT_ANALYSIS_COMPLETED",
                time_step=time_step,
                frame=frame,
                position=fault_obj.position,
                fault=fault_name,
                details={"repair_resource": fault_obj.repair_resource},
            )
        current_remaining = (
            fault_obj.remaining_repair_resource if fault_obj.analyzed else None
        )
        if (
            previous["remaining"] is not None
            and current_remaining is not None
            and current_remaining < previous["remaining"]
        ):
            self._emit_event(
                "REPAIR_PROGRESS",
                time_step=time_step,
                frame=frame,
                position=fault_obj.position,
                fault=fault_name,
                details={
                    "resource_used": previous["remaining"] - current_remaining,
                    "remaining_resource": current_remaining,
                },
            )
        if previous["active"] and not fault_obj.active:
            self._emit_event(
                "REPAIR_COMPLETED",
                time_step=time_step,
                frame=frame,
                position=fault_obj.position,
                fault=fault_name,
            )

    for switch_name, previous_state in before["switches"].items():
        switch_obj = self.switches[switch_name]
        if (
            switch_obj.operation_type == "manual"
            and switch_obj.true_state != previous_state
        ):
            self._emit_event(
                "MS_OPERATION_COMPLETED",
                time_step=time_step,
                frame=frame,
                position=self.nodes[switch_obj.node_name].position,
                switch=switch_name,
                details={"target_state": switch_obj.true_state},
            )

    if not before["source_available"] and (
        self.transmission_source is not None
        and self.transmission_source.available
    ):
        self._emit_event(
            "SOURCE_RESTORED",
            time_step=time_step,
            frame=frame,
            position=self.nodes[self.transmission_source.node_name].position,
            details={"source": self.transmission_source.name},
        )

    for node_name, was_energized in before["loads"].items():
        is_energized = self.nodes[node_name].energized
        if is_energized != was_energized:
            self._emit_event(
                "LOAD_RESTORED" if is_energized else "LOAD_LOST",
                time_step=time_step,
                frame=frame,
                position=self.nodes[node_name].position,
                load=node_name,
                details={
                    "p_kw": self.nodes[node_name].active_power,
                    "weight": self.nodes[node_name].load_weight,
                },
            )
    return self.runtime_snapshot()


def _v23_execute_current_step(
    self: DSRDashboardBackend,
    mobile_commands: Dict[str, Dict[str, object]],
    switch_operations: list[tuple[str, str]] | list[list[str]],
    frames_per_step: int = 1,
    collect_frame_snapshots: bool = False,
) -> Dict[str, object]:
    """Execute Switch commands first, then RC/Scott/MPS commands atomically."""

    self._require_loaded()
    if frames_per_step <= 0:
        raise ValueError("frames_per_step must be positive.")

    candidate = copy.deepcopy(self)
    candidate.last_step_events = []
    step_start = candidate.time_step
    step_start_snapshot = candidate.runtime_snapshot()
    normalized_switches: Tuple[Tuple[str, str], ...] = tuple()
    if switch_operations:
        normalized_switches = candidate._normalize_switch_operations(
            switch_operations
        )
        previously_energized_nodes = {
            name: node.energized
            for name, node in candidate.nodes.items()
        }
        previous_switch_states = {
            name: candidate.switches[name].true_state
            for name, _ in normalized_switches
        }
        candidate._apply_switch_operations(normalized_switches)
        candidate._validate_applied_switch_batch(
            operations=normalized_switches,
            previously_energized_nodes=previously_energized_nodes,
        )
        for switch_name, target_state in normalized_switches:
            if previous_switch_states[switch_name] != target_state:
                switch_obj = candidate.switches[switch_name]
                candidate._emit_event(
                    "REMOTE_SWITCH_OPERATED",
                    time_step=step_start,
                    frame=0,
                    position=candidate.nodes[switch_obj.node_name].position,
                    switch=switch_name,
                    details={"target_state": target_state},
                )

    # The step from k-1 to k is blocked until the source-connected topology is safe.
    candidate._validate_restoration_topology()

    mobile_commands = mobile_commands or {}
    unknown_names = set(mobile_commands).difference(candidate.mobile_units)
    if unknown_names:
        raise KeyError(
            "Unknown mobile unit(s): " + ", ".join(sorted(unknown_names))
        )

    applied_commands: Dict[str, str] = {}
    continued_repair_units: set[str] = set()
    explicit_repair_units: set[str] = set()
    for unit in candidate.mobile_units.values():
        if (
            unit.fault_analysis_enabled
            and not unit.analysis_locked
            and unit.state != "OperatingSwitch"
        ):
            arrival_faults = unit._known_unanalyzed_fault_names_at_position(
                faults=candidate.faults,
                event_id=candidate.event_id,
            )
            unit.begin_fault_analysis(arrival_faults, candidate.faults)

    for unit_name, unit in candidate.mobile_units.items():
        command = mobile_commands.get(unit_name, {"type": "continue"}) or {}
        command_type = str(command.get("type", "continue")).strip().lower()

        if unit.state == "OperatingSwitch":
            if command_type not in {"", "continue", "keep"}:
                raise ValueError(
                    f"{unit_name} is operating {unit.switch_operation_name} "
                    "and is locked until completion."
                )
            applied_commands[unit_name] = (
                f"Continue operating {unit.switch_operation_name}"
            )
            continue

        if unit.analysis_locked:
            if command_type not in {"", "continue", "keep"}:
                raise ValueError(
                    f"{unit_name} is locked for one automatic fault-analysis "
                    "step and cannot accept commands."
                )
            applied_commands[unit_name] = (
                "Automatic fault analysis: "
                + ", ".join(unit.analysis_fault_names)
            )
            continue

        if command_type in {"", "continue", "keep"}:
            if unit_name in candidate.repair_crews:
                if unit.last_command_type == "repair" and unit.last_repair_fault_name:
                    fault_name = unit.last_repair_fault_name
                    fault = candidate.faults.get(fault_name)
                    maximum = (
                        candidate.maximum_repair_amount(unit_name, fault_name)
                        if fault is not None
                        else 0
                    )
                    if (
                        fault is not None
                        and fault.active
                        and fault.is_known(candidate.event_id)
                        and fault.position == unit.position
                        and maximum > 0
                    ):
                        amount = min(unit.last_repair_amount, maximum)
                        unit.schedule_repair(
                            fault_name=fault_name,
                            resource_amount=amount,
                            faults=candidate.faults,
                            event_id=candidate.event_id,
                        )
                        continued_repair_units.add(unit_name)
                        applied_commands[unit_name] = (
                            f"Continue Repaire Fault {fault_name} with {amount}"
                        )
                    else:
                        unit.idle()
                        applied_commands[unit_name] = (
                            "Continue; previous repair is complete or unavailable"
                        )
                else:
                    applied_commands[unit_name] = "Continue"
            elif unit_name in candidate.mps_units:
                if unit.state == "Supplying":
                    applied_commands[unit_name] = (
                        f"Continue Supply P={unit.current_p:g}, Q={unit.current_q:g}"
                    )
                else:
                    applied_commands[unit_name] = "Continue"
            else:
                applied_commands[unit_name] = "Continue"
            continue

        if command_type == "move":
            waypoints = [tuple(point) for point in command.get("waypoints", [])]
            unit.move(waypoints)
            unit.set_patrol_context(
                command.get("patrol_task_id"),
                command.get("patrol_area"),
                command.get("patrol_required_lines", ()),
            )
            applied_commands[unit_name] = f"Move through {waypoints}"
            continue

        if command_type == "stay":
            unit.idle()
            applied_commands[unit_name] = "Stay"
            continue

        if command_type in {"repair", "repaire"}:
            if unit_name not in candidate.repair_crews:
                raise ValueError(f"{unit_name} cannot receive a Repaire command.")
            fault_name = str(command.get("fault", "")).strip()
            if not fault_name:
                raise ValueError(f"Select one fault for {unit_name} Repaire.")
            try:
                amount = int(command.get("amount"))
            except (TypeError, ValueError) as error:
                raise ValueError("Repair resource must be an integer.") from error
            unit.schedule_repair(
                fault_name=fault_name,
                resource_amount=amount,
                faults=candidate.faults,
                event_id=candidate.event_id,
            )
            explicit_repair_units.add(unit_name)
            applied_commands[unit_name] = (
                f"Repaire Fault {fault_name} with {amount}"
            )
            continue

        if command_type == "supply":
            if unit_name not in candidate.mps_units:
                raise ValueError(f"{unit_name} is not an MPS.")
            unit.schedule_supply(
                active_power=command.get("p"),
                reactive_power=command.get("q"),
                nodes=candidate.nodes,
                event_id=candidate.event_id,
            )
            applied_commands[unit_name] = (
                f"Supply P={unit.current_p:g} kW, Q={unit.current_q:g} kVar"
            )
            continue

        if command_type == "operate_switch":
            if unit_name not in candidate.repair_crews:
                raise ValueError(
                    f"{unit_name} cannot physically operate a manual switch."
                )
            switch_name = str(command.get("switch", "")).strip()
            target_state = str(command.get("target_state", "")).strip().lower()
            if switch_name not in candidate.switches:
                raise KeyError(f"Unknown switch: {switch_name}")
            switch_obj = candidate.switches[switch_name]
            unit.schedule_switch_operation(
                switch=switch_obj,
                target_state=target_state,
                nodes=candidate.nodes,
            )
            candidate._emit_event(
                "MS_OPERATION_STARTED",
                time_step=step_start,
                frame=0,
                unit=unit_name,
                position=unit.position,
                switch=switch_name,
                details={
                    "target_state": target_state,
                    "operation_time_min": switch_obj.operation_time_min,
                },
            )
            applied_commands[unit_name] = (
                f"Operate {switch_name} to {target_state}"
            )
            continue

        raise ValueError(
            f"Unsupported command for {unit_name}: {command_type}"
        )

    manual_assignments: Dict[str, list[str]] = {}
    for crew in candidate.repair_crews.values():
        if crew.state == "OperatingSwitch" and crew.switch_operation_name:
            manual_assignments.setdefault(
                crew.switch_operation_name, []
            ).append(crew.name)
    duplicate_manual = {
        name: crews
        for name, crews in manual_assignments.items()
        if len(crews) > 1
    }
    if duplicate_manual:
        switch_name, crews = next(iter(duplicate_manual.items()))
        raise ValueError(
            f"Manual switch {switch_name} cannot be operated by multiple RCs: "
            + ", ".join(crews)
        )

    repair_assignments: Dict[str, list[RepairCrew]] = {}
    for crew in candidate.repair_crews.values():
        if crew.state == "Repairing" and crew.repair_fault_name:
            repair_assignments.setdefault(
                crew.repair_fault_name, []
            ).append(crew)
    for fault_name, crews in repair_assignments.items():
        remaining = candidate.faults[fault_name].remaining_repair_resource
        explicit_requested = sum(
            crew.pending_repair_amount
            for crew in crews if crew.name in explicit_repair_units
        )
        if explicit_requested > remaining:
            raise ValueError(
                f"Aggregate repair request for Fault {fault_name} is "
                f"{explicit_requested}, exceeding remaining requirement {remaining}."
            )
        available_for_continue = remaining - explicit_requested
        for crew in sorted(crews, key=lambda item: item.name):
            if crew.name not in continued_repair_units:
                continue
            allowed = min(crew.pending_repair_amount, available_for_continue)
            if allowed <= 0:
                crew.idle()
                continue
            crew.pending_repair_amount = allowed
            crew.last_repair_amount = allowed
            available_for_continue -= allowed

        requested = sum(
            crew.pending_repair_amount
            for crew in candidate.repair_crews.values()
            if crew.state == "Repairing" and crew.repair_fault_name == fault_name
        )
        if requested > remaining:
            raise ValueError(
                f"Aggregate repair request for Fault {fault_name} is "
                f"{requested}, exceeding remaining requirement {remaining}."
            )

    auto_disconnected: list[str] = []
    for mps in candidate.mps_units.values():
        required_energy = (
            mps.current_p * candidate.minutes_per_step / 60.0
        )
        if mps.state == "Supplying" and mps.energy + 1e-9 < required_energy:
            mps.disconnect()
            auto_disconnected.append(mps.name)
            applied_commands[mps.name] = (
                "Auto-disconnected because stored energy cannot support one step"
            )

    candidate._refresh_derived_states()
    candidate._evaluate_power_balance(strict=True)
    candidate._refresh_derived_states()
    candidate._emit_transition_events(
        step_start_snapshot,
        time_step=step_start,
        frame=0,
        unit_results=[],
    )
    LegalityChecker(
        nodes=candidate.nodes,
        electrical_lines=candidate.electrical_lines,
    ).check_no_energized_cycle()

    next_time_step = candidate.time_step + 1
    frame_results: list[Dict[str, object]] = []
    frame_snapshots: list[DSRDashboardBackend] = []
    for unit in candidate.mobile_units.values():
        unit.prepare_time_step()

    for frame_number in range(1, frames_per_step + 1):
        before_frame = candidate.runtime_snapshot()
        unit_results = []
        for unit in candidate.mobile_units.values():
            unit_results.append(
                unit.advance_frame(
                    frame_duration=1.0 / frames_per_step,
                    frame_number=frame_number,
                    frames_per_step=frames_per_step,
                    event_id=candidate.event_id,
                    nodes=candidate.nodes,
                    electrical_lines=candidate.electrical_lines,
                    switches=candidate.switches,
                    faults=candidate.faults,
                )
            )
        candidate.current_frame = frame_number
        if frame_number == frames_per_step:
            candidate.time_step = next_time_step
            candidate.current_frame = 0
        candidate._refresh_derived_states()
        candidate._emit_transition_events(
            before_frame,
            time_step=step_start,
            frame=frame_number,
            unit_results=unit_results,
        )
        candidate.stage = (
            f"Time step {candidate.time_step} completed"
            if frame_number == frames_per_step
            else f"Time step {next_time_step}, frame {frame_number}/{frames_per_step}"
        )
        frame_results.append({
            "frame": frame_number,
            "unit_results": unit_results,
        })
        if collect_frame_snapshots:
            frame_snapshots.append(copy.deepcopy(candidate))

    candidate._validate_restoration_topology()
    candidate._evaluate_power_balance(strict=True)
    candidate._refresh_derived_states()
    LegalityChecker(
        nodes=candidate.nodes,
        electrical_lines=candidate.electrical_lines,
    ).check_no_energized_cycle()

    actual_repair_by_unit: Dict[str, int] = {}
    for frame_result in frame_results:
        for unit_result in frame_result["unit_results"]:
            used = int(unit_result.get("used_repair_resource", 0))
            if used:
                name = str(unit_result["name"])
                actual_repair_by_unit[name] = (
                    actual_repair_by_unit.get(name, 0) + used
                )

    candidate.last_switch_preview = tuple()
    candidate.stage = f"Command step {candidate.time_step} completed"
    self.__dict__.clear()
    self.__dict__.update(candidate.__dict__)
    return {
        "time_step": self.time_step,
        "switch_operations": normalized_switches,
        "mobile_commands": applied_commands,
        "auto_disconnected_mps": auto_disconnected,
        "unit_results": frame_results[-1]["unit_results"],
        "frame_results": frame_results,
        "frame_snapshots": frame_snapshots,
        "actual_repair_resource": actual_repair_by_unit,
        "events": list(self.last_step_events),
    }


def _v23_source_rows(self: DSRDashboardBackend) -> list[list[str]]:
    source = self.transmission_source
    if source is None:
        return []
    return [[
        source.name,
        source.node_name,
        "Available" if source.available else "Outage",
        str(source.restoration_time_step),
        f"{source.restoration_time_step * self.minutes_per_step:g}",
        self._format_load_value(source.active_power),
        self._format_load_value(source.reactive_power),
        ", ".join(source.isolated_switch_names) or "-",
    ]]


def _v23_mps_rows(self: DSRDashboardBackend) -> list[list[str]]:
    rows = []
    for mps in self.mps_units.values():
        rows.append([
            mps.name,
            mps.state,
            str(mps.position),
            self._format_load_value(mps.energy),
            self._format_load_value(mps.p_limit),
            self._format_load_value(mps.s_limit),
            self._format_load_value(mps.current_p),
            self._format_load_value(mps.current_q),
            ", ".join(mps.connected_area_names) or "-",
        ])
    return rows


DSRDashboardBackend.__post_init__ = _v23_backend_post_init
DSRDashboardBackend.mobile_units = property(_v23_mobile_units)
DSRDashboardBackend._load_topology = _v23_load_topology
DSRDashboardBackend._load_mobile_units = _v23_load_mobile_units
DSRDashboardBackend._node_components = _v23_node_components
DSRDashboardBackend._component_details = _v23_component_details
DSRDashboardBackend._node_at_position = _v23_node_at_position
DSRDashboardBackend._evaluate_power_balance = _v23_evaluate_power_balance
DSRDashboardBackend._refresh_derived_states = _v23_refresh_derived_states
DSRDashboardBackend._isolate_source_fault = _v23_isolate_source_fault
DSRDashboardBackend.load_map = _v23_load_map
DSRDashboardBackend.load_faults_and_unknown = _v23_load_faults_and_unknown
DSRDashboardBackend._validate_restoration_topology = _v23_validate_restoration_topology
DSRDashboardBackend.mps_supply_context = _v23_mps_supply_context
DSRDashboardBackend.execute_current_step = _v23_execute_current_step
DSRDashboardBackend.source_rows = _v23_source_rows
DSRDashboardBackend.mps_rows = _v23_mps_rows


def _v23_validate_switch_plan(
    self: DSRDashboardBackend,
    switch_operations: list[tuple[str, str]] | list[list[str]],
) -> tuple[DSRDashboardBackend, Tuple[Tuple[str, str], ...]]:
    """Validate a provisional Switch plan and return its candidate topology.

    The actual simulator state is not changed. The returned candidate may be
    used to calculate MPS supply contexts after the provisional Switch actions.
    """

    self._require_loaded()
    candidate = copy.deepcopy(self)
    normalized: Tuple[Tuple[str, str], ...] = tuple()

    if switch_operations:
        normalized = candidate._normalize_switch_operations(
            switch_operations
        )
        previously_energized_nodes = {
            name: node.energized
            for name, node in candidate.nodes.items()
        }
        candidate._apply_switch_operations(normalized)
        candidate._validate_applied_switch_batch(
            operations=normalized,
            previously_energized_nodes=previously_energized_nodes,
        )

    candidate._validate_restoration_topology()
    candidate._refresh_derived_states()
    candidate.stage = "Validated provisional Switch topology"
    return candidate, normalized


DSRDashboardBackend.validate_switch_plan = _v23_validate_switch_plan

# ---------------------------------------------------------------------------
# V23 game scoring, configurable unknown regions, signed power balance,
# and text-command support.
# ---------------------------------------------------------------------------

@dataclass
class GameScore:
    """Track the monotonic-restoration score for one simulator game."""

    max_time_step: int = 20
    score: float = 0.0
    power_history: list[float] = field(default_factory=list)
    started: bool = False
    locked: bool = False
    ended: bool = False
    end_reason: str = ""
    violation_step: Optional[int] = None
    completion_step: Optional[int] = None
    projected_steps: int = 0

    def __post_init__(self) -> None:
        self.configure(self.max_time_step)

    def configure(self, max_time_step: int) -> None:
        """Set the maximum completed time step for a new game."""

        if isinstance(max_time_step, bool):
            raise ValueError("Maximum time step N must be a positive integer.")
        try:
            normalized = int(max_time_step)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Maximum time step N must be a positive integer."
            ) from error
        if normalized < 1:
            raise ValueError("Maximum time step N must be at least 1.")
        self.max_time_step = normalized

    def reset(self) -> None:
        """Reset runtime score data while preserving the configured N."""

        self.score = 0.0
        self.power_history = []
        self.started = False
        self.locked = False
        self.ended = False
        self.end_reason = ""
        self.violation_step = None
        self.completion_step = None
        self.projected_steps = 0

    def start(self, restored_active_power: float) -> None:
        """Start a game and count P0 immediately."""

        self.reset()
        power = float(restored_active_power)
        self.power_history = [power]
        self.score = power
        self.started = True

    def record(self, time_step: int, restored_active_power: float) -> None:
        """Record Pi and update the score after one completed time step."""

        if not self.started:
            raise RuntimeError("Load a fault scenario before scoring a step.")
        if self.ended:
            raise RuntimeError("The game has already ended.")

        time_step = int(time_step)
        power = float(restored_active_power)
        previous_power = self.power_history[-1]
        self.power_history.append(power)

        if self.locked:
            return

        if power < previous_power - 1e-9:
            self.score = 0.0
            self.locked = True
            self.violation_step = time_step
            return

        self.score += power

    def finish_at_limit(self, time_step: int) -> None:
        """Finish normally when the maximum time step is reached."""

        self.ended = True
        self.completion_step = int(time_step)
        self.end_reason = "Maximum time step reached"

    def finish_early(
        self,
        time_step: int,
        fully_restored_active_power: float,
    ) -> None:
        """Project the fully restored power through N and end the game."""

        time_step = int(time_step)
        remaining_steps = max(0, self.max_time_step - time_step)
        if not self.locked:
            self.score += remaining_steps * float(
                fully_restored_active_power
            )
        self.projected_steps = remaining_steps
        self.ended = True
        self.completion_step = time_step
        self.end_reason = "All faults cleared and all loads restored"

    @property
    def current_power(self) -> float:
        """Return the latest recorded restored active power."""

        return self.power_history[-1] if self.power_history else 0.0

    @property
    def status(self) -> str:
        """Return a compact human-readable game status."""

        if self.ended:
            return "Completed"
        if self.locked:
            return "Score locked at zero"
        if self.started:
            return "Running"
        return "Ready"

    def to_payload(self) -> Dict[str, object]:
        """Return a JSON-serializable score snapshot."""

        return {
            "max_time_step": self.max_time_step,
            "score": self.score,
            "power_history": list(self.power_history),
            "current_power": self.current_power,
            "started": self.started,
            "locked": self.locked,
            "ended": self.ended,
            "status": self.status,
            "end_reason": self.end_reason,
            "violation_step": self.violation_step,
            "completion_step": self.completion_step,
            "projected_steps": self.projected_steps,
        }


_v23_game_base_post_init = DSRDashboardBackend.__post_init__
_v23_game_base_load_map = DSRDashboardBackend.load_map
_v23_game_base_validate_switch_plan = DSRDashboardBackend.validate_switch_plan
_v23_game_base_execute_current_step = DSRDashboardBackend.execute_current_step


def _v23_game_post_init(self: DSRDashboardBackend) -> None:
    _v23_game_base_post_init(self)
    self.game_score = GameScore(max_time_step=20)
    self.configured_unknown_regions: list[Dict[str, object]] = []
    self.configured_unknown_regions = self.default_unknown_region_payload()


def _v23_default_unknown_region_payload(
    self: DSRDashboardBackend,
) -> list[Dict[str, object]]:
    """Read the default list of unknown rectangles from the scenario file."""

    path = self.data_dir / "UnknownState_v1.json"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return self._normalize_unknown_regions(data.get("unknown_regions", []))


def _v23_normalize_unknown_regions(
    self: DSRDashboardBackend,
    unknown_regions: Optional[list[Dict[str, object]]],
) -> list[Dict[str, object]]:
    """Validate rectangles and normalize corner order without map clipping."""

    if unknown_regions is None:
        return self.default_unknown_region_payload()
    if not isinstance(unknown_regions, list):
        raise ValueError("unknown_regions must be a list of rectangles.")

    normalized: list[Dict[str, object]] = []
    for index, raw_region in enumerate(unknown_regions, start=1):
        if not isinstance(raw_region, dict):
            raise ValueError(
                f"Unknown region {index} must be a JSON object."
            )
        raw_lower = raw_region.get("lower_left")
        raw_upper = raw_region.get("upper_right")
        if (
            not isinstance(raw_lower, (list, tuple))
            or not isinstance(raw_upper, (list, tuple))
            or len(raw_lower) != 2
            or len(raw_upper) != 2
        ):
            raise ValueError(
                f"Unknown region {index} requires lower_left=[x,y] "
                "and upper_right=[x,y]."
            )
        coordinates = [*raw_lower, *raw_upper]
        if not all(
            isinstance(value, (int, float, np.integer, np.floating))
            and np.isfinite(float(value))
            for value in coordinates
        ):
            raise ValueError(
                f"Unknown region {index} coordinates must be finite numbers."
            )

        x_1, y_1 = float(raw_lower[0]), float(raw_lower[1])
        x_2, y_2 = float(raw_upper[0]), float(raw_upper[1])

        def preserve_integer(value: float):
            return int(value) if value.is_integer() else value

        lower_left = [
            preserve_integer(min(x_1, x_2)),
            preserve_integer(min(y_1, y_2)),
        ]
        upper_right = [
            preserve_integer(max(x_1, x_2)),
            preserve_integer(max(y_1, y_2)),
        ]
        name = str(raw_region.get("name", "")).strip() or f"Region{index}"
        normalized.append({
            "name": name,
            "lower_left": lower_left,
            "upper_right": upper_right,
        })
    return normalized


def _v23_unknown_region_payload(
    self: DSRDashboardBackend,
) -> list[Dict[str, object]]:
    """Return the currently configured unknown rectangles."""

    if self.unknown_state is not None:
        return [
            {
                "name": region.name,
                "lower_left": list(region.lower_left),
                "upper_right": list(region.upper_right),
            }
            for region in self.unknown_state.unknown_regions
        ]
    return [copy.deepcopy(item) for item in self.configured_unknown_regions]


def _v23_configure_max_time_step(
    self: DSRDashboardBackend,
    max_time_step: int,
) -> None:
    """Configure N before the fault scenario starts."""

    if self.game_score.started:
        try:
            requested = int(max_time_step)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Maximum time step N must be a positive integer."
            ) from error
        if requested == self.game_score.max_time_step:
            return
        raise RuntimeError(
            "Maximum time step N cannot change after the game starts. "
            "Load the base map first."
        )
    self.game_score.configure(max_time_step)


def _v23_restored_active_power(self: DSRDashboardBackend) -> float:
    """Return signed active power of all currently energized loads."""

    return float(sum(
        node.active_power
        for node in self.nodes.values()
        if node.has_load and node.energized
    ))


def _v23_total_active_load(self: DSRDashboardBackend) -> float:
    """Return signed active power of every configured load."""

    return float(sum(
        node.active_power
        for node in self.nodes.values()
        if node.has_load
    ))


def _v23_all_faults_cleared(self: DSRDashboardBackend) -> bool:
    """Return whether ordinary and transmission-source faults are cleared."""

    ordinary_faults_cleared = all(
        not fault.active
        for fault in self.faults.values()
    )
    source_fault_cleared = (
        self.transmission_source is None
        or not self.transmission_source.fault_active
    )
    return ordinary_faults_cleared and source_fault_cleared


def _v23_all_loads_restored(self: DSRDashboardBackend) -> bool:
    """Return whether every load-bearing Node is energized."""

    load_nodes = [node for node in self.nodes.values() if node.has_load]
    return bool(load_nodes) and all(node.energized for node in load_nodes)


def _v23_game_payload(self: DSRDashboardBackend) -> Dict[str, object]:
    """Return score and completion state for the web and text interfaces."""

    payload = self.game_score.to_payload()
    payload.update({
        "time_step": self.time_step,
        "restored_active_power": self.restored_active_power(),
        "total_active_load": self.total_active_load(),
        "all_faults_cleared": self.all_faults_cleared(),
        "all_loads_restored": self.all_loads_restored(),
        "commands_enabled": self.game_score.started and not self.game_score.ended,
    })
    return payload


def _v23_signed_evaluate_power_balance(
    self: DSRDashboardBackend,
    strict: bool = False,
) -> set[str]:
    """Evaluate source balance while allowing signed grid power."""

    tolerance = 1e-6
    if self.transmission_source is None:
        return set()
    self.transmission_source.update_availability(self.time_step)
    self.transmission_source.active_power = 0.0
    self.transmission_source.reactive_power = 0.0
    active_source_nodes: set[str] = set()
    cluster_rows: list[list[str]] = []

    for mps in self.mps_units.values():
        mps.connected_node_name = self._node_at_position(mps.position)
        mps.connected_area_names = tuple()
        mps.connected_load_p = 0.0
        mps.connected_load_q = 0.0
        mps.grid_connected = False

    for component_nodes in self._node_components():
        details = self._component_details(component_nodes)
        supplying_mps = [
            mps
            for mps in self.mps_units.values()
            if (
                mps.state == "Supplying"
                and mps.connected_node_name in component_nodes
                and mps.current_p > -tolerance
            )
        ]
        grid_connected = (
            self.transmission_source.available
            and self.transmission_source.node_name in component_nodes
        )
        mps_p = sum(mps.current_p for mps in supplying_mps)
        mps_q = sum(mps.current_q for mps in supplying_mps)
        load_p = float(details["load_p"])
        load_q = float(details["load_q"])
        safe = bool(details["safe"])

        for mps in self.mps_units.values():
            if mps.connected_node_name in component_nodes:
                mps.connected_area_names = details["areas"]
                mps.connected_load_p = load_p
                mps.connected_load_q = load_q
                mps.grid_connected = grid_connected

        valid_supply = False
        grid_p = 0.0
        grid_q = 0.0
        error_message = ""

        if (grid_connected or supplying_mps) and not safe:
            error_message = (
                "Power cannot be applied to an unsafe electrical cluster: "
                + "; ".join(details["unsafe_items"][:6])
            )
        elif grid_connected:
            # The transmission grid is the balancing source. Negative P/Q
            # means the grid absorbs net export from negative loads or MPS.
            grid_p = load_p - mps_p
            grid_q = load_q - mps_q
            valid_supply = True
        elif supplying_mps:
            if (
                abs(mps_p - load_p) > tolerance
                or abs(mps_q - load_q) > tolerance
            ):
                error_message = (
                    f"Islanded cluster requires P={load_p:g} kW and "
                    f"Q={load_q:g} kVar, but its MPS units provide "
                    f"P={mps_p:g} kW and Q={mps_q:g} kVar."
                )
            else:
                valid_supply = True

        if strict and error_message:
            raise ValueError(error_message)

        if valid_supply:
            if grid_connected:
                active_source_nodes.add(self.transmission_source.node_name)
                self.transmission_source.active_power += grid_p
                self.transmission_source.reactive_power += grid_q
            for mps in supplying_mps:
                active_source_nodes.add(mps.connected_node_name)

        cluster_rows.append([
            ", ".join(details["areas"]) or "Transmission only",
            ", ".join(details["nodes"]),
            self._format_load_value(load_p),
            self._format_load_value(load_q),
            self._format_load_value(mps_p),
            self._format_load_value(mps_q),
            self._format_load_value(grid_p if valid_supply else 0.0),
            self._format_load_value(grid_q if valid_supply else 0.0),
            "Energized" if valid_supply else "De-energized",
        ])

    self.power_cluster_rows_cache = cluster_rows
    return active_source_nodes


def _v23_game_load_map(
    self: DSRDashboardBackend,
    render_output: bool = True,
) -> str:
    """Load the base map and reset the game clock and score."""

    result = _v23_game_base_load_map(self, render_output=False)
    self.game_score.reset()
    self.event_history = []
    self.last_step_events = []
    self.next_event_sequence = 1
    self.configured_unknown_regions = self.default_unknown_region_payload()
    self.stage = "Base map loaded; configure N and unknown regions"
    return self.render("base_map.png") if render_output else result


def _v23_game_load_faults_and_unknown(
    self: DSRDashboardBackend,
    render_output: bool = True,
    unknown_regions: Optional[list[Dict[str, object]]] = None,
    fault_scenario: Optional[list[Dict[str, object]]] = None,
    unit_initial_positions: Optional[Dict[str, object]] = None,
) -> str:
    """Load a fresh scored scenario with a union of unknown rectangles."""

    if self.game_score.ended and not self.stage.startswith("Base map"):
        raise RuntimeError(
            "The game has ended. Load the base map before loading a new scenario."
        )

    self._load_topology()
    if self.graph is None:
        raise RuntimeError("Graph loading failed.")

    if fault_scenario is None:
        self.faults = Fault.load_all(
            json_path=self._path("Faults_v1.json"),
            graph=self.graph,
            nodes=self.nodes,
            electrical_lines=self.electrical_lines,
            areas=self.areas,
        )
    else:
        if not isinstance(fault_scenario, list) or not fault_scenario:
            raise ValueError("fault_scenario must be a non-empty list.")
        self.faults = {}
        for index, raw_fault in enumerate(fault_scenario, start=1):
            if not isinstance(raw_fault, dict):
                raise ValueError(f"Fault override {index} must be an object.")
            name = str(raw_fault.get("name", f"F{index}")).strip()
            if not name or name in self.faults:
                raise ValueError(f"Invalid or duplicate fault name: {name!r}")
            raw_position = raw_fault.get("position")
            if (
                not isinstance(raw_position, (list, tuple))
                or len(raw_position) != 2
            ):
                raise ValueError(
                    f"Fault {name} position must contain two coordinates."
                )
            fault = Fault(
                name=name,
                position=(int(raw_position[0]), int(raw_position[1])),
                repair_resource=int(raw_fault.get("repair_resource", 1)),
                active=bool(raw_fault.get("active", True)),
            )
            fault.bind(
                graph=self.graph,
                nodes=self.nodes,
                electrical_lines=self.electrical_lines,
                areas=self.areas,
            )
            self.faults[name] = fault
    for line in self.electrical_lines.values():
        line.update_state()

    self.fault_isolation_switch_names = Fault.isolate_affected_areas(
        faults=self.faults,
        areas=self.areas,
        switches=self.switches,
    )
    for line in self.electrical_lines.values():
        line.update_state()

    base_unknown_state = UnknownState.from_json(
        self._path("UnknownState_v1.json")
    )
    self.configured_unknown_regions = self._normalize_unknown_regions(
        unknown_regions
        if unknown_regions is not None
        else self.configured_unknown_regions
    )
    regions = tuple(
        UnknownRegion(
            name=str(region["name"]),
            lower_left=tuple(region["lower_left"]),
            upper_right=tuple(region["upper_right"]),
        )
        for region in self.configured_unknown_regions
    )
    self.unknown_state = UnknownState(
        event_id=base_unknown_state.event_id,
        default_inspected=base_unknown_state.default_inspected,
        unknown_regions=regions,
        open_switches_in_unknown_regions=(
            base_unknown_state.open_switches_in_unknown_regions
        ),
        open_switches_adjacent_to_unknown_areas=(
            base_unknown_state.open_switches_adjacent_to_unknown_areas
        ),
    )
    self.event_id = self.unknown_state.event_id
    self.unknown_safety_switch_names = self.unknown_state.apply(
        graph=self.graph,
        nodes=self.nodes,
        electrical_lines=self.electrical_lines,
        switches=self.switches,
        areas=self.areas,
    )

    self._load_mobile_units()
    if unit_initial_positions is not None:
        if not isinstance(unit_initial_positions, dict):
            raise ValueError("unit_initial_positions must be an object.")
        unknown_units = set(unit_initial_positions).difference(
            self.mobile_units
        )
        if unknown_units:
            raise ValueError(
                "Unknown mobile units in initial-position overrides: "
                + ", ".join(sorted(unknown_units))
            )
        for unit_name, raw_position in unit_initial_positions.items():
            if (
                not isinstance(raw_position, (list, tuple))
                or len(raw_position) != 2
            ):
                raise ValueError(
                    f"Initial position for {unit_name} must have two coordinates."
                )
            position = (int(raw_position[0]), int(raw_position[1]))
            if not self.graph.contains(position):
                raise ValueError(
                    f"Initial position for {unit_name} is outside the graph: "
                    f"{position}"
                )
            unit = self.mobile_units[unit_name]
            unit.initial_position = position
            unit.position = position
            unit.idle()
    for mobile_unit in self.mobile_units.values():
        newly_inspected = mobile_unit.inspect_and_refresh(
            event_id=self.event_id,
            nodes=self.nodes,
            electrical_lines=self.electrical_lines,
            switches=self.switches,
        )
        newly_discovered = mobile_unit._newly_discovered_fault_names(
            newly_inspected_positions=newly_inspected,
            faults=self.faults,
        )
        mobile_unit.begin_fault_analysis(newly_discovered, self.faults)

    if self.transmission_source is None:
        raise RuntimeError("Transmission source configuration was not loaded.")
    self.transmission_source.scenario_fault_enabled = True
    self.transmission_source.update_availability(self.time_step)
    if self.transmission_source.fault_active:
        self._isolate_source_fault()
    else:
        self.source_fault_switch_names = ()
        self.transmission_source.isolated_switch_names = ()
    self._refresh_derived_states()

    self.game_score.start(self.restored_active_power())
    self.event_history = []
    self.last_step_events = []
    self.next_event_sequence = 1
    self.stage = "Scored fault scenario loaded at time step 0"

    if self.all_faults_cleared() and self.all_loads_restored():
        self.game_score.finish_early(
            time_step=0,
            fully_restored_active_power=self.total_active_load(),
        )
        self.stage = "Game completed automatically at time step 0"

    return (
        self.render("fault_unknown_source_map.png")
        if render_output
        else ""
    )


def _v23_game_validate_switch_plan(
    self: DSRDashboardBackend,
    switch_operations: list[tuple[str, str]] | list[list[str]],
) -> tuple[DSRDashboardBackend, Tuple[Tuple[str, str], ...]]:
    """Reject provisional commands after game completion."""

    if self.game_score.ended:
        raise RuntimeError(
            "The game has ended. Load the base map to start again."
        )
    if not self.game_score.started:
        raise RuntimeError("Load the fault scenario before issuing commands.")
    return _v23_game_base_validate_switch_plan(self, switch_operations)


def _v23_game_execute_current_step(
    self: DSRDashboardBackend,
    mobile_commands: Dict[str, Dict[str, object]],
    switch_operations: list[tuple[str, str]] | list[list[str]],
    frames_per_step: int = 1,
    collect_frame_snapshots: bool = False,
) -> Dict[str, object]:
    """Execute one atomic step, update the score, and end when required."""

    if not self.game_score.started:
        raise RuntimeError("Load the fault scenario before issuing commands.")
    if self.game_score.ended:
        raise RuntimeError(
            "The game has ended. Load the base map to start again."
        )
    if self.time_step >= self.game_score.max_time_step:
        self.game_score.finish_at_limit(self.time_step)
        raise RuntimeError("The maximum time step has already been reached.")

    was_ended = self.game_score.ended
    result = _v23_game_base_execute_current_step(
        self,
        mobile_commands=mobile_commands,
        switch_operations=switch_operations,
        frames_per_step=frames_per_step,
        collect_frame_snapshots=collect_frame_snapshots,
    )

    restored_power = self.restored_active_power()
    self.game_score.record(self.time_step, restored_power)

    if (
        self.all_faults_cleared()
        and self.all_loads_restored()
        and self.time_step < self.game_score.max_time_step
    ):
        self.game_score.finish_early(
            time_step=self.time_step,
            fully_restored_active_power=self.total_active_load(),
        )
        self.stage = (
            f"Game completed early at time step {self.time_step}; "
            f"score projected through step {self.game_score.max_time_step}"
        )
    elif self.time_step >= self.game_score.max_time_step:
        self.game_score.finish_at_limit(self.time_step)
        self.stage = (
            f"Game ended at maximum time step "
            f"{self.game_score.max_time_step}"
        )
    elif self.game_score.locked:
        self.stage = (
            f"Time step {self.time_step} completed; restored load decreased "
            "and the score is permanently locked at zero"
        )

    if not was_ended and self.game_score.ended:
        self._emit_event(
            "GAME_ENDED",
            time_step=self.time_step,
            frame=0,
            details={
                "reason": self.game_score.end_reason,
                "score": self.game_score.score,
            },
        )

    result["game"] = self.game_payload()
    result["events"] = list(self.last_step_events)
    return result


def _v23_validate_command_batch(
    self: DSRDashboardBackend,
    mobile_commands: Dict[str, Dict[str, object]],
    switch_operations: list[tuple[str, str]] | list[list[str]],
) -> "DSRDashboardBackend":
    """Predict one complete command step without mutating live state."""

    candidate = copy.deepcopy(self)
    candidate.execute_current_step(
        mobile_commands=mobile_commands,
        switch_operations=switch_operations,
        frames_per_step=self.physics_frames_per_step,
        collect_frame_snapshots=False,
    )
    return candidate


DSRDashboardBackend.__post_init__ = _v23_game_post_init
DSRDashboardBackend.default_unknown_region_payload = _v23_default_unknown_region_payload
DSRDashboardBackend._normalize_unknown_regions = _v23_normalize_unknown_regions
DSRDashboardBackend.unknown_region_payload = _v23_unknown_region_payload
DSRDashboardBackend.configure_max_time_step = _v23_configure_max_time_step
DSRDashboardBackend.restored_active_power = _v23_restored_active_power
DSRDashboardBackend.total_active_load = _v23_total_active_load
DSRDashboardBackend.all_faults_cleared = _v23_all_faults_cleared
DSRDashboardBackend.all_loads_restored = _v23_all_loads_restored
DSRDashboardBackend.game_payload = _v23_game_payload
DSRDashboardBackend.clock_payload = _v23_clock_payload
DSRDashboardBackend._emit_event = _v23_emit_event
DSRDashboardBackend.runtime_snapshot = _v23_runtime_snapshot
DSRDashboardBackend._emit_transition_events = _v23_emit_transition_events
DSRDashboardBackend._evaluate_power_balance = _v23_signed_evaluate_power_balance
DSRDashboardBackend.load_map = _v23_game_load_map
DSRDashboardBackend.load_faults_and_unknown = _v23_game_load_faults_and_unknown
DSRDashboardBackend.validate_switch_plan = _v23_game_validate_switch_plan
DSRDashboardBackend.execute_current_step = _v23_game_execute_current_step
DSRDashboardBackend.validate_command_batch = _v23_validate_command_batch
