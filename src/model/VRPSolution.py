from __future__ import annotations

import datetime
import json
from pathlib import Path

import plotly.graph_objects as go

from src.model.VRP import VRP, DistanceUnit


class VRPSolution:
    """
    Class to represent a solution to the Capacitated Vehicle Routing Problem (CVRP).

    Attributes:
    - num_vehicles (int): The number of vehicles used in the solution.
    - locations (list of tuples): The coordinates of the locations.
    - objective (float): The objective value of the solution.
    - total_distance (float): The total distance traveled by all vehicles.
    - routes (list of lists): The routes (indices) taken by each vehicle.
    - distances (list of floats): The distance traveled by each vehicle.
    - capacities (int or list of ints): The capacity of each vehicle.
    - use_depot (bool): Whether the vehicles start and return to the depot after visiting all locations.
    - loads (list of lists): The load of each vehicle at each location of its route.
    - depot (int): The index of the depot location.
    - use_capacity (bool): Whether the solution uses vehicle capacity or not.
    - run_time (float): The total runtime of the solver.
    - qpu_access_time (float): The total runtime of the QPU.
    - local_run_time (float): The total runtime of the local machine, including pre-processing and queues.
    """

    COLOR_LIST = [
        "blue",
        "red",
        "green",
        "purple",
        "orange",
        "brown",
        "pink",
        "grey",
        "black",
        "cyan",
        "magenta",
        "yellow",
    ]

    RESULTS_PATH = "results"

    def __init__(
        self,
        num_vehicles: int,
        locations: list[tuple[float, float]],
        objective: float,
        total_distance: int,
        routes: list[list[int]],
        distances: list[int],
        depot: int | None,
        capacities: int | list[int] | None,
        loads: list[list[int]] | None,
        run_time: float | None,
        qpu_access_time: float | None,
        local_run_time: float | None,
        location_names: list[str] | None,
        distance_unit: DistanceUnit,
    ):
        self.num_vehicles = num_vehicles
        self.locations = locations
        self.objective = objective
        self.total_distance = total_distance
        self.routes = routes
        self.distances = distances
        self.loads = loads
        self.depot = depot
        self.run_time = run_time
        self.qpu_access_time = qpu_access_time
        self.local_run_time = local_run_time
        self.location_names = location_names or [str(i) for i in range(len(locations))]
        self.distance_unit = distance_unit
        self.use_depot = depot is not None
        self.capacities = capacities
        self.use_capacity = capacities is not None

    def display(
        self, file_name: str = None, results_path: str = RESULTS_PATH, fig_height=None
    ):
        """
        Display the solution using a plotly figure.
        Saves the figure to an HTML file if a file name is provided.
        """

        fig = go.Figure()

        for vehicle_id in range(self.num_vehicles):
            route_coordinates = [
                (self.locations[node][0], self.locations[node][1])
                for node in self.routes[vehicle_id]
            ]

            color = self.COLOR_LIST[vehicle_id % len(self.COLOR_LIST)]

            capacity = (
                self.capacities[vehicle_id]
                if isinstance(self.capacities, list)
                else self.capacities
            )
            legend_group = f"Vehicle {vehicle_id + 1}"
            legend_name = (
                f"Vehicle {vehicle_id + 1} ({capacity})"
                if self.use_capacity
                else f"Vehicle {vehicle_id + 1}"
            )

            # Draw routes
            fig.add_trace(
                go.Scatter(
                    x=[loc[0] for loc in route_coordinates],
                    y=[loc[1] for loc in route_coordinates],
                    mode="lines",
                    line=dict(width=5, color=color),
                    name=legend_name,
                    legendgroup=legend_group,
                )
            )

            # Draw annotations
            for i in range(len(route_coordinates) - 1):
                loc_name = self.location_names[self.routes[vehicle_id][i]]

                self.plot_direction(
                    fig,
                    route_coordinates[i],
                    route_coordinates[i + 1],
                    color,
                    5,
                    legend_group,
                )
                self.plot_location(
                    fig,
                    route_coordinates[i],
                    color,
                    loc_name,
                    legend_group,
                    vehicle_id,
                    i,
                )

            if not self.use_depot and len(route_coordinates) > 0:
                self.plot_location(
                    fig,
                    route_coordinates[-1],
                    color,
                    self.location_names[self.routes[vehicle_id][-1]],
                    legend_group,
                    vehicle_id,
                    len(route_coordinates) - 1,
                )

        if self.use_depot:
            self.plot_location(
                fig, self.locations[self.depot], "gray", self.location_names[self.depot]
            )

        layout_args = {
            "xaxis_title": "X Coordinate",
            "yaxis_title": "Y Coordinate",
            "legend": dict(
                title=(
                    f"Total Distance: {self.parse_distance(self.total_distance)}"
                    if self.distance_unit == DistanceUnit.METERS
                    else f"Total Time: {datetime.timedelta(seconds=self.total_distance)}"
                ),
                orientation="h",
                yanchor="bottom",
                y=1.02,
            ),
        }
        if fig_height is not None:
            layout_args["height"] = fig_height
        fig.update_layout(**layout_args)

        fig.show()

        if file_name is not None:
            html_path = f"{results_path}/html"
            Path(html_path).mkdir(parents=True, exist_ok=True)
            fig.write_html(f"{html_path}/{file_name}.html")

    def plot_direction(self, fig, loc1, loc2, color, line_width, legend_group=None):
        """
        Plot an arrow representing the direction from coord1 to coord2 with the given color and line width.
        """
        x_mid = (loc1[0] + loc2[0]) / 2
        y_mid = (loc1[1] + loc2[1]) / 2

        fig.add_trace(
            go.Scatter(
                x=[loc1[0], x_mid],
                y=[loc1[1], y_mid],
                mode="lines+markers",
                line=dict(width=line_width, color=color),
                marker=dict(size=20, symbol="arrow-up", angleref="previous"),
                hoverinfo="skip",
                showlegend=False,
                legendgroup=legend_group,
            )
        )

    def plot_location(
        self,
        fig,
        loc,
        color,
        name,
        legend_group=None,
        vehicle_id=None,
        route_id=None,
    ):
        """
        Plot a location with the given color and legend group.
        """
        hovertext = (
            "Starting Point"
            if vehicle_id is None
            else (
                f"Vehicle {vehicle_id + 1}: {self.loads[vehicle_id][route_id]} passengers"
                if self.use_capacity
                else f"Vehicle {vehicle_id + 1}"
            )
        )

        # Add the 'outline' text
        fig.add_trace(
            go.Scatter(
                x=[loc[0]],
                y=[loc[1]],
                mode="text",
                text=name,
                textposition="middle center",
                textfont=dict(color="#6b6a6a", size=15.5),  # Slightly larger
                showlegend=False,
                hoverinfo="none",  # Disable hover for the 'outline' text
                legendgroup=legend_group,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[loc[0]],
                y=[loc[1]],
                mode="markers+text",
                marker=dict(size=50, symbol="circle", color=color, line_width=2),
                text=name,
                textposition="middle center",
                textfont=dict(color="white", size=15),
                showlegend=False,
                hoverinfo="text",
                hovertext=hovertext,
                legendgroup=legend_group,
            )
        )

    def print(self):
        """Print the solution to the console."""

        print()

        if self.run_time is not None:
            print(f"Solver runtime: {self.run_time}µs")

        if self.qpu_access_time is not None:
            print(f"QPU access time: {self.qpu_access_time}µs")

        if self.local_run_time is not None:
            print(f"Local runtime: {self.local_run_time}µs")

        print(f"\nObjective: {self.objective}\n")

        for vehicle_id in range(self.num_vehicles):
            print(f"Route for vehicle {vehicle_id}:")
            print("  ", end="")

            for i, node in enumerate(self.routes[vehicle_id]):
                if self.use_capacity:
                    print(f"{node} [{self.loads[vehicle_id][i]}P]", end="")
                else:
                    print(f"{node}", end="")
                if i < len(self.routes[vehicle_id]) - 1:
                    print(" -> ", end="")

            if self.distance_unit == DistanceUnit.METERS:
                print(
                    f"\nDistance of the route: {self.parse_distance(self.distances[vehicle_id])}\n"
                )
            else:
                print(
                    f"\nTime of the route: {datetime.timedelta(seconds=self.distances[vehicle_id])}\n"
                )

        if self.distance_unit == DistanceUnit.METERS:
            print(
                f"Total distance of all routes: {self.parse_distance(self.total_distance)}"
            )
        else:
            print(
                f"Total time of all routes: {datetime.timedelta(seconds=self.total_distance)}"
            )

    def save_json(self, file_name: str, results_path: str = RESULTS_PATH):
        """
        Save the solution to a JSON file.
        """

        json_path = f"{results_path}/json"
        Path(json_path).mkdir(parents=True, exist_ok=True)

        with open(f"{json_path}/{file_name}.json", "w") as file:
            json.dump(self, file, default=lambda o: o.__dict__, indent=4)

    @staticmethod
    def parse_distance(distance: float) -> str:
        """
        Parse the distance to a string with the correct unit (meters or kilometers).
        """

        return f"{distance // 1000}km" if distance > 10000 else f"{distance}m"

    @staticmethod
    def from_json(file_name: str, results_path: str = RESULTS_PATH) -> VRPSolution:
        """
        Load a solution from a JSON file.
        """

        with open(f"{results_path}/json/{file_name}.json", "r") as file:
            data = json.load(file)

        return VRPSolution(
            data["num_vehicles"],
            [(loc[0], loc[1]) for loc in data["locations"]],
            data["objective"],
            data["total_distance"],
            data["routes"],
            data["distances"],
            data["depot"],
            data["capacities"],
            data["loads"],
            data["run_time"],
            data["qpu_access_time"],
            data["local_run_time"],
            data["location_names"],
            data["distance_unit"],
        )

    @staticmethod
    def from_model(
        model: VRP,
        objective: float,
        total_distance: int,
        routes: list[list[int]],
        distances: list[int],
        loads: list[list[int]] = None,
        run_time: float = None,
        qpu_access_time: float = None,
        local_run_time: float = None,
    ):
        return VRPSolution(
            model.num_vehicles,
            model.locations,
            objective,
            total_distance,
            routes,
            distances,
            model.depot,
            getattr(model, "capacities", None) or getattr(model, "capacity", None),
            loads,
            run_time,
            qpu_access_time,
            local_run_time,
            model.location_names,
            model.distance_unit,
        )
