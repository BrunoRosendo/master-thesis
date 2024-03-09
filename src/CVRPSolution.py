import plotly.express as px
import plotly.graph_objects as go
import numpy as np


class CVRPSolution:

    COLOR_LIST = [  # TODO: Refine colors
        "blue",
        "red",
        "green",
        "purple",
        "orange",
        "yellow",
        "pink",
        "brown",
        "grey",
        "black",
        "cyan",
        "magenta",
    ]

    def __init__(
        self,
        num_vehicles,
        locations,
        objective,
        total_distance,
        routes,
        distances,
        depot,
        loads=None,
    ):
        self.num_vehicles = num_vehicles
        self.locations = locations
        self.objective = objective
        self.total_distance = total_distance
        self.routes = routes
        self.distances = distances
        self.loads = loads
        self.depot = depot
        self.use_capacity = loads is not None

    def display(self):  # TODO: fix hover info, loads, distance, total_distance
        """
        Display the solution using a plotly figure.
        """
        fig = go.Figure()

        for vehicle_id in range(self.num_vehicles):
            route_coordinates = [
                (self.locations[node][0], self.locations[node][1])
                for node in self.routes[vehicle_id]
            ]

            color = self.COLOR_LIST[vehicle_id % len(self.COLOR_LIST)]

            # Draw routes
            fig.add_trace(
                go.Scatter(
                    x=[loc[0] for loc in route_coordinates],
                    y=[loc[1] for loc in route_coordinates],
                    mode="lines",
                    line=dict(width=5, color=color),
                    name=f"Vehicle {vehicle_id + 1}",
                )
            )

            # Draw annotations
            for i in range(len(route_coordinates) - 1):
                self.plot_direction(
                    fig, route_coordinates[i], route_coordinates[i + 1], color, 5
                )
                self.plot_location(fig, route_coordinates[i], color)

        self.plot_location(fig, self.locations[self.depot], "gray")

        fig.update_layout(
            xaxis_title="X Coordinate",
            yaxis_title="Y Coordinate",
        )

        fig.show()

    def plot_direction(self, fig, loc1, loc2, color, line_width):
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
            )
        )

    def plot_location(self, fig, loc, color):
        fig.add_trace(
            go.Scatter(
                x=[loc[0]],
                y=[loc[1]],
                mode="markers+text",
                marker=dict(size=50, symbol="circle", color=color, line_width=2),
                text=str(
                    self.locations.index(loc)
                ),  # Display the index of the location
                textposition="middle center",
                textfont=dict(color="white", size=15),
                showlegend=False,
            )
        )

    def print(self):
        """Print the solution to the console."""
        print(f"Objective: {self.objective}\n")

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

            print(f"\nDistance of the route: {self.distances[vehicle_id]}m\n")

        print(f"Total distance of all routes: {self.total_distance}m")
