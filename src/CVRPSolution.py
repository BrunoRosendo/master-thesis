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
        loads=None,
    ):
        self.num_vehicles = num_vehicles
        self.locations = locations
        self.objective = objective
        self.total_distance = total_distance
        self.routes = routes
        self.distances = distances
        self.loads = loads
        self.use_capacity = loads is not None

    def display(self):  # TODO: extract methods, paint 0 black, fix hover info
        fig = go.Figure()

        # Draw routes
        for vehicle_id in range(self.num_vehicles):
            route_coordinates = [
                (self.locations[node][0], self.locations[node][1])
                for node in self.routes[vehicle_id]
            ]

            color = self.COLOR_LIST[vehicle_id % len(self.COLOR_LIST)]

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
                x_start, y_start = route_coordinates[i]
                x_end, y_end = route_coordinates[i + 1]
                x_mid = (x_start + x_end) / 2
                y_mid = (y_start + y_end) / 2

                # Draw arrows
                fig.add_trace(
                    go.Scatter(
                        x=[x_start, x_mid],
                        y=[y_start, y_mid],
                        mode="lines+markers",
                        line=dict(width=5, color=color),
                        marker=dict(size=20, symbol="arrow-up", angleref="previous"),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

                # Draw locations
                fig.add_trace(
                    go.Scatter(
                        x=[x_start],
                        y=[y_start],
                        mode="markers+text",
                        marker=dict(
                            size=50, symbol="circle", color=color, line_width=2
                        ),
                        text=str(
                            self.locations.index(route_coordinates[i])
                        ),  # Display the index of the location
                        textposition="middle center",
                        textfont=dict(color="white", size=15),
                        showlegend=False,
                    )
                )

        fig.update_layout(
            xaxis_title="X Coordinate",
            yaxis_title="Y Coordinate",
        )

        fig.show()

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
