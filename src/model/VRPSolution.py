import plotly.graph_objects as go


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
    - loads (list of lists): The load of each vehicle at each location of its route.
    - depot (int): The index of the depot location.
    - use_capacity (bool): Whether the solution uses vehicle capacity or not.
    """

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
        num_vehicles: int,
        locations: list[tuple[int, int]],
        objective: float,
        total_distance: int,
        routes: list[list[int]],
        distances: list[int],
        depot: int,
        loads: list[list[int]] = None,
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

    def display(self):
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
            legend_group = f"Vehicle {vehicle_id + 1}"

            # Draw routes
            fig.add_trace(
                go.Scatter(
                    x=[loc[0] for loc in route_coordinates],
                    y=[loc[1] for loc in route_coordinates],
                    mode="lines",
                    line=dict(width=5, color=color),
                    name=f"Vehicle {vehicle_id + 1}: {self.distances[vehicle_id]}m",
                    legendgroup=legend_group,
                )
            )

            # Draw annotations
            for i in range(len(route_coordinates) - 1):
                self.plot_direction(
                    fig,
                    route_coordinates[i],
                    route_coordinates[i + 1],
                    color,
                    5,
                    legend_group,
                )
                self.plot_location(
                    fig, route_coordinates[i], color, legend_group, vehicle_id, i
                )

        self.plot_location(fig, self.locations[self.depot], "gray")

        fig.update_layout(
            xaxis_title="X Coordinate",
            yaxis_title="Y Coordinate",
            legend=dict(
                title=f"Total Distance: {self.total_distance}m",
                orientation="h",
                yanchor="bottom",
                y=1.02,
            ),
        )

        fig.show()

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
        self, fig, loc, color, legend_group=None, vehicle_id=None, route_id=None
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
                hoverinfo="text",
                hovertext=hovertext,
                legendgroup=legend_group,
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