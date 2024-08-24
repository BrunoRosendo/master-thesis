from abc import ABC

from src.model.VRPSolution import VRPSolution
from src.solver.VRPSolver import VRPSolver


class QuboSolver(VRPSolver, ABC):
    def convert_qubo_result(
        self,
        var_dict: dict[str, float],
        objective: float,
        run_time: float,
        local_run_time: float,
        qpu_access_time: float = None,
    ) -> VRPSolution:
        """
        Convert the final variables into a VRPSolution result.
        """

        use_depot = self.model.depot is not None
        route_starts = self.model.get_result_route_starts(var_dict)

        routes = []
        loads = []
        distances = []
        total_distance = 0

        for i in range(self.model.num_vehicles):
            route = []
            route_loads = []
            route_distance = 0
            cur_load = 0

            index = route_starts[i] if i < len(route_starts) else None
            previous_index = self.model.depot if use_depot else index
            if use_depot:
                route.append(self.model.depot)
                route_loads.append(0)

            while index is not None:
                route_distance += self.model.distance_matrix[previous_index][index]
                cur_load += self.model.get_location_demand(index)
                route.append(index)
                route_loads.append(cur_load)

                if use_depot and index == self.model.depot:
                    break

                previous_index = index
                index = self.model.get_result_next_location(var_dict, index)

            routes.append(route)
            distances.append(route_distance)
            loads.append(route_loads)
            total_distance += route_distance

        return VRPSolution.from_model(
            self.model,
            objective,
            total_distance,
            routes,
            distances,
            loads,
            run_time=run_time,
            qpu_access_time=qpu_access_time,
            local_run_time=local_run_time,
        )
