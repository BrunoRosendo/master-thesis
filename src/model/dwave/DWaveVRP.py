from abc import ABC, abstractmethod

from dimod import ConstrainedQuadraticModel, SampleSet

from src.model.VRP import VRP


class DWaveVRP(ABC, VRP):
    """
    A class to represent a DWave Ocean formulation of the VRP model.

    Attributes:
        num_vehicles (int): Number of vehicles available.
        trips (list): List of tuples, where each tuple contains the pickup and delivery locations, and the amount of customers for a trip.
        distance_matrix (list): Matrix with the distance between each pair of locations.
        locations (list): List of coordinates for each location.
        use_deliveries (bool): Whether the problem uses deliveries or not.
        cqm (ConstrainedQuadraticModel): DWave Ocean model for the VRP
    """

    def __init__(
        self,
        num_vehicles: int,
        trips: list[tuple[int, int, int]],
        distance_matrix: list[list[int]],
        locations: list[tuple[int, int]],
        use_deliveries: bool,
        simplify: bool,
    ):
        super().__init__(
            num_vehicles, trips, distance_matrix, locations, use_deliveries
        )

        self.simplify = simplify
        self.cqm = ConstrainedQuadraticModel()
        self.build_cqm()

    def build_cqm(self):
        """
        Builds the CQM DWave model for VRP.
        """

        self.create_vars()
        self.create_objective()
        self.create_constraints()

    def constrained_quadratic_model(self) -> ConstrainedQuadraticModel:
        """
        Returns the constrained quadratic model for VRP, based on DWave Ocean.
        Removes unnecessary constraints if simplify is set to True.
        """

        if self.simplify:
            self.cqm.fix_variables(self.get_simplified_variables())
        return self.cqm

    @abstractmethod
    def get_simplified_variables(self) -> dict[str, int]:
        """
        Get the variables that should be replaced during the simplification and their values.
        """
        pass

    @abstractmethod
    def create_vars(self):
        """
        Create the variables for the CQM model.
        """
        pass

    @abstractmethod
    def create_objective(self):
        """
        Create the objective function for the CQM model.
        """
        pass

    @abstractmethod
    def create_constraints(self):
        """
        Create the constraints for the CQM model.
        """
        pass

    def re_add_variables(self, var_dict: dict[str, float]) -> dict[str, float]:
        """
        Re-add the variables that were removed during the simplification.
        """

        replaced_vars = self.get_simplified_variables()
        for var_name, value in replaced_vars.items():
            var_dict[var_name] = float(value)
        return var_dict

    def build_var_dict(self, result: SampleSet) -> (dict[str, float], float):
        """
        Builds a dictionary of variable names and their values from the result.
        Assumes the order of energy returned by the sampler.
        It takes the simplification step into consideration.

        Returns:
            dict[str, float]: Dictionary of variable names and their values.
            float: Energy of the best solution.
        """

        for i, var_dict in enumerate(result):
            is_feasible = result.record.is_feasible[i]
            energy = result.record.energy[i]
            if is_feasible:
                if self.simplify:
                    var_dict = self.re_add_variables(dict(var_dict))
                return var_dict, energy

        raise Exception("The solution is infeasible, aborting!")

    @abstractmethod
    def get_result_route_starts(self, var_dict: dict[str, float]) -> list[int]:
        """
        Get the starting location for each route from the variable dictionary.
        """
        pass

    @abstractmethod
    def get_result_next_location(
        self, var_dict: dict[str, float], cur_location: int
    ) -> int | None:
        """
        Get the next location for a route from the variable dictionary.
        """
        pass

    @abstractmethod
    def get_var_name(self, i: int, j: int, k: int | None) -> str:
        """
        Get the variable name for the given indices.
        """
        pass

    def get_var(
        self, var_dict: dict[str, float], i: int, j: int, k: int | None = None
    ) -> float:
        """
        Get the variable value for the given indices.
        """

        var_name = self.get_var_name(i, j, k)
        return round(var_dict[var_name])
