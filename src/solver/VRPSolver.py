import time
from abc import ABC, abstractmethod
from typing import Callable, Any

from src.model.VRP import VRP
from src.model.VRPSolution import VRPSolution


class VRPSolver(ABC):
    """
    Abstract class for solving the Capacitated Vehicle Routing Problem (CVRP).

    Attributes:
        track_progress (bool): Whether to track the progress of the solver or not.
        model (VRP): VRP instance of the model.
        run_time (int): Time taken to run the solver (measured locally).
    """

    def __init__(
        self,
        model: VRP,
        track_progress: bool,
    ):
        self.model = model
        self.track_progress = track_progress
        self.run_time: int | None = None

    @abstractmethod
    def _solve_cvrp(self) -> Any:
        """
        Solve the CVRP with a specific solver.
        """
        pass

    @abstractmethod
    def _convert_solution(self, result: Any, local_run_time: float) -> VRPSolution:
        """
        Convert the result from the solver to a CVRP solution.
        """
        pass

    def solve(self) -> VRPSolution:
        """
        Solve the CVRP.
        """

        result, execution_time = self.measure_time(self._solve_cvrp)
        return self._convert_solution(result, execution_time)

    @staticmethod
    def measure_time(
        fun: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> tuple[Any, int]:
        """
        Measure the execution time of a function.
        Returns the result and the execution time in microseconds.
        """

        start_time = time.perf_counter_ns()
        result = fun(*args, **kwargs)
        execution_time = (
            time.perf_counter_ns() - start_time
        ) // 1000  # Convert to microseconds

        return result, execution_time
