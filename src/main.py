from dotenv import load_dotenv
from dwave.system import DWaveSampler, LeapHybridCQMSampler
from ortools.constraint_solver import routing_enums_pb2

from src.model.VRPSolution import VRPSolution
from src.solver.ClassicSolver import ClassicSolver
from src.solver.cost_functions import euclidean_distance
from src.solver.qubo.DWaveSolver import DWaveSolver
from src.solver.qubo.QiskitSolver import QiskitSolver, get_backend_sampler

if __name__ == "__main__":
    cvrp = ClassicSolver(
        2,
        [12, 10],
        [
            (46, 32),
            (40, 0),
            (91, 0),
            (0, 8),
            (57, 40),
            (91, 40),
            (11, 48),
            (23, 48),
        ],
        [
            (1, 6, 5),
            (2, 7, 6),
            (4, 3, 4),
            (5, 0, 2),
            (5, 1, 3)
        ],
        True,
        # local_search_metaheuristic=routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
        # time_limit_seconds=2,
        # solution_strategy=routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION,
        # classical_solver=True,
    )
    result = cvrp.solve()
    result.display()
