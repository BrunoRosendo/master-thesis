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
        6,
        15,
        [
            (46, 32),
            (23, 0),
            (91, 0),
            (0, 8),
            (11, 8),
            (57, 16),
            (80, 16),
            (34, 24),
            (69, 24),
            (57, 40),
            (91, 40),
            (11, 48),
            (23, 48),
            (34, 56),
            (68, 56),
            (0, 64),
            (80, 64),
            (46, 72),
            (0, 80),
            (91, 80),
            (23, 88),
            (68, 88),
            (34, 96),
            (80, 96),
            (11, 104),
            (57, 104),
        ],
        [
            (1, 6, 5),
            (2, 10, 6),
            (4, 3, 4),
            (0, 9, 2),
            (7, 8, 7),
            (15, 11, 4),
            (13, 12, 6),
            (5, 14, 4),
            (16, 17, 7),
            (19, 18, 4),
            (20, 21, 6),
            (22, 23, 7),
            (24, 25, 3),
        ],
        True,
        local_search_metaheuristic=routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
        time_limit_seconds=1,
        solution_strategy=routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION,
        # classical_solver=True,
    )
    # result = cvrp.solve()
    # result.save_json("rpp-n26-k6-or2")
    result = VRPSolution.from_json("rpp-n26-k6-or")
    result.display()
