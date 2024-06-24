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
        1,
        [12],
        [
            (456, 320),
            (400, 0)
        ],
        [
            (0, 1, 1),
        ],
        True,
        # local_search_metaheuristic=routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
        # solution_strategy=routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC,
        # time_limit_seconds=2,
        # classical_solver=True,
    )
    result = cvrp.solve()
    result.display()
