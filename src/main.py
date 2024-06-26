from dotenv import load_dotenv
from dwave.system import DWaveSampler, LeapHybridCQMSampler
from ortools.constraint_solver import routing_enums_pb2

from src.model.VRPSolution import VRPSolution
from src.solver.ClassicSolver import ClassicSolver
from src.solver.cost_functions import euclidean_distance
from src.solver.qubo.DWaveSolver import DWaveSolver
from src.solver.qubo.QiskitSolver import QiskitSolver, get_backend_sampler

if __name__ == "__main__":
    cvrp = QiskitSolver(
        1,
        None,
        [(5, 5), (15, 10)],
        [],
        False,
    )
    result = cvrp.solve()
    result.save_json("vrp-n2-k1-qiskit2")
    # result = VRPSolution.from_json("rpp-n8-k2-cplex")
    result.display()
