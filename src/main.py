from dotenv import load_dotenv
from dwave.system import DWaveSampler, LeapHybridCQMSampler

from src.model.VRPSolution import VRPSolution
from src.solver.ClassicSolver import ClassicSolver
from src.solver.cost_functions import euclidean_distance
from src.solver.qubo.DWaveSolver import DWaveSolver
from src.solver.qubo.QiskitSolver import QiskitSolver, get_backend_sampler

if __name__ == "__main__":
    cvrp = ClassicSolver(
        8,
        35,
        [
            (30, 40),
            (37, 52),
            (49, 49),
            (52, 64),
            (31, 62),
            (52, 33),
            (42, 41),
            (52, 41),
            (57, 58),
            (62, 42),
            (42, 57),
            (27, 68),
            (43, 67),
            (58, 48),
            (58, 27),
            (37, 69)
        ],
        [
            (1, 0, 19),
            (2, 0, 30),
            (3, 0, 16),
            (4, 0, 23),
            (5, 0, 11),
            (6, 0, 31),
            (7, 0, 15),
            (8, 0, 28),
            (9, 0, 8),
            (10, 0, 8),
            (11, 0, 7),
            (12, 0, 14),
            (13, 0, 6),
            (14, 0, 19),
            (15, 0, 11)
        ],
        False,
        cost_function=euclidean_distance,
        # classical_solver=True,
    )
    result = cvrp.solve()
    # result = VRPSolution.from_json("300+302-3v")
    # result.display()
    # result.save_json("tmp")
    print(result.objective)
# [64, 62, 87, 55, 68, 42, 60, 24]