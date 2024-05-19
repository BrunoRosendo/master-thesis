from dotenv import load_dotenv

from src.solver.distance_functions import euclidean_distance
from src.solver.qubo.CplexSolver import CplexSolver

if __name__ == "__main__":
    load_dotenv()

    cvrp = CplexSolver(
        2,
        10,
        [
            (456, 320),
            (228, 0),
            (912, 0),
            (0, 80),
        ],
        [
            (0, 1, 5),
            (3, 2, 6),
        ],
        True,
        classical_solver=True,
        distance_function=euclidean_distance,
    )
    result = cvrp.solve()
    result.display()
