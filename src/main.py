from dotenv import load_dotenv

from src.solver.qubo.CplexSolver import CplexSolver

if __name__ == "__main__":
    load_dotenv()

    cvrp = CplexSolver(
        4,
        10,
        [
            (456, 320),
            (228, 0),
            (912, 0),
            (0, 80),
            (114, 80),
            (570, 160),
            (798, 160),
            (342, 240),
            (684, 240),
            (570, 400),
            (912, 400),
            (114, 480),
            (228, 480),
            (342, 560),
            (684, 560),
            (0, 640),
            (798, 640),
        ],
        [
            (1, 6, 5),
            (2, 10, 6),
            (4, 3, 4),
            (5, 9, 2),
            (7, 8, 7),
            (15, 11, 4),
            (13, 12, 6),
            (16, 14, 4),
        ],
        True,
        classical_solver=True,
    )
    result = cvrp.solve()
    result.display()
