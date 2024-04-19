from dotenv import load_dotenv

from src.solver.ClassicSolver import ClassicSolver

if __name__ == "__main__":
    load_dotenv()

    cvrp = ClassicSolver(
        1,
        None,
        [
            (456, 320),
            (228, 0),
        ],
        [
            (0, 1, 6),
        ],
        True,
    )
    result = cvrp.solve()
    result.print()
