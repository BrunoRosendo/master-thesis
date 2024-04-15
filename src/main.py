from src.solver.ClassicSolver import ClassicSolver

if __name__ == "__main__":
    cvrp = ClassicSolver(
        2,
        None,
        [
            (456, 320),
            (228, 0),
            (912, 0),
            (0, 80),
            (114, 80),
            # (570, 160),
            # (798, 160),
            # (342, 240),
            # (684, 240),
            # (570, 400),
            # (912, 400),
            # (114, 480),
            # (228, 480),
            # (342, 560),
            # (684, 560),
            # (0, 640),
            # (798, 640),
        ],
        [
            (0, 1, 6),
            (2, 3, 5),
        ],
        True,
    )
    result = cvrp.solve()
    result.display()
