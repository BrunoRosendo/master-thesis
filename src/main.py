from src.solver.QuboSolver import QuboSolver

if __name__ == "__main__":
    # cvrp = QuboSolver(
    #     1,
    #     None,
    #     [
    #         (456, 320),
    #         (228, 0),
    #         (912, 0),
    #     ],
    #     [(1, 2, 5)],
    #     False,
    #     warm_start=True,
    # )
    cvrp = QuboSolver(
        2,
        10,
        [
            (456, 320),
            (228, 0),
            (912, 0),
            (0, 80),
            (114, 80),
        ],
        [
            (0, 3, 10),
            (2, 1, 6),
            (3, 4, 4),
        ],
        True,
        True,
    )
    result = cvrp.solve()
    result.display()
