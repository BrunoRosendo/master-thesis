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
        1,
        None,
        [
            (456, 320),
            (228, 0),
        ],
        [(1, 3, 7)],
        False,
        False,
        warm_start=True,
    )
    result = cvrp.solve()
    result.display()
