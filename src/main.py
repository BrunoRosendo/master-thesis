from src.solver.qubo.DWaveSolver import DWaveSolver

if __name__ == "__main__":
    cvrp = DWaveSolver(
        2,
        None,
        [
            (456, 320),
            (228, 0),
            (912, 0),
            (0, 80),
            (114, 80),
            (570, 160),
            (798, 160),
            (342, 240),
        ],
        [
            (2, 1, 6),
            (3, 0, 5),
            (4, 5, 8),
            (6, 7, 2),
        ],
        True,
        True,
        # sampler=DWaveSampler(),
        # num_reads=3500,
        # time_limit=10,
    )
    result = cvrp.solve()
    result.display()
