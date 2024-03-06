from ClassicCVRP import ClassicCVRP

if __name__ == "__main__":
    cvrp = ClassicCVRP(
        [15, 15, 15, 15],
        0,
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
            [1, 6, 1],
            [2, 10, 1],
            [4, 3, 1],
            [5, 9, 1],
            [7, 8, 1],
            [15, 11, 1],
            [13, 12, 1],
            [16, 14, 1],
        ],
    )
    cvrp.solve()