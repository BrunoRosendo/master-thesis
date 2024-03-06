from ClassicCVRP import ClassicCVRP

if __name__ == "__main__":
    cvrp = ClassicCVRP(
        [15, 15, 15, 15], 0, [], [0, 1, 1, 2, 4, 2, 4, 8, 8, 1, 2, 1, 2, 4, 4, 8, 8]
    )
    cvrp.solve()
