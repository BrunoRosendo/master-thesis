import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

# Data points
x = np.array([100, 250, 400, 550, 700, 850, 1000])
y = np.array([0, 2, 2, 3, 5, 6, 7])

# Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress(x, y)

# Line of best fit
line = slope * x + intercept

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color="blue")
plt.plot(x, line, color="red", label="Best Fit Line")
plt.title("Number of Reads vs BKS Frequency")
plt.xlabel("Number of Reads")
plt.ylabel("Number of Times BKS was achieved")
plt.xticks(x)  # Set x-axis to show the point values
plt.ylim(0, 10)  # Set y-axis to go up to 10
plt.legend()
plt.grid(True)
plt.show()
