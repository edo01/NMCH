import matplotlib.pyplot as plt


euler_times = []
exact_times = []
with open("times.txt", "r") as file:
    for line in file:
        euler_time, exact_time = map(float, line.strip().split(","))
        euler_times.append(euler_time)
        exact_times.append(exact_time)


plt.plot(euler_times, label="Euler Scheme")
plt.plot(exact_times, label="Exact Simulation")
plt.xlabel("Parameter Set Index")
plt.ylabel("Execution Time (milliseconds)")
plt.title("Execution Time Comparison: Euler Scheme vs Exact Simulation")
plt.legend()
plt.show()
