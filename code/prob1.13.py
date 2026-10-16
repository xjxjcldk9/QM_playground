import matplotlib.pyplot as plt
import numpy as np


t_domain = np.random.rand(10000) * 2 * np.pi
x_domain = np.linspace(-0.99, 0.99, 5000)

real_density = 1 / np.pi / np.sqrt(1 - x_domain**2)


plt.figure(figsize=(10, 6))

# The Histogram of samples
# density=True is CRITICAL to match the scale of the PDF
plt.hist(np.cos(t_domain), bins=40, density=True, alpha=0.6,
         color='skyblue', label='Sample Frequency')

# The Real Density Function
plt.plot(x_domain, real_density, 'r-', linewidth=2, label='Real PDF')

plt.title("Frequency Plot: Random Samples vs. Real Density")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.show()
