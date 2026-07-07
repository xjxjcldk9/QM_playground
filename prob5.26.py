import numpy as np
import matplotlib.pyplot as plt

x_neg = np.linspace(-20, -0.01, 2000)
x_pos = np.linspace(0.01, 20, 2000)

y_neg = np.cosh(x_neg) - 1.5 * np.sinh(x_neg) / x_neg
y_pos = np.cos(x_pos) - 1.5 * np.sin(x_pos) / x_pos

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(x_neg, y_neg, 'b', label=r'$\cosh(x) - 1.5\,\sinh(x)/x$')
ax.plot(x_pos, y_pos, 'r', label=r'$\cos(x) - 1.5\,\sin(x)/x$')

ax.axhline(1, color='gray', linestyle='--', linewidth=0.8)
ax.axhline(-1, color='gray', linestyle='--', linewidth=0.8)
ax.fill_between([-20, 20], -1, 1, color='yellow', alpha=0.2, label='allowed region $[-1, 1]$')

ax.axvline(0, color='k', linewidth=0.5)
ax.set_ylim(-3, 3)
ax.set_xlim(-20, 20)
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title('Prob 5.26')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('prob5.26.png', dpi=150)
plt.show()
print("Saved prob5.26.png")
