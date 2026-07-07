import numpy as np
import matplotlib.pyplot as plt

E = np.linspace(0.001, 50, 10000)
z = np.sqrt(2 * E)
rhs = np.cos(z) + np.sin(z) / z

# allowed bands: |rhs| <= 1
allowed = np.abs(rhs) <= 1
q = np.full_like(E, np.nan)
q[allowed] = np.arccos(rhs[allowed])  # q in [0, pi]

fig, ax = plt.subplots(figsize=(10, 5))

# plot both +q and -q for full Brillouin zone
ax.plot(E,  q, 'b', linewidth=1)
ax.plot(E, -q, 'b', linewidth=1)

ax.set_xlabel('E')
ax.set_ylabel('q')
ax.set_title('Prob 5.28: Band structure\n'
             r'$\cos(q) = \cos(\sqrt{2E}) + \sin(\sqrt{2E})/\sqrt{2E}$')
ax.set_ylim(-np.pi, np.pi)
ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
ax.axhline(np.pi,  color='gray', linestyle='--', linewidth=0.7)
ax.axhline(-np.pi, color='gray', linestyle='--', linewidth=0.7)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('prob5.28.png', dpi=150)
plt.show()
print("Saved prob5.28.png")
