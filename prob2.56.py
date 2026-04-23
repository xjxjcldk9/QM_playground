import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

K_INITS = [3.0, 5.0, 6.999]
IC = [[0, 1], [1, 0], [0, 1]]
SLIDER_HALF_WIDTH = 0.001

T_SPAN = (0, 10)
T_EVAL = np.linspace(*T_SPAN, 500)


def solve(k, ic):
    return solve_ivp(lambda t, y: [y[1], (t**2 - k) * y[0]],
                     T_SPAN, ic, t_eval=T_EVAL)


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
plt.subplots_adjust(bottom=0.25, wspace=0.35)

lines = []
sliders = []

for i, (ax, k_init) in enumerate(zip(axes, K_INITS)):
    sol = solve(k_init, IC[i])
    line, = ax.plot(sol.t, sol.y[0], lw=1.5)
    ax.set_xlabel("t")
    ax.set_ylabel("y(t)")
    ax.set_title(f"K ≈ {k_init}")
    ax.set_ylim(-10, 10)
    ax.axhline(0, color='gray', lw=0.5)
    lines.append(line)

    left = 0.1 + i * 0.3
    ax_slider = plt.axes([left + 0.02, 0.08, 0.22, 0.04])
    slider = Slider(ax_slider, f"K{i+1}",
                    k_init - SLIDER_HALF_WIDTH,
                    k_init + SLIDER_HALF_WIDTH,
                    valinit=k_init)
    sliders.append(slider)


def make_update(idx, ax):
    def update(val):
        sol = solve(val, IC[idx])
        lines[idx].set_ydata(sol.y[0])
        ax.set_ylim(-10, 10)
        fig.canvas.draw_idle()
    return update


for i, ax in enumerate(axes):
    sliders[i].on_changed(make_update(i, ax))

fig.suptitle("Solution of y'' = (t² − K)·y", y=1.01)
plt.show()
