import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from scipy.integrate import solve_ivp

K_INITS = [2.5, 9.8, 22.19, 39.4]
SLIDER_HALF_WIDTH = 0.1
IC = [[1, 0], [0, 1], [1, 0], [0, 1]]

T_SPAN = (0, 1)
T_EVAL = np.linspace(*T_SPAN, 500)


def solve(k, ic):
    return solve_ivp(lambda _, y: [y[1], -k * y[0]],
                     T_SPAN, ic, t_eval=T_EVAL)


fig, axes_grid = plt.subplots(2, 2, figsize=(12, 9))
plt.subplots_adjust(hspace=0.45, wspace=0.35, bottom=0.08)
axes = axes_grid.flatten()

lines = []
sliders = []

SLIDER_POSITIONS = [
    [0.1,  0.5,  0.35, 0.03],
    [0.55, 0.5,  0.35, 0.03],
    [0.1,  0.03, 0.35, 0.03],
    [0.55, 0.03, 0.35, 0.03],
]

for i, (ax, k_init, ic) in enumerate(zip(axes, K_INITS, IC)):
    sol = solve(k_init, ic)
    line, = ax.plot(sol.t, sol.y[0], lw=1.5)
    ax.set_xlabel("t")
    ax.set_ylabel("y(t)")
    ax.set_title(f"K ≈ {k_init},  IC = {ic},  y(1) = {sol.y[0][-1]:.8f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 1)
    ax.axhline(0, color='gray', lw=0.5)
    lines.append(line)

    ax_slider = plt.axes(SLIDER_POSITIONS[i])
    slider = Slider(ax_slider, f"K{i+1}",
                    k_init - SLIDER_HALF_WIDTH,
                    k_init + SLIDER_HALF_WIDTH,
                    valinit=k_init)
    sliders.append(slider)


def make_update(idx, ax):
    def update(val):
        sol = solve(val, IC[idx])
        lines[idx].set_ydata(sol.y[0])
        ax.set_ylim(-1, 1)
        ax.set_title(
            f"K ≈ {val:.4f},  IC = {IC[idx]},  y(1) = {sol.y[0][-1]:.4f}")
        fig.canvas.draw_idle()
    return update


for i, ax in enumerate(axes):
    sliders[i].on_changed(make_update(i, ax))

fig.suptitle("Solution of y'' = −K·y")
plt.show()
