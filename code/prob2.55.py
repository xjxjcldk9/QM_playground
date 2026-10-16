import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

K = 0.9

T_SPAN = (0, 10)
T_EVAL = np.linspace(*T_SPAN, 500)


def solve(k):
    return solve_ivp(lambda t, y: [y[1], (t**2 - k) * y[0]],
                     T_SPAN, [1, 0], t_eval=T_EVAL)


fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)

sol = solve(K)
line, = ax.plot(sol.t, sol.y[0], lw=1.5)
ax.set_xlabel("t")
ax.set_ylabel("y(t)")
ax.set_title("Solution of y'' = (t² − K)·y")
ax.set_ylim(-10, 10)
ax.axhline(0, color='gray', lw=0.5)

ax_slider = plt.axes([0.15, 0.05, 0.7, 0.04])
slider = Slider(ax_slider, "K", 0.9999, 1.0001, valinit=K)


def update(val):
    k = slider.val
    sol = solve(k)
    line.set_ydata(sol.y[0])
    ax.set_ylim(-10, 10)
    fig.canvas.draw_idle()


slider.on_changed(update)

plt.show()
