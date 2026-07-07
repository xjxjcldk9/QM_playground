import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox

N_INITS = [1.0, 2.0, 3.0]
X_SPAN = (1e-6, 30)
X_EVAL = np.linspace(*X_SPAN, 2000)
EPS = X_SPAN[0]

L_VALUES = [0, 1, 2]
COLORS = ["tab:blue", "tab:orange", "tab:green"]


def ic(l):
    if l == 0:
        return [EPS, 1.0]
    return [EPS ** (l + 1), (l + 1) * EPS**l]


def solve(n, l):
    def rhs(x, y):
        u, up = y
        return [up, (1 - 2 * n / x + l * (l + 1) / x**2) * u]

    return solve_ivp(rhs, X_SPAN, ic(l), t_eval=X_EVAL, max_step=0.05)


fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
plt.subplots_adjust(bottom=0.25, hspace=0.4)

lines = []
titles = []
for ax, l, color, n0 in zip(axes, L_VALUES, COLORS, N_INITS):
    sol = solve(n0, l)
    (line,) = ax.plot(sol.t, sol.y[0], lw=1.5, color=color)
    lines.append(line)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_ylim(-2, 2)
    ax.set_ylabel(f"u (l={l})")
    title = ax.set_title(f"l = {l},  n = {n0:.7g}", fontsize=10)
    titles.append(title)

axes[-1].set_xlabel("x")
fig.suptitle(r"$u'' = \left(1 - \dfrac{2n}{x} + \dfrac{l(l+1)}{x^2}\right)u$", fontsize=13)

sliders = []
textboxes = []

for i, (l, color, n0) in enumerate(zip(L_VALUES, COLORS, N_INITS)):
    row_y = 0.16 - i * 0.06
    # Slider occupies left 75% of the row
    ax_s = plt.axes([0.12, row_y, 0.62, 0.03])
    s = Slider(ax_s, f"l={l}", 0.5, 5.0, valinit=n0, color=color)
    s.valtext.set_visible(False)
    sliders.append(s)
    # TextBox occupies the remaining right portion
    ax_t = plt.axes([0.78, row_y, 0.13, 0.03])
    tb = TextBox(ax_t, "", initial=f"{n0:.7g}")
    textboxes.append(tb)


def redraw(idx, n):
    l = L_VALUES[idx]
    sol = solve(n, l)
    lines[idx].set_ydata(sol.y[0])
    titles[idx].set_text(f"l = {l},  n = {n:.7g}")
    fig.canvas.draw_idle()


def make_slider_update(idx):
    def update(val):
        n = sliders[idx].val
        textboxes[idx].set_val(f"{n:.7g}")
        redraw(idx, n)
    return update


def make_text_submit(idx):
    def submit(text):
        try:
            n = float(text)
        except ValueError:
            return
        n = max(0.5, min(5.0, n))
        sliders[idx].set_val(n)  # also triggers slider callback
    return submit


for i in range(len(L_VALUES)):
    sliders[i].on_changed(make_slider_update(i))
    textboxes[i].on_submit(make_text_submit(i))

plt.show()
