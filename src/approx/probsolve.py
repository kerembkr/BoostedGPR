!pip install probnum

import probnum as pn
import numpy as np
from probnum.problems.zoo.linalg import random_spd_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

rng = np.random.default_rng(42)  # for reproducibility
n = 25  # dimensionality

# generate linear system
spectrum = 10 * np.linspace(0.5, 1, n) ** 4
A = random_spd_matrix(rng=rng, dim=n, spectrum=spectrum)
b = rng.normal(size=(n, 1))

print("Matrix condition: {:.2f}".format(np.linalg.cond(A)))
print("Eigenvalues: {}".format(np.linalg.eigvalsh(A)))

# Plot linear system
fig, axes = plt.subplots(
    nrows=1,
    ncols=4,
    figsize=(5, 3.5),
    sharey=True,
    squeeze=False,
    gridspec_kw={"width_ratios": [4, 0.25, 0.25, 0.25]},
)

vmax = np.max(np.hstack([A, b]))
vmin = np.min(np.hstack([A, b]))

# normalize diverging colobar, such that it is centered at zero
norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

axes[0, 0].imshow(A, cmap="bwr", norm=norm)
axes[0, 0].set_title("$A$", fontsize=24)
axes[0, 1].text(0.5, A.shape[0] / 2, "$x$", va="center", ha="center", fontsize=32)
axes[0, 1].axis("off")
axes[0, 2].text(0.5, A.shape[0] / 2, "$=$", va="center", ha="center", fontsize=32)
axes[0, 2].axis("off")
axes[0, 3].imshow(b, cmap="bwr", norm=norm)
axes[0, 3].set_title("$b$", fontsize=24)
for ax in axes[0, :]:
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()


