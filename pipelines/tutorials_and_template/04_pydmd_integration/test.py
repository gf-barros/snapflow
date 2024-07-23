
from pydmd import DMD
from pydmd.plotter import plot_summary

# Build an exact DMD model with 12 spatiotemporal modes.
dmd = DMD(svd_rank=12)

# Fit the DMD model.
# X = (n, m) numpy array of time-varying snapshot data.
dmd.fit(X)

# Plot a summary of the key spatiotemporal modes.
plot_summary(dmd)