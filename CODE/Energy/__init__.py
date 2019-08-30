from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import numpy as np

s1 = np.array([0., 0, 1, 2, 1, 0, 1, 0, 0, 2, 1, 0, 0])
s2 = np.array([0., 1, 2, 3, 1, 0, 0, 0, 2, 1, 0, 0, 0])
distance, path = dtw.warping_paths(s1, s2)
print(distance)
#dtwvis.plot_warping(s1, s2, path, filename="warp.png")
