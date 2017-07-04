import numpy as np

def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))

v = [ 1.35021589, 8.08326303]
r = [1, 0]
print(angle_between(v,r))