import numpy as np
import itertools
import pickle
import matplotlib.pyplot as plt
from matplotlib import pylab
import numpy.polynomial.polynomial as poly
from sklearn.cluster import KMeans
# for ignoring the warning of fitting the points using polynomial curves
import warnings
warnings.simplefilter('ignore', np.RankWarning)
#----------

flag = True

def kmeans_clustering(points):
   kmeans = KMeans(n_clusters=4, max_iter=300, tol=0.01).fit(points)
   centroids = kmeans.cluster_centers_
   labels = kmeans.labels_
   colors = ["g.", "r.", "b.", "y.", "m."]

   for i in range(len(labels)):
       plt.plot(points[i][0], points[i][1], colors[labels[i]], markersize=10)
       plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=150, linewidths=5, zorder=10)
       #print 'centroids, x,y coordinates: ', centroids
   plt.show()
   return centroids

def angle_between_clockwise(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))

def angle_between(v1, v2):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def calculate_vector(kmeans_centroids, num_cluster, touchpad_center):
    degree = []
    relative_vector = list(kmeans_centroids - touchpad_center)
    all_comb = list(itertools.combinations(range(num_cluster), 2))
    for i, j in all_comb:
        degree.append(np.degrees(angle_between(relative_vector[i], relative_vector[j])))
    # print all_comb[degree.index(max(degree))]
    # ----------- might have error------------------
    index = sorted(list(all_comb[degree.index(max(degree))]), reverse=True)

    for i in index:
        del relative_vector[i]
    final_direction = sum(relative_vector) / 3
    positive_x_axis = (1, 0)
    final_angle = angle_between(positive_x_axis, final_direction)
    return np.degrees(final_angle)


def fitting_poly(points):
    # get x and y vectors
    x = points[:, 0]
    y = points[:, 1]
    # calculate polynomial
    coeffs = np.polyfit(x, y, 4)
    ffit = np.poly1d(coeffs)
    xp = np.linspace(0, 30, 100)
    plt.plot(x, y, '.', xp, ffit(xp), '-')
    plt.show()

    return ffit

def calculate_critical_points(ffit):
    crit = ffit.deriv().r
    r_crit = crit[crit.imag == 0].real
    test = ffit.deriv(2)(r_crit)
    x_max = r_crit[test < 0]
    y_max = ffit(x_max)
    print('x_max', x_max)
    print('y_max', y_max)
    # plt.plot(x_max, y_max, 'r.', 14,13, 'g.')
    plt.scatter(x_max, y_max, marker="x", s=150, linewidths=5, zorder=10, color='r')
    plt.scatter(14, 13, marker="x", s=150, linewidths=5, zorder=10, color='g')
    xc = np.arange(0, 30, 0.02)
    yc = ffit(xc)
    plt.plot(xc, yc)
    plt.xlim([0, 30])
    plt.show()
    return [x_max[0], y_max[0]]

def calculate_final_angle(crit_pts, touchpad_center=[14, 13]):
    vec = np.array(crit_pts) - np.array(touchpad_center)
    print('#### final vec: ', vec)
    positive_x_axis = (1, 0)
    final_angle = angle_between(positive_x_axis, vec)
    print('#### final angle: ', np.degrees(final_angle))


def load_points():
    with open('points.pickle', 'rb') as f:
        points_list = pickle.load(f)
    return points_list

points_list = load_points()
for i in points_list:
    points = kmeans_clustering(i)
    if flag == True:
        ffit = fitting_poly(points)
        crit_pts = calculate_critical_points(ffit)
        print(crit_pts)
        calculate_final_angle(crit_pts)
    else:
        print('##### Not in the fitting mode. calculate by hands. ####')


# points = kmeans_clustering(points_list[2])
# if flag == True:
#     ffit = fitting_poly(points)
#     crit_pts = calculate_critical_points(ffit)
#     print(crit_pts)
#     calculate_final_angle(crit_pts)
# else:
#     print('##### Not in the fitting mode. calculate by hands. ####')