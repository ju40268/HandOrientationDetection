import numpy as np
import itertools
import pickle
import matplotlib.pyplot as plt
from matplotlib import pylab
import numpy.polynomial.polynomial as poly
from sklearn.cluster import KMeans
# for ignoring the warning of fitting the points using polynomial curves
import heapq
import warnings
warnings.simplefilter('ignore', np.RankWarning)
#----------

flag = False
# Flag == True, curve fitting
# Flag == False, calculate by hand

def kmeans_clustering(points): # return the center of all the clusters
   kmeans = KMeans(n_clusters=4, max_iter=300, tol=0.01).fit(points)
   centroids = kmeans.cluster_centers_
   labels = kmeans.labels_
   colors = ["g.", "r.", "b.", "y.", "m."]
   for i in range(len(labels)):
       # plotting all the possible points
       plt.plot(points[i][0], points[i][1], colors[labels[i]], markersize=10)
       # plotting the center
       plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=150, linewidths=5, zorder=10)

   # plotting the center
   plt.scatter(14, 13, marker="x", s=150, linewidths=5, zorder=10)
   plt.annotate('center', xy=(14,13), color='red', fontsize=16)
   # plotting the label # --> belongs to what clusters
   for i in range(len(centroids)):
       plt.annotate(str(tuple(centroids[i])), xy=tuple(centroids[i]), fontsize=10)
   #plt.annotate('center', xy=(14, 13), xytext=(14, 14),
   #            arrowprops=dict(facecolor='red', shrink=0.05))

   # save image for further check
   plt.savefig('Cluster_output.jpg')
   # plt.show()
   return centroids

def angle_between_clockwise(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))

def angle_between(v1, v2):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def check_index(index):
    # only for simply checking cluster
    for i in index:
        return {
            0: 'green',
            1: 'red',
            2: 'blue',
            3: 'yellow',
            4: 'purple',
        }[i]

def calculate_vector(kmeans_centroids, num_cluster, touchpad_center=[14,13]):

    dist = []
    degree = []
    compare = []
    print(kmeans_centroids)
    relative_vector = list(kmeans_centroids - touchpad_center)
    print(relative_vector)
    all_comb = list(itertools.combinations(range(num_cluster), 2))
    for i, j in all_comb:
        print(i,j)
        degree.append(np.degrees(angle_between(relative_vector[i], relative_vector[j])))
    # print all_comb[degree.index(max(degree))]
    # ----------- might have error------------------
    print(degree)
    index = sorted(list(all_comb[degree.index(max(degree))]), reverse=True)
    check_index(index)
    positive_x_axis = (1, 0)
    for i in index:
        compare.append(angle_between_clockwise(relative_vector[i], positive_x_axis))
    # TODO: decide which mode for 4 fingers(w/ picky or w/ thumb)

    # decide which is the right finger
    right = kmeans_centroids[index[1]] if compare[0] > compare[1] else kmeans_centroids[index[0]]

    # https://stackoverflow.com/questions/10062954/valueerror-the-truth-value-of-an-array-with-more-than-one-element-is-ambiguous
    # for i != right
    for i in kmeans_centroids:
        # if (i - right).any():
        #     dist.append(np.linalg.norm(right - i))
        dist.append(np.linalg.norm(right - i))


    # https://stackoverflow.com/questions/2474015/getting-the-index-of-the-returned-max-or-min-item-using-max-min-on-a-list
    # nearest = dist.index(min(dist))
    # https://stackoverflow.com/questions/19697504/second-largest-number-in-list-python

    # pick up the second min number for nearest point
    min_dist = min(dist)
    # nearest = dist.index(min(i for i in dist if i != min_dist))
    nearest_xy = kmeans_centroids[dist.index(min(i for i in dist if i != min_dist))]

    # calculating the final middle finger direction and the according angle
    middle_finger_vec = relative_vector[dist.index(min(i for i in dist if i != min_dist))]
    final_angle = angle_between_clockwise(middle_finger_vec, positive_x_axis)


    #------checking--------------------
    print('######## CHECKING BLOCK ######### ')
    print('#### index', index)
    print('#### dist', dist)
    print('#### nearest_xy', nearest_xy)
    print('#### right', right)
    print('#### final finger vec: ', middle_finger_vec)
    print('#### final angle: ', final_angle)

    return nearest_xy


# for parameter passing?
# error for SyntaxError: non-default argument follows default argument
# https://stackoverflow.com/questions/16932825/why-non-default-arguments-cant-follows-default-argument

# draw line in plt
# https://stackoverflow.com/questions/36470343/how-to-draw-a-line-with-matplotlib

def plot_result(centroids, nearest_xy, touchpad_center=[14,13]):
    plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=150, linewidths=5, zorder=10)
    plt.scatter(touchpad_center[0], touchpad_center[1], marker="x", s=150, linewidths=5, zorder=10)
    # plt.annotate('center', xy=tuple(touchpad_center), color='red', fontsize=16)
    plt.plot(touchpad_center, nearest_xy,  marker='*')
    plt.savefig('with_line_final_degree.jpg')
    # plt.show()



def fitting_poly(points):
    # get x and y vectors
    x = points[:, 0]
    y = points[:, 1]
    # calculate polynomial and fit every point on it.
    coeffs = np.polyfit(x, y, 4)
    ffit = np.poly1d(coeffs)
    # set up display area, draw
    xp = np.linspace(0, 30, 100)
    plt.plot(x, y, '.', xp, ffit(xp), '-')
    plt.show()
    # return curve
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
    if len(x_max) == 2:
        # not sure why x_max return greater length sometimes?
        print('x length greater than 1')
        return [x_max[1], y_max[1]]
    else:
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


#---------------------------------------------------------------------------------
# ------ main program start here. --------------------------


# ------ traverse all the points(img) ----------------------
# for i in points_list:
#     points = kmeans_clustering(i)
#     if flag == True:
#         print('##### In the curve fitting mode. calculate critical points.####')
#         ffit = fitting_poly(points)
#         crit_pts = calculate_critical_points(ffit)
#         print(crit_pts)
#         calculate_final_angle(crit_pts)
#     else:
#         print('##### Not in the fitting mode. calculate by hands. ####')


# ------ traverse single point(img) ----------------------



if __name__ == "__main__":
    points_list = load_points()
    points = kmeans_clustering(points_list[2])
    if flag == True:
        print('##### In the curve fitting mode. calculate critical points.####')
        ffit = fitting_poly(points)
        crit_pts = calculate_critical_points(ffit)
        print(crit_pts)
        calculate_final_angle(crit_pts)
    else:
        print('##### Not in the fitting mode. calculate by hands. ####')
        nearest_xy = calculate_vector(points, num_cluster=4, touchpad_center=[14, 13])
        plot_result(points,nearest_xy)