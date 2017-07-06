import preprocess
import file_operation
import pickle
import os
from sklearn import preprocessing
from sklearn.cluster import KMeans
import itertools
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import csv
from math import acos
from math import sqrt
from math import pi

#--------------------------------------------------------------------------------
#for representing the final angle around [0-350 counter-clock wise]
def length(v):
    return sqrt(v[0]**2+v[1]**2)
def dot_product(v,w):
   return v[0]*w[0]+v[1]*w[1]
def determinant(v,w):
   return v[0]*w[1]-v[1]*w[0]
def inner_angle(v,w):
   cosx=dot_product(v,w)/(length(v)*length(w))
   rad=acos(cosx) # in radians
   return rad*180/pi # returns degrees
def angle_counter_clockwise(A, B):
    inner=inner_angle(A,B)
    det = determinant(A,B)
    if det<0: #this is a property of the det. If the det < 0 then B is clockwise of A
        return 360-inner
    else: # if the det > 0 then A is immediately clockwise of B
        return inner
#---------------------------------------------------------------------------------
def angle_between(v1, v2):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
#---------------------------------------------------------------------------------
def calculate_vector(kmeans_centroids, num_cluster, touchpad_center):
    degree = []
    degree_counter = []
    relative_vector = []
    temp_vector = list(kmeans_centroids - touchpad_center)
    #normalize all vector to unit vector
    for v in temp_vector:
        relative_vector.append(v / np.linalg.norm(v))
    all_comb = list(itertools.combinations(range(num_cluster), 2))
    for i, j in all_comb:
        degree.append(np.degrees(angle_between(relative_vector[i], relative_vector[j])))
    # print all_comb[degree.index(max(degree))]
    # ----------- might have error------------------
    index = sorted(list(all_comb[degree.index(max(degree))]), reverse=True)

    for i in index:
        del relative_vector[i]
    #print('relative vector: ')
    #print(relative_vector)
    final_direction = sum(relative_vector) / 3
    positive_x_axis = (1, 0)
    final_angle = angle_counter_clockwise(positive_x_axis, final_direction)
    print(int(final_angle))
    return np.degrees(final_angle)

#---------------------------------------------------------------------------------
def write_angle(filename, slice_timestamp, index_list, angle_list):
    pair_dir = file_operation.data_output('pair')
    if not os.path.exists(pair_dir):
        print('No timestamp angle pair directory, now creating one.')
        os.makedirs(pair_dir)
    _, tail = os.path.split(filename)
    # extract the .txt extend file name
    f = open(pair_dir + tail[:-4] + "_timestamp_angle_pair.csv", "w")
    w = csv.writer(f)
    #repitition_angle = []
    # for i in angle_list:
    #     repitition_angle.append( [str(i)] * 162 + ['write command']*2)

    # angle_list = list(itertools.chain.from_iterable(slice_timestamp))
    # print angle_list
    [identity, timestamp] = zip(*slice_timestamp)
    #print 'identity', identity
    #timestamp_angle = zip(identity, timestamp, frame_list)
    #print timestamp_angle
    timestamp_angle = zip(index_list, identity, timestamp, angle_list)
    # print timestamp_angle
    w.writerows(timestamp_angle)
    # print timestamp_angle
    f.close()
#------------------------------------------------------------------
def get_timestamp(header,time_stamp):
    #print 'escape starting point'
    #slice_time_stamp = time_stamp.iloc[starting_point+1:ending_point]
    #print slice_time_stamp
    #return slice_time_stamp.values.tolist()
    #print 'time stamp: ', time_stamp
    #print 'header: ', header
    #header = [x + 40 for x in header]
    # TODO: why some timestamp is shifted? --> then cause duplicated timestamp
    sliced = time_stamp.iloc[header].values.tolist()
    return sliced

def save_pickle(obj):
    pickle_dir = file_operation.data_output('pickle')
    if not os.path.exists(pickle_dir):
        print('No reference pickle output directory, now creating one.')
        os.makedirs(pickle_dir)
    with open('ref.pickle', 'w') as f:  # Python 3: open(..., 'wb')
        pickle.dump(obj, f)

def load_ref(obj='ref.pickle'):
    with open(obj) as f:  # Python 3: open(..., 'rb')
        ref_lsit = pickle.load(f)
        # print ref_lsit
    return  ref_lsit

def gradient():
    # https://math.stackexchange.com/questions/1394455/how-do-i-compute-the-gradient-vector-of-pixels-in-an-image
    #worth a try! calculate the gradient for the img
    print('calculating the gradient for the image...')

# -----------------------------------------------------------------
# TODO: find better threshold for each
def threshold(threshold_output):
    label_x, label_y = np.where(threshold_output == 0)
    points = list(zip(label_y, 23 - label_x))
    # the transformation of coordinate from upper-left to lower-left
    # plt.scatter(label_y, 23 - label_x)
    # plt.show()
    # plt.imshow(threshold_output, interpolation = 'nearest')
    # plt.show()
    return points

# -----------------------------------------------------------------
# TODO: not decide if the total # of cluster is not enough
def kmeans_clustering(points):
    # kmeans = KMeans(n_clusters=5, max_iter=300, tol=0.01).fit(points)
    kmeans = KMeans(n_clusters=5).fit(points)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    '''
    colors = ["g.", "r.", "b.", "y.", "m."]
    for i in range(len(labels)):
        plt.plot(points[i][0], points[i][1], colors[labels[i]], markersize=10)
        plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=150, linewidths=5, zorder=10)
        plt.show()
        print 'centroids, x,y coordinates: ', centroids
    '''
    return centroids


def save_img(img, index, filename):
    # print 'saving img ' + str(index)
    filtered_dir = file_operation.data_output('filtered')
    if not os.path.exists(filtered_dir):
        print('No filtered img directory, now creating one.')
        os.makedirs(filtered_dir)
    _, tail = os.path.split(filename)
    # extract the .txt extend file name
    plt.imsave(filtered_dir + tail[:-4] + '_' + str(index) + '.png', img, cmap=plt.cm.GnBu)

#-----------------------------------------------------------------------------
def determine_lift(index, blurred_img):
    if np.std(blurred_img) < 0.1: # threshold not yet finalized
        print('frame#', index, 'hands lifted')
        return False
    else:
        return True

#------------------------------------------------------------------------------
def img_processing(img_list, filename):
    angle_list = []
    index_list = []
    for i in range(len(img_list)):
        blurred_img = ndimage.gaussian_filter(img_list[i], sigma=0.8)
        filtered = [0 if x < np.mean(blurred_img) - np.std(blurred_img) else 1 for x in blurred_img]
        # save_img(np.transpose(np.flipud(blurred_img)).reshape(23, 28), i, filename='filtered_____')
        # save_img(np.transpose(np.flipud(filtered)).reshape(23, 28), i, filename='binary_____')
        # points = threshold(np.transpose(np.flipud(filtered)).reshape(23, 28))
        index_list.append(i)
        if determine_lift(i, blurred_img):
            save_img(np.transpose(np.flipud(blurred_img)).reshape(23, 28),i,filename)
            points = threshold(np.transpose(np.flipud(filtered)).reshape(23, 28))
            kmeans_centroids = kmeans_clustering(points)
            final_angle = calculate_vector(kmeans_centroids, 5, touchpad_center=[14,13])
            angle_list.append(final_angle)
        else:
            angle_list.append('hand lifted')
    return index_list, angle_list

def detect_online(filename,img_list, data_numeric, num_frame, header, time_stamp):
    print('in detect')
    index_list, angle_list = img_processing(img_list,filename)
    slice_timestamp = get_timestamp(header, time_stamp)
    write_angle(filename, slice_timestamp, index_list, angle_list)