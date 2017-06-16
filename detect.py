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
#---------------------------------------------------------------------------------
def angle_between(v1, v2):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
#---------------------------------------------------------------------------------
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

#---------------------------------------------------------------------------------
def write_angle(filename, slice_timestamp, index_list, angle_list):
    pair_dir = file_operation.data_output('pair')
    if not os.path.exists(pair_dir):
        print 'No timestamp angle pair directory, now creating one.'
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
        print 'No reference pickle output directory, now creating one.'
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
    print 'calculating the gradient for the image...'

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

def threshold2(threshold_output): # for palm-on cases
    label_x, label_y = np.where(threshold_output < -0.5)
    points = list(zip(label_y, 23 - label_x))
    # the transformation of coordinate from upper-left to lower-left
    #plt.scatter(label_y, 23 - label_x)
    #plt.show()
    #plt.imshow(threshold_output, interpolation = 'nearest')
    #plt.show()
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
    img_dir = file_operation.data_output('img')
    if not os.path.exists(img_dir):
        print 'No contour img directory, now creating one.'
        os.makedirs(img_dir)
    _, tail = os.path.split(filename)
    # extract the .txt extend file name
    plt.imsave(img_dir + tail[:-4] + '_' + str(index) + '.jpg', img, cmap=plt.cm.GnBu)

#-----------------------------------------------------------------------------
def determine_lift(index, blurred_img):
    if np.std(blurred_img) < 0.1: # threshold not yet finalized
        print 'frame#', index, 'hands lifted'
        return False
    else:
        return True

#------------------------------------------------------------------------------
def distance(point1, point2):
    return np.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)
#------------------------------------------------------------------------------
def most(img, box_size):
    max_x = 0
    max_y = 0
    mmax = 0
    for x in range(img.shape[0]-box_size[0]):
        for y in range(img.shape[1]-box_size[1]):
            if np.sum(img[x:x+box_size[0],y:y+box_size[1]])> mmax:
                mmax = np.sum(img[x:x+box_size[0],y:y+box_size[1]])
                max_x = x+int(box_size[0]/2)+1
                max_y = y+int(box_size[1]/2)+1
    return np.array([max_x, max_y])
#------------------------------------------------------------------------------
def remove_palm(img_ori):
    scale = 4 # resize scale
    img =ndimage.zoom(img_ori, scale, order=3) # interpolation: cubic
    img = ndimage.grey_dilation(img, size=(int(scale/2), int(scale/2)))
    img = (img < np.mean(img))*1

    box_size = [int((img.shape[0]+img.shape[1])/8),int((img.shape[0]+img.shape[1])/8)] # to be determined
    palmrt = most(img, box_size) # right-top corner of palm
    img[palmrt[0]-int(box_size[0]/2)+1:palmrt[0]+int(box_size[0]/2), palmrt[1]-int(box_size[1]/2)+1:palmrt[1]+int(box_size[1]/2)] = 0
    palmlb = most(img, box_size) # left-bottom corner of palm
    # ----brute force for checking----
    if distance(palmrt,palmlb) > 12*scale: # would be wrong if too far away, try again
        img[palmlb[0]-int(box_size[0]/2)+1:palmlb[0]+int(box_size[0]/2), palmlb[1]-int(box_size[1]/2)+1:palmlb[1]+int(box_size[1]/2)] = 0
        palmlb = most(img, box_size)
        if distance(palmrt,palmlb) > 12*scale: # would be wrong if too far away, try again
            img[palmlb[0]-int(box_size[0]/2)+1:palmlb[0]+int(box_size[0]/2), palmlb[1]-int(box_size[1]/2)+1:palmlb[1]+int(box_size[1]/2)] = 0
            palmlb = most(img, box_size)
    #----------------------------------
    palmrt = palmrt/scale
    palmlb = palmlb/scale	
    center = ( palmrt*3 + palmlb ) / 4 # to be determined
    for x in range(23):
        for y in range(28):
            if distance([x,y],center)<12: # radius to be determined
                img_ori[x,y] = 1

    return img_ori
#------------------------------------------------------------------------------
def img_processing(img_list):
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
            points = threshold(np.transpose(np.flipud(filtered)).reshape(23, 28))
            if len(points)<100: # finger only
                print len(points),"finger only"
                kmeans_centroids = kmeans_clustering(points)
                final_angle = calculate_vector(kmeans_centroids, 5, touchpad_center=[14,13])
                angle_list.append([final_angle,"finger only"])
            else: # palm on
                print len(points), "palm on"
                no_palm = remove_palm(np.transpose(np.flipud(img_list[i])).reshape(23, 28))
                points = threshold2(no_palm)
                kmeans_centroids = kmeans_clustering(points)
                final_angle = calculate_vector(kmeans_centroids, 5, touchpad_center=[14,13])
                angle_list.append([final_angle,"palm on"])
        else:
            angle_list.append('hand lifted')
    return index_list, angle_list

def detect_online(filename,img_list, data_numeric, num_frame, header, time_stamp):
    print 'in detect'
    index_list, angle_list = img_processing(img_list)
    slice_timestamp = get_timestamp(header, time_stamp)
    write_angle(filename, slice_timestamp, index_list, angle_list)
