"""
@Target: data preprocessing procedure for gaudi HID++ output
@author: Chia-Ju Chen
"""
# -----------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from skimage import measure
from sklearn import preprocessing
from sklearn.cluster import KMeans
import itertools
from scipy import misc
from scipy import ndimage
import csv


# -----------------------------------------------------------------
def cut_tail(sum_frame):
    # size of each dataframe, (81,8)
    w, l = sum_frame.shape
    # print 'w', w, 'l', l
    # reshape each dataframe ---> (1,648) ---> in python: (0, 647)
    flat_data = sum_frame.reshape(1, w * l)
    # delete 4 tail element
    mask = np.ones(w * l, dtype=bool)
    mask[[647, 646, 645, 644]] = False
    remove_tail = flat_data[0][mask]
    return remove_tail


# -----------------------------------------------------------------
def preprocess_data_offline(data_numeric_temp, num_frame, group_len):
    sum_temp = []
    flat_data = []
    for i in range(0, num_frame / group_len):
        sum_temp.append(data_numeric_temp[i] + data_numeric_temp[i + 1])
    print 'len for sum_temp: ', len(sum_temp)
    for i in range(len(sum_temp) - 1):
        flat_data.append(cut_tail(sum_temp[i]) - cut_tail(sum_temp[i + 1]))
    print 'len for flat data', len(flat_data)
    for i in range(len(sum_temp) - 1):
        scaled_list = preprocessing.scale(flat_data[i])
        temp_list = [
            np.mean(scaled_list) if x > np.mean(scaled_list) + np.std(scaled_list) or x < np.mean(scaled_list) - np.std(
                scaled_list) else x for x in scaled_list]
        greyscale = np.transpose(np.flipud(temp_list)).reshape(23, 28)
        plt.imshow(greyscale)
        plt.savefig(str(i) + '.png')
        # threshold img (after deciding threshold 0,1)
        filter_list = [0 if x < np.mean(scaled_list) + tol else 1 for x in temp_list]
    return np.transpose(np.flipud(filter_list)).reshape(23, 28)


def save_img(img, index, filename):
    plt.imshow(img)
    plt.axis('off')
    # plt.savefig(str(index)+".gif",bbox_inches='tight', pad_inches=0)
    plt.imsave('C:/Users/cchen19/Desktop/Output_contour/' + filename + '_' + str(index) + '.jpg', img, cmap=plt.cm.GnBu)
    # plt.show()


# -----------------------------------------------------------------
def preprocess_data_online(data_numeric, num_frame, filename, touchpad_center):
    flat_data = []
    angle_list = []
    for i in range(1, len(data_numeric)):
        flat_data.append(cut_tail(data_numeric[i]) - cut_tail(data_numeric[0]))
    print 'len for flat data', len(flat_data)
    for i in range(len(data_numeric)-1):
        scaled_list = preprocessing.scale(flat_data[i])
        remove_glitch = [
            np.mean(scaled_list) if x > np.mean(scaled_list) + np.std(scaled_list) or x < np.mean(scaled_list) - np.std(
                scaled_list) else x for x in scaled_list]
        # blurring the image --> better image output
        blurred_img = ndimage.gaussian_filter(remove_glitch, sigma=0.8)
        #save_img(np.transpose(np.flipud(blurred_img)).reshape(23, 28), i, filename)
        filtered = [0 if x < np.mean(blurred_img) - np.std(blurred_img) else 1 for x in blurred_img]
        greyscale = np.transpose(np.flipud(filtered)).reshape(23, 28)
        #print 'Gaussian 1D Blur'
        # plt.imshow(greyscale)
        # plt.show()
        points = threshold(greyscale)
        num_cluster = 5
        kmeans_centroids = kmeans_clustering(points)
        final_angle = calculate_vector(kmeans_centroids, num_cluster, touchpad_center)
        angle_list.append(final_angle)
        #print final_angle
    return angle_list

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
# TODO: skipped the timestamp header, neeeeed to keep it
def parse_data(df, line_num, header_padding):
    frame_line = []
    # skip the first line for command 00 31 00 --> +2
    starting_point = line_num[header_padding]
    ending_point = line_num[-2]

    print 'starting point: ', starting_point
    print 'ending point:', ending_point

    print line_num
    num_frame = (len(line_num) - header_padding)/2 - 1 # still skip the final frame in case not complete
    for i in range(num_frame):
        skipheader = line_num[header_padding + i * 2]
        data = df.iloc[skipheader + 2 : skipheader + 164 : 2]
        frame_line.append(data.apply(lambda x: x.astype(str).map(lambda x: int(x, base=16))).as_matrix())


    return  frame_line, num_frame, starting_point, ending_point


# ------------------------------------------------------------------
def blocks(files, size=65536):
    while True:
        b = files.read(size)
        if not b: break
        yield b

# -----------------------------------------------------------------
def read_file(filename, lookup):
    line_num = []
    total_line_num = 0
    with open(filename) as outputfile:
        for num, line in enumerate(outputfile, 1):
            if lookup in line:
                line_num.append(num)
    '''
    with open(filename, "r") as f:
        total_line_num = sum(bl.count("\n") for bl in blocks(f))
    '''

    return pd.read_csv(filename, delim_whitespace=True, header=None, usecols=range(2)), pd.read_csv(filename, delim_whitespace=True, header=None, usecols=range(7, 23, 2)), line_num


# -----------------------------------------------------------------
# TODO: not decide if the total # of cluster is not enough
def kmeans_clustering(points):
    kmeans = KMeans(n_clusters=5, max_iter=300, tol=0.01).fit(points)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    colors = ["g.", "r.", "b.", "y.", "m."]
    '''
   for i in range(len(labels)):
       plt.plot(points[i][0], points[i][1], colors[labels[i]], markersize=10)
       plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=150, linewidths=5, zorder=10)
       plt.show()
       print 'centroids, x,y coordinates: ', centroids
   '''
    return centroids


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

#------------------------------------------------------------------
def write_angle(slice_timestamp, angle_list):
    #time_stamp_record.to_csv('output.csv', sep='\t', encoding='utf-8', index=False, header=T)
    repitition_angle = []
    for i in angle_list:
        repitition_angle.append([str(i)] * 164 + ['write command']*2)
    angle_list = list(itertools.chain.from_iterable(repitition_angle))
    timestampe_angle = zip(slice_timestamp, angle_list)

    print timestampe_angle
    timestampe_angle.to_csv('output.csv', sep='\t', encoding='utf-8', index=False)
#------------------------------------------------------------------
def escape_starting_point(starting_point, ending_point,time_stamp):
    print 'escape starting point'
    slice_time_stamp = time_stamp.iloc[starting_point+2:ending_point]
    return slice_time_stamp.values.tolist()

# -----------------------------------------------------------------
# TODO: filter out the abnormal frames
def main_offline():
    df, line_num = read_file('Output_750millisec.txt', '11 01 0A 0F 31 00 00 00 00')
    data_numeric = parse_data(df, num_frame, line_num, header_padding)
    coutour_img = preprocess_data_offline(data_numeric, num_frame, group_len)
    points = threshold(coutour_img)
    kmeans_centroids = kmeans_clustering(points)
    final_angle = calculate_vector(kmeans_centroids)
    print final_angle


# -----------------------------------------------------------------
# TODO: filter out the abnormal frames
def main_online(filename, header_padding, touchpad_center):
    # filename =  raw_input("input file name: ")
    #filename = os.path.abspath('hand_data/Output_5points')
    time_stamp, df, line_num = read_file(filename + '.txt', '11 01 0A 0F 31 00 00 00 00')
    data_numeric, num_frame, starting_point, ending_point = parse_data(df, line_num, header_padding)
    angle_list = preprocess_data_online(data_numeric, num_frame, filename, touchpad_center)
    #print angle_list
    slice_timestamp = escape_starting_point(starting_point, ending_point, time_stamp)
    #print slice_timestamp
    write_angle(slice_timestamp, angle_list)
