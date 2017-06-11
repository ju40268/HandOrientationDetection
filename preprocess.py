"""
@Target: data preprocessing procedure for gaudi HID++ output
@author: Chia-Ju Chen
"""
# -----------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import file_operation
import pickle
from sklearn import preprocessing

# -----------------------------------------------------------------
def cut_tail(sum_frame):
    # size of each dataframe, (81,8)
    w, l = sum_frame.shape
    # print 'w', w, 'l', l
    # reshape each dataframe ---> (1,648) ---> in python: (0, 647)
    flat_data = sum_frame.reshape(1, w * l)
    # delete 4 tail element
    mask = np.ones(w * l, dtype=bool)
    mask[list(range(w*l-4, w*l))] = False
    remove_tail = flat_data[0][mask]
    return remove_tail

#------------------------------------------------------------------
#TODO: not sure why white noise cannot used as reference frame?
def gen_white_noise():
    #generate white noise frame as reference frame for every other to substract from
    white_noise = np.random.normal(0.5, 0.5, size=644)
    #ref_frame = np.transpose(np.flipud(white_noise).reshape(23, 28))
    #save_img(ref_frame, 0, 'white_noise')
    return white_noise

def load_ref(obj='ref.pickle'):
    with open(obj) as f:  # Python 3: open(..., 'rb')
        ref_lsit = pickle.load(f)
        # print ref_lsit
    return  ref_lsit

def save_img(img, index, filename):
    # print 'saving img ' + str(index)
    img_dir = file_operation.data_output('img')
    if not os.path.exists(img_dir):
        print 'No contour img directory, now creating one.'
        os.makedirs(img_dir)
    _, tail = os.path.split(filename)
    # extract the .txt extend file name
    plt.imsave(img_dir + tail[:-4] + '_' + str(index) + '.jpg', img, cmap=plt.cm.GnBu)

def save_csv(data_numeric, num_frame, filename):
    flat_data = []
    for i in range(1, len(data_numeric)):
        flat_data.append(cut_tail(data_numeric[i]) - load_ref())
    csv_dir = file_operation.data_output('csv')

    if not os.path.exists(csv_dir):
        print 'No csv directory, now creating one.'
        os.makedirs(csv_dir)
    _, tail = os.path.split(filename)
    csv_name = csv_dir + tail[:-4] + '_' + 'difference_not_scale.csv'

    try:
        with open(csv_name, 'wb') as f:
            print f
            for i in range(len(data_numeric) - 1):
                f.write('frame#'+str(i)+'\n')
                array = np.transpose(np.flipud(flat_data[i])).reshape(23, 28)
                csv.writer(f).writerows(array)
        f.close()

    except IOError:
        print 'file still opening, in lock. Please close all the corresponding csv file'

# -----------------------------------------------------------------
def gen_img(data_numeric, num_frame, filename, touchpad_center):
    flat_data = []
    img_list = []
    ref_list = load_ref()
    for i in range(1, len(data_numeric)):
        flat_data.append(cut_tail(data_numeric[i]) - ref_list)
    print 'len for flat data', len(flat_data)

    for i in range(len(data_numeric) - 1):
        scaled_list = preprocessing.scale(flat_data[i])
        remove_glitch = [
            np.mean(scaled_list) if x > np.mean(scaled_list) + np.std(scaled_list) or x < np.mean(
                scaled_list) - np.std(
                scaled_list) else x for x in scaled_list]
        # blurring the image --> better image output
        img_list.append(remove_glitch)

        #------------------------------------------------------------------------------
        # here saving the output img(not blurred)
        save_img(np.transpose(np.flipud(remove_glitch)).reshape(23, 28), i, filename)

    return img_list

# -----------------------------------------------------------------
# TODO[Checked]: skipped the timestamp header, neeeeed to keep it
def parse_data(df, line_num, header_padding):
    frame_line = []
    header = []
    # skip the first line for command 00 31 00 --> +2
    starting_point = line_num[header_padding]
    ending_point = line_num[-2]
    print 'starting point: ', starting_point
    print 'ending point: ', ending_point

    num_frame = (len(line_num) - header_padding) / 2 - 1  # still skip the final frame in case not complete
    for i in range(num_frame):
        # dealing with NaN first, sanity check.
        skipheader = line_num[header_padding + i * 2]
        full_data = df.iloc[skipheader + 2: skipheader + 164]
        if full_data.isnull().values.any():
            print 'containing NaN in output signal at: ', skipheader
            continue
        else:
            data = df.iloc[skipheader + 2: skipheader + 164: 2]
            frame_line.append(data.apply(lambda x: x.astype(str).map(lambda x: int(x, base=16))).as_matrix())
            header.append(skipheader)

    print 'gather all the header: ', header
    return  frame_line, num_frame, header


# ------------------------------------------------------------------
# calculating number of lines
def blocks(files, size=65536):
    while True:
        b = files.read(size)
        if not b: break
        yield b

# ------------------------------------------------------------------
def read_file(filename, lookup):
    line_num = []
    total_line_num = 0
    with open(filename) as outputfile:
        for num, line in enumerate(outputfile, 1):
            if lookup in line:
                line_num.append(num)

    print line_num
    return pd.read_csv(filename, delim_whitespace=True, header=None, usecols=range(2)), pd.read_csv(filename, delim_whitespace=True, header=None, usecols=range(7, 23, 2)), line_num


# -----------------------------------------------------------------
# TODO: filter out the abnormal frames
def preprocess_online(filename, header_padding, touchpad_center):
    #TODO: check if there exist a fixed pattern for write command?
    time_stamp, df, line_num = read_file(filename, '11 03 0A 0F 31 00 00 00 00')
    if not line_num:
        print 'list empty, try another lookup pattern'
        time_stamp, df, line_num = read_file(filename, '11 01 0A 0F 31 00 00 00 00')
    #-----------------------------------------------------------------------------
    data_numeric, num_frame, header = parse_data(df, line_num, header_padding)
    save_csv(data_numeric, num_frame, filename)
    img_list = gen_img(data_numeric, num_frame, filename, touchpad_center)

    return filename, img_list, data_numeric, num_frame, header, time_stamp