import preprocess
import os
import file_operation
import detect
#import test_module
# main function start here!
'''
TODO:
(1)[Checked!]import the data file path
(2)robust way: down-sampling to filter out the abnormal angles
(3)[checked!]run file all-in-one --> traverse all the file
(4)[Checked!]log out both the timestamp and the angle to a csv file --> calculate the accuracy
(5)fix the file path problems(in the relative path, if the directory not exist, create one)
(6)[checked!]do not set up the end point manually
'''
'''
:param: left to be tuned.
'''

if __name__ == "__main__":
    f = file_operation.pwd() + 'Output_tilt.txt'
    # file_operation.img_output()
    #print 'Now processing with: ' + f
    filename, img_list, data_numeric, num_frame, header, time_stamp = preprocess.preprocess_online(f, header_padding=0, touchpad_center=[14, 13])
    detect.detect_online(filename,img_list, data_numeric, num_frame, header, time_stamp)
    #---------------------------------------------------------------------------
    # if traversing all the file
    '''
    # if traversing all the file
    f_list = file_operation.traverse()
    for f in f_list:
        abs_f =  file_operation.pwd() + f
        print 'Now processing with: ' + f + '.....'
        detect.main_online(abs_f, header_padding=0, touchpad_center=[14, 13])
    '''