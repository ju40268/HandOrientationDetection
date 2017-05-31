import detect
import test_module
# main function start here!
'''
TODO: 
 (1)import the data file path
 (2)robust way: down-sampling to filter out the abnormal angles
 (3)run file all-in-one --> traverse all the file
 (4)log out both the timestamp and the angle to a csv file --> calculate the accuracy
 (5)fix the file path problems
 (6)[checked!]do not set up the end point manually
'''
'''
:param: left to be tuned.
'''

if __name__ == "__main__":
    filename = 'hand_raw_data/Output_5points'
    header_padding = 10
    touchpad_center = [14,13]
    detect.main_online(filename, header_padding, touchpad_center)
