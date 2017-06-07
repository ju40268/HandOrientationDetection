import detect
import os

# import test_module
# main function start here!
'''
TODO:
(1)import the data file path
(2)robust way: down-sampling to filter out the abnormal angles
(3)[checked!]run file all-in-one --> traverse all the file
(4)[Checked!]log out both the timestamp and the angle to a csv file --> calculate the accuracy
(5)fix the file path problems(in the relative path, if the directory not exist, create one)
(6)[checked!]do not set up the end point manually
'''
'''
:param: left to be tuned.
'''

directory = 'C:/Users/cchen19/PycharmProjects/hand_orientation_detection/hand_raw_data'
file_list = []
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        # print(os.path.join(directory, filename))
        file_list.append('hand_raw_data/' + filename)
        continue
    else:
        continue

if __name__ == "__main__":
    for f in file_list:
        print f
        header_padding = 20
        touchpad_center = [14, 13]
        detect.main_online(f, header_padding, touchpad_center)

