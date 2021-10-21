from vsum_tools import *
from vasnet import *

segment_length = 2 #in seconds

sampling_rate = 2 #in frame per second
model_root_dir = '/content/drive/MyDrive/VSum/Colab/UIT-VSUM'
model_file_path = model_root_dir + '/vasnet_inceptionv3_avg_2048/models/tvsum_splits_1_0.6187176441788906.tar.pth'

model_func, preprocess_func, target_size = model_picker('inceptionv3', 'max')
import h5py

import cv2

# %rm test.h5
# new_h5 = h5py.File('test.h5', 'w')

video_path = '/working/video.mp4'

video = cv2.VideoCapture(video_path)

got, frame = video.read()
print(got)

fps = (video.get(cv2.CAP_PROP_FPS))
frameCount = video.get(cv2.CAP_PROP_FRAME_COUNT)
size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
size_param = "%dx%d"%size 
padding = lambda i: '0'*(6-len(str(i))) + str(i)

print(fps, frameCount, size)

changepoints = [(i, i+ int(segment_length*fps)-1) for i in range(0, int(frameCount), int(segment_length*fps))]
nfps = [x[1] - x[0] + 1 for x in changepoints]

picks = []
i = 0
while True:
  picks += [int(fps*i + x* int(fps//sampling_rate)) for x in range(sampling_rate)]
  i += 1
  if picks[-1] > frameCount :
    break

picks = [i for i in picks if i < frameCount]



features = extract_features4video(model_func, preprocess_func, target_size, picks, video_path)

print(changepoints, nfps, picks, features, sep = '\n')


aonet = AONet()

aonet.initialize(f_len = len(features[0]))
aonet.load_model(model_file_path)
print("load model successfull")
predict = aonet.eval(features)
# print(p)
threshold = 0.5

summary = generate_summary(predict, np.array(changepoints), int(frameCount),nfps, picks)

print(summary)
print(sum(summary), len(summary))




first_video_path = 'video.mp4'


sum_video_path = "/test_watch_summary/"

generate_summary_video(first_video_path, sum_video_path, summary)
