import os
import time
import cv2
import face_model
import numpy as np

class Args():
	"""docstring for Args"""
	def __init__(self):
		self.image_size = '112,112'
		self.gpu = 0
		self.model = '../models/model-y1-test2/model,0'
		self.ga_model = '../gender-age/model/model,0'
		self.threshold = 1.24
		self.flip = 0
		self.det = 0

args = Args()
model = face_model.FaceModel(args)		

# 自己的人脸图片路径
imgs_dir = '../datasets/star_data/star_images/'
# 提取人脸图片保存的路径
face_save_dst = '../datasets/star_data/star_face/'

folders = os.listdir(imgs_dir)

cnt = 0
for folder in folders:
	imgs = os.path.join(imgs_dir, folder)
	
	start = time.time()
	img_root_path = os.path.join(imgs_dir, folder)
	
	try:
		pic = cv2.imread(img_root_path)
		pic = model.get_input(pic)

		filename = os.path.join(face_save_dst, 'star' +str(cnt) + '.jpg')
		cv2.imwrite(filename, np.transpose(pic, (1, 2, 0))[:, :, ::-1])
		print('save face picture as: ' + filename)
		cnt += 1
	except:
		continue

	#end = time.time()
	#interval = end - start
	#print('interval time: ',interval)
	