import face_model
import argparse
import cv2
import sys
import numpy as np

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')

def main(args):
	model = face_model.FaceModel(args)
	video_capture = cv2.VideoCapture(0)

	# init load
	img2 = cv2.imread('unknown.jpg')
	img2 = model.get_input(img2)
	features2 = model.get_feature(img2)

	while True:
		# Grab a single frame of video
	    ret, frame = video_capture.read()

	    # Resize frame of video to 1/4 size for faster face recognition processing
	    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

	    bounding_boxes, landmarks, face_img = model.get_bbox_and_landmarks(small_frame)
	    bounding_boxes = bounding_boxes.astype(int)

	    features1 = model.get_feature(face_img)
	    # compare_faces
	    # TODO

	    for b in bounding_boxes:
	        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
	        b *= 4
	        cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 1)

	    cv2.imshow('Video', frame)

	    # Hit 'q' on the keyboard to quit!
	    if cv2.waitKey(1) & 0xFF == ord('q'):
	        break

	video_capture.release()
	cv2.destroyAllWindows()

def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    """
    Compare a list of face encodings against a candidate encoding to see if they match.

    :param known_face_encodings: A list of known face encodings
    :param face_encoding_to_check: A single face encoding to compare against the list
    :param tolerance: How much distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance.
    :return: A list of True/False values indicating which known_face_encodings match the face encoding to check
    """
    return list(face_distance(known_face_encodings, face_encoding_to_check) <= tolerance)

def face_distance(face_encodings, face_to_compare):
    """
    Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
    for each comparison face. The distance tells you how similar the faces are.

    :param faces: List of face encodings to compare
    :param face_to_compare: A face encoding to compare against
    :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
    """
    if len(face_encodings) == 0:
        return np.empty((0))

    return np.linalg.norm(face_encodings - face_to_compare, axis=1)


if __name__ == '__main__':
	args = parser.parse_args()
	main(args)
