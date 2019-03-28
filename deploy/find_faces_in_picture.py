import cv2
import face_model
import argparse

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()


def show_bboxes(img, bounding_boxes, facial_landmarks=[]):
    """Draw bounding boxes and facial landmarks.

    Arguments:
        img: an instance of cv2 Image.
        bounding_boxes: a float numpy array of shape [n, 5].
        facial_landmarks: a float numpy array of shape [n, 10].

    Returns:
        an instance of cv2 Image.
    """
    img_copy = img.copy()

    if bounding_boxes is None or facial_landmarks is None:
        return img_copy

    bounding_boxes = bounding_boxes.astype(int)
    facial_landmarks = facial_landmarks.astype(int)
    for b in bounding_boxes:
        cv2.rectangle(img_copy, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 1)
    
    for p in facial_landmarks:
        for i in range(5):
            center = (p[i], p[i + 5])
            cv2.ellipse(img_copy, center, (1, 1), 0, 0, 360, (255, 0, 0), 1)
    return img_copy           

model = face_model.FaceModel(args)
img = cv2.imread('test.jpeg')
#img = cv2.imread('flower_x.jpg')

bounding_boxes, landmarks, _ = model.get_bbox_and_landmarks(img)
# print('bbox', bounding_boxes)
# print('landmarks', landmarks)

bboxes_img = show_bboxes(img, bounding_boxes, landmarks)
cv2.imshow("show_bboxes", bboxes_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
