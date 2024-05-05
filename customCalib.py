import cv2
import numpy as np

akaze = None
matcher = None
ref_img = None
kpts_ref = None
desc_ref = None

N = 5 # leads to less noisy results?
# N = 10
MIN_MATCH_COUNT = 4
DIST_RATIO = 0.7

def init():
    global akaze, matcher, ref_img, kpts_ref, desc_ref
    # Load the reference image
    ref_img = cv2.imread('ref.jpg', cv2.IMREAD_GRAYSCALE)

    # Create AKAZE detector and descriptor
    akaze = cv2.AKAZE_create()
    kpts_ref, desc_ref = akaze.detectAndCompute(ref_img, None)
 
    # matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE_HAMMING)
    matcher = cv2.BFMatcher()

def isHomographyValid(M):
    if M is None:
        return False

    # Calculate the determinant of the matrix
    det = M[0, 0] * M[1, 1] - M[1, 0] * M[0, 1]

    # Check if the determinant is in the range
    return 1/N < abs(det) and abs(det) < N

def process(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # TODO Check if needed to flip horizontally

    # Detect and compute the keypoints and descriptors
    kpts, desc = akaze.detectAndCompute(gray, None)

    # Match the descriptors
    matches = matcher.knnMatch(desc_ref, desc, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < DIST_RATIO * n.distance:
            good_matches.append(m)

    if not len(good_matches) > MIN_MATCH_COUNT:
        return frame

    # Find homography
    src_pts = np.float32([kpts_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kpts[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if not isHomographyValid(M):
        return frame

    # Draw a rectangle that marks the found model in the frame
    h, w = ref_img.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    return cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
            
    # else:
    #     # Draw the matches
    #     frame = cv2.drawMatches(ref_img, kpts_ref, gray, kpts, good_matches, None)

init()

cap = cv2.VideoCapture(0)

try:
    while cap.isOpened():

        succes, img = cap.read()

        k = cv2.waitKey(5)

        if k == 27:
            break

        output = process(img)
        cv2.imshow('Output', output)

except Exception as e:
    print(e)
finally:
    # Release and destroy all windows before termination
    cap.release()

cv2.destroyAllWindows()