import cv2
import numpy as np
import traceback

akaze = None
matcher = None
ref_img = None
kpts_ref = None
desc_ref = None
objp = None

N = 5 # leads to less noisy results?
# N = 10
MIN_MATCH_COUNT = 4
DIST_RATIO = 0.7

is_calibrated = False
objpoints = []
imgpoints = []
cameraMatrix = None
dist = None

def init():
    global akaze, matcher, ref_img, kpts_ref, desc_ref, objp
    # Load the reference image
    ref_img = cv2.imread('ref.jpg', cv2.IMREAD_GRAYSCALE)
    w = ref_img.shape[1] / 1000.
    h = ref_img.shape[0] / 1000.
    objp = [
        [0., 0., 0.],
        [0., h, 0.],
        [w, h, 0.],
        [w, 0., 0.]
    ]

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
    global is_calibrated, objpoints, imgpoints, cameraMatrix, dist

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
    corners = cv2.perspectiveTransform(pts, M)

    # return cv2.polylines(frame, [np.int32(corners)], True, 255, 3, cv2.LINE_AA)

    if is_calibrated:
        # Estimate the pose
        # retval, rvecs, tvecs, inliers = cv2.solvePnPRansac(np.array(objp, dtype=np.float32), corners, cameraMatrix, dist)
        retval, rvecs, tvecs = cv2.solvePnP(np.array(objp, dtype=np.float32), corners, cameraMatrix, dist)

        # Show coordinate axes using drawFrameAxes
        return cv2.drawFrameAxes(frame, cameraMatrix, dist, rvecs, tvecs, 0.1)

    objpoints.append(objp)
    imgpoints.append(corners)
    print(len(objpoints))

    if len(objpoints) > 20:
        print('Calibrating...')
        frameSize = (frame.shape[1], frame.shape[0])
        # Calculate the camera calibration parameters
        print(np.array(objpoints).shape)
        print(np.array(imgpoints).shape)
        ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(
            np.array(objpoints, dtype=np.float32),
            np.array(imgpoints, dtype=np.float32),
            frameSize, None, None)
        is_calibrated = True

    return frame

            
    # else:
    #     # Draw the matches
    #     frame = cv2.drawMatches(ref_img, kpts_ref, gray, kpts, good_matches, None)

init()

cap = cv2.VideoCapture(0)
frameCount = 0

try:
    while cap.isOpened():

        succes, img = cap.read()

        k = cv2.waitKey(5)

        if k == 27:
            break

        frameCount += 1
        # Process the frame every 30 frames when not calibrated to get varied poses
        if (frameCount % 30 == 0 and not is_calibrated) or is_calibrated:
            output = process(img)
        elif is_calibrated:
            output = process(img)
        else:
            output = img
        cv2.imshow('Output', output)

except Exception:
    print(traceback.format_exc())
finally:
    # Release and destroy all windows before termination
    cap.release()

cv2.destroyAllWindows()