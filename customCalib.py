import cv2

akaze = None
matcher = None
ref_img = None
kpts_ref = None
desc_ref = None

def init():
    global akaze, matcher, ref_img, kpts_ref, desc_ref
    # Load the reference image
    ref_img = cv2.imread('ref.jpg', cv2.IMREAD_GRAYSCALE)

    # Create AKAZE detector and descriptor
    akaze = cv2.AKAZE_create()
    kpts_ref, desc_ref = akaze.detectAndCompute(ref_img, None)
 
    # matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE_HAMMING)
    matcher = cv2.BFMatcher()
 
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
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Draw the matches
    img_matches = cv2.drawMatches(ref_img, kpts_ref, gray, kpts, good_matches, None)

    # Display the image
    cv2.imshow('Matches', img_matches)

init()

cap = cv2.VideoCapture(0)

try:
    while cap.isOpened():

        succes, img = cap.read()

        k = cv2.waitKey(5)

        if k == 27:
            break

        # cv2.imshow('Img',img)
        process(img)
except Exception as e:
    print(e)
finally:
    # Release and destroy all windows before termination
    cap.release()

cv2.destroyAllWindows()