import os
import cv2

sample = cv2.imread("SOCOFing/Altered/Altered-Hard/150__M_Right_index_finger_Obl.BMP")

best_score = 0 # best matching score for each image
filename = None
image = None # the original image
kp1, kp2, mp = None, None, None # key point of sample image, original image, and matched points

counter = 0
for file in [file for file in os.listdir("SOCOFing/Real")]:
    
    if counter % 100 == 0:
        print(counter)
    counter += 1
    
    fingerprint_image = cv2.imread("SOCOFing/Real/" + file)
    
    # create SIFT object to detect keypoints and compute descriptors
    sift = cv2.SIFT_create()
    
    # get keypoints and descriptors for sample image and fingerprint image
    keypoints1, descriptors1 = sift.detectAndCompute(sample, None)
    keypoints2, descriptors2 = sift.detectAndCompute(fingerprint_image, None)
    
    # find matches
    matches = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 10}, {}).knnMatch(descriptors1, descriptors2, k=2)
    
    # find relevant matches
    match_points = []
    for p, q in matches:
        if p.distance < 0.1 * q.distance:
            match_points.append(p)
            
    keypoints = min(len(keypoints1), len(keypoints2))
    
    # calculate score
    if len(match_points) / keypoints * 100 > best_score:
        best_score = len(match_points) / keypoints * 100
        filename = file
        image = fingerprint_image
        kp1, kp2, mp = keypoints1, keypoints2, match_points

# printing the result
if filename == None:
    print("NO MATCHES FOUND")
else:        
    print("BEST MATCH: " + filename)
    print("SCORE: " + str(best_score))

result = cv2.drawMatches(sample, kp1, image, kp2, mp, None)
result = cv2.resize(result, None, fx=4, fy=4)
cv2.imwrite("result.jpg", result)
cv2.imshow("Result", result)
cv2.waitKey(0)