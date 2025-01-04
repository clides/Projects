this project uses a dataset from kaggle (https://www.kaggle.com/datasets/ruizgara/socofing/data) to find real fingerprints when given an altered one

Key takeaways:
- Using cv2 library
- Using SIFT object to detect keypoints and descriptors
- Using FLANN (kth nearest neighbour) to find all the matches between original image and altered image
- Using Lowe’s Ratio Test to filter relevant matches
