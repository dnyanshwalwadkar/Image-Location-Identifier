import cv2
import numpy as np
import sys
import os  
original = cv2.imread("E:\\mass technologies\\Project 2018-19\\Ashlesha & Group\\images\\original_golden_bridge.jpg")
image_to_compare = cv2.imread("E:\\mass technologies\\Project 2018-19\\Ashlesha & Group\\images\\fashion_girl_187287.jpg")

# print(original)
# print(image_to_compare)
# 1) Check if 2 images are equals
# if original.shape == image_to_compare.shape:
#     print("The images have same size and channels")
#     difference = cv2.subtract(original, image_to_compare)
#     b, g, r = cv2.split(difference)

#     if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
#         print("The images are completely Equal")
#     else:
#         print("The images are NOT equal")

# 2) Check for similarities between the 2 images
sift = cv2.xfeatures2d.SIFT_create()
kp_1, desc_1 = sift.detectAndCompute(original, None)
kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)

index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(desc_1, desc_2, k=2)

good_points = []
for m, n in matches:
    if m.distance < 0.6*n.distance:
        good_points.append(m)

# Define how similar they are
number_keypoints = 0
if len(kp_1) <= len(kp_2):
    number_keypoints = len(kp_1)
else:
    number_keypoints = len(kp_2)

# print("Keypoints 1ST Image: " + str(len(kp_1)))
# print("Keypoints 2ND Image: " + str(len(kp_2)))
# print("GOOD Matches:", len(good_points))
percentagematch=len(good_points) / number_keypoints * 100
print("How good it's the match: ", percentagematch)
if percentagematch>36.87:
    print('Two images are same')
else:
    print('Two images are differant')

result = cv2.drawMatches(original, kp_1, image_to_compare, kp_2, good_points, None)
# print(result)
imagepath='E:\\mass technologies\\Project 2018-19\\Ashlesha & Group\\images\\original_golden_bridge.jpg'
print(imagepath)
name=os.path.basename(imagepath)
print(name)
height=np.size(original,0)
width=np.size(original,1)
size1=round((os.path.getsize(imagepath)/1024),2)

imagepath1="E:\\mass technologies\\Project 2018-19\\Ashlesha & Group\\images\\duplicate.jpg"
print(imagepath1)
name1=os.path.basename(imagepath1)
height1=np.size(image_to_compare,0)
width1=np.size(image_to_compare,1)
size2=round((os.path.getsize(imagepath1)/1024),2)
cv2.imshow("result", cv2.resize(result, None, fx=0.4, fy=0.4))
cv2.imwrite("feature_matching.jpg", result)
cv2.imshow("File Name "+name+" Height "+str(height)+" Width "+str(width)+" Size in Kb "+str(size1), cv2.resize(original, None, fx=0.4, fy=0.4))

cv2.imshow("File Name "+name1+" Height "+str(height1)+" Width "+str(width1)+" Size in Kb "+str(size2), cv2.resize(image_to_compare, None, fx=0.4, fy=0.4))

k=cv2.waitKey(10000)
    
    
cv2.destroyAllWindows()
