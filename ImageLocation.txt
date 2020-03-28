import cv2
import os
import MySQLdb

path="F:\\Intenship Project\\ImageProcessing\\images\\"


sift = cv2.xfeatures2d.SIFT_create()
img = cv2.imread("F:\\Intenship Project\\ImageProcessing\\images\\6.jpg")
#manual image input to program
kk1,ds1 = sift.detectAndCompute(img,None)
file = os.listdir(path)
print(file)

for image in file:

    img1 = cv2.imread(path+image)

    kk,ds= sift.detectAndCompute(img1,None)
    #print(ds.size)
    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(ds, ds1, k=2)
    good_points = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good_points.append(m)
    number_keypoints = 0
    if len(kk) <= len(kk1):
        number_keypoints = len(kk)
    else:
        number_keypoints = len(kk1)

    percentageMatch = len(good_points) / number_keypoints * 100
    print("\npercentage match is ")
    print(percentageMatch)

    if percentageMatch > 50.00:
        print('------------  Two images are same  ---------------')
        print(image)
        db = MySQLdb.connect("localhost","root","root","images" )
        cursor = db.cursor()
        cursor.execute("SELECT * from image where name='%s'" %image)
        data = cursor.fetchall()
        print("Image Data :")
        print(data)

    else:
        print('-------------Two images are differant--------------')

db.close()



