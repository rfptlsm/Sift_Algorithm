import cv2
import glob
import numpy as np


def main():
    file_name = 'shanghai'
    path_1 = cv2.imread('./AdobePanoramasDataset/' + file_name + '-00.png')
    path_2 = glob.glob('./AdobePanoramasDataset/' + file_name + '-*')
    img_gray = cv2.cvtColor(path_1, cv2.COLOR_BGR2GRAY)

    image_resize = cv2.resize(img_gray, (460, 480))

    sift = cv2.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(image_resize, None)

    for i in path_2:
        print(i)
        img2 = cv2.imread(i)
        img_gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        image_resize2 = cv2.resize(img_gray2, (460, 480))
        keypoints_2, descriptors_2 = sift.detectAndCompute(image_resize2, None)

        match = cv2.BFMatcher()
        matches = match.knnMatch(descriptors_1, descriptors_2, k=2)
        good_matches = np.array([m1 for m1, m2 in matches if m1.distance < m2.distance / 3])

        img_result = cv2.drawMatches(image_resize, keypoints_1, image_resize2, keypoints_2, good_matches, None)

        cv2.namedWindow('SIFT')
        cv2.moveWindow('SIFT', 150, 50)
        cv2.imshow("SIFT", img_result)

        if cv2.waitKey(2000) == ord('q'):
            return
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()