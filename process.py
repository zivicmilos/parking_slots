# import libraries here
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt


def count_cars(image_path):
    """
    Procedura prima putanju do fotografije i vraca broj prebrojanih automobila. Koristiti ovu putanju koja vec dolazi
    kroz argument procedure i ne hardkodirati nove putanje u kodu.

    Ova procedura se poziva automatski iz main procedure i taj deo koda nije potrebno menjati niti implementirati.

    :param image_path: <String> Putanja do ulazne fotografije.
    :return: <int>  Broj prebrojanih automobila
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cars = 0
    if img.shape[1] > 500:
        gray_img = cv2.resize(gray_img, (1500, 1500))
        adapt_bin_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 355, 35)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))

        adapt_bin_img = cv2.erode(adapt_bin_img, kernel, iterations=2)
        adapt_bin_img = cv2.dilate(adapt_bin_img, kernel, iterations=2)

        im_floodfill = adapt_bin_img.copy()
        h, w = adapt_bin_img.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(im_floodfill, mask, (0, 0), 255)
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        im_out = adapt_bin_img | im_floodfill_inv
        adapt_bin_img = im_out.copy()

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        adapt_bin_img = cv2.dilate(adapt_bin_img, kernel, iterations=10)
        adapt_bin_img = cv2.erode(adapt_bin_img, kernel, iterations=10)

        img_contours, contours, hierarchy = cv2.findContours(adapt_bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if 10000 < cv2.contourArea(contour) < 100000:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 3)
                cars += 1
    else:
        gray_img = cv2.resize(gray_img, (1500, 1500))
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        gray_img = cv2.filter2D(gray_img, -1, kernel)
        adapt_bin_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 355, 35)

        im_floodfill = adapt_bin_img.copy()
        h, w = adapt_bin_img.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(im_floodfill, mask, (0, 0), 255)
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        im_out = adapt_bin_img | im_floodfill_inv
        adapt_bin_img = im_out.copy()

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (4, 4))
        adapt_bin_img = cv2.dilate(adapt_bin_img, kernel, iterations=2)
        adapt_bin_img = cv2.erode(adapt_bin_img, kernel, iterations=2)

        img_contours, contours, hierarchy = cv2.findContours(adapt_bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if 5000 < cv2.contourArea(contour):
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 3)
                cars += 1

    car_count = cars
    # TODO - Prebrojati auta i vratiti njihov broj kao povratnu vrednost ove procedure

    return car_count