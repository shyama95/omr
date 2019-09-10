import argparse
import cv2
import math
import numpy as np
from pyzbar import pyzbar
from os import walk
import os
import numpy as np

CORNER_FEATS = (
    0.322965313273202,
    0.19188334690998524,
    1.1514327482234812,
    0.998754685666376,
)

TRANSF_SIZE = 1024


def normalize(im):
    return cv2.normalize(im, np.zeros(im.shape), 0, 255, norm_type=cv2.NORM_MINMAX)

def get_approx_contour(contour, tol=.01):
    """Get rid of 'useless' points in the contour"""
    epsilon = tol * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)

def get_contours(image_gray):
    im2, contours, hierarchy = cv2.findContours(
        image_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return map(get_approx_contour, contours)

def get_corners(contours):
    return sorted(
        contours,
        key=lambda c: features_distance(CORNER_FEATS, get_features(c)))[:4]

def get_bounding_rect(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return np.int0(box)

def get_convex_hull(contour):
    return cv2.convexHull(contour)

def get_contour_area_by_hull_area(contour):
    return (cv2.contourArea(contour) /
            cv2.contourArea(get_convex_hull(contour)))

def get_contour_area_by_bounding_box_area(contour):
    return (cv2.contourArea(contour) /
            cv2.contourArea(get_bounding_rect(contour)))

def get_contour_perim_by_hull_perim(contour):
    return (cv2.arcLength(contour, True) /
            cv2.arcLength(get_convex_hull(contour), True))

def get_contour_perim_by_bounding_box_perim(contour):
    return (cv2.arcLength(contour, True) /
            cv2.arcLength(get_bounding_rect(contour), True))

def get_features(contour):
    try:
        return (
            get_contour_area_by_hull_area(contour),
            get_contour_area_by_bounding_box_area(contour),
            get_contour_perim_by_hull_perim(contour),
            get_contour_perim_by_bounding_box_perim(contour),
        )
    except ZeroDivisionError:
        return 4*[np.inf]

def features_distance(f1, f2):
    return np.linalg.norm(np.array(f1) - np.array(f2))

# Default mutable arguments should be harmless here
def draw_point(point, img, radius=5, color=(0, 0, 255)):
    cv2.circle(img, tuple(point), radius, color, -1)

def get_centroid(contour):
    m = cv2.moments(contour)
    x = int(m["m10"] / m["m00"])
    y = int(m["m01"] / m["m00"])
    return (x, y)

def order_points(points):
    """Order points counter-clockwise-ly."""
    origin = np.mean(points, axis=0)

    def positive_angle(p):
        x, y = p - origin
        ang = np.arctan2(y, x)
        return 2 * np.pi + ang if ang < 0 else ang

    return sorted(points, key=positive_angle)

def get_outmost_points(contours):
    all_points = np.concatenate(contours)
    return get_bounding_rect(all_points)

def perspective_transform(img, points):
    """Transform img so that points are the new corners"""

    source = np.array(
        points,
        dtype="float32")

    dest = np.array([
        [TRANSF_SIZE, TRANSF_SIZE],
        [0, TRANSF_SIZE],
        [0, 0],
        [TRANSF_SIZE, 0]],
        dtype="float32")

    img_dest = img.copy()
    transf = cv2.getPerspectiveTransform(source, dest)
    warped = cv2.warpPerspective(img, transf, (TRANSF_SIZE, TRANSF_SIZE))
    return warped

def sheet_coord_to_transf_coord(x, y):
    return list(map(lambda n: int(np.round(n)), (
        TRANSF_SIZE * x/740,
        TRANSF_SIZE * (1 - y/1230)
    )))

def get_question_patch(transf, q_number):
    # Top left
    tl = sheet_coord_to_transf_coord(
        200,
        1030 - 100 * (q_number - 1)
    )

    # Bottom right
    br = sheet_coord_to_transf_coord(
        650,
        980 - 100 * (q_number - 1)
    )
    return transf[tl[1]:br[1], tl[0]:br[0]]

def get_question_patches(transf):
    for i in range(1, 11):
        yield get_question_patch(transf, i)

def get_alternative_patches(question_patch):
    for i in range(5):
        x0, _ = sheet_coord_to_transf_coord(100 * i, 0)
        x1, _ = sheet_coord_to_transf_coord(50 + 100 * i, 0)
        yield question_patch[:, x0:x1]

def draw_marked_alternative(question_patch, index):
    cx, cy = sheet_coord_to_transf_coord(
        50 * (2 * index + .5),
        50/2)
    draw_point((cx, TRANSF_SIZE - cy), question_patch, radius=5, color=(0, 255, 0))

def get_marked_alternative(alternative_patches):
    means = list(map(np.mean, alternative_patches))
    sorted_means = sorted(means)
    
    # print(means)
    # print(sorted_means[0]/sorted_means[1])
    
    maximum = sorted_means[-1]

    # Eliminate no answers marked case
    if sorted_means[0] > (0.8*maximum):
        return None
    # Eliminate double marking case
    elif sorted_means[0] < (0.8*maximum) and sorted_means[1] < (0.8*maximum):
        return None

    return np.argmin(means)

def get_letter(alt_index):
    return ["A", "B", "C", "D", "E"][alt_index] if alt_index is not None else "N/A"

def get_answers(source_file, file):
    """Run the full pipeline:

        - Load image
        - Convert to grayscale
        - Filter out high frequencies with a Gaussian kernel
        - Apply threshold
        - Find contours
        - Find corners among all contours
        - Find 'outmost' points of all corners
        - Apply perpsective transform to get a bird's eye view
        - Scan each line for the marked answer
    """

    im_orig = cv2.imread(source_file)

    blurred = cv2.GaussianBlur(im_orig, (9, 9), 10)

    cv2.imwrite("../test_images/b_blur.png", blurred)

    im = normalize(cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY))

    ret, im = cv2.threshold(im, 200, 255, cv2.THRESH_BINARY)
    
    # cv2.imwrite("../test_images/b_th1.png", im)
    cv2.imwrite("tmp/tmp_"+file, im)
    imNorm = im.copy()

    contours = get_contours(im)

    corners = get_corners(contours)

    outmost = order_points(get_outmost_points(corners))

    transf = perspective_transform(imNorm, outmost)

    answers = []
    for i, q_patch in enumerate(get_question_patches(transf)):
        alt_index = get_marked_alternative(get_alternative_patches(q_patch))

        if alt_index is not None:
            draw_marked_alternative(q_patch, alt_index)

        answers.append(get_letter(alt_index))

    #cv2.imshow('orig', im_orig)
    #cv2.imshow('blurred', blurred)
    #cv2.imshow('bw', im)

    return answers, transf

def get_uid(image):
   # image = cv2.imread(source_file)
   barcodes = pyzbar.decode(image) 
   barcodeData = "" 
   
   # loop over the detected barcodes
   for barcode in barcodes:
        # extract the bounding box location of the barcode and draw the
        # bounding box surrounding the barcode on the image
        (x, y, w, h) = barcode.rect
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # the barcode data is a bytes object so if we want to draw it on
        # our output image we need to convert it to a string first
        barcodeData = barcode.data.decode("utf-8")
        barcodeType = barcode.type

        # draw the barcode data and barcode type on the image
        text = "{} ({})".format(barcodeData, barcodeType)
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 255), 2)
        cv2.imwrite("../test_images/0out.png", image)

   return barcodeData 

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        help="Input image filename",
        required=True,
        type=str)

    parser.add_argument(
        "--output",
        help="Output image filename",
        type=str)

    parser.add_argument(
        "--show",
        action="store_true",
        help="Displays annotated image")

    args = parser.parse_args()

    splitImageCmd = "if [ -e tmp ]; then rm -rf tmp; fi && mkdir tmp && convert -density 300 -depth 8 -quality 100 " + args.input + " tmp/answer_sheet.png"
    os.system(splitImageCmd)

    answerKey = np.loadtxt(open("answerkey.csv", "r"), delimiter=",", dtype = 'str')    
    answerKey = answerKey[:,1]
    
    for (dirpath, dirnames, filenames) in walk("tmp/"):
        files = filenames
     
    for file in files:
        imagePath = "tmp/" + file
        image = cv2.imread(imagePath) 	
        
	# uid = get_uid(imagePath)        
        uid = get_uid(image)

        if(uid == ''):
            # print("Barcode not read")

            blurred = cv2.GaussianBlur(image, (9, 9), 10)
            image = normalize(cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY))
            ret, image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
            cv2.imwrite("../test_images/out_" + file + "_tmp.png", image)
                                                                     
            uid = get_uid(image)
            

        if(1):
            print("ID : {}".format(uid))

            answers, im = get_answers(imagePath, file)
            score = 0

            for i, answer in enumerate(answers):
                solution = answerKey[i]

                print("Q{}: {}".format(i + 1, answer))
                # print(solution)

                if(answer == solution):
                    score += 10
            
            print("Socre = {}".format(score))
            cv2.imwrite("tmp/out_"+file, im)
            print("Wrote image to {}".format("tmp/out_"+file))

            if args.show:
               cv2.imshow('trans', im)

        else:
            print("Failed to read unique id")

if __name__ == '__main__':
    main()
