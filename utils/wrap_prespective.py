"""
    This Script will detect the plate rectangle and then 
    apply wrap perspective ("bird's eye view") to this rectangle
    start with the method ==>  make_page_upright()
"""

import cv2
import numpy as np
from .plate_mask import best_fit_mask


def resize(image, width=None, height=None):
    """
        Arguments:
            image {np.array} -- The image to be resized

        Keyword Arguments:
            width {[int]} -- The new width for the image after resizing
            height {[int]} -- The new height of the image after resizing
            show {bool} -- Show image resized or not

        Returns:
            [np.array] -- Image resized
    """
    if width is None and height is None:
        return image

    if width is None:
        r = height / image.shape[0]
        width = int(r * image.shape[1])
    elif height is None:
        r = width / image.shape[1]
        height = int(r * image.shape[0])

    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    return resized


def order_corner_points_clockwise(points):
    """
    this method will rearrange the given corner point in a clockwise direction
    """
    # Initialize a List of coordinates to be requested
    # So that the first entry in the list is the top right,
    # The second entry is the top right, and the third is
    # Bottom right, fourth is bottom left
    rect = np.zeros((4, 2), dtype="float32")

    #! very important
    #! ==========
    # ? The image is a numpy array
    # ? The first column from the left is the vertical axis values
    # ? The second column from the north is the values ​​for the horizontal axis

    # The upper left point will contain the smallest summation,
    #  while The right bottom point will contain the largest plural
    axis_sum = np.sum(points, axis=1)
    rect[0] = points[np.argmin(axis_sum)]
    rect[2] = points[np.argmax(axis_sum)]

    # Now, calculate the difference between the points
    # Top right of the point will have the smallest difference,
    # While the bottom left will be the biggest difference
    axis_diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(axis_diff)]
    rect[3] = points[np.argmax(axis_diff)]

    # The four corners arranged
    return rect


def is_rectangle(corners, image_area):
    """
    this method will check if the given corners points represents
    the corners of a rectangle because the licence plate is rectangular 🤣

    Returns:
        bool: whether it's a rectangle or not
    """
    corners = order_corner_points_clockwise(corners)
    tl, tr, br, bl = corners

    #! Displacements in Y values
    #!==========================
    top = abs(tl[-1] - tr[-1])
    bottom = abs(bl[-1] - br[-1])

    #! Displacements in X values
    #!==========================
    right = abs(br[0] - tr[0])
    left = abs(bl[0] - tl[0])

    #! Width & Height
    #! ===================
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Calculate the height of the new image that will be
    # The maximum distance between the upper right and the lower right
    # or the Y coordinates of top left and Y coordinates of bottom left
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    diagonal = np.sqrt(((tl[0] - br[0]) ** 2) + ((tl[1] - br[1]) ** 2))

    diagonal_bigger_than_width_and_height = diagonal > max(maxHeight, maxWidth)

    does_nearby_points_on_the_same_base_line = any(
        [x < 30 for x in [top, bottom, left, right]])

    #! I chose 3500 because regular plate is 70px height by 140px width
    #! so this area would be 9800
    #! I chose the logic to be 3500 because the input image might be smaller
    #! than this standard width and height and I fear if I set the logic too
    #! high that we miss the correct ROI
    corners_area = (maxHeight * maxWidth)
    diff = abs(corners_area - image_area)
    area_is_big = diff < 2000

    dim_bool = [
        area_is_big,
        diagonal_bigger_than_width_and_height,
        does_nearby_points_on_the_same_base_line, ]

    return all(dim_bool)


def get_corner_points(biggest_contour):
    """
        cv2.arclength(contour, closed)
        ------------
            The function calculates the perimeter
            closed: [True OR False] Is the shape closed

        cv2.approxPolyDP(contour, epsilon, closed)
        --------------------------------------------
            this method returns the smallest number of points that can represent 
            the corners of the contour

            epsilon: This is the permissible error percentage
            closed: [True OR False] Is the shape closed
    """
    # the perimeter
    peri = cv2.arcLength(biggest_contour, True)
    # contour corner
    corners = cv2.approxPolyDP(biggest_contour, 0.02 * peri, True)
    corners = np.squeeze(corners)

    return corners


def apply_top_view(image, pts):
    """
    this method will apply "bird's eye view" using the given corner points ,

    Returns:
        the new image in bird's eye view
    """
    (tl, tr, br, bl) = pts

    # Calculate the width of the new image that will be
    # The maximum distance between the bottom right and the bottom left
    # x- Coordinator or top right and x coordinate of top left
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Calculate the height of the new image that will be
    # The maximum distance between the upper right and the lower right
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Now we have the new image dimensions, build
    # Set destination points to get "bird's eye view",
    # (I.e. top-down view) of the image, and again define points
    # Top left, top right, bottom right, and bottom left arrangement
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Calculate the wrap perspective transformation matrix and then apply it
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # the new image
    return warped


def make_plate_upright(src):
    """
    this method will detect the plate rectangle and then 
    apply wrap perspective ("bird's eye view") to this rectangle

    Parameters:
        src: either the image path or a numpy array
    Returns:
        the new image in bird's eye view
    """

    image = None
    if isinstance(src, str):  # The image path was given
        image = cv2.imread(src)
    else:  # image itself was given
        image = src

    original_height, original_width = image.shape[:2]
    new_height = 70

    h_ratio = original_height / new_height

    #!==============
    # ? Step # 1
    # ? apply some Image Processing
    #!==============
    small_image = resize(image, height=new_height)
    w_ratio = original_width / small_image.shape[1]

    #!==============
    # ? Step # 2
    # ? Determine the boundaries of the plate
    #!==============
    mask, biggest_contour = best_fit_mask(small_image, 3)
    if mask is None or biggest_contour is None:
        return image

    #!==============
    # ? Step # 3
    # ? Determine the corner points of the plate
    #!==============
    image_area = original_height*original_width
    unordered_corners = get_corner_points(biggest_contour)
    if len(unordered_corners) == 0:
        print("NO CORNERS")
        return image

    corners = order_corner_points_clockwise(unordered_corners)

    #!==============
    # ? Step # 4
    # ? Apply "bird's eye view"
    #!==============
    new_corners = np.zeros((4, 2), dtype="float32")
    copy = image.copy()
    for i in range(len(corners)):
        new_corners[i] = [corners[i][0] * w_ratio, corners[i][1] * h_ratio]
        cv2.circle(copy, tuple(map(int, new_corners[i])), 6, (0, 0, 255), -1)

    return apply_top_view(image, new_corners)
