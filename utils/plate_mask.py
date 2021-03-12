import numpy as np
import cv2


def bgr_pixel_to_hsv(value):
    pixel_bgr = np.uint8([[list(value)]])
    hsv = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2HSV)
    return hsv[0][0]


def find_dominant_color(filename):
    # Resizing parameters
    width, height = 150, 150
    image = pilImage.open(filename)
    image = image.resize((width, height), resample=0)
    # Get colors from image object
    pixels = image.getcolors(width * height)
    # Sort them by count number(first element of tuple)
    sorted_pixels = sorted(pixels, key=lambda t: t[0])
    # Get the most frequent color
    dominant_color = sorted_pixels[-1][1]
    return dominant_color


def my_method(bgr_image, how_many_colors=1):
    img = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    unique, counts = np.unique(img.reshape(-1, 3), axis=0, return_counts=True)

    if how_many_colors == 1:
        return unique[np.argmax(counts)]  # !dominant color in rgb

    elif how_many_colors > 1:
        # top_indices = np.argsort(counts)[-how_many_colors:]
        top_indices = np.argpartition(
            counts, how_many_colors)[-how_many_colors:]

        return unique[top_indices]

    return None


def check_color(hsv_color):
    h, s, v = hsv_color
    margin = 15
    #! Opencv: v=77 & s=35  EQUALS in gimp: v=30% & s=13.7%
    is_black = v < 77
    #! Opencv: v=255 & s=35 EQUALS in gimp: v=90% & s=13.7%
    is_white = v > 120 and s < 45
    color_type = None
    if is_white:
        color_type = True
        #! we will mask the black color then inverse the mask because we want to
        #! get red of the red diagonal line in some white color plates
        return [
            np.array([0, 0, 0], dtype=np.uint8),
            np.array([255, 255, 150], dtype=np.uint8),
            color_type, ]

        # return [
        #     np.array([0, 0, 120], dtype=np.uint8),
        #     np.array([178, 45, 255], dtype=np.uint8)
        # ]

    elif is_black:
        color_type = False
        return [
            np.array([0, 0, 0], dtype=np.uint8),
            np.array([255, 255, 150], dtype=np.uint8),
            color_type, ]

    return [
        np.array([max(h-margin, 0), 50, 50], dtype=np.uint8),
        np.array([min(h+margin, 178), 255, 255], dtype=np.uint8),
        color_type, ]


def best_fit_mask(bgr_image, how_many_colors=1):
    resize = cv2.resize(bgr_image, (15, 15))
    colors = my_method(resize, how_many_colors)  # !RGB Colors
    hsv_img = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)  # ! HSV Image
    biggest_mean = 0
    best_mask = None
    best_contour = None
    for i, test_color in enumerate(colors):
        h, s, v = bgr_pixel_to_hsv(test_color[::-1])
        lower_hsv, upper_hsv, color_type = check_color([h, s, v])
        mask = cv2.inRange(hsv_img, lower_hsv, upper_hsv)
        if color_type is not None and color_type:
            mask = cv2.bitwise_not(mask)

        #! Preprocess the mask
        # ! we applied MORPH_CLOSE in two seprate steps because we wanted the dilation
        #! to be bigger than erosion
        errode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, errode_kernel)

        errode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, errode_kernel)

        cnts = cv2.findContours(mask,
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
        if len(cnts):
            biggest_contour = sorted(
                cnts, key=cv2.contourArea, reverse=True)[0]

            img_masked_mean = cv2.bitwise_and(
                bgr_image, bgr_image, mask=mask).mean()
            #! we will choose the must that will show the most parts of the image
            if img_masked_mean > biggest_mean:
                biggest_mean = img_masked_mean
                #! we need a copy because we will change the "mask" variable in place
                best_mask = mask.copy()
                best_contour = biggest_contour.copy()

    return best_mask, best_contour
