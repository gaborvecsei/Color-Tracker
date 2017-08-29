import numpy as np
import cv2


def calculate_distance(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def crop_out_polygon_convex(rgb_image, point_array):
    point_array = np.array(point_array)
    point_array = np.reshape(cv2.convexHull(point_array), point_array.shape)
    mask = np.zeros(rgb_image.shape, dtype=np.uint8)
    roi_corners = np.array([point_array], dtype=np.int32)
    channel_count = get_image_channel(rgb_image)
    ignore_mask_color = (255,) * channel_count
    cv2.fillConvexPoly(mask, roi_corners, ignore_mask_color)
    masked_image = cv2.bitwise_and(rgb_image, mask)
    return masked_image


def get_image_channel(image):
    if len(image.shape) == 3:
        h, w, c = image.shape
    else:
        raise Exception("Image is not RGB image")
    return c


def overrides(interface_class):
    def overrider(method):
        if method.__name__ in dir(interface_class):

            pass
        else:
            raise Exception("Function <{0}> not found in Parent Class".format(method.__name__))
        return method

    return overrider
