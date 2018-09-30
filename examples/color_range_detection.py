import color_tracker

# Init camera
cam = color_tracker.WebCamera(video_src=0)
cam.start_camera()

# Init Range detector
detector = color_tracker.HSVColorRangeDetector(camera=cam)
lower, upper, kernel = detector.detect()

# Print out the selected values
# (best practice is to save as numpy arrays and then you can load it whenever you want it)
print("Lower HSV color is: {0}".format(lower))
print("Upper HSV color is: {0}".format(upper))
print("Kernel shape is:\n{0}".format(kernel.shape))
