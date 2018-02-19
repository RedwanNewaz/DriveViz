# DriveViz
This repository discuss how to visualize the driving behavior data.
In my [previous repository](https://github.com/RedwanNewaz/rosbag_decoder), I have shown how to decode naturalistic driving behavior in image format. However, the main challenge is how to visuzalize multiple sensors information into a one image format. The information we are processing here are as follows

* CAN Image (16x100)pixels
* Frontal Camera Image (480x640)pixels
* Depth map from LiDar sensor (40x1030)pixels

Here, we use camera image as a reference image then resize the depth image into smaller size and fit it into bottom at camera image. For this resizing open cv resize option is sufficient enough. On the other hand, resizing CAN image is a little bit tricky. Since CAN image is very small, to see any pattern with naked eyes, we need to increase the resolution of each pixel. Here, we repeatedly use the same pixel value for (5,3) gird cells. Thus, we enlarge the (16x100) pixel CAN image to (80x300)pixel.

The process of concatenating three images are computationally expensive. Therefore, in order to accelerate and show a smooth video, we transform and process the images using multiple to threads. Finally, we add the labelling option that can show the anomaly detection performace of your deep learning network with repsect to input images.   
