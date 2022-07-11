# kalman_filters_course
Repository for the Kalman Filter Course from thinkautonomous.ai https://www.thinkautonomous.ai/kalman-filters

In this course a Kalman Filter was used to track a bicycle in a public setting. For the kinematic modeling, a Constant Velocity (1st order) model and a Constant Acceleration (2nd order) model were chosen, the results can be compared and constrasted in the output video below.

The bounding boxes were obtained via YOLO v5 which proved to be a more than adequate detector for this use case. The Kalman Filter is able to contiuously track the bicycle even though YOLO drops measurements through a few slight occlusion gaps. The performance of the 1st order model (blue) is much better than that of the 2nd order model (green). The noisy measurements clearly have a larger impact on the 2nd order model and it also has trouble extrapolating through the measurement gaps

- Constant Velocity     --> Blue
- Constant Acceleration --> Green


https://user-images.githubusercontent.com/60835780/178173010-4047a7ed-40b2-412f-92a0-b75e65cc9f9c.mp4

