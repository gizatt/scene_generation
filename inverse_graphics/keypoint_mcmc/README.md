## Keypoint Detection + MCMC to Sim Variables

Idea in this one is to:

1) Train a network to spit out keypoints.
2) From the resulting images, infer the number, poses, + shapes of objects that
could have generated those output images.

My two applications:

1) Amazon prime packages, varying in # / shape / poses
2) Mugs or shoes, varying widely in shape

Spirals:
1) Use Faster or MaskRCNN's bbox detection and assume they're right. Spit out a heatmap
of keypoints as head operating over that bbox. Do MCMC assuming the bbox detection was good
(or totally erroneous), but don't try to associate points with other bboxes.
2) Instead directly estimate a keypoint heatmap on the original image + some
form of part association field, to capture uncertainty over the correspondence
to which object. Do MCMC that uses the PAF.