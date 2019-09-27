
#ToRender="test_set_ims untrained_post_proj untrained_pre_proj trained_post_proj trained_pre_proj"
ToRender="test_set_ims"

for val in $ToRender; do
	echo "Rendering: " $val
	ffmpeg -y -r 1 -i $val/%07d.png -vcodec libx264 $val\_1hz.mp4
	ffmpeg -y -r 2 -i $val/%07d.png -vcodec libx264 $val\_2hz.mp4
	#ffmpeg -y -r 4 -i $val/%07d.png -vcodec libx264 $val\_4hz.mp4
done
