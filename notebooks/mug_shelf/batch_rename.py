import os

target_dir = "test_set_ims"
k = 0
while True:
	if k == 0:
		candidate = target_dir + "/meshcat.png"
	else:
		candidate = target_dir + ("/meshcat (%d).png" % k)
	print("Looking for %s" % candidate)
	if os.path.exists(candidate):
		os.system("cp \"%s\" %s" % (candidate, target_dir + "/%07d.png" % k))
		print("Found")
		k += 1
	else:
		break
