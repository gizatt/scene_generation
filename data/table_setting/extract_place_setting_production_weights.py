import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
#target_run_filename = "icra_runs/nominal/newpriors_1/fixed_elbo_2019-09-14-06-09-1568456802/events.out.tfevents.1568456802.ProblemChild"
target_run_filename = "icra_runs/nominal/1_evennewerpriors/fixed_elbo_2019-09-14-23-09-1568517439/events.out.tfevents.1568517439.ProblemChild"
all_weights = np.zeros((500, 256))
try:
    for event in tf.train.summary_iterator(target_run_filename):
        for value in event.summary.value:
            if "place_setting_production_weights" in value.tag:
                _, num = value.tag.split("/")
                all_weights[event.step, int(num)] = value.simple_value
except Exception as e:
    print("Except ", e)
print(all_weights)
all_weights = all_weights.T

np.save("place_setting_production_weights_across_epochs.npy", all_weights)
all_weights = np.log(all_weights + 1E-6)
plt.imshow(all_weights)
plt.xlabel("Epoch")
plt.ylabel("Log-prob of corresponding place setting production weight")
plt.show()