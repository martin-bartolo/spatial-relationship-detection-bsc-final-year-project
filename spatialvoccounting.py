import numpy as np
import matplotlib.pyplot as plt

x = ["above", "across", "against", "along", "around", "at_the_level_of",
                        "behind", "beyond", "below", "far_from", "in", "in_front_of", "near",
                        "next_to","none","on", "opposite", "outside_of", "under"]

y = [2, 0, 0, 0, 0, 0, 84, 0, 1, 0, 10, 37, 36, 102, 0, 32, 0, 0, 6]

fig, ax = plt.subplots()
ax.bar(x,y)
fig.autofmt_xdate()
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()