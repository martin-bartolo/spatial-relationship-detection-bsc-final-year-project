import matplotlib.pyplot as plt
import numpy as np

human_scores = np.array([[0.875,0.917,0.1875,0.708,0.875],#above
                [0.781,0.5,0.531,0.75,0.781,0.469,0.5],#against
                [0.304,0.321],#around
                [0.844,0.5,0.1875,0.583,1,0.8125,0.8125,0.333],#at the level of
                [0.3125,1,1,1,0.656,0.667,0.906,1,0.969,0.917],#behind
                [0.3125,0.958,0.906,0.781,0.917,0.9375,0.667,0.469,0.542,1],#far from
                [1,0,0.83,0.917,0.9375],#in
                [1,1,0.969,1,0.969,1,0.25,0.917,0.958],#in front of
                [1,0.75,0.375,0.708,0.656,0.917,1,1,1,0.833],#near
                [0.25,0.458,0.938,1,0.1875,0.708,0.375,1],#next to
                [1,0.0833,0,1,1,1,1,1,1],#on
                [0.875,0.542,0.125,0.594,0.688,0.0417,0.0833,0.281,0,0.0833]#under
                ])

model_scores = np.array([[0.593,0.554,0.675,0.676,0.593],#above
                [0.517,0.509,0.674,0.673,0.719,0.797,0.842,],#against
                [0.515,0.994],#around
                [0.558,0.555,0.65,0.613,0.784,0.764,0.801,0.906],#at the level of
                [0.554,0.533,0.69,0.621,0.767,0.741,0.869,0.806,0.990,0.931],#behind
                [0.53,0.577,0.675,0.681,0.706,0.949,0.863,0.799,0.759,0.937],#far from
                [0.534,0.508,0.685,0.742,0.768],#in
                [0.553,0.579,0.632,0.747,0.717,0.86,0.858,0.97,0.945],#in front of
                [0.549,0.588,0.669,0.684,0.758,0.767,0.812,0.868,0.913,0.947],#near
                [0.507,0.522,0.613,0.727,0.778,0.839,0.883,0.927],#next to
                [0.507,0.551,0.655,0.658,0.716,0.791,0.843,0.744,0.993],#on
                [0.559,0.527,0.641,0.758,0.782,0.732,0.891,0.892,0.982,0.917]#under
                ])

labels = ['above', 'against', 'around', 'at the level of', 'behind', 'far from', 'in', 'in front of', 'near', 'next to', 'on', 'under']
colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'k', 'm']

for i in range(human_scores.shape[0]):
    plt.scatter(human_scores[i], model_scores[i], c=colours[i] ,label = labels[i])
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.xticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1])
    plt.yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1])
    plt.xlabel("Human Scores")
    plt.ylabel("Model Scores")
    plt.grid(True)
    plt.show()

#plt.axline((0.5,0.5),(1,1))
