import matplotlib
matplotlib.use('Qt5Agg')
from operator import mod

import math
from datetime import datetime

import os
import numpy as np

import argparse
import anogan
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def make_TF_label(y_input, predict_input):
    max_len = len(y_input)
    output = []

    for i in range(0,max_len):
        output.append(y_input[i] == predict_input[i])


def anomaly_detection(test_img, label = "3"):
    model = anogan.anomaly_detector(g_label = label, d_label=label)

    residual_loss, discrimitive_loss = anogan.compute_anomaly_score(model, test_img.reshape(1, 28, 28, 1), iterations=500, d_label=label)

    return residual_loss, discrimitive_loss


def compute_anomaly_score(test_img, label_list):
    anomaly_score_list = []

    for taret_label in label_list:
        r_score,d_score = anomaly_detection(test_img, label=taret_label)
        anomaly_score_list.append([r_score, d_score ])

    return anomaly_score_list

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

X_test = np.load("MNIST_X.npy", allow_pickle=True)
Y_test = np.load("MNIST_Y.npy", allow_pickle=True)

e_X_test = np.load("EMNIST_BM_X_14.npy", allow_pickle=True)
e_Y_test = np.load("EMNIST_BM_Y.npy", allow_pickle=True)
#X_test = e_X_test

generated_img = anogan.generate(25, "3")

score_total = []
count = 0

pred_list = []
true_list = []

syu_score_list = []
thres_score_list = []

score_list = []

dect_true = 0
dect_false = 0
start_time = datetime.today()
raw_start_time = datetime.today()

true_list = []

# Needed when you want to continue from last record
#score_list = np.load("./score_m_only_all_s_1000.npy", allow_pickle=True) 
#score_list = score_list.tolist()

for idx in range(0,1000):
    img_size = math.floor(np.concatenate((X_test[idx]+1)**2).sum())**(1/2)

    score= compute_anomaly_score(X_test[idx],['0','1','2','3','4','5','6','7','8','9'],img_size)

    score_list.append(score)

    if mod(idx+1,20) == 0:
        print(((idx+1)/100)*10,"%")
        print("Spent time: " + str(datetime.today() - start_time))
        print("To save at idx: "+str(idx))
        start_time = datetime.today()

        score_list_np = np.array(score_list)
        np.save("score_m_only_all_s_"+str(idx+1),score_list_np)

print("SCORING DONE")
