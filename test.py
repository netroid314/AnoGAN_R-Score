import matplotlib
matplotlib.use('Qt5Agg')

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

def make_TF_label(y_input, predict_input):
    max_len = len(y_input)
    output = []

    for i in range(0,max_len):
        output.append(y_input[i] == predict_input[i])

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

m_X_test = np.load("MNIST_X.npy", allow_pickle=True)
m_Y_test = np.load("MNIST_Y.npy", allow_pickle=True)

e_X_test = np.load("EMNIST_BM_X.npy", allow_pickle=True)
e_X_test20 = np.load("EMNIST_BM_X_20.npy", allow_pickle=True)
e_X_test14 = np.load("EMNIST_BM_X_14.npy", allow_pickle=True)

e_Y_test = np.load("EMNIST_BM_Y.npy", allow_pickle=True)

e_X_test = e_X_test[:,:,:,None]
m_X_test = m_X_test[:,:,:,None]


obj_threshold = 0
obj_square = 2

def get_object_size(img):
    #re_img = np.where(img < 0.5, -1, img)
    
    size = obj_threshold + np.concatenate(((img+1)*(1/2))**obj_square).sum()

    #log_const = 10
    #size = math.log(log_const + np.concatenate(((img+1)*(1/2))**2).sum(),log_const) 

    ##re_img = (img+1)*(1/2)
    ##row_max = np.max(re_img, axis=0)
    ##col_max = np.max(re_img, axis=1)
    ##row = len(np.where(row_max > 0.3)[0])
    ##col = len(np.where(col_max > 0.3)[0])

    #size = np.concatenate(((img+1)*(1/2))**4).sum()**(1/30)
    #size = 1 + (np.concatenate(((img+1)*(1/2))**1).sum()/(28*28))
    return size/100

def f_score(f, precision, recall):
    return ((f**2+1)*precision*recall/(recall+(f**2)*precision))

print("testing...")
score_total = []
count = 0

pred_list = []
true_list = []

score_list_m = []
score_list_e = []
score_list_e20 = []
score_list_e14 = []

score_list_m= np.load("./score_m_only_all_s_1000.npy", allow_pickle=True)
score_list_e = np.load("./score_eb_only_all_ns_500.npy", allow_pickle=True)
score_list_e20 = np.load("./score_eb20_only_all_ns_500.npy", allow_pickle=True)
score_list_e14 = np.load("./score_eb14_only_all_ns_500.npy", allow_pickle=True)

target_number = 0

m_len = len(score_list_m)
e_len = len(score_list_e)
e20_len = len(score_list_e20)
e14_len = len(score_list_e14)

true_list = []
thres_score_list = []
syu_score_list = []

true_label = [target_number]

lamda = 1
slamda = 1


for i in range(0, e_len):
    true_list.append(1)

for i in range(0, e20_len):
    true_list.append(1)

for i in range(0, e14_len):
    true_list.append(1)

for i in range(0, m_len):
    true_list.append(0)

count = 0
 
# This section is for when test in multi label situation
#new_m = []
#new_e = []
#idx_list_m = []
#idx_list_e = []
#for i in range(0, m_len):
#    if m_Y_test[i] in true_label:
#        new_m.append(score_list_m[i])
#        true_list.append(0)
#        idx_list_m.append(i)
#        count += 1
#    else:
#        new_e.append(score_list_m[i])
#        true_list.append(1)
#        idx_list_e.append(i)
#
#score_list_m = new_m
#score_list_e = new_e[0: 0 + math.floor(len(score_list_m) * 0.5)]
#m_len = len(score_list_m)
#e_len = len(score_list_e)
#true_list= []
#for i in range(0, e_len):
#    true_list.append(1)
#for i in range(0, m_len):
#    true_list.append(0)

iter_range = range(target_number, target_number+1)

for i in range(0, e_len):
    score = []
    for j in iter_range:
        score.append(score_list_e[i][j][0] * lamda + score_list_e[i][j][1] * (1-lamda))

    thres_score_list.append(score)

for i in range(0, e20_len):
    score = []
    for j in iter_range:
        score.append(score_list_e20[i][j][0] * lamda + score_list_e20[i][j][1] * (1-lamda))

    thres_score_list.append(score)

for i in range(0, e14_len):
    score = []
    for j in iter_range:
        score.append(score_list_e14[i][j][0] * lamda + score_list_e14[i][j][1] * (1-lamda))

    thres_score_list.append(score)

new_score_n_norm = []
new_score_n_anomaly = []

for i in range(0, m_len):
    score = []
    for j in iter_range:
        score.append(score_list_m[i][j][0] * lamda + score_list_m[i][j][1] * (1-lamda))

    thres_score_list.append(score)

    if m_Y_test[i] in true_label:
        new_score_n_norm.append(thres_score_list[i])
    else:
        new_score_n_anomaly.append(thres_score_list[i])

#----------------------------------------------------------------------------------------------#
obj_list =[]

for i in range(0, e_len):
    score = []
    obj_size = get_object_size(e_X_test[i])
    obj_list.append(obj_size)
    
    for j in iter_range:
        score.append( (((score_list_e[i][j][0]+0)/obj_size)) * slamda + score_list_e[i][j][1] * (1-slamda))

    syu_score_list.append(score)

for i in range(0, e20_len):
    score = []
    obj_size = get_object_size(e_X_test20[i])
    obj_list.append(obj_size)
    
    for j in iter_range:
        score.append( (((score_list_e20[i][j][0]+0)/obj_size)) * slamda + score_list_e20[i][j][1] * (1-slamda))

    syu_score_list.append(score)

for i in range(0, e14_len):
    score = []
    obj_size = get_object_size(e_X_test14[i])
    obj_list.append(obj_size)
    
    for j in iter_range:
        score.append( (((score_list_e14[i][j][0]+0)/obj_size)) * slamda + score_list_e14[i][j][1] * (1-slamda))

    syu_score_list.append(score)

new_score_s_norm = []
new_score_s_anomaly = []
new_label_norm=[]
new_label_anomaly=[]

for i in range(0, m_len):
    score = []
    obj_size = get_object_size(m_X_test[i])
    obj_list.append(obj_size)

    for j in iter_range:
        score.append((((score_list_m[i][j][0]/obj_size))) * slamda + score_list_m[i][j][1] * (1-slamda))

    syu_score_list.append(score)

    if m_Y_test[i] in true_label:
        new_score_s_norm.append(syu_score_list[i])
        new_label_norm.append(0)
    else:
        new_score_s_anomaly.append(syu_score_list[i])
        new_label_anomaly.append(1)


re_thres_score_list = []
re_syu_score_list = []
re_true_list = []
for i in range(0,len(thres_score_list)):
    re_thres_score_list.append(min(thres_score_list[i]))

for i in range(0,len(syu_score_list)):
    re_syu_score_list.append(min(syu_score_list[i]))

for i in range(0,len(true_list)):
    re_true_list.append(true_list[i])

thres_score_list = re_thres_score_list
syu_score_list = re_syu_score_list
true_list = re_true_list

plt.hist(thres_score_list[0:e_len], histtype='bar',alpha=0.5, color='r', rwidth=1.25)
plt.hist(thres_score_list[e_len:e_len+e20_len], histtype='bar',alpha=0.5, color='b', rwidth=1.25)
plt.hist(thres_score_list[e_len+e20_len:e_len+e20_len+e14_len], histtype='bar',alpha=0.5, color='y', rwidth=1.25)
plt.hist(thres_score_list[e_len+e20_len+e14_len:e_len+e20_len+e14_len+m_len], histtype='bar',alpha=0.5, color='g', rwidth=1.25)
plt.xlim([0,600])
plt.ylim([0,600])
plt.figure()

plt.title('threshold: '+str(obj_threshold) + ', s: '+str(obj_square))
plt.hist(syu_score_list[0:e_len], histtype='bar',alpha=0.5, color='r', rwidth=1.25)
plt.hist(syu_score_list[e_len:e_len+e20_len], histtype='bar',alpha=0.5, color='b', rwidth=1.25)
plt.hist(syu_score_list[e_len+e20_len:e_len+e20_len+e14_len], histtype='bar',alpha=0.5, bins = 20, color='y', rwidth=1.25)
plt.hist(syu_score_list[e_len+e20_len+e14_len:e_len+e20_len+e14_len+m_len], histtype='bar',alpha=0.5, color='g', rwidth=1.25)
plt.xlim([0,1000])
plt.ylim([0,600])
plt.figure()

plt.hist(obj_list[0:e_len], histtype='bar',alpha=0.5, color='r')
plt.hist(obj_list[e_len:e_len+e20_len], histtype='bar',alpha=0.5, color='b')
plt.hist(obj_list[e_len+e20_len:e_len+e20_len+e14_len], histtype='bar',alpha=0.5, color='y')
plt.hist(obj_list[e_len+e20_len+e14_len:e_len+e20_len+e14_len+m_len], histtype='bar',alpha=0.5, color='g')
#plt.show() 


# Belows are for normal one
print('----------------------------------------')
normal_AUPRC = average_precision_score(true_list, thres_score_list)
print("AUPRC: "+ str(normal_AUPRC))

normal_AUC = roc_auc_score(true_list, thres_score_list)
print("AUC ", normal_AUC)

precision, recall, thresholds = precision_recall_curve(true_list, thres_score_list)

precision = list(precision)
recall = list(recall)
thresholds = list(thresholds)

precision.reverse()
recall.reverse()
thresholds.reverse()

f1_score = []
for i in range(0, len(precision)):
    f1_score.append(f_score(2,precision[i], recall[i]))

normal_f1 = max(f1_score)
print("Max F1 Score: ", str(normal_f1))

thresholds = [0] + thresholds
fig = plt.figure()
fig.set_size_inches(15, 15)
plt.plot(recall, precision)
plt.fill_between(recall, precision, 0, facecolor="red", alpha=0.2)
plt.xlabel("recall", fontsize = 24)
plt.ylabel("precision", fontsize = 24)
plt.text(recall[-1]/2, precision[-1]/2, 'AUPRC : ' + str(round(normal_AUPRC,3)), fontsize = 40)
#plt.show()


# Belows are for revision
syu_AUPRC = average_precision_score(true_list, syu_score_list)
print("AUPRC: "+ str(syu_AUPRC))

syu_AUC = roc_auc_score(true_list, syu_score_list)
print("AUC: ", syu_AUC)

precision, recall, thresholds = precision_recall_curve(true_list, syu_score_list)

precision = list(precision)
recall = list(recall)
thresholds = list(thresholds)

precision.reverse()
recall.reverse()
thresholds.reverse()

f1_score = []
for i in range(0, len(precision)):
    f1_score.append(f_score(2, precision[i], recall[i]))

syu_f1 = max(f1_score)
print("Max F1 Score: ", str(syu_f1))

thresholds = [0] + thresholds
fig = plt.figure()
fig.set_size_inches(15, 15)
plt.plot(recall, precision)
plt.fill_between(recall, precision, 0, facecolor="red", alpha=0.2)
plt.xlabel("recall", fontsize = 24)
plt.ylabel("precision", fontsize = 24)
plt.text(recall[-1]/2, precision[-1]/2, 'AUPRC : ' + str(round(syu_AUPRC,3)), fontsize = 40)
#plt.show()

print('----------------------------------------')
print("Diffrential")
print("AUPRC: ",syu_AUPRC - normal_AUPRC)
print("F1: ", syu_f1 - normal_f1)
print("AUC: ", syu_AUC - normal_AUC)