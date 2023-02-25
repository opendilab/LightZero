import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import copy
module_path = os.path.dirname(__file__)

def open_file_oneseed(path : str) -> list:
    fp=open(path,'r')
    read_list=[]
    step = []
    value = []
    for line in fp:
        line=line.replace("\n","")
        read_list.append(list(line.split(",")))
    fp.close()
    label = read_list[0]
    database = read_list[1:]
    for i in range(len(database)):
        step.append(float(database[i][1]))
        value.append(float(database[i][2]))
    return {'step':np.array(step), 'value':np.array(value)}

def open_file(path : str) -> list:
    fp=open(path,'r')
    read_list=[]
    step = []
    value = []
    step2 = []
    value2 = []
    value3 = []
    for line in fp:
        line=line.replace("\n","")
        read_list.append(list(line.split(",")))
    fp.close()
    label = read_list[0]
    database = read_list[1:]
    for i in range(len(database)):
        if i%1 == 0:
            if database[i][2]=='':
                database[i][2]=database[i-1][2]
            # if database[i][3]=='':
            #     database[i][3]=database[i-1][3]
            # if database[i][4]=='':
                # database[i][4]=database[i-1][4]
            step.append(float(database[i][1]))
            value.append(float(database[i][2]))
            # value2.append(float(database[i][3]))
            # value3.append(float(database[i][4]))
    # return {'step':np.array(step), 'value':np.array(value), 'value2':np.array(value2)}
    return {'step':np.array(step), 'value':np.array(value)}


def moving_average(interval, windowsize):
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'same')
    return re


from matplotlib.ticker import FuncFormatter
def y_update_scale_value(temp, position):
    # result = temp//1e+6
    # return "{}M".format(int(result))
    if temp/1e+6 % 1 == 0:
        result = int(temp//1e+6)
    else:
        result = temp/1e+6
    return "{}M".format(result)

sns.set()
# windowsize=10
windowsize=5

f = plt.figure(figsize=(7, 5.5))
size1=23
size2=16
size3=2.5
size4=12



data = open_file('./run-mspacman_sampled_efficientzero_k9_ns50_upc1000_ic1_rr03_seed0_log_serial-tag-evaluator_step_reward_mean.csv')
data_deepcopy = copy.deepcopy(data)
step_seeds = np.concatenate((data['step'],data['step'],data['step']))
data['value'] = moving_average(data['value'], windowsize)
# data['value2'] = moving_average(data['value2'], windowsize)
# data['value3'] = moving_average(data['value3'], windowsize)
data['value'][:windowsize]=data_deepcopy['value'][:windowsize]
# data['value2'][:windowsize]=data_deepcopy['value2'][:windowsize]
# data['value3'][:windowsize]=data_deepcopy['value3'][:windowsize]
data['value'][-windowsize:]=data_deepcopy['value'][-windowsize:]
# data['value2'][-windowsize:]=data_deepcopy['value2'][-windowsize:]
# data['value3'][-windowsize:]=data_deepcopy['value3'][-windowsize:]
# value_seeds = np.concatenate((data['value'],data['value2'],data['value2']))
value_seeds = np.concatenate((data['value'],data['value'],data['value']))
sns.lineplot(x=step_seeds,y=value_seeds,label='Sampled EfficientZero (K=9)')

data = open_file('./run-mspacman_sampled_efficientzero_k5_ns50_upc1000_ic1_rr03_seed0_log_serial-tag-evaluator_step_reward_mean.csv')
data_deepcopy = copy.deepcopy(data)
step_seeds = np.concatenate((data['step'],data['step'],data['step']))
data['value'] = moving_average(data['value'], windowsize)
# data['value2'] = moving_average(data['value2'], windowsize)
# data['value3'] = moving_average(data['value3'], windowsize)
data['value'][:windowsize]=data_deepcopy['value'][:windowsize]
# data['value2'][:windowsize]=data_deepcopy['value2'][:windowsize]
# data['value3'][:windowsize]=data_deepcopy['value3'][:windowsize]
data['value'][-windowsize:]=data_deepcopy['value'][-windowsize:]
# data['value2'][-windowsize:]=data_deepcopy['value2'][-windowsize:]
# data['value3'][-windowsize:]=data_deepcopy['value3'][-windowsize:]
# value_seeds = np.concatenate((data['value'],data['value2'],data['value2']))
value_seeds = np.concatenate((data['value'],data['value'],data['value']))
sns.lineplot(x=step_seeds,y=value_seeds,label='Sampled EfficientZero (K=5)')

data = open_file('./run-mspacman_sampled_efficientzero_k3_ns50_upc1000_ic1_rr03_seed0_log_serial-tag-evaluator_step_reward_mean.csv')
data_deepcopy = copy.deepcopy(data)
step_seeds = np.concatenate((data['step'],data['step'],data['step']))
data['value'] = moving_average(data['value'], windowsize)
# data['value2'] = moving_average(data['value2'], windowsize)
# data['value3'] = moving_average(data['value3'], windowsize)
data['value'][:windowsize]=data_deepcopy['value'][:windowsize]
# data['value2'][:windowsize]=data_deepcopy['value2'][:windowsize]
# data['value3'][:windowsize]=data_deepcopy['value3'][:windowsize]
data['value'][-windowsize:]=data_deepcopy['value'][-windowsize:]
# data['value2'][-windowsize:]=data_deepcopy['value2'][-windowsize:]
# data['value3'][-windowsize:]=data_deepcopy['value3'][-windowsize:]
# value_seeds = np.concatenate((data['value'],data['value2'],data['value2']))
value_seeds = np.concatenate((data['value'],data['value'],data['value']))
sns.lineplot(x=step_seeds,y=value_seeds,label='Sampled EfficientZero (K=3)')

data = open_file('./run-mspacman_sampled_efficientzero_k2_ns50_upc1000_ic1_rr03_seed0_log_serial-tag-evaluator_step_reward_mean.csv')
data_deepcopy = copy.deepcopy(data)
step_seeds = np.concatenate((data['step'],data['step'],data['step']))
data['value'] = moving_average(data['value'], windowsize)
# data['value2'] = moving_average(data['value2'], windowsize)
# data['value3'] = moving_average(data['value3'], windowsize)
data['value'][:windowsize]=data_deepcopy['value'][:windowsize]
# data['value2'][:windowsize]=data_deepcopy['value2'][:windowsize]
# data['value3'][:windowsize]=data_deepcopy['value3'][:windowsize]
data['value'][-windowsize:]=data_deepcopy['value'][-windowsize:]
# data['value2'][-windowsize:]=data_deepcopy['value2'][-windowsize:]
# data['value3'][-windowsize:]=data_deepcopy['value3'][-windowsize:]
# value_seeds = np.concatenate((data['value'],data['value2'],data['value2']))
value_seeds = np.concatenate((data['value'],data['value'],data['value']))
sns.lineplot(x=step_seeds,y=value_seeds,label='Sampled EfficientZero (K=2)')

plt.gca().xaxis.set_major_formatter(FuncFormatter(y_update_scale_value))
plt.tick_params(axis='both', labelsize=size2)
plt.title('MsPacmanNoFrameskip-v4',fontsize=size1)
plt.legend(loc='lower right',fontsize=size4) # 显示图例
plt.xlabel('Env Steps',fontsize=size1)
plt.ylabel('Return',fontsize=size1)
plt.tight_layout()
f.savefig('mspacman_rr03_k_500k.pdf',bbox_inches='tight')
plt.show()