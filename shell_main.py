import sys
import os
import subprocess
import time

batch_size =[30]#[30,50,64]# [30,50,64]
num_filters=[128]#[128,100,80,64]#[128,100]
l2_reg_lambda=[0]#[0,0.0001]#[0,0.1,0.01,1,2,3]
learning_rate =[0.001]#[0.001,0.003,0.005,0.0001]# [0.01,0.0001]
dropout_keep_prob =[0.5]#[0.1,0.4,0.5,0.6]# [0.1,0.4,0.5,0.6,1]
count = 0
epoch = 50
print('batch_size :',batch_size)
print('num_filters :',num_filters)
print('l2_reg_lambda :',l2_reg_lambda)
print('learning_rate :',learning_rate)
print('dropout_keep_prob :',dropout_keep_prob)
for batch in batch_size:
	for num in num_filters:
		for d in dropout_keep_prob:
				for l2 in l2_reg_lambda:
					for rate in learning_rate:
						print ('The ', count, 'excue\n')
						count += 1
						subprocess.call('python train.py --batch_size %d --num_filters %d --l2_reg_lambda %f --learning_rate %f --num_epochs %d --dropout_keep_prob %f' % (batch,num,l2,rate,epoch,d), shell = True)
