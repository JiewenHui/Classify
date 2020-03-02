import sys
import os
import subprocess
import time
batch_size = [30,50,64]
num_filters=[128,100]
l2_reg_lambda=[0,0.1,0.01,1,2,3]
learning_rate = [0.01,0.0001]
dropout_keep_prob = [0.1,0.4,0.5,0.6,1]
count = 0
for batch in batch_size:
	for num in num_filters:
		for d in dropout_keep_prob:
				for l2 in l2_reg_lambda:
					for rate in learning_rate:
						print ('The ', count, 'excue\n')
						count += 1
						if learning_rate== 0.0001:
							subprocess.call('python train.py --batch_size %d --num_filters %d --l2_reg_lambda %f --learning_rate %f --num_epochs %d --dropout_keep_prob %f' % (batch,num,l2,rate,50,d), shell = True)
						else:
							subprocess.call('python train.py --batch_size %d --num_filters %d --l2_reg_lambda %f --learning_rate %f --num_epochs %d --dropout_keep_prob %f' % (batch,num,l2,rate,50,d), shell = True)