# coding = utf-8
# motif walk based alias
# author 'Hu Qian'
'''
func explanations
read_motif:             load motif files
node_neighbor_indexsum: built all neighbor nodes of the node index sum in different type motif
preprocess_motif_probs: caculate Alias sample
motif_walk:             motif walk sequence priority: Alias select motif type > find max weight motif > random choice
'''

import pandas as pd
import numpy as np
from scipy import sparse
from collections import defaultdict
from alias import create_alias_table, alias_sample
import os
import random
import time
import copy
import itertools
from collections import Counter
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
import sys
sys.path.append('C:/MBRep/')

class MotifWalk():

	def __init__(self, dataset, walk_length=100):
		self.dataset = dataset
		self.walk_length = walk_length
		if self.dataset == 'db':
			self.motif_file_dir = './data/db/motif_processed'
			self.relation_path = './data/db/db_sub_link.dat'
		if self.dataset == 'mo':
			self.motif_file_dir = './data/mo/motif_processed'
			self.relation_path = './data/mo/mo_sub_link.dat'

	def read_motif(self):
		motif_dict = {}
		motif_length_dict = {}
		for root, _, files in os.walk(self.motif_file_dir):
			for file in files:
				tmp_df = pd.read_csv(root + '/' + file, sep='\t', header=None)
				motif_dict[file.replace('.csv', '')] = np.array(tmp_df)
				motif_length_dict[file.replace('.csv', '')] = len(tmp_df)

		return motif_dict, motif_length_dict

	def walk(self):
		def preprocess_motif_probs():
			# creat alias table
			motif_pro = [float(value) / sum(motif_length_dict.values()) for value in motif_length_dict.values()]
			accept, alias = create_alias_table(motif_pro)
			return accept, alias
		# motif_dict as motif_type as keys, motif_np as values
		motif_dict, motif_length_dict = self.read_motif()
		# motif_index_dict {'APA':{index_list}}
		motif_index_dict = {}
		for motype, motnp in motif_dict.items():
			motif_index_dict[motype] = list(range(len(motnp)))
		# motype_list for index motif type
		motype_list = list(motif_length_dict.keys())
		# accept, alias are Alias sample
		accept, alias = preprocess_motif_probs()
		# walk priority: alias > weight > random choice
		for mot_type in motype_list:
			walks = [] # collect all walk
			mot_np = motif_dict[mot_type]
			def motif_walk(index):
				start0 = time.process_time()
				walk = [] # motif sequence
				walk.append(tuple(mot_np[index]))
				while len(walk) < self.walk_length:
					# use Alias sample to choose next motif type
					next_mot_type = motype_list[alias_sample(accept, alias)]
					next_mot_np = motif_dict[next_mot_type]
					index_seleced = random.choice(motif_index_dict[next_mot_type])
					next_mot = tuple(next_mot_np[index_seleced])
					walk.append(next_mot)
				print('\r'+'each walk needs:', time.process_time() - start0, end='')
				return walk

			pool = ThreadPool(16)
			start = time.process_time()
			walks = pool.map(motif_walk, range(len(mot_np)))
			pool.close()
			pool.join()
			# for index in range(len(mot_np)):
			# 	walk = motif_walk(index)
			# 	walks.append(walk)
			df = pd.DataFrame(walks)
			df.to_csv('./data/'+self.dataset+'/sentence_noweight/'+mot_type+'_walk.dat', sep='\t', header=0, index=0)
			print('\r',mot_type+' walk needs:', time.process_time() - start,'\n',time.strftime('%y-%m-%d %I:%M:%S %p'))


if __name__ == '__main__':
	MotifWalk('db').walk()