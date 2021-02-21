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
import itertools
from collections import Counter
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
import sys
sys.path.append('C:/MbRep/')

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
		def motif_index():
			tmp_df = pd.read_csv(self.relation_path, sep='\t', header=None)
			adj_dict = {}
			for _, row in tqdm(tmp_df.iterrows()):
				if row[0] not in adj_dict.keys():
					adj_dict[row[0]] = set(tuple([row[1]]))
				else:
					adj_dict[row[0]].add(row[1])
			for key, value in adj_dict.items():
				adj_dict[key] = list(value)
			print('Nodes -> neighbor ready!')
			# mot_mask_dict {motif_type:{node:motif_np mask}}
			mot_mask_dict = {}
			tmp_np = np.array(tmp_df)
			all_nodes = set(itertools.chain(*tmp_np))
			for key, value in motif_dict.items():
				value = np.array(value)
				index_dict = {}
				for node in all_nodes:
					if node in value:
						index_dict[node] = sparse.coo_matrix((value == node))
				mot_mask_dict[key] = index_dict
			print('Node -> motif_mask ready!')
			return adj_dict, mot_mask_dict

		def node_neighbor_indexsum():
			adj_dict, mot_mask_dict = motif_index()
			# node_neighbor_indexsum {node : {motif_type : sum(motif_np mask of node's neighbor)}}
			node_neighbor_indexsum = {}
			for n, ngh in tqdm(adj_dict.items()):
				ngb_sum = {}
				for motype in motif_dict.keys():
					tmp_list = [mot_mask_dict[motype][i].toarray()
					            for i in ngh
					            if i in mot_mask_dict[motype].keys()]
					if tmp_list:
						ngb_sum[motype] = sparse.coo_matrix(sum(tmp_list))
						node_neighbor_indexsum[n] = ngb_sum
			print('Nodes neighbor -> motif_index_sum ready!')
			return node_neighbor_indexsum

		def preprocess_motif_probs():
			# creat alias table
			motif_pro = [float(value) / sum(motif_length_dict.values()) for value in motif_length_dict.values()]
			accept, alias = create_alias_table(motif_pro)
			return accept, alias

		# motif_dict as motif_type as keys, motif_np as values
		motif_dict, motif_length_dict = self.read_motif()
		# motif_index_dict {'APA':{motif:index}}
		motif_index_dict = {}
		for motype, motnp in motif_dict.items():
			index_dict = {}
			for ind in range(len(motnp)):
				index_dict[tuple(motnp[ind])] = ind
			motif_index_dict[motype] = index_dict

		# motype_list for index motif type
		motype_list = list(motif_length_dict.keys())
		# lookup dict is realized by node->motif_type_sum index dict
		lud = node_neighbor_indexsum()
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
				# esxit_mot_index_dict {'motif type' : [index of exsit motif in walk]}
				exsit_mot_index_dict = defaultdict(list)
				exsit_mot_index_dict[mot_type].append(motif_index_dict[mot_type][walk[-1]])
				while len(walk) < self.walk_length:
					# use Alias sample to choose next motif type
					next_mot_type = motype_list[alias_sample(accept, alias)]
					next_mot_np = motif_dict[next_mot_type]
					np_list = [lud[i][next_mot_type].toarray()
					           for i in walk[-1]
					           if i in lud.keys() and next_mot_type in lud[i].keys()]
					if np_list:
						tmp_zero_np = sum(np_list).sum(axis=1)
						if next_mot_type in exsit_mot_index_dict.keys():
							# set the index of motif which has been in walk weight to 0
							tmp_zero_np[exsit_mot_index_dict[next_mot_type]] = 0

						max_count = tmp_zero_np.max()
						index_for_choice = np.where(tmp_zero_np == max_count)[0]
						index_seleced = random.choice(index_for_choice)
						next_mot = tuple(next_mot_np[index_seleced])
						walk.append(next_mot)
						exsit_mot_index_dict[next_mot_type].append(motif_index_dict[next_mot_type][walk[-1]])
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
			df.to_csv('./data/'+self.dataset+'/cold_start/sentence/'+mot_type+'_walk.dat', sep='\t', header=0, index=0)
			print('\r',mot_type+' walk needs:', time.process_time() - start,'\n',time.strftime('%y-%m-%d %I:%M:%S %p'))


if __name__ == '__main__':
	MotifWalk('db').walk()