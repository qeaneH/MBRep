import os
import random
from tqdm import tqdm
from itertools import combinations
from collections import Counter
import itertools
seed = 2020

class ColdstartPreprocess():
	def testset_preprocess(self):
		testset_links = pd.read_csv('./data/cn/cn_test_link.dat', sep='\t', header=None)
		node_type = pd.read_csv('./data/cn/cn_nodes_type.dat', sep='\t', header=None)
		author_node = set(node_type[node_type[1]==0][0].tolist())
		paper_node = set(node_type[node_type[1]==1][0].tolist())
		venue_node = set(node_type[node_type[1]==2][0].tolist())

		# random choose 10% nodes in test set as cold-start nodes
		a = []
		p = []
		v = []
		for node in list(set(testset_links[1].tolist()+testset_links[0].tolist())):
			if node in author_node:
				a.append(node)
			if node in paper_node:
				p.append(node)
			if node in venue_node:
				v.append(node)
		random.seed(seed)
		a_new = random.sample(a,446) # len(a)*10% = 446
		p_new = random.sample(p,1202) # len(p)*10% = 1202
		v_new = random.sample(v,1) # len(v)*10% = 1
		return a_new, p_new, v_new

	def motif_preprocess(self):
		a_new, p_new, v_new = self.testset_preprocess()
		cold_start_nodes = a_new + p_new + v_new
		motif_dict = {}
		for root, _, files in os.walk('./data/cn/motif_processed'):
			for file in files:
				tmp_df = pd.read_csv(root + '/' + file, sep='\t', header=None)
				motif_dict[file.replace('.csv', '')] = np.array(tmp_df)

		# remove motif instances containing a_new,p_new,v_new from orignal network
		for key, value in motif_dict.items():
			tmp_list = []
			for cs_node in tqdm(cold_start_nodes):
				tmp_list.append(value == cs_node)
			remain_mot_index = np.where(sum(tmp_list).sum(axis=1)==0)
			pd.DataFrame(value[remain_mot_index]).to_csv(
				'./data/cn/cold_start/motif_remain/' + key + '.csv', sep='\t', index=0, header=0)
			cs_newmotif_index = np.where(sum(tmp_list).sum(axis=1)!=0)
			pd.DataFrame(value[cs_newmotif_index]).to_csv(
				'./data/cn/cold_start/motif_new/' + key + '.csv', sep='\t', index=0, header=0)
			print(key, 'ready!')