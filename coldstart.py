import networkx as nx
import numpy as np
import pandas as pd
import os
import random
from tqdm import tqdm
from itertools import combinations
from collections import Counter
from gensim.models import Word2Vec
import itertools
seed = 2020

class ColdstartPreprocess():
	def __init__(self,dataset):
		self.dataset = dataset
		if dataset == 'cn':
			self.cold_start_sub_link = './data/cn/cold_start/cold_start_sub_link.csv'
			self.motif_remain = './data/cn/cold_start/motif_remain'
			self.motif_new = './data/cn/cold_start/motif_new'
			self.test_link = './data/cn/cn_test_link.dat'
			self.sentence = './data/cn/cold_start/sentence'

	def read_original_node(self):
		tmp_np = np.array(pd.read_csv(self.cold_start_sub_link, sep='\t', header=None))
		nodes = set(itertools.chain(*tmp_np))
		return nodes

	def new_node(self):
		# actually cold-start nodes account for more than 10% ,and the number of nodes we removed from the test set
		# would result in a corresponding decrease in the number of nodes of the test set
		l = []
		for root, _, files in os.walk(self.motif_remain):
			for file in files:
				df = pd.read_csv(root + '/' + file, sep='\t', header=None)
				l.append(df)
		df1 = pd.concat(l, sort=False)
		nodes_remain = list(set(itertools.chain(*np.array(df1))))

		test_set = np.array(pd.read_csv(self.test_link, sep='\t', header=None))
		nodes_intest = list(set(itertools.chain(*test_set)))

		cold_start_nodes = []
		for node in nodes_intest:
			if node not in nodes_remain:
				cold_start_nodes.append(node)
		# finally cold start nodes:
		# author 978, paper 1732, venue 1
		return cold_start_nodes

	def read_motif(self,motif_dir):
		motype_dict = {}
		for root, _, files in os.walk(motif_dir):
			for file in files:
				tmp_np = np.array(pd.read_csv(root + '/' + file, sep='\t', header=None))
				motif_set = set([tuple(tmp_np[i]) for i in range(len(tmp_np))])
				for mot in motif_set:
					motype_dict[mot] = file.replace('.csv', '')
		print('motype_dict ready!')
		return motype_dict

	def new_motif(self,motif_dir):
		motif_dict = {}
		for root, _, files in os.walk(motif_dir):
			for file in files:
				tmp_df = pd.read_csv(root + '/' + file, sep='\t', header=None)
				motif_dict[file.replace('.csv', '')] = np.array(tmp_df)
		return motif_dict

	def skipgram(self):
		file_list = []
		sentence_dir = self.sentence
		for _, _, files in os.walk(sentence_dir):
			for file in files:
				tmp_df = pd.read_csv(sentence_dir+'/'+file, sep='\t', header=None)
				file_list.append(tmp_df)
		sentence_df = pd.concat(file_list,axis=0)
		sentence_df = sentence_df.dropna(axis=0).reset_index(drop=True)
		sentence = [list(sentence_df.iloc[i]) for i in sentence_df.index]

		def train(embed_size=200, window_size=5, workers=3, iter=5, negative=5, **kwargs):
			kwargs["sentences"] = sentence
			kwargs["min_count"] = kwargs.get("min_count", 0)
			kwargs["size"] = embed_size
			kwargs["sg"] = 1
			kwargs["hs"] = 0
			kwargs["workers"] = workers
			kwargs["window"] = window_size
			kwargs["iter"] = iter
			kwargs["negative"] = negative
			print('Learning embedding vectors')
			model = Word2Vec(**kwargs)
			print('Learning embedding vectors done!')
			return model

		model = train(iter = 3)
		embeddings = {}
		for motif in sentence_df[0]:
			embeddings[motif] = model.wv[motif]
		# motif_emb_df index:motif(str)
		motif_emb_df = pd.DataFrame(embeddings).T
		return motif_emb_df

	def node_emb(self):
		motif_emb_df = self.skipgram()
		motif_emb_index = np.array(motif_emb_df.reset_index(drop=True))
		# creat motif array, same index, with motifs in 3 columns (str)
		motif_list_str = [i.replace('(','').replace(')','') for i in motif_emb_df.index]
		# all motif list type:int
		motif_array = np.array([i.split(', ') for i in motif_list_str], dtype=int)
		# all nodes type:int
		nodes = self.read_original_node()
		# motype_dict {(motif) : motif type}
		motype_dict = self.read_motif(self.motif_remain)
		motif_df = pd.DataFrame(motif_array)
		# motif_df
		# 0---1---2---3
		# m o t i f---motype
		tmp_list = []
		for ind in range(len(motif_array)):
			if tuple(motif_array[ind]) in motype_dict.keys():
				tmp_list.append(motype_dict[tuple(motif_array[ind])])
		motif_df[3] = tmp_list

		# note that different motif type maybe contains the same motif
		# that is to say: the same motif may satisfy different motif type!!!
		motype_set = set(motype_dict.values())
		node_emb = {}
		print('nodes cold start ...')
		cs_nodes = self.new_node()
		new_motype_dict = self.new_motif(self.motif_new)
		# cn_mt_rn_dict = {cold start node : {motif type : [relate nodes]}}
		# cn_mt_rn_dict means:
		# cold start node -> can generate motif type -> with these nodes of original network
		cn_mt_rn_dict = {}
		for cs_node in tqdm(cs_nodes):
			tmp_dict = {}
			for mtype, mnp in new_motype_dict.items():
				if cs_node in mnp:
					cs_relate_nodelist = list(set(mnp[np.where(mnp==cs_node)[0]].flatten()))
					for new_node in cs_nodes:
						if new_node in cs_relate_nodelist:
							cs_relate_nodelist.remove(new_node)
					tmp_dict[mtype] = cs_relate_nodelist
			cn_mt_rn_dict[cs_node] = tmp_dict

		# calculate cold start node embedding
		for cs_node in tqdm(cs_nodes):
			tmp_dict = {}
			for motype in ['APA','APP','APP1','APP2','APV','PAP','PPP','PPP1','PPP2','PPP3','PVP','PVP1','VPP','VPP1']:
				tmp_sum = []
				# motif_df.index == motif_array.index
				if motype in cn_mt_rn_dict[cs_node].keys():
					for relate_node in cn_mt_rn_dict[cs_node][motype]:
						relate_index = np.where(motif_array==relate_node)[0]
						# need_sum_index : the cold start node generate 'motype'-class motif with original nodes
						# we sum and average these 'motype'-class motif embedding for cold start nodes
						need_sum_index = list(set(motif_df[motif_df[3]==motype].index) & set(relate_index))

						tmp_sum.append(sum(motif_emb_index[need_sum_index])/(len(need_sum_index)+1))
					if tmp_sum:
						tmp_dict[motype] = sum(tmp_sum)/len(tmp_sum)
					else:
						tmp_dict[motype] = np.zeros(np.shape(motif_emb_index)[1])
				else:
					tmp_dict[motype] = np.zeros(np.shape(motif_emb_index)[1])

			df2 = pd.DataFrame(tmp_dict).T
			df2.sort_index(inplace = True)
			node_emb[cs_node] = np.array(df2).flatten()

		node_emb_df = pd.DataFrame(node_emb).T
		node_emb_df.to_csv('./data/' + self.dataset + '/cold_start/coldstart_nodes_emb.dat', sep='\t', header=0, index=True)
		print('node emb write out!')
		return node_emb

if __name__ == '__main__':
	ColdstartPreprocess('cn').node_emb()

