# coding = utf-8
# learn motif embeddings use word2vec
import pandas as pd
import numpy as np
import itertools
import os
import time
from tqdm import tqdm
from collections import Counter
from collections import defaultdict
from gensim.models import Word2Vec

class GetEmedding():
	def __init__(self, dataset, embsize):
		self.dataset = dataset
		self.embsize = embsize
		if self.dataset == 'mo':
			self.motif_file_dir = './data/mo/motif_processed'
			self.sentence_dir = './data/mo/sentence/'
			self.relation = './data/mo/mo_sub_link.dat'
		if self.dataset == 'db':
			self.motif_file_dir = './data/db/motif_processed'
			self.sentence_dir = './data/db/sentence_noweight/'
			self.relation = './data/db/db_sub_link.dat'

	def read_motif(self):
		motype_dict = {}
		for root, _, files in os.walk(self.motif_file_dir):
			for file in files:
				tmp_np = np.array(pd.read_csv(root + '/' + file, sep='\t', header=None))
				motif_set = set([tuple(tmp_np[i]) for i in range(len(tmp_np))])
				for mot in motif_set:
					motype_dict[mot] = file.replace('.csv', '')
		print('motype_dict ready!')
		return motype_dict

	def read_original_node(self):
		tmp_np = np.array(pd.read_csv(self.relation, sep='\t', header=None))
		nodes = set(itertools.chain(*tmp_np))
		return nodes

	def get_node_emb(self):
		file_list = []
		for _, _, files in os.walk(self.sentence_dir):
			for file in files:
				tmp_df = pd.read_csv(self.sentence_dir+'/'+file, sep='\t', header=None)
				file_list.append(tmp_df)
		sentence_df = pd.concat(file_list,axis=0)
		sentence_df = sentence_df.dropna(axis=0).reset_index(drop=True)
		sentence = [list(sentence_df.iloc[i]) for i in sentence_df.index]

		def train(embed_size=self.embsize, window_size=5, workers=10, iter=5, negative=5, **kwargs):
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
		motif_emb_index = np.array(motif_emb_df.reset_index(drop=True))
		# creat motif array, same index, with motifs in 3 columns (str)
		motif_list_str = [i.replace('(','').replace(')','') for i in motif_emb_df.index]
		# all motif list type:int
		motif_array = np.array([i.split(', ') for i in motif_list_str], dtype=int)
		# all nodes type:int
		nodes = self.read_original_node()
		# motype_dict {(motif) : motif type}
		motype_dict = self.read_motif()
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
		for node in tqdm(nodes):
			index = np.where(motif_array == node)[0]
			# tmp_dict = {motype:[emb sum]}
			tmp_dict = {}
			for motype in motype_set:
				# motif_df.index == motif_array.index
				tmpl = set(motif_df[motif_df[3] == motype].index)
				# sum motif emb of motype
				sumindex = list(set(index) & tmpl)
				if sumindex:
					tmp_dict[motype] = sum(motif_emb_index[sumindex])/(len(sumindex)+1)
				else:
					tmp_dict[motype] = np.zeros(np.shape(motif_emb_index)[1])
			df1 = pd.DataFrame(tmp_dict).T
			df1.sort_index(inplace = True)
			node_emb[node] = np.array(df1).flatten()

		node_emb_df = pd.DataFrame(node_emb).T
		node_emb_df.to_csv('./data/' + self.dataset + '/' + str(self.embsize)+'emb.dat', sep='\t', header=0, index=True)
		print('node emb write out!')
		return node_emb



if __name__ == '__main__':
	for size in [10,25,50,100,200]:
		GetEmedding('mo',size).get_node_emb()


