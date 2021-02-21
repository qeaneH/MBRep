# coding = utf-8

import networkx as nx
import numpy as np
import pandas as pd

from tqdm import tqdm
from collections import Counter
import os
from shutil import copyfile
import time

import dotmotif
from dotmotif.executors import NetworkXExecutor

class ExtractMotif():
	def __init__(self, dataset):
		self.dataset = dataset

	def cn_G(self):
		links = pd.read_csv('./data/cn/cold_start/cold_start_sub_link.csv', sep='\t', header=None)
		node_type = pd.read_csv('./data/cn/cn_nodes_type.dat', sep='\t', header=None)
		author_node = set(node_type[node_type[1]==0][0].tolist())
		paper_node = set(node_type[node_type[1]==1][0].tolist())
		venue_node = set(node_type[node_type[1]==2][0].tolist())
		G = nx.DiGraph()
		G.add_edges_from([tuple(i) for i in links.values])
		for node in G.nodes:
			if node in author_node:
				G.nodes[node]['type'] = 'author'
			if node in paper_node:
				G.nodes[node]['type'] = 'paper'
			if node in venue_node:
				G.nodes[node]['type'] = 'venue'
		return G

	def mo_G(self):
		links = pd.read_csv('./data/mo/mo_sub_link.dat', sep='\t', header=None)
		G = nx.DiGraph()
		G.add_edges_from([tuple(i) for i in links.values])
		for node in G.nodes():
			if 0 <= node <= 20:
				G.nodes[node]['type'] = 'occupation' # 21
			if 101 <= node <= 118:
				G.nodes[node]['type'] = 'genre' # 18
			if 10001 <= node <= 10201:
				G.nodes[node]['type'] = 'user' # 200
			if 20001 <= node <= 23952:
				G.nodes[node]['type'] = 'movie' # 1595
		return G

	def db_G(self):
		links = pd.read_csv('./data/db/db_sub_link.dat', sep='\t', header=None)
		G = nx.DiGraph()
		G.add_edges_from([tuple(i) for i in links.values])
		for node in G.nodes():
			if 0 <= node <= 22347:
				G.nodes[node]['type'] = 'book'  #8213
			if 30001 <= node <= 31076:
				G.nodes[node]['type'] = 'user'  #1000
			if 40008 <= node <= 40450:
				G.nodes[node]['type'] = 'location'  #135
			if 50002 <= node <= 60801:
				G.nodes[node]['type'] = 'author'  #4255
			if 70001 <= node <= 71815:
				G.nodes[node]['type'] = 'publisher'  #983
		return G

	def find_motif(self):
		# find motif and write out motif_df use dotmotif
		'''
		dotmotif need to cite
		@article{matelsky_2020_dotmotif,
				doi = {10.1101/2020.06.08.140533},
				url = {https://www.biorxiv.org/content/10.1101/2020.06.08.140533v1},
				year = 2020,
				month = {june},
				publisher = {BiorXiv},
				author = {Matelsky, Jordan K. and Reilly, Elizabeth P. and Johnson,Erik C. and Wester, Brock A. and Gray-Roncal, William},
				title = {{Connectome subgraph isomorphisms and graph queries with DotMotif}},
				journal = {BiorXiv}
				}
		'''
		if self.dataset == 'cn':
			G = self.cn_G()
		if self.dataset == 'mo':
			G = self.mo_G()
		if self.dataset == 'db':
			G = self.db_G()

		for root, _, files in os.walk('./data/'+ self.dataset + '/motif'):
			for file in files:
				start = time.process_time()
				motif_file_name = root + '/' + file
				dm = dotmotif.dotmotif().from_motif(motif_file_name)
				tmp_df = pd.DataFrame()
				tmp_list = []
				for i in NetworkXExecutor(graph=G).find(dm):
					tmp_list.append(i)
				if tmp_list:
					tmp_df = tmp_df.append(tmp_list, ignore_index=True)
					tmp_df = tmp_df.reindex(columns=['A', 'B', 'C']) # see floder '/motif'
					tmp_df.to_csv('./data/'+ self.dataset + '/motif_unprocess/' + file, sep='\t', index=0, header=0)
				else:
					print(file, ' not exsit')
				elapsed = time.process_time() - start
				print(file, 'motif created. time cost:', elapsed)
				print(time.strftime('%y-%m-%d %I:%M:%S %p'))

	def symetric_motif_process(self):
		# for symmetric motif, need to delete the three same node motifs
		# check below in my paper
		if self.dataset == 'cn':
			sym_motif = ['APA', 'PAP', 'PVP', 'PPP', 'PPP3', 'PPP4']
		if self.dataset == 'mo':
			sym_motif = ['GMM', 'GMU', 'MGG', 'MUU', 'OUM', 'OUU', 'UMM', 'UOO']
		if self.dataset == 'db':
			sym_motif = ['LUU', 'ULL', 'UBB', 'BUU', 'ABB', 'BAA', 'PBB', 'BPP']

		file_dir = './data/' + self.dataset + '/motif_unprocess'

		def check_symetric(df):
			check_set = set()
			del_list = []
			for index in df.index:
				every_row = df.iloc[index].tolist() 
				every_row.sort()
				if tuple(every_row) not in check_set:
					check_set.add(tuple(every_row))
				else:
					del_list.append(index)
			df = df.drop(del_list)
			return df

		for root, _, files in os.walk(file_dir):
			new_floder = root.replace('unprocess','') + 'processed'
			if not os.path.exists(new_floder):
				os.makedirs(new_floder)
			for file in files:
				if file.replace('.csv','') in sym_motif:
					print('symetric processing ',file)
					start = time.process_time()
					tmp_df = pd.read_csv(root + '/' + file, sep='\t', header=None)
					if not tmp_df.empty:
						tmp_df = check_symetric(tmp_df)
						tmp_df.to_csv(new_floder + '/' + file, sep='\t', index=0, header=0)
						elapsed = time.process_time() - start
						print(file, 'motif processed. time cost:', elapsed,time.strftime('%y-%m-%d %I:%M:%S %p'))
				else:
					print('copy un-symetric motif ',file)
					copyfile(root + '/' + file, new_floder + '/' + file)

if __name__ == '__main__':
	ExtractMotif('db').find_motif()
	ExtractMotif('db').symetric_motif_process()
