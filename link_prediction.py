# coding = utf-8

import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning

seed = 1
max_iter = 3000
np.random.seed(seed)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

class LinkPred():
	def __init__(self,dataset):
		if dataset == 'mo':
			self.test_link_path = './data/mo/mo_test_link.dat'
			self.node_emb_path = './data4deepwalk/mo10.embeddings'

	def node_emb_dict(self):
		emb_df = pd.read_csv(self.node_emb_path, sep='\s', header=None)
		emb_df.set_index([0], inplace=True)
		emb_dict = emb_df.T.to_dict('list')

		# node2vec metapath2vec
		# tmp_emb = pd.read_csv(self.node_emb_path, index_col=0, sep='\s', header=None, engine='python')
		# emb_dict = tmp_emb.T.to_dict('list')
		return emb_dict

	def cross_validation(self, edge_embs, edge_labels):
		auc, mrr = [], []
		seed_nodes, num_nodes = np.array(list(edge_embs.keys())), len(edge_embs)

		skf = KFold(n_splits=5, shuffle=True, random_state=seed)
		for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros((num_nodes, 1)), np.zeros(num_nodes))):

			print(f'Start Evaluation Fold {fold}!')
			train_edge_embs, test_edge_embs, train_edge_labels, test_edge_labels = [], [], [], []
			for each in train_idx:
				train_edge_embs.append(edge_embs[seed_nodes[each]])
				train_edge_labels.append(edge_labels[seed_nodes[each]])
			for each in test_idx:
				test_edge_embs.append(edge_embs[seed_nodes[each]])
				test_edge_labels.append(edge_labels[seed_nodes[each]])
			train_edge_embs, test_edge_embs, train_edge_labels, test_edge_labels = np.concatenate(
				train_edge_embs), np.concatenate(test_edge_embs), np.concatenate(train_edge_labels), np.concatenate(
				test_edge_labels)

			clf = LinearSVC(random_state=seed, max_iter=max_iter)
			clf.fit(train_edge_embs, train_edge_labels)
			preds = clf.predict(test_edge_embs)
			auc.append(roc_auc_score(test_edge_labels, preds))

			confidence = clf.decision_function(test_edge_embs)
			curr_mrr, conf_num = [], 0
			for each in test_idx:
				test_edge_conf = np.argsort(-confidence[conf_num:conf_num + len(edge_labels[seed_nodes[each]])])
				rank = np.empty_like(test_edge_conf)
				rank[test_edge_conf] = np.arange(len(test_edge_conf))
				tmp = rank[np.argwhere(edge_labels[seed_nodes[each]] == 1).flatten()]
				if tmp.any():
					curr_mrr.append(1 / (1 + np.min(tmp)))
				conf_num += len(rank)
			mrr.append(np.mean(curr_mrr))
			assert conf_num == len(confidence)
		print(np.mean(auc), np.mean(mrr))

		return np.mean(auc), np.mean(mrr)

	def lp_evaluate(self):
		emb_dict = self.node_emb_dict()
		posi, nega = defaultdict(set), defaultdict(set)
		with open(self.test_link_path, 'r') as test_file:
			for line in test_file:
				left, right, label = line[:-1].split('\t')
				if label == '1':
					posi[int(left)].add(int(right))
				elif label == '0':
					nega[int(left)].add(int(right))

		edge_embs, edge_labels = defaultdict(list), defaultdict(list)
		for left, rights in posi.items():
			for right in rights:
				edge_embs[left].append(np.array(emb_dict[left]) * np.array(emb_dict[right]))
				edge_labels[left].append(1)
		for left, rights in nega.items():
			for right in rights:
				edge_embs[left].append(np.array(emb_dict[left]) * np.array(emb_dict[right]))
				edge_labels[left].append(0)

		for node in edge_embs:
			edge_embs[node] = np.array(edge_embs[node])
			edge_labels[node] = np.array(edge_labels[node])

		auc, mrr = self.cross_validation(edge_embs, edge_labels)

		return auc, mrr


if __name__ == '__main__':
	LinkPred('mo').lp_evaluate()