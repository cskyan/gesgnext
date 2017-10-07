#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2016 by Caspar. All rights reserved.
# File Name: gsc.py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2016-10-18 22:17:47
###########################################################################
#

import os
import sys
import operator

import numpy as np
import scipy as sp
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler, Normalizer

from bionlp.spider import geoxml as geo
from bionlp.util import fs, io, func, ontology
from bionlp import nlp


if sys.platform.startswith('win32'):
	DATA_PATH = 'D:\\data\\gesgnext'
	BIONLP_PATH = 'D:\\data\\bionlp'
elif sys.platform.startswith('linux2'):
	DATA_PATH = os.path.join(os.path.expanduser('~'), 'data', 'gesgnext')
	BIONLP_PATH = os.path.join(os.path.expanduser('~'), 'data', 'bionlp')
GEO_PATH = geo.GEO_PATH
ONTO_PATH = ontology.DATA_PATH
RXNAV_PATH = os.path.join(BIONLP_PATH, 'rxnav')
BIOGRID_PATH = os.path.join(BIONLP_PATH, 'biogrid')
HGNC_PATH = os.path.join(BIONLP_PATH, 'hgnc')
DNORM_PATH = os.path.join(BIONLP_PATH, 'dnorm')
SC=';;'

LABEL2FILE = {'gene perturbation':'single_gene_perturbations-v1.0.csv', 'drug perturbation':'single_drug_perturbations-v1.0.csv', 'disease signature':'disease_signatures-v1.0.csv'}
LABEL2ONTO = {'gene perturbation':'PRGE', 'drug perturbation':'CHED', 'disease signature':'DISO'}
LABEL2DB = {'gene perturbation':'go', 'drug perturbation':'dbvocab', 'disease signature':'do'}
LABEL2ID = {'gene perturbation':'gene', 'drug perturbation':'drug', 'disease signature':'dz'}

DB2LANG = {'go':'', 'dron':'', 'do':'', 'dbvocab':'en', 'dgidb':'', 'dgnet':'en'}
DB2IDNS = {'go':'OBO', 'dron':'OBO', 'do':'OBO', 'dbvocab':'DBID', 'dgidb':{'gene':[('dgigene', 'DGIDB_GENE')], 'drug':[('dgidrug', 'DGIDB_DRUG')]}, 'dgnet':{'gene':[('obo', 'OBO')], 'disease':[('omim', 'OMIM')]}}
DB2IDN = {'go':'go_id', 'dron':'dron_id', 'do':'do_id', 'dbvocab':'drugbank_id'}
DB2ONTON = {'go':'gene_symbol', 'dron':'drug_name', 'do':'disease_name', 'dbvocab':'drug_name'}
DB2PRDS = {'go':{'idprd':[('RDFS', 'label'), ('OBOWL', 'hasExactSynonym'), ('OBOWL', 'hasRelatedSynonym')], 'lbprds':[('RDFS', 'label')]}, 'dron':{'idprd':[('RDFS', 'label'), ('OBOWL', 'hasExactSynonym'), ('OBOWL', 'hasRelatedSynonym')], 'lbprds':[('RDFS', 'label')]}, 'do':{'idprd':[('RDFS', 'label'), ('OBOWL', 'hasExactSynonym'), ('OBOWL', 'hasRelatedSynonym')], 'lbprds':[('RDFS', 'label')]}, 'dbvocab':{'idprd':[('DBV', 'common-name'), ('DBV', 'term')], 'lbprds':[('DBV', 'common-name')]}}
DB2INTPRDS = {'go':[('obowl', 'OBOWL')], 'dron':[('obowl', 'OBOWL')], 'do':[('obowl', 'OBOWL')], 'dbvocab':[], 'dgidb':[('dgidbv', 'DGIDBV')], 'dgnet':[]}
DB2ATTR = {'go':{'noid':False}, 'dron':{'noid':False}, 'do':{'noid':False}, 'dbvocab':{'noid':True}, 'dgidb':{'noid':True}, 'dgnet':{'noid':False}}


def get_geos(type='gse', fmt='xml'):
	geo_cachef = os.path.join(GEO_PATH, fmt, '%s_doc.pkl' % type)
	if (os.path.exists(geo_cachef)):
		return io.read_obj(geo_cachef)
	geo_docs = {}
	for lb, fname in LABEL2FILE.iteritems():
		excel_df = pd.read_csv(os.path.join(GEO_PATH, fname))
		if (fmt == 'soft'):
			from bionlp.spider import geo as geo
			for gse in geo.fetch_geo(excel_df['geo_id'], saved_path=os.path.join(GEO_PATH, fmt)):
				if (type == 'gse'):
					geo_id, geo_data, _ = geo.parse_geo(gse, with_samp=False)
					geo_record = [(geo_id, geo_data)]
				elif (type == 'gsm'):
					geo_id, geo_data, samples = geo.parse_geo(gse)
					geo_record = samples
				for geo_id, geo_data in geo_record:
					if (geo_docs.has_key(geo_id)):
						geo_docs[geo_id][1].append(lb)
					else:
						geo_docs[geo_id] = (geo_data, [lb])
		else:
			if (type == 'gse'):
				lb_folder = os.path.join(GEO_PATH, fmt, os.path.splitext(fname)[0])
			elif (type == 'gsm'):
				lb_folder = os.path.join(GEO_PATH, fmt, os.path.splitext(fname)[0], 'samples')
			for f in fs.listf(lb_folder):
				geo_id = os.path.splitext(f)[0]
				if (geo_docs.has_key(geo_id)):
					geo_docs[geo_id][1].append(lb)
				else:
					geo_docs[geo_id] = (geo.parse_geo(os.path.join(lb_folder, f), type=type, fmt=fmt), [lb])
	io.write_obj(geo_docs, geo_cachef)
	return geo_docs
	
	
def get_data(geo_docs, type='gse', **kwargs):
	if (type == 'gse'):
		return get_data_gse(geo_docs, **kwargs)
	elif (type == 'gsm'):
		return get_data_gsm(geo_docs, **kwargs)


def get_data_gse(geo_docs, from_file=None, ft_type='tfidf', max_df=1.0, min_df=1, fmt='npz', spfmt='csr'):
	# Read from local files
	if (from_file is not None):
		if (type(from_file) == bool):
			file_name = 'gse_X.npz' if (fmt == 'npz') else 'gse_X.csv'
		else:
			file_name = from_file
		print 'Reading file: %s and gse_Y.%s' % (file_name, fmt)
		if (fmt == 'npz'):
			return io.read_df(os.path.join(DATA_PATH, file_name), with_idx=True, sparse_fmt=spfmt), io.read_df(os.path.join(DATA_PATH, 'gse_Y.npz'), with_idx=True)
		else:
			return pd.read_csv(os.path.join(DATA_PATH, file_name), index_col=0, encoding='utf8'), pd.read_csv(os.path.join(DATA_PATH, 'gse_Y.csv'), index_col=0, encoding='utf8')

	## Feature Sets
	ft_geoid, ft_title, ft_summary, ft_keywords, label = [[] for i in range(5)]
	ft_order = ['title', 'summary', 'keywords']
	ft_name = {'title':'Title', 'summary':'Summary', 'keywords':'Keywords'}
	ft_dic = {'title':ft_title, 'summary':ft_summary, 'keywords':ft_keywords}
	vft_dic, gse_stat = [{} for i in range(2)]
	
	for geo_id, geo_data in geo_docs.iteritems():
		ft_geoid.append(geo_id)
		for fset in ft_order:
			ft_dic[fset].append(geo_data[0][fset])
		label.append(geo_data[1])
		for lb in geo_data[1]:
			gse_stat[lb] = gse_stat.setdefault(lb, 0) + 1

	## Feature Construction

	def tokenizer(text):
		tokens = nlp.tokenize(text)
		tokens = nlp.del_punct(tokens)
		tokens = nlp.lemmatize(tokens)
		tokens = nlp.stem(tokens)
		return tokens
	
	Vectorizer = TfidfVectorizer if ft_type == 'tfidf' else CountVectorizer
	title_vctrz, summary_vctrz, kw_vctrz = [Vectorizer(analyzer='word', tokenizer=tokenizer, ngram_range=(1, 2), stop_words='english', lowercase=True, max_df=max_df, min_df=min_df, binary=True if ft_type=='binary' else False) for i in range(3)]
	vctrz_dic = dict(zip(ft_order, [title_vctrz, summary_vctrz, kw_vctrz]))
	for fset in ft_order:
		ft_mt = vctrz_dic[fset].fit_transform(ft_dic[fset]).tocsr()
		classes = [cls[0] for cls in sorted(vctrz_dic[fset].vocabulary_.items(), key=operator.itemgetter(1))]
		vft_dic[fset] = (ft_mt, classes)
		
	## Label Construction
	mlb = MultiLabelBinarizer()
	bin_label = (mlb.fit_transform(label), mlb.classes_)
	
	## Generate the features as well as the labels to form a completed dataset
	feat_mt = sp.sparse.hstack([vft_dic[fset][0] for fset in ft_order])
	feat_cols = ['%s_%s' % (fset, w) for fset in ft_order for w in vft_dic[fset][1]]
	feat_df = pd.DataFrame(feat_mt.todense(), index=ft_geoid, columns=feat_cols)
	label_df = pd.DataFrame(bin_label[0], index=ft_geoid, columns=bin_label[1])
	
	## Sampling
	obj_samp_idx = np.random.random_integers(0, feat_df.shape[0] - 1, size=200).tolist()
	ft_samp_idx = np.random.random_integers(0, feat_df.shape[1] - 1, size=1000).tolist()
	samp_feat_df = feat_df.iloc[obj_samp_idx, ft_samp_idx]
	samp_lb_df = label_df.iloc[obj_samp_idx,:]

	## Output the dataset
	if (fmt == 'npz'):
		io.write_df(feat_df, os.path.join(DATA_PATH, 'gse_X.npz'), with_idx=True, sparse_fmt=spfmt, compress=True)
		io.write_df(label_df, os.path.join(DATA_PATH, 'gse_Y.npz'), with_idx=True)
		io.write_df(samp_feat_df, os.path.join(DATA_PATH, 'sample_gse_X.npz'), with_idx=True, sparse_fmt=spfmt, compress=True)
		io.write_df(samp_lb_df, os.path.join(DATA_PATH, 'sample_gse_Y.npz'), with_idx=True)
	else:
		feat_df.to_csv(os.path.join(DATA_PATH, 'gse_X.csv'), encoding='utf8')
		label_df.to_csv(os.path.join(DATA_PATH, 'gse_Y.csv'), encoding='utf8')
		samp_feat_df.to_csv(os.path.join(DATA_PATH, 'sample_gse_X.csv'), encoding='utf8')
		samp_lb_df.to_csv(os.path.join(DATA_PATH, 'sample_gse_Y.csv'), encoding='utf8')
	return feat_df, label_df
	
	
def get_data_gsm(geo_docs, from_file=None, ft_type='tfidf', max_df=1.0, min_df=1, d2v_params={'size':100, 'np':1}, fmt='npz', spfmt='csr'):
	# Read from local files
	if (from_file is not None):
		if (type(from_file) == bool):
			file_name = 'gsm_X.npz' if (fmt == 'npz') else 'gsm_X.csv'
		else:
			file_name = from_file
		X_fname = os.path.splitext(file_name)
		X_files = sorted([x for x in os.listdir(DATA_PATH) if x.startswith(X_fname[0]+'_') and x.endswith(X_fname[1])])
		Y_files = [x.replace('X', 'y') for x in X_files]
		label_files = [x.replace('X', 'lb') for x in X_files]
		print 'Reading file: %s, %s and %s' % (X_files, Y_files, label_files)
		if (fmt == 'npz'):
			return [io.read_df(os.path.join(DATA_PATH, x), with_idx=True, sparse_fmt=spfmt) for x in X_files], [io.read_df(os.path.join(DATA_PATH, y), with_idx=True) for y in Y_files], [io.read_df(os.path.join(DATA_PATH, lb), with_idx=True, sparse_fmt=spfmt) for lb in label_files]
		else:
			return [pd.read_csv(os.path.join(DATA_PATH, x), index_col=0, encoding='utf8') for x in X_files], [pd.read_csv(os.path.join(DATA_PATH, y), index_col=0, encoding='utf8') for y in Y_files], [pd.read_csv(os.path.join(DATA_PATH, lb), index_col=0, encoding='utf8') for lb in label_files]

	## Feature Sets
	ft_geoid, ft_title, ft_desc, ft_trait, ft_source, label = [[] for i in range(6)]
	ft_order = ['title', 'description', 'trait', 'source']
	ft_name = {'title':'Title', 'description':'Description', 'trait':'Characteristics', 'source':'Source'}
	ft_dic = {'title':ft_title, 'description':ft_desc, 'trait':ft_trait, 'source':ft_source}
	vft_dic, gsm_stat = [{} for i in range(2)]

	## Feature Construction
	if (ft_type != 'd2v'):
		for geo_id, geo_data in geo_docs.iteritems():
			ft_geoid.append(geo_id)
			for fset in ft_order:
				ft_dic[fset].append(geo_data[0][fset])
			label.append(geo_data[1])
			for lb in geo_data[1]:
				gsm_stat[lb] = gsm_stat.setdefault(lb, 0) + 1

		def tokenizer(text):
			tokens = nlp.tokenize(text)
			tokens = nlp.del_punct(tokens)
			tokens = nlp.lemmatize(tokens)
			tokens = nlp.stem(tokens)
			return tokens
		
		Vectorizer = TfidfVectorizer if ft_type == 'tfidf' else CountVectorizer
		title_vctrz, desc_vctrz, trait_vctrz, source_vctrz = [Vectorizer(analyzer='word', tokenizer=tokenizer, ngram_range=(1, 2), stop_words='english', lowercase=True, max_df=max_df, min_df=min_df, binary=True if ft_type=='binary' else False) for i in range(4)]
		vctrz_dic = dict(zip(ft_order, [title_vctrz, desc_vctrz, trait_vctrz, source_vctrz]))
		for fset in ft_order:
			ft_mt = vctrz_dic[fset].fit_transform(ft_dic[fset]).tocsr()
			classes = [cls[0] for cls in sorted(vctrz_dic[fset].vocabulary_.items(), key=operator.itemgetter(1))]
			vft_dic[fset] = (ft_mt, classes)
		
		## Generate the features data
		feat_mt = sp.sparse.hstack([vft_dic[fset][0] for fset in ft_order])
		feat_cols = ['%s_%s' % (fset, w) for fset in ft_order for w in vft_dic[fset][1]]
		feat_df = pd.DataFrame(feat_mt.todense(), index=ft_geoid, columns=feat_cols)
	else:
		from gensim.models.doc2vec import TaggedDocument, Doc2Vec
		d2v_dfs = dict.fromkeys(ft_order)
		for geo_id, geo_data in geo_docs.iteritems():
			ft_geoid.append(geo_id)
			for fset in ft_order:
				ft_dic[fset].append(TaggedDocument(words=geo_data[0][fset], tags=[geo_id]))
			label.append(geo_data[1])
			for lb in geo_data[1]:
				gsm_stat[lb] = gsm_stat.setdefault(lb, 0) + 1
		for fset in ft_order:		
			model = Doc2Vec(ft_dic[fset], size=d2v_params.setdefault('size', 100), window=d2v_params.setdefault('window', 8), min_count=d2v_params.setdefault('min_count', 5), workers=d2v_params.setdefault('np', 1), dbow_words=0 if fset=='description' else 1)
			model.save(os.path.join(DATA_PATH, 'd2v_%s.mdl' % fset))
			# mms = MinMaxScaler()
			# d2v_dfs[fset] = pd.DataFrame(mms.fit_transform(model.docvecs[range(model.docvecs.count)]), index=ft_geoid)
			d2v_dfs[fset] = pd.DataFrame(model.docvecs[range(model.docvecs.count)], index=ft_geoid)
		feat_df = pd.concat([d2v_dfs[fset] for fset in ft_order], axis=1, join_axes=[d2v_dfs[ft_order[0]].index])
		spfmt = None
	
	## Label Construction
	mlb = MultiLabelBinarizer()
	bin_label = (mlb.fit_transform(label), mlb.classes_)
	label_df = pd.DataFrame(bin_label[0], index=ft_geoid, columns=bin_label[1], dtype='int8')
	
	## Sampling
	obj_samp_idx = np.random.random_integers(0, feat_df.shape[0] - 1, size=200).tolist()
	ft_samp_idx = np.random.random_integers(0, feat_df.shape[1] - 1, size=1000).tolist()
	samp_feat_df = feat_df.iloc[obj_samp_idx, ft_samp_idx]
	samp_lb_df = label_df.iloc[obj_samp_idx,:]

	## Output the combined dataset
	if (fmt == 'npz'):
		io.write_df(feat_df, os.path.join(DATA_PATH, 'gsm_X.npz'), with_idx=True, sparse_fmt=spfmt, compress=True)
		io.write_df(label_df, os.path.join(DATA_PATH, 'gsm_Y.npz'), with_idx=True)
		io.write_df(samp_feat_df, os.path.join(DATA_PATH, 'sample_gsm_X.npz'), with_idx=True, sparse_fmt=spfmt, compress=True)
		io.write_df(samp_lb_df, os.path.join(DATA_PATH, 'sample_gsm_Y.npz'), with_idx=True)
	else:
		feat_df.to_csv(os.path.join(DATA_PATH, 'gsm_X.csv'), encoding='utf8')
		label_df.to_csv(os.path.join(DATA_PATH, 'gsm_Y.csv'), encoding='utf8')
		samp_feat_df.to_csv(os.path.join(DATA_PATH, 'sample_gsm_X.csv'), encoding='utf8')
		samp_lb_df.to_csv(os.path.join(DATA_PATH, 'sample_gsm_Y.csv'), encoding='utf8')
		
	## Split the dataset into different parts according to the labels, create labels indicating whether a sample is part of the control or treatment group, and mark the cluster labels
	Xs, Ys, labels = [[] for i in range(3)]
	for i in xrange(label_df.shape[1]):
		lb = label_df.columns[i]
		sub_X = feat_df[label_df[lb] == 1]
		sub_Y = pd.DataFrame(np.zeros((sub_X.shape[0], 2)), index=sub_X.index, columns=['ctrl', 'pert'], dtype='int8')
		excel_df = pd.read_csv(os.path.join(GEO_PATH, LABEL2FILE[lb]))
		# control-perturbation labels
		for slb in ['ctrl_ids', 'pert_ids']:
			sub_Y.loc[func.flatten_list([ids_str.split('|') for ids_str in excel_df[slb]]), slb.split('_')[0]] = 1
		# cluster labels
		label_list = [[] for x in range(sub_X.shape[0])]
		lb_dict = {}
		for j, (ctrl_ids, pert_ids) in enumerate(zip(excel_df['ctrl_ids'], excel_df['pert_ids'])):
			for ctrl_id in ctrl_ids.split('|'):
				label_list[sub_X.index.get_loc(ctrl_id)].append(2 * j)
			for pert_id in pert_ids.split('|'):
				label_list[sub_X.index.get_loc(pert_id)].append(2 * j + 1)
		for j, lbs in enumerate(label_list):
			for lb in lbs:
				lb_dict.setdefault(lb, set([])).add(j)
		clusters = list(set([tuple(sorted(x)) for x in lb_dict.values()]))
		# sub_label = pd.DataFrame(-1 * np.ones((sub_X.shape[0], 1)), index=sub_X.index, columns=['cltlb'], dtype='int64')
		sub_label = pd.DataFrame(np.zeros((sub_X.shape[0], len(clusters))), index=sub_X.index, dtype='int64')
		for j, clt in enumerate(clusters):
			# for sample in clt:
				# if (sub_label.ix[sample,'cltlb'] == -1 or len(clusters[sub_label.ix[sample,'cltlb']]) < len(clusters[j])):
					# sub_label.ix[sample,'cltlb'] = j
			sub_label.ix[clt, j] = 1
		if (fmt == 'npz'):
			io.write_df(sub_X, os.path.join(DATA_PATH, 'gsm_X_%i.npz' % i), with_idx=True, sparse_fmt=spfmt, compress=True)
			io.write_df(sub_Y, os.path.join(DATA_PATH, 'gsm_y_%i.npz' % i), with_idx=True)
			io.write_df(sub_label, os.path.join(DATA_PATH, 'gsm_lb_%i.npz' % i), with_idx=True, sparse_fmt=spfmt, compress=True)
		else:
			sub_X.to_csv(os.path.join(DATA_PATH, 'gsm_X_%i.csv' % i), encoding='utf8')
			sub_Y.to_csv(os.path.join(DATA_PATH, 'gsm_y_%i.csv' % i), encoding='utf8')
			sub_label.to_csv(os.path.join(DATA_PATH, 'gsm_lb_%i.csv' % i), encoding='utf8')
		Xs.append(sub_X)
		Ys.append(sub_Y)
		labels.append(sub_label)
		del [sub_X, sub_Y, sub_label]
		
	return Xs, Ys, labels
	
	
def get_mltl_npz(type='gse', lbs=[], mltlx=True, spfmt='csr'):
	if (len(lbs) == 0):
		return None, None
	Xs, Ys, labels = [[] for i in range(3)]
	for lb in lbs:
		if (mltlx):
			Xs.append(io.read_df(os.path.join(DATA_PATH, '%s_X_%s.npz' % (type, str(lb).split('_')[0])), with_idx=True, sparse_fmt=spfmt))
			labels.append(io.read_df(os.path.join(DATA_PATH, '%s_lb_%s.npz' % (type, str(lb).split('_')[0])), with_idx=True, sparse_fmt=spfmt))
		Ys.append(io.read_df(os.path.join(DATA_PATH, '%s_y_%s.npz' % (type, lb)), with_col=False, with_idx=True))
	if (not mltlx):
		Xs.append(io.read_df(os.path.join(DATA_PATH, '%s_X.npz' % type), with_idx=True, sparse_fmt=spfmt))
	if (type == 'gse'):
		return Xs, Ys
	elif (type == 'gsm'):
		return Xs, Ys, labels