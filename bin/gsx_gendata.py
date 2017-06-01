#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2016 by Caspar. All rights reserved.
# File Name: gsx_gendata.py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2016-10-18 22:15:59
###########################################################################
#

import os
import logging
import ast
from optparse import OptionParser

import numpy as np
import scipy as sp
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, Normalizer
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD

import bionlp.spider.geo as geo
from bionlp import ftslct, ftdecomp
from bionlp.util import fs, io, func

import gsc

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
PAR_DIR = os.path.abspath(os.path.join(FILE_DIR, os.path.pardir))
CONFIG_FILE = os.path.join(PAR_DIR, 'etc', 'config.yaml')
SPDR_MAP = {'gsc':gsc, 'geo':geo}
SC=';;'

opts, args = {}, []
cfgr = None
spdr = geo


def gen_data(type='gse'):
	if (type == 'gse'):
		return gen_data_gse()
	elif (type == 'gsm'):
		return gen_data_gsm()


def gen_data_gse():
	if (opts.local):
		X, Y = spdr.get_data(None, type='gse', from_file=True)
	else:
		geo_docs = spdr.get_geos(type='gse', fmt='soft')
		X, Y = spdr.get_data(geo_docs, type='gse', ft_type=opts.type, max_df=ast.literal_eval(opts.maxdf), min_df=ast.literal_eval(opts.mindf), fmt=opts.fmt, spfmt=opts.spfmt)

	for i in range(Y.shape[1]):
		y = Y.iloc[:,i]
		if (opts.fmt == 'npz'):
			io.write_df(y, os.path.join(spdr.DATA_PATH, 'gse_y_%s.npz' % i), with_col=False, with_idx=True)
		else:
			y.to_csv(os.path.join(spdr.DATA_PATH, 'gse_y_%s.csv' % i), encoding='utf8')
		
		
def gen_data_gsm():
	if (opts.local):
		Xs, Ys, labels = spdr.get_data(None, type='gsm', from_file=True)
	else:
		geo_docs = spdr.get_geos(type='gsm', fmt='soft')
		Xs, Ys, labels = spdr.get_data(geo_docs, type='gsm', ft_type=opts.type, max_df=ast.literal_eval(opts.maxdf), min_df=ast.literal_eval(opts.mindf), fmt=opts.fmt, spfmt=opts.spfmt)
		
	# Feature Selection
	for i, (X, Y, Z) in enumerate(zip(Xs, Ys, labels)):
		stat, _ = ftslct.utopk(X.values, Y.values, ftslct.decision_tree, fn=500)
		io.write_npz(stat, os.path.join(spdr.DATA_PATH, 'ftw.npz'))
		cln_X = X.iloc[:,stat.argsort()[-500:][::-1]]
		print 'The size of data has been changed from %s to %s.' % (X.shape, cln_X.shape)
		
		if (opts.fmt == 'npz'):
			io.write_df(cln_X, os.path.join(spdr.DATA_PATH, 'cln_gsm_X_%i.npz' % i), with_idx=True, sparse_fmt=opts.spfmt, compress=True)
		else:
			cln_X.to_csv(os.path.join(spdr.DATA_PATH, 'cln_gsm_X_%i.csv' % i), encoding='utf8')
		del X, cln_X

	for i in xrange(len(Ys)):
		for j in xrange(Ys[i].shape[1]):
			y = Ys[i].iloc[:,j]
			if (opts.fmt == 'npz'):
				io.write_df(y, os.path.join(spdr.DATA_PATH, 'gsm_y_%i_%i.npz' % (i, j)), with_col=False, with_idx=True)
			else:
				y.to_csv(os.path.join(spdr.DATA_PATH, 'gsm_y_%i_%i.csv' % (i, j)), encoding='utf8')
		
		
def decomp_data(method='LDA', n_components=100):
	if (opts.ftype == 'gse'):
		X, Y = spdr.get_data(None, type='gse', from_file=True, ft_type=opts.type, max_df=ast.literal_eval(opts.maxdf), min_df=ast.literal_eval(opts.mindf), fmt=opts.fmt, spfmt=opts.spfmt)
	elif (opts.ftype == 'gsm'):
		Xs, Ys, _ = gsc.get_mltl_npz(type='gsm', lbs=[opts.pid], mltlx=True, spfmt=opts.spfmt)
		X, Y = Xs[0], Ys[0]
	method = method.upper()
	n_components = min(n_components, X.shape[1])
	core_model = None
	if (method == 'LDA'):
		model = make_pipeline(LatentDirichletAllocation(n_topics=n_components, learning_method='online', learning_offset=50., max_iter=5, n_jobs=opts.np, random_state=0), Normalizer(copy=False))
	elif (method == 'NMF'):
		model = make_pipeline(NMF(n_components=n_components, random_state=0, alpha=.1, l1_ratio=.5), Normalizer(copy=False), MinMaxScaler(copy=False))
	elif (method == 'LSI'):
		model = make_pipeline(TruncatedSVD(n_components), Normalizer(copy=False), MinMaxScaler(copy=False))
	elif (method == 'TSNE'):
#		from sklearn.manifold import TSNE
		# model = make_pipeline(TSNE(n_components=n_components, random_state=0), Normalizer(copy=False), MinMaxScaler(copy=False))
		from MulticoreTSNE import MulticoreTSNE as TSNE
		model = TSNE(random_state=0, n_jobs=opts.np)
		core_model = model
		# model = make_pipeline(ftdecomp.DecompTransformer(n_components, ftdecomp.t_sne, initial_dims=min(15*n_components, X.shape[1]), perplexity=30.0), Normalizer(copy=False), MinMaxScaler(copy=False))
	if (core_model is None):
		core_model = model.steps[0][1]
	if (opts.prefix == 'all'):
		td_cols = X.columns
	else:
		# Only apply dimension reduction on specific columns
		td_cols = np.array(map(lambda x: True if any(x.startswith(prefix) for prefix in opts.prefix.split(SC)) else False, X.columns))
	td_X = X.loc[:,td_cols]
	new_td_X = model.fit_transform(td_X.as_matrix())
	if (method == 'LSI'):
		print('Explained Variance Ratio:\n%s' % core_model.explained_variance_ratio_)
		print('Total and Average Explained Variance Ratio: %s, %s' % (core_model.explained_variance_ratio_.sum(), core_model.explained_variance_ratio_.mean()))
	if (opts.prefix == 'all'):
		columns = range(new_td_X.shape[1]) if not hasattr(core_model, 'components_') else td_X.columns[core_model.components_.argmax(axis=1)]
		new_X = pd.DataFrame(new_td_X, index=X.index, columns=['tp_%s' % x for x in columns])
	else:
		columns = range(new_td_X.shape[1]) if not hasattr(core_model, 'components_') else td_X.columns[core_model.components_.argmax(axis=1)]
		# Concatenate the components and the columns are not applied dimension reduction on
		new_X = pd.concat([pd.DataFrame(new_td_X, index=X.index, columns=['tp_%s' % x for x in columns]), X.loc[:,np.logical_not(td_cols)]], axis=1)
	if (opts.fmt == 'npz'):
		io.write_df(new_X, os.path.join(spdr.DATA_PATH, '%s%i_%s_X%s.npz' % (method.lower(), n_components, opts.ftype, ('_%i' % opts.pid if opts.ftype=='gsm' else ''))), with_idx=True, sparse_fmt=opts.spfmt, compress=True)
	else:
		new_X.to_csv(os.path.join(spdr.DATA_PATH, '%s%i_%s_X%s.csv' % (method.lower(), n_components, opts.ftype, ('_%i' % opts.pid if opts.ftype=='gsm' else ''))), encoding='utf8')
		
		
def add_cns():
	Xs, Ys, labels = spdr.get_data(None, type='gsm', from_file=True)
	for i, (X, y, z) in enumerate(zip(Xs, Ys, labels)):
		le = LabelEncoder()
		encoded_lb = (le.fit_transform(X.index), le.classes_)
		gseid_df = pd.DataFrame(encoded_lb[0], index=X.index, columns=['gse_id'])
		new_X = pd.concat([X, gseid_df, y], axis=1, join_axes=[X.index])
		print 'The size of data has been changed from %s to %s.' % (X.shape, new_X.shape)
		if (opts.fmt == 'npz'):
			io.write_df(new_X, os.path.join(spdr.DATA_PATH, 'new_gsm_X_%i.npz' % i), with_idx=True, sparse_fmt=opts.spfmt, compress=True)
		else:
			new_X.to_csv(os.path.join(spdr.DATA_PATH, 'new_gsm_X_%i.csv' % i), encoding='utf8')


def main():
	if (opts.method is None):
		return
	elif (opts.method == 'gen'):
		gen_data(type=opts.ftype)
	elif (opts.method == 'decomp'):
		decomp_data(method=opts.decomp.upper(), n_components=opts.cmpn)
	elif (opts.method == 'pcns'):
		add_cns()
	

if __name__ == '__main__':
	# Parse commandline arguments
	op = OptionParser()
	op.add_option('-p', '--pid', action='store', type='int', dest='pid', help='indicate the process ID')
	op.add_option('-n', '--np', default=-1, action='store', type='int', dest='np', help='indicate the number of processes used for training')
	op.add_option('-f', '--fmt', default='npz', help='data stored format: csv or npz [default: %default]')
	op.add_option('-s', '--spfmt', default='csr', help='sparse data stored format: csr or csc [default: %default]')
	op.add_option('-l', '--local', default=False, action='store_true', dest='local', help='read data from the preprocessed data matrix file')
	op.add_option('-t', '--type', default='tfidf', help='feature type: binary, numeric, tfidf [default: %default]')
	op.add_option('-a', '--mindf', default='1', type='str', dest='mindf', help='lower document frequency threshold for term ignorance')
	op.add_option('-b', '--maxdf', default='1.0', type='str', dest='maxdf', help='upper document frequency threshold for term ignorance')
	op.add_option('-d', '--decomp', default='LDA', help='decomposition method to use: LDA, NMF, LSI or TSNE [default: %default]')
	op.add_option('-c', '--cmpn', default=100, type='int', dest='cmpn', help='number of components that used in clustering model')
	op.add_option('-j', '--prefix', default='all', type='str', dest='prefix', help='prefixes of the column names that the decomposition method acts on, for example, \'-j lem;;nn;;ner\' means columns that starts with \'lem_\', \'nn_\', or \'ner_\'')
	op.add_option('-e', '--ftype', default='gse', type='str', dest='ftype', help='the document type used to generate data')
	op.add_option('-i', '--input', default='gsc', help='input source: gsc or geo [default: %default]')
	op.add_option('-m', '--method', help='main method to run')
	op.add_option('-v', '--verbose', action='store_true', dest='verbose', default=False, help='display detailed information')

	(opts, args) = op.parse_args()
	if len(args) > 0:
		op.print_help()
		op.error('Please input options instead of arguments.')
		exit(1)

	# Logging setting
	logging.basicConfig(level=logging.INFO if opts.verbose else logging.ERROR, format='%(asctime)s %(levelname)s %(message)s')
	# Parse config file
	spdr = SPDR_MAP[opts.input]
	if (os.path.exists(CONFIG_FILE)):
		cfgr = io.cfg_reader(CONFIG_FILE)
		spdr_cfg = cfgr('bionlp.spider.%s' % opts.input, 'init')
		if (len(spdr_cfg) > 0 and spdr_cfg['DATA_PATH'] is not None and os.path.exists(spdr_cfg['DATA_PATH'])):
			spdr.DATA_PATH = spdr_cfg['DATA_PATH']
		if (len(spdr_cfg) > 0 and spdr_cfg['GEO_PATH'] is not None and os.path.exists(spdr_cfg['GEO_PATH'])):
			spdr.GEO_PATH = spdr_cfg['GEO_PATH']

	main()