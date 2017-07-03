#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2016 by Caspar. All rights reserved.
# File Name: gsx_extrc.py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2016-03-16 15:56:16
###########################################################################
#

import os
import re
import sys
import ast
import time
import bisect
import logging
import operator
import itertools
import collections
from shutil import copyfile
from optparse import OptionParser

import numpy as np
import scipy as sp
import scipy.stats as stats
import pandas as pd

from sklearn.preprocessing import label_binarize
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectPercentile, SelectFpr, SelectFromModel, chi2, f_classif
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, Perceptron, PassiveAggressiveClassifier, LassoCV, LassoLarsCV, LassoLarsIC, RandomizedLasso
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.metrics.pairwise import pairwise_distances

from bionlp.spider import geo, annot, hgnc, dnorm, rxnav, sparql
from bionlp.model.fzcmeans import FZCMeans, CNSFZCMeans
from bionlp.model.lda import LDACluster
from bionlp.model import kerasext, kallima
from bionlp.util import fs, io, func, ontology
from bionlp import ftslct, txtclf, dstclc, txtclt, nlp

import gsc
import gsx_helper as helper

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
PAR_DIR = os.path.abspath(os.path.join(FILE_DIR, os.path.pardir))
CONFIG_FILE = os.path.join(PAR_DIR, 'etc', 'config.yaml')
SPDR_MAP = {'gsc':gsc, 'geo':geo}
KERAS_DEVID, KERAS_DEV, KERAS_DEVINIT, KERAS_DEVARS = 0, '', False, {}
SC=';;'

opts, args = {}, []
cfgr = None
spdr = geo


def load_data(type='gse', pid=-1, fmt='npz', spfmt='csr'):
	print 'Loading data for %s...' % type.upper()
	try:
		if (type == 'gsm-clf'):
			if (pid == -1):
				Xs, Ys, _ = spdr.get_data(None, type='gsm', from_file=True, fmt=fmt, spfmt=spfmt)
			else:
				Xs, Ys, _ = spdr.get_mltl_npz(type='gsm', lbs=['%i_%i' % (int(pid/2), int(pid%2))], spfmt=spfmt)
			return Xs, Ys
		elif (type == 'gsm-clt'):
			if (pid == -1):
				Xs, Ys, labels = spdr.get_data(None, type='gsm', from_file=True, fmt=fmt, spfmt=spfmt)
			else:
				Xs, Ys, labels = spdr.get_mltl_npz(type='gsm', lbs=[pid], spfmt=spfmt)
			gsm2gse = get_gsm2gse(data_path=spdr.DATA_PATH)
			return Xs, Ys, labels, gsm2gse
		elif (type == 'sgn'):
			if (pid == -1):
				Xs, Ys, labels = spdr.get_data(None, type='gsm', from_file=True, fmt=fmt, spfmt=spfmt)
			else:
				Xs, Ys, labels = spdr.get_mltl_npz(type='gsm', lbs=[pid], spfmt=spfmt)
			gsm2gse = get_gsm2gse(data_path=spdr.DATA_PATH)
			return Xs, Ys, labels, gsm2gse
		if (pid == -1):
			# From combined data file
			X, Y = spdr.get_data(None, type=type, from_file=True, fmt=fmt, spfmt=spfmt)
		else:
			# From splited data file
			Xs, Ys = spdr.get_mltl_npz(type=type, lbs=[pid], mltlx=False, spfmt=spfmt)
			X, Y = Xs[0], Ys[0]
		return X, Y
	except Exception as e:
		print e
		print 'Can not find the data files!'
		exit(1)
		
		
def get_gsm2gse(data_path):
	return io.read_df(os.path.join(data_path, 'gsm2gse.npz'), with_idx=True)


def build_model(mdl_func, mdl_t, mdl_name, tuned=False, pr=None, mltl=False, **kwargs):
	if (tuned and bool(pr)==False):
		print 'Have not provided parameter writer!'
		return None
	if (mltl):
		return OneVsRestClassifier(mdl_func(**func.update_dict(pr(mdl_t, mdl_name) if tuned else {}, kwargs)), n_jobs=opts.np)
	else:
		return mdl_func(**func.update_dict(pr(mdl_t, mdl_name) if tuned else {}, kwargs))

		
# Keras Deep Learning
def gen_keras(input_dim, output_dim, model='simple', **kwargs):
	udargs = []
	mdl_map = {'simple':(simple_nn, 'clf'), 'tunable':(tunable_nn, 'clf'), 'cfkmeans':(cfkmeans_nn, 'clt')}
	udargs.extend(['input_dim', 'output_dim', 'backend', 'device', 'session'])
	mdl = mdl_map[model]
	return kerasext.gen_mdl(input_dim, output_dim, mdl[0], mdl[1], backend=opts.dend, verbose=opts.verbose, udargs=udargs, **kwargs)
	
	
# Constraint Fuzzy K-means Neural Network
def _cfkmeans_loss(Y_true, Y):
	import keras.backend as K
	return K.mean(Y)
	

def cfkmeans_nn(input_dim=1, output_dim=1, constraint_dim=0, batch_size=32, backend='th', device='', session=None, internal_dim=64, metric='euclidean', gamma=0.01, **kwargs):
	from keras.layers import Input, Lambda, merge
	from keras.optimizers import SGD
	from bionlp.model.cfkmeans import CFKU, CFKD, CFKC
	import keras.backend as K
	with kerasext.gen_cntxt(backend, device):
		X_input = Input(shape=(input_dim,), dtype=K.floatx(), name='X')
		C_input = Input(shape=(constraint_dim,), name='CI')
		cfku = CFKU(output_dim=output_dim, input_dim=input_dim, batch_size=batch_size, name='U', session=session)(X_input)
		cfkd = CFKD(output_dim=output_dim, input_dim=input_dim, metric=metric, batch_size=batch_size, name='D', session=session)([X_input, cfku])
		loss = merge([cfku, cfkd], mode='mul', name='L')
		rglz = Lambda(lambda x: gamma * K.tanh(x), name='R')(cfku)
		constr = CFKC(output_dim=output_dim, input_dim=input_dim, batch_size=batch_size, name='C', session=session)([C_input, cfku, cfkd])
		J = merge([loss, rglz, constr], mode='sum', name='J')
		model = kerasext.gen_cltmdl(context=dict(backend=backend, device=device), session=session, input=[X_input, C_input], output=[J], constraint_dim=constraint_dim)
		optmzr = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
		model.compile(loss=_cfkmeans_loss, optimizer=optmzr, metrics=['accuracy', 'mse'])
	return model


# Tunable Deep Learning Model
def tunable_nn(input_dim=1, output_dim=1, backend='th', device='', session=None, internal_dim=64, layer_num=3, init='uniform', activation='tanh', dropout_ratio=0.5):
	from keras.layers import Dense, Dropout
	from keras.optimizers import SGD
	with kerasext.gen_cntxt(backend, device):
		model = kerasext.gen_mlseq(context=dict(backend=backend, device=device), session=session)
		model.add(Dense(output_dim=internal_dim, input_dim=input_dim, init=init, activation=activation))
		model.add(Dropout(dropout_ratio))
		for i in xrange(layer_num):
			model.add(Dense(output_dim=internal_dim, init=init, activation=activation))
			model.add(Dropout(dropout_ratio))
		model.add(Dense(output_dim=output_dim, init=init, activation='sigmoid'))
		sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
		model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy', 'mae'])
	return model


# Simple Deep Learning Model
def simple_nn(input_dim=1, output_dim=1, backend='th', device='', session=None, internal_dim=64):
	from keras.layers import Dense, Dropout
	from keras.optimizers import SGD
	with kerasext.gen_cntxt(backend, device):
		model = kerasext.gen_mlseq(context=dict(backend=backend, device=device), session=session)
		model.add(Dense(output_dim=internal_dim, input_dim=input_dim, init='uniform', activation='tanh'))
		model.add(Dropout(0.5))
		model.add(Dense(output_dim=internal_dim, init='uniform', activation='tanh'))
		model.add(Dropout(0.5))
		model.add(Dense(output_dim=output_dim, init='uniform', activation='sigmoid'))
		sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
		model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy', 'mae'])
	return model
	

# Neural Network Classification Models
def gen_nnclfs(input_dim, output_dim, other_clfs=None, **kwargs):
	def nnclfs(tuned=False, glb_filtnames=[], glb_clfnames=[]):
		tuned = tuned or opts.best
		common_cfg = cfgr('gsx_extrc', 'common')
		pr = io.param_reader(os.path.join(PAR_DIR, 'etc', '%s.yaml' % common_cfg.setdefault('mdl_cfg', 'mdlcfg')))
		clf_names = []
		for clf_name, clf in [
			('3L64U Neural Network', gen_keras(input_dim, output_dim, model='simple', internal_dim=64)),
			('3L96U Neural Network', gen_keras(input_dim, output_dim, model='simple', internal_dim=96))
		]:
			yield clf_name, clf
			clf_names.append(clf_name)
		if (other_clfs is not None):
			for clf_name, clf in other_clfs(tuned, glb_filtnames, glb_clfnames):
				yield clf_name, clf
				clf_names.append(clf_name)
		if (len(glb_clfnames) < len(clf_names)):
			del glb_clfnames[:]
			glb_clfnames.extend(clf_names)
	return nnclfs


# Feature Filtering Models
def gen_featfilt(tuned=False, glb_filtnames=[], **kwargs):
	tuned = tuned or opts.best
	common_cfg = cfgr('gsx_extrc', 'common')
	pr = io.param_reader(os.path.join(PAR_DIR, 'etc', '%s.yaml' % common_cfg.setdefault('mdl_cfg', 'mdlcfg')))
	filt_names = []
	for filt_name, filter in [
#		('Var Cut', VarianceThreshold()),
#		('Chi2 Pval on FPR', SelectFpr(chi2, alpha=0.05)),
#		('ANOVA-F Pval on FPR', SelectFpr(f_classif, alpha=0.05)),
#		('Chi2 Top K Perc', SelectPercentile(chi2, percentile=30)),
#		('ANOVA-F Top K Perc', SelectPercentile(f_classif, percentile=30)),
#		('Chi2 Top K', SelectKBest(chi2, k=1000)),
#		('ANOVA-F Top K', SelectKBest(f_classif, k=1000)),
#		('LinearSVC', LinearSVC(loss='squared_hinge', dual=False, **pr('Classifier', 'LinearSVC') if tuned else {})),
#		('Logistic Regression', SelectFromModel(LogisticRegression(dual=False, **pr('Feature Selection', 'Logistic Regression') if tuned else {}))),
#		('Lasso', SelectFromModel(LassoCV(cv=6), threshold=0.16)),
#		('Lasso-LARS', SelectFromModel(LassoLarsCV(cv=6))),
#		('Lasso-LARS-IC', SelectFromModel(LassoLarsIC(criterion='aic'), threshold=0.16)),
#		('Randomized Lasso', SelectFromModel(RandomizedLasso(random_state=0))),
#		('Extra Trees Regressor', SelectFromModel(ExtraTreesRegressor(100))),
		# ('U102-GSS502', ftslct.MSelectKBest(ftslct.gen_ftslct_func(ftslct.utopk, filtfunc=ftslct.gss_coef, fn=100), k=500)),
		# ('GSS502', ftslct.MSelectKBest(ftslct.gss_coef, k=500)),
#		('Combined Model', FeatureUnion([('Var Cut', VarianceThreshold()), ('Chi2 Top K', SelectKBest(chi2, k=1000))])),
		('No Feature Filtering', None)
	]:
		yield filt_name, filter
		filt_names.append(filt_name)
	if (len(glb_filtnames) < len(filt_names)):
		del glb_filtnames[:]
		glb_filtnames.extend(filt_names)


# Classification Models
def gen_clfs(tuned=False, glb_clfnames=[], **kwargs):
	tuned = tuned or opts.best
	common_cfg = cfgr('gsx_extrc', 'common')
	pr = io.param_reader(os.path.join(PAR_DIR, 'etc', '%s.yaml' % common_cfg.setdefault('mdl_cfg', 'mdlcfg')))
	clf_names = []
	for clf_name, clf in [
#		('RidgeClassifier', RidgeClassifier(tol=1e-2, solver='lsqr')),
#		('Perceptron', build_model(Perceptron, 'Classifier', 'Perceptron', tuned=tuned, pr=pr, mltl=opts.mltl, n_jobs=1 if opts.mltl else opts.np)),
#		('Passive-Aggressive', PassiveAggressiveClassifier(n_iter=50, n_jobs=1 if opts.mltl else opts.np)),
#		('kNN', KNeighborsClassifier(n_neighbors=100, n_jobs=1 if opts.mltl else opts.np)),
#		('NearestCentroid', NearestCentroid()),
#		('BernoulliNB', BernoulliNB()),
#		('MultinomialNB', MultinomialNB()),
#		('ExtraTrees', build_model(ExtraTreesClassifier, 'Classifier', 'Extra Trees', tuned=tuned, pr=pr, mltl=opts.mltl, n_jobs=opts.np)),
		('RandomForest', build_model(RandomForestClassifier, 'Classifier', 'Random Forest', tuned=tuned, pr=pr, mltl=opts.mltl, n_jobs=1 if opts.mltl else opts.np, random_state=0)),
#		('RandomForest', Pipeline([('clf', build_model(RandomForestClassifier, 'Classifier', 'Random Forest', tuned=tuned, pr=pr, n_jobs=opts.np, random_state=0))])),
#		('BaggingkNN', BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5, n_jobs=1 if opts.mltl else opts.np, random_state=0)),
#		('BaggingLinearSVC', build_model(BaggingClassifier, 'Classifier', 'Bagging LinearSVC', tuned=tuned, pr=pr, mltl=opts.mltl, base_estimator=build_model(LinearSVC, 'Classifier', 'LinearSVC', tuned=tuned, pr=pr, mltl=opts.mltl, loss='squared_hinge', dual=False), n_jobs=1 if opts.mltl else opts.np, random_state=0)(LinearSVC(), max_samples=0.5, max_features=0.5)),
#		('LinSVM', build_model(LinearSVC, 'Classifier', 'LinearSVC', tuned=tuned, pr=pr, mltl=opts.mltl, loss='squared_hinge', dual=False)),
		('RbfSVM', build_model(SVC, 'Classifier', 'RBF SVM', tuned=tuned, pr=pr, mltl=opts.mltl))
	]:
		yield clf_name, clf
		clf_names.append(clf_name)
	if (len(glb_clfnames) < len(clf_names)):
		del glb_clfnames[:]
		glb_clfnames.extend(clf_names)
		

# Benchmark Neural Network Models
def gen_bmnn_models(input_dim, output_dim, other_clfs=None, **kwargs):
	def bmnn_models(tuned=False, glb_filtnames=[], glb_clfnames=[]):
		# Feature Filtering Model
		for filt_name, filter in gen_featfilt(tuned, glb_filtnames):
			# Classification Model
			clf_iter = gen_nnclfs(input_dim, output_dim, other_clfs)
			for clf_name, clf in clf_iter(tuned, glb_clfnames):
				yield filt_name, filter, clf_name, clf
				del clf
			del filter
	return bmnn_models
	

# Benchmark Models
def gen_bm_models(tuned=False, glb_filtnames=[], glb_clfnames=[], **kwargs):
	# Feature Filtering Model
	for filt_name, filter in gen_featfilt(tuned, glb_filtnames):
		# Classification Model
		for clf_name, clf in gen_clfs(tuned, glb_clfnames):
			yield filt_name, filter, clf_name, clf
			del clf
		del filter
		
	
# Combined Models	
def gen_cb_models(tuned=False, glb_filtnames=[], glb_clfnames=[], **kwargs):
	tuned = tuned or opts.best
	common_cfg = cfgr('gsx_extrc', 'common')
	pr = io.param_reader(os.path.join(PAR_DIR, 'etc', '%s.yaml' % common_cfg.setdefault('mdl_cfg', 'mdlcfg')))
#	filtref_func = ftslct.filtref(os.path.join(spdr.DATA_PATH, 'X.npz'), os.path.join(spdr.DATA_PATH, 'union_filt_X.npz'))
	for mdl_name, mdl in [
		# ('RandomForest', Pipeline([('clf', build_model(RandomForestClassifier, 'Classifier', 'Random Forest', tuned=tuned, pr=pr, mltl=opts.mltl, n_jobs=1 if opts.mltl else opts.np, random_state=0))])),
		('UDT-RF', Pipeline([('featfilt', ftslct.MSelectKBest(ftslct.utopk, filtfunc=ftslct.decision_tree, k=500, fn=100)), ('clf', build_model(RandomForestClassifier, 'Classifier', 'Random Forest', tuned=tuned, pr=pr, mltl=opts.mltl, n_jobs=1 if opts.mltl else opts.np, random_state=0))])),
		# ('RandomForest', Pipeline([('featfilt', SelectFromModel(DecisionTreeClassifier(criterion='entropy', class_weight='balanced', random_state=0))), ('clf', build_model(RandomForestClassifier, 'Classifier', 'Random Forest', tuned=tuned, pr=pr, mltl=opts.mltl, n_jobs=1 if opts.mltl else opts.np, random_state=0))])),
		# ('DF-RbfSVM', Pipeline([('featfilt', ftslct.MSelectOverValue(ftslct.filtref(os.path.join(spdr.DATA_PATH, 'X.npz'), os.path.join(spdr.DATA_PATH, 'union_filt_X.npz'), os.path.join(spdr.DATA_PATH, 'orig_X.npz')))), ('clf', build_model(SVC, 'Classifier', 'RBF SVM', tuned=tuned, pr=pr, mltl=opts.mltl, probability=True))])),
		('RbfSVM', Pipeline([('clf', build_model(SVC, 'Classifier', 'RBF SVM', tuned=tuned, pr=pr, mltl=opts.mltl, probability=True))])),
		# ('L1-LinSVC', Pipeline([('clf', build_model(LinearSVC, 'Classifier', 'LinearSVC', tuned=tuned, pr=pr, mltl=opts.mltl, loss='squared_hinge', dual=False))])),
		# ('Perceptron', Pipeline([('clf', build_model(Perceptron, 'Classifier', 'Perceptron', tuned=tuned, pr=pr, mltl=opts.mltl, n_jobs=1 if opts.mltl else opts.np))])),
		# ('MNB', Pipeline([('clf', build_model(MultinomialNB, 'Classifier', 'MultinomialNB', tuned=tuned, pr=pr, mltl=opts.mltl))])),
#		('5NN', Pipeline([('clf', build_model(KNeighborsClassifier, 'Classifier', 'kNN', tuned=tuned, pr=pr, mltl=opts.mltl, n_neighbors=5, n_jobs=1 if opts.mltl else opts.np))])),
		# ('MEM', Pipeline([('clf', build_model(LogisticRegression, 'Classifier', 'Logistic Regression', tuned=tuned, pr=pr, mltl=opts.mltl, dual=False))])),
		# ('LinearSVC with L2 penalty [Ft Filt] & Perceptron [CLF]', Pipeline([('featfilt', SelectFromModel(build_model(LinearSVC, 'Feature Selection', 'LinearSVC', tuned=tuned, pr=pr, mltl=opts.mltl, loss='squared_hinge', dual=False, penalty='l2'))), ('clf', build_model(Perceptron, 'Classifier', 'Perceptron', tuned=tuned, pr=pr, n_jobs=opts.np))])),
		# ('ExtraTrees', Pipeline([('clf', build_model(ExtraTreesClassifier, 'Classifier', 'Extra Trees', tuned=tuned, pr=pr, mltl=opts.mltl, n_jobs=opts.np))])),
#		('Random Forest', Pipeline([('clf', build_model(RandomForestClassifier, 'Classifier', 'Random Forest', tuned=tuned, pr=pr, n_jobs=opts.np, random_state=0))]))
	]:
		yield mdl_name, mdl
		
		
# Neural Network Clustering model
def gen_nnclt_models(input_dim, output_dim, constraint_dim=0, batch_size=32, other_clts=None, **kwargs):
	def nnclt(tuned=False, glb_filtnames=[], glb_cltnames=[], **kwargs):
		tuned = tuned or opts.best
		common_cfg = cfgr('gsx_extrc', 'common')
		pr = io.param_reader(os.path.join(PAR_DIR, 'etc', '%s.yaml' % common_cfg.setdefault('mdl_cfg', 'mdlcfg')))
		clt_names = []
		for clt_name, clt in [
			('3L64U Neural Network', gen_keras(input_dim, output_dim, model='cfkmeans', constraint_dim=constraint_dim, batch_size=batch_size, internal_dim=64, metric='manhattan', gamma=0.01)),
			('3L96U Neural Network', gen_keras(input_dim, output_dim, model='cfkmeans', constraint_dim=constraint_dim, batch_size=batch_size, internal_dim=96, metric='manhattan', gamma=0.01))
		]:
			yield clt_name, clt
			clt_names.append(clt_name)
		if (other_clts is not None):
			for clt_name, clt in other_clts(tuned, glb_filtnames, glb_clfnames):
				yield clt_name, clt
				clt_names.append(clt_name)
		if (len(glb_cltnames) < len(clt_names)):
			del glb_cltnames[:]
			glb_cltnames.extend(clt_names)
	return nnclt
		
		
# Clustering model
def gen_clt_models(tuned=False, glb_filtnames=[], glb_cltnames=[], **kwargs):
	tuned = tuned or opts.best
	common_cfg = cfgr('gsx_extrc', 'common')
	pr = io.param_reader(os.path.join(PAR_DIR, 'etc', '%s.yaml' % common_cfg.setdefault('mdl_cfg', 'mdlcfg')))
	clt_names = []
	for clt_name, clt in [
		# ('SJI-AGGLM', Pipeline([('distcalc', dstclc.gen_dstclc(dstclc.sji)), ('clt', AgglomerativeClustering(metric='precomputed'))])),
		# ('Manh-DBSCAN', Pipeline([('distcalc', dstclc.gen_dstclc(pairwise_distances, kw_args={'metric':'manhattan', 'n_jobs':opts.np})), ('clt', DBSCAN(min_samples=2, metric='precomputed', n_jobs=opts.np))])),
		# ('Manh-DBSCAN', DBSCAN(min_samples=2, metric='manhattan', algorithm='ball_tree', n_jobs=opts.np)),
		('FuzzyCmeans', FZCMeans(n_clusters=100, random_state=0)),
		('LDA', LDACluster(n_clusters=100, learning_method='online', learning_offset=50., max_iter=5, n_jobs=opts.np, random_state=0)),
	]:
		yield clt_name, clt
		clt_names.append(clt_name)
	if (len(glb_cltnames) < len(clt_names)):
		del glb_cltnames[:]
		glb_cltnames.extend(clt_names)
		
		
def gen_cbclt_models(tuned=False, glb_filtnames=[], glb_clfnames=[], **kwargs):
	import hdbscan
	tuned = tuned or opts.best
	common_cfg = cfgr('gsx_extrc', 'common')
	pr = io.param_reader(os.path.join(PAR_DIR, 'etc', '%s.yaml' % common_cfg.setdefault('mdl_cfg', 'mdlcfg')))
	for mdl_name, mdl in [
		# ('CNZ-DBSCAN', Pipeline([('distcalc', dstclc.gen_dstclc(dstclc.cns_dist, kw_args={'metric':'euclidean', 'C':kwargs.setdefault('constraint', None), 'a':0.4, 'n_jobs':opts.np})), ('clt', DBSCAN(metric='precomputed', n_jobs=opts.np))])),
		# ('CNZ-HDBSCAN', Pipeline([('distcalc', dstclc.gen_dstclc(dstclc.cns_dist, kw_args={'metric':'euclidean', 'C':kwargs.setdefault('constraint', None), 'a':0.4, 'n_jobs':opts.np})), ('clt', hdbscan.HDBSCAN(min_cluster_size=2, metric='precomputed', n_jobs=opts.np))])),
		# ('ManhCNZ-DBSCAN', Pipeline([('distcalc', dstclc.gen_dstclc(dstclc.cns_dist, kw_args={'metric':'manhattan', 'C':kwargs.setdefault('constraint', None), 'a':0.4, 'n_jobs':opts.np})), ('clt', DBSCAN(metric='precomputed', n_jobs=opts.np))])),
		# ('ManhCNZ-HDBSCAN', Pipeline([('distcalc', dstclc.gen_dstclc(dstclc.cns_dist, kw_args={'metric':'manhattan', 'C':kwargs.setdefault('constraint', None), 'a':0.4, 'n_jobs':opts.np})), ('clt', hdbscan.HDBSCAN(min_cluster_size=2, metric='precomputed', n_jobs=opts.np))])),
		('Kallima', Pipeline([('clt', kallima.Kallima(metric='euclidean', method='mstcut', cut_method='normcut', cut_step=0.01, cns_ratio=0.5, nn_method='rnn', nn_param=0.5, max_cltnum=1500, coarse=0.4, rcexp=1, cond=0.3, cross_merge=False, merge_all=False, save_g=True, n_jobs=opts.np))])),
		# ('Manh-DBSCAN', Pipeline([('clt', DBSCAN(metric='manhattan', algorithm='ball_tree', n_jobs=opts.np))])),
		# ('FuzzyCmeans', Pipeline([('clt', FZCMeans(n_clusters=1500, random_state=0))])),
		# ('CNSFuzzyCmeans', Pipeline([('clt', CNSFZCMeans(n_clusters=1500, a=0.4, random_state=0, n_jobs=opts.np))])),
		# ('LDA', Pipeline([('clt', LDACluster(n_clusters=1500, learning_method='online', learning_offset=50., max_iter=5, n_jobs=opts.np, random_state=0))])),
	]:
		yield mdl_name, mdl
		
		
# Models with parameter range
def gen_nnmdl_params(input_dim, output_dim, rdtune=False):
	common_cfg = cfgr('gsx_extrc', 'common')
	pr = io.param_reader(os.path.join(PAR_DIR, 'etc', '%s.yaml' % common_cfg.setdefault('mdl_cfg', 'mdlcfg')))
	if (rdtune):
		for mdl_name, mdl, params in [
			('Neural Network', gen_keras(input_dim, output_dim, model='tunable'), {
				'param_grid':dict(
					internal_dim=np.logspace(6, 9, num=4, base=2, dtype='int'),
					layer_num=np.logspace(2, 6, num=5, base=2, dtype='int'),
					dropout_ratio=np.logspace(-0.301, 0, num=10).tolist(),
					init=['uniform', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'],
					activation=['tanh', 'sigmoid', 'hard_sigmoid', 'relu', 'linear', 'softplus', 'softsign']),
				'n_iter':30
			})
		]:
			yield mdl_name, mdl, params
	else:
		for mdl_name, mdl, params in [
			('Neural Network', gen_keras(input_dim, output_dim, model='tunable'), {
				'param_grid':dict(
					internal_dim=np.logspace(6, 9, num=4, base=2, dtype='int'),
					layer_num=np.logspace(2, 6, num=5, base=2, dtype='int'),
					dropout_ratio=np.logspace(-0.301, 0, num=10).tolist(),
					init=['uniform', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'],
					activation=['tanh', 'sigmoid', 'hard_sigmoid', 'relu', 'linear', 'softplus', 'softsign'])
			})
		]:
			yield mdl_name, mdl, params


# Models with parameter range
def gen_mdl_params(rdtune=False):
	common_cfg = cfgr('gsx_extrc', 'common')
	pr = io.param_reader(os.path.join(PAR_DIR, 'etc', '%s.yaml' % common_cfg.setdefault('mdl_cfg', 'mdlcfg')))
	if (rdtune):
		for mdl_name, mdl, params in [
			# ('Logistic Regression', LogisticRegression(dual=False), {
				# 'param_dist':dict(
					# penalty=['l1', 'l2'],
					# C=np.logspace(-5, 5, 11),
					# tol=np.logspace(-6, 3, 10)),
				# 'n_iter':30
			# }),
			# ('LinearSVC', LinearSVC(dual=False), {
				# 'param_dist':dict(
					# penalty=['l1', 'l2'],
					# C=np.logspace(-5, 5, 11),
					# tol=np.logspace(-6, 3, 10)),
				# 'n_iter':30
			# }),
			# ('Perceptron', Perceptron(), {
				# 'param_dist':dict(
					# alpha=np.logspace(-6, 3, 10),
					# n_iter=stats.randint(3, 20)),
				# 'n_iter':30
			# }),
			# ('MultinomialNB', MultinomialNB(), {
				# 'param_dist':dict(
					# alpha=np.logspace(-6, 3, 10),
					# fit_prior=[True, False]),
				# 'n_iter':30
			# }),
			# ('SVM', SVC(), {
				# 'param_dist':dict(
					# kernel=['linear', 'rbf', 'poly'],
					# C=np.logspace(-5, 5, 11),
					# gamma=np.logspace(-6, 3, 10)),
				# 'n_iter':30
			# }),
			# ('Extra Trees', ExtraTreesClassifier(random_state=0), {
				# 'param_dist':dict(
					# n_estimators=[50, 100] + range(200, 1001, 200),
					# max_features=np.linspace(0.5, 1, 6).tolist()+['sqrt', 'log2'],
					# min_samples_leaf=[1]+range(10, 101, 10),
					# class_weight=['balanced', None]),
				# 'n_iter':30
			# }),
			('Random Forest', RandomForestClassifier(random_state=0), {
				'param_dist':dict(
					n_estimators=[50, 100] + range(200, 1001, 200),
					max_features=np.linspace(0.5, 1, 6).tolist()+['sqrt', 'log2'],
					max_depth=[None] + range(10,101,10),
					min_samples_leaf=[1]+range(10, 101, 10),
					class_weight=['balanced', None]),
				'n_iter':30
			}),
			# ('Bagging MNB', BaggingClassifier(base_estimator=MultinomialNB(), random_state=0), {
				# 'param_dist':dict(
					# n_estimators=[20, 50, 100] + range(200, 601, 200),
					# max_samples=np.linspace(0.5, 1, 6),
					# max_features=np.linspace(0.5, 1, 6),
					# bootstrap=[True, False],
					# bootstrap_features=[True, False]),
				# 'n_iter':30
			# }),
			# ('AdaBoost MNB', AdaBoostClassifier(base_estimator=MultinomialNB(), algorithm='SAMME', random_state=0), {
				# 'param_dist':dict(
					# n_estimators=[20, 50, 100] + range(200, 601, 200),
					# learning_rate=np.linspace(0.5, 1, 6)),
				# 'n_iter':30
			# }),
			# ('GBoost', GradientBoostingClassifier(random_state=0), {
				# 'param_dist':dict(
					# n_estimators=[20, 50, 100] + range(200, 601, 200),
					# subsample = np.linspace(0.5, 1, 6),
					# max_features=np.linspace(0.5, 1, 6).tolist()+['sqrt', 'log2'],
					# min_samples_leaf=[1]+range(10, 101, 10),
					# learning_rate=np.linspace(0.5, 1, 6),
					# loss=['deviance', 'exponential']),
				# 'n_iter':30
			# }),
			# ('UGSS & RF', Pipeline([('featfilt', ftslct.MSelectKBest(ftslct.utopk, filtfunc=ftslct.gss_coef, fn=4000)), ('clf', RandomForestClassifier())]), {
				# 'param_dist':dict(
					# featfilt__k=np.logspace(np.log2(250), np.log2(32000), 8, base=2).astype('int')),
				# 'n_iter':8
			# })
		]:
			yield mdl_name, mdl, params
	else:
		for mdl_name, mdl, params in [
			# ('Logistic Regression', LogisticRegression(dual=False), {
				# 'param_grid':dict(
					# penalty=['l1', 'l2'],
					# C=np.logspace(-5, 5, 11),
					# tol=np.logspace(-6, 3, 10))
			# }),
			# ('LinearSVC', LinearSVC(dual=False), {
				# 'param_grid':dict(
					# penalty=['l1', 'l2'],
					# C=np.logspace(-5, 5, 11),
					# tol=np.logspace(-6, 3, 10))
			# }),
			# ('Perceptron', Perceptron(), {
				# 'param_grid':dict(
					# alpha =np.logspace(-5, 5, 11),
					# n_iter=range(3, 20, 3))
			# }),
			# ('MultinomialNB', MultinomialNB(), {
				# 'param_grid':dict(
					# alpha=np.logspace(-6, 3, 10),
					# fit_prior=[True, False])
			# }),
			# ('SVM', SVC(), {
				# 'param_grid':dict(
					# kernel=['linear', 'rbf', 'poly'],
					# C=np.logspace(-5, 5, 11),
					# gamma=np.logspace(-6, 3, 10))
			# }),
			# ('Extra Trees', ExtraTreesClassifier(random_state=0), {
				# 'param_grid':dict(
					# n_estimators=[50, 100] + range(200, 1001, 200),
					# max_features=np.linspace(0.5, 1, 6).tolist()+['sqrt', 'log2'],
					# min_samples_leaf=[1]+range(10, 101, 10),
					# class_weight=['balanced', None])
			# }),
			('Random Forest', RandomForestClassifier(random_state=0), {
				'param_grid':dict(
					n_estimators=[50, 100] + range(200, 1001, 200),
					max_features=np.linspace(0.5, 1, 6).tolist()+['sqrt', 'log2'],
					max_depth=[None] + range(10,101,10),
					min_samples_leaf=[1]+range(10, 101, 10),
					class_weight=['balanced', None])
			}),
			# ('Bagging MNB', BaggingClassifier(base_estimator=MultinomialNB(), random_state=0), {
				# 'param_grid':dict(
					# n_estimators=[50, 100] + range(200, 1001, 200),
					# max_samples=np.linspace(0.5, 1, 6),
					# max_features=np.linspace(0.5, 1, 6),
					# bootstrap=[True, False],
					# bootstrap_features=[True, False])
			# }),
			# ('AdaBoost MNB', AdaBoostClassifier(base_estimator=MultinomialNB(), algorithm='SAMME', random_state=0), {
				# 'param_grid':dict(
					# n_estimators=[50, 100] + range(200, 1001, 200),
					# learning_rate=np.linspace(0.5, 1, 6))
			# }),
			# ('GBoost', GradientBoostingClassifier(random_state=0), {
				# 'param_grid':dict(
					# n_estimators=[50, 100] + range(200, 1001, 200),
					# subsample = np.linspace(0.5, 1, 6),
					# max_features=np.linspace(0.5, 1, 6).tolist()+['sqrt', 'log2'],
					# min_samples_leaf=[1]+range(10, 101, 10),
					# learning_rate = np.linspace(0.5, 1, 6),
					# loss=['deviance', 'exponential'])
			# }),
			# ('UDT & RF', Pipeline([('featfilt', ftslct.MSelectKBest(ftslct.utopk, filtfunc=ftslct.decision_tree, fn=4000)), ('clf', RandomForestClassifier())]), {
				# 'param_grid':dict(
					# featfilt__k=np.logspace(np.log2(250), np.log2(32000), 8, base=2).astype('int'))
			# }),
			# ('DT & RF', Pipeline([('featfilt', ftslct.MSelectKBest(ftslct.decision_tree)), ('clf', RandomForestClassifier())]), {
				# 'param_grid':dict(
					# featfilt__k=np.logspace(np.log2(250), np.log2(32000), 8, base=2).astype('int'))
			# }),
			# ('UNGL & RF', Pipeline([('featfilt', ftslct.MSelectKBest(ftslct.utopk, filtfunc=ftslct.ngl_coef, fn=4000)), ('clf', RandomForestClassifier())]), {
				# 'param_grid':dict(
					# featfilt__k=np.logspace(np.log2(250), np.log2(32000), 8, base=2).astype('int'))
			# }),
			# ('NGL & RF', Pipeline([('featfilt', ftslct.MSelectKBest(ftslct.ngl_coef)), ('clf', RandomForestClassifier())]), {
				# 'param_grid':dict(
					# featfilt__k=np.logspace(np.log2(250), np.log2(32000), 8, base=2).astype('int'))
			# }),
			# ('UGSS & RF', Pipeline([('featfilt', ftslct.MSelectKBest(ftslct.utopk, filtfunc=ftslct.gss_coef, fn=4000)), ('clf', RandomForestClassifier())]), {
				# 'param_grid':dict(
					# featfilt__k=np.logspace(np.log2(250), np.log2(32000), 8, base=2).astype('int'))
			# }),
			# ('GSS & RF', Pipeline([('featfilt', ftslct.MSelectKBest(ftslct.gss_coef)), ('clf', RandomForestClassifier())]), {
				# 'param_grid':dict(
					# featfilt__k=np.logspace(np.log2(250), np.log2(32000), 8, base=2).astype('int'))
			# })
		]:
			yield mdl_name, mdl, params
			
			
def all():
	gse_clf()
	gsm_clf()
	gen_sgn()


def gse_clf():
	global cfgr

	if (opts.mltl):
		pid = -1
	else:
		pid = opts.pid
	print 'Process ID: %s' % pid
	
	## Load data for GSE
	gse_X, gse_Y = load_data(type='gse', pid=pid, fmt=opts.fmt, spfmt=opts.spfmt)
	if (opts.mltl):
		gse_Y = gse_Y.as_matrix()
		if (len(gse_Y.shape) == 1 or gse_Y.shape[1] == 1):
			gse_Y = gse_Y.reshape((gse_Y.shape[0],))
	else:
		gse_Y = gse_Y.as_matrix().reshape((gse_Y.shape[0],))
	
	## Cross validation for GSE
	print 'Cross validation for GSE'
	gse_filt_names, gse_clf_names, gse_pl_names = [[] for i in range(3)]
	gse_pl_set = set([])
	gse_model_iter = gen_cb_models if opts.comb else gen_bm_models
	if (opts.dend is not None):
		# gse_model_iter = gen_cb_models if opts.comb else gen_bmnn_models(gse_X.shape[1], gse_Y.shape[1] if len(gse_Y.shape) == 2 else 1, gse_model_iter)
		gse_model_iter = gen_cb_models if opts.comb else gen_bmnn_models(gse_X.shape[1], gse_Y.shape[1] if len(gse_Y.shape) == 2 else 1, None)
	model_param = dict(tuned=opts.best, glb_filtnames=gse_filt_names, glb_clfnames=gse_clf_names)
	global_param = dict(comb=opts.comb, pl_names=gse_pl_names, pl_set=gse_pl_set)
	txtclf.cross_validate(gse_X, gse_Y, gse_model_iter, model_param, avg=opts.avg, kfold=opts.kfold, cfg_param=cfgr('bionlp.txtclf', 'cross_validate'), global_param=global_param, lbid=pid)


def gsm_clf():
	global cfgr

	if (opts.mltl):
		pid = -1
	else:
		pid = opts.pid
	print 'Process ID: %s' % pid
	
	## Load data for GSM
	gsm_Xs, gsm_Ys = load_data(type='gsm-clf', pid=pid, fmt=opts.fmt, spfmt=opts.spfmt)
	if (opts.mltl):
		gsm_Ys = [Y.as_matrix() if len(Y.shape) > 1 and Y.shape[1] > 1 else Y.as_matrix().reshape((Y.shape[0],)) for Y in gsm_Ys]
	else:
		gsm_Ys = [Y.as_matrix().reshape((Y.shape[0],)) for Y in gsm_Ys]
	
	## Cross validation for GSM
	print 'Cross validation for GSM'
	for X, Y in zip(gsm_Xs, gsm_Ys):
		gsm_filt_names, gsm_clf_names, gsm_pl_names = [[] for i in range(3)]
		gsm_pl_set = set([])
		gsm_model_iter = gen_cb_models if opts.comb else gen_bm_models
		if (opts.dend is not None):
			# gsm_model_iter = gen_cb_models if opts.comb else gen_bmnn_models(X.shape[1], Y.shape[1] if len(Y.shape) == 2 else 1, gsm_model_iter)
			gsm_model_iter = gen_cb_models if opts.comb else gen_bmnn_models(X.shape[1], Y.shape[1] if len(Y.shape) == 2 else 1, None)
		model_param = dict(tuned=opts.best, glb_filtnames=gsm_filt_names, glb_clfnames=gsm_clf_names)
		global_param = dict(comb=opts.comb, pl_names=gsm_pl_names, pl_set=gsm_pl_set)
		txtclf.cross_validate(X, Y, gsm_model_iter, model_param, avg=opts.avg, kfold=opts.kfold, cfg_param=cfgr('bionlp.txtclf', 'cross_validate'), global_param=global_param, lbid=pid)
		
		
def gsm_clt():
	global cfgr

	if (opts.mltl):
		pid = -1
	else:
		pid = opts.pid
	print 'Process ID: %s' % pid
	
	## Load data for GSM
	Xs, Ys, labels, gsm2gse = load_data(type='gsm-clt', pid=pid, fmt=opts.fmt, spfmt=opts.spfmt)
	labels = [lbs.as_matrix() if len(lbs.shape) > 1 and lbs.shape[1] > 1 else lbs.as_matrix().reshape((lbs.shape[0],)) for lbs in labels]
	
	## Clustering for GSM
	print 'Clustering for GSM...'
	# orig_wd = os.getcwd()
	for i, (X, y, c) in enumerate(zip(Xs, labels, Ys)):
		# Switch working directory
		# new_wd = os.path.join(orig_wd, str(i) if pid == -1 else str(pid))
		# fs.mkdir(new_wd)
		# os.chdir(new_wd)
		c = c.as_matrix()
		# Transform the GEO IDs into constraints
		# gse_ids = gsm2gse.loc[X.index]
		# gseidc = label_binarize(gse_ids.as_matrix(), classes=gse_ids.gse_id.value_counts().index)
		# c = np.hstack((c, gseidc))
		filt_names, clt_names, pl_names = [[] for j in range(3)]
		pl_set = set([])
		model_iter = gen_cbclt_models if opts.comb else gen_clt_models
		if (opts.dend is not None):
			y = label_binarize(y, classes=list(set([l for l in y.reshape((-1,)) if l != -1])))
			# model_iter = gen_nnclt_models(input_dim=X.shape[1], output_dim=y.shape[1] if len(y.shape) == 2 else 1, constraint_dim=c.shape[1] if len(c.shape) == 2 else 1, batch_size=opts.bsize, other_clts=gsm_model_iter)
			model_iter = gen_nnclt_models(input_dim=X.shape[1], output_dim=y.shape[1] if len(y.shape) == 2 else 1, constraint_dim=c.shape[1] if len(c.shape) == 2 else 1, batch_size=opts.bsize, other_clts=None)
		model_param = dict(tuned=opts.best, glb_filtnames=filt_names, glb_cltnames=clt_names, is_fuzzy=opts.fuzzy, is_nn=False if opts.dend is None else True, constraint=c)
		global_param = dict(comb=opts.comb, pl_names=pl_names, pl_set=pl_set)
		# txtclt.cross_validate(X, y, model_iter, model_param, kfold=opts.kfold, cfg_param=cfgr('bionlp.txtclt', 'cross_validate'), global_param=global_param, lbid=pid)
		txtclt.clustering(X, model_iter, model_param, cfg_param=cfgr('bionlp.txtclt', 'clustering'), global_param=global_param, lbid=pid)
		# os.chdir(orig_wd)

		
def _filt_ent(entities, onto_lb):
	filtered = []
	txt = nlp.clean_txt('\n'.join([e['word'] for e in entities]))
	loc = np.cumsum([0] + [len(e['word']) + 1 for e in entities])
	if (onto_lb == 'PRGE'):
		succeeded, trial_num = False, 0
		while (not succeeded and trial_num < 20):
			try:
				df = hgnc.symbol_checker(txt).dropna()
				succeeded = True
			except RuntimeError as e:
				trial_num += 1
				time.sleep(5)
		if (df.empty): return []
		idx = [bisect.bisect_left(loc, txt.find(x)) for x in df['Input']]
	elif (onto_lb == 'DISO'):
		df = dnorm.annot_dss(txt)
		if (df.empty): return []
		idx = [bisect.bisect_left(loc, loc_s) for loc_s in df['start']]
	elif (onto_lb == 'CHED'):
		c = rxnav.RxNavAPI('drugs')
		idx = [i for i, e in enumerate(entities) if len(c.call(name=nlp.clean_txt(e['word']))['concept_group']) > 0]
	else:
		return entities
	return [entities[i] for i in idx if i < len(entities)]

	
def gen_sgn():
	global cfgr
	common_cfg = cfgr('gsx_extrc', 'common')
	sgn_cfg = cfgr('gsx_extrc', 'gen_sgn')

	if (opts.mltl):
		pid = -1
	else:
		pid = opts.pid
	print 'Process ID: %s' % pid
	
	if (opts.thrshd != 'mean' and opts.thrshd != 'min'):
		opts.thrshd = float(ast.literal_eval(opts.thrshd))
	
	if (len(sgn_cfg) > 0):
		method = sgn_cfg['method']
		format = sgn_cfg['format']
		sample_dir = os.path.join('.', 'samples') if sgn_cfg['sample_dir'] is None else sgn_cfg['sample_dir']
		ge_dir = spdr.GEO_PATH if sgn_cfg['ge_dir'] is None else sgn_cfg['ge_dir']
		dge_dir = spdr.GEO_PATH if sgn_cfg['dge_dir'] is None else sgn_cfg['dge_dir']
	
	## Load data for GSM and the association
	Xs, Ys, labels, gsm2gse = load_data(type='sgn', pid=pid, fmt=opts.fmt, spfmt=opts.spfmt)
	## Load GEO and GSM Documents
	gse_docs, gsm_docs = spdr.get_geos(type='gse'), spdr.get_geos(type='gsm')
	
	## Generating GSM Cluster Pairs
	pair_dfs = []
	for i, (X, Y, z) in enumerate(zip(Xs, Ys, labels)):
		lbid = i if (pid == -1) else pid
		io.inst_print('Generating pairs of GSM sample clusters for dataset %i ...' % lbid)
		pair_df = helper._gsmclt_pair(X, Y, z, gsm2gse, lbid, thrshd=opts.thrshd, cache_path=opts.cache)
		pair_dfs.append(pair_df)

	## Generating Basic Signatures
	presgn_dfs = []
	for i, (X, Y, z, pair_df) in enumerate(zip(Xs, Ys, labels, pair_dfs)):
		lbid = i if (pid == -1) else pid
		sgn_fname = 'pre_sgn_%s.npz' % lbid
		cachef = os.path.join(opts.cache, sgn_fname)
		if (os.path.exists(cachef)):
			io.inst_print('Reading cache for basic signatures of dataset %i ...' % lbid)
			presgn_dfs.append(io.read_df(cachef, with_idx=True))
		else:
			io.inst_print('Generating the basic signatures for dataset %i ...' % lbid)
			platforms, organisms, tissues = [[] for x in range(3)]
			for gse_id, ctrl_str, pert_str in zip(pair_df['geo_id'], pair_df['ctrl_ids'], pair_df['pert_ids']):
				gsm_doc_list = [gsm_docs[gsm_id][0] for gsm_id in ctrl_str.split('|') + pert_str.split('|')]
				# Label the terms in the GEO documents that associated with each signature
				pf_count, og_cout, ts_count = collections.Counter([doc['platform'] for doc in gsm_doc_list]).most_common(1), collections.Counter([doc['organism'] for doc in gsm_doc_list]).most_common(1), collections.Counter([doc['tissue'] for doc in gsm_doc_list if doc.has_key('tissue') and doc['tissue'] != ''] + [doc['tissue_type'] for doc in gsm_doc_list if doc.has_key('tissue_type') and doc['tissue_type'] != '']).most_common(1)
				platforms.append(pf_count[0][0] if len(pf_count) > 0 else '')
				organisms.append(og_cout[0][0] if len(og_cout) > 0 else '')
				tissues.append(ts_count[0][0] if len(ts_count) > 0 else '')
			columns = ['Platforms', 'Organisms', 'Tissues']
			preannot_df = pd.DataFrame.from_items([(k, v) for k, v in zip(columns, [platforms, organisms, tissues])], columns=columns)
			preannot_df.index = pair_df.index
			presgn_df = pd.concat([pair_df, preannot_df], axis=1, join_axes=[pair_df.index], copy=False)
			presgn_df.index.name = 'id'
			io.write_df(presgn_df, 'pre_sgn_%s.npz' % lbid, with_idx=True)
			presgn_df.to_excel('pre_sgn_%s.xlsx' % lbid, encoding='utf8')
			presgn_dfs.append(presgn_df)
		# Calculate the Differential Gene Expression
		ds_lb = gse_docs[pair_df['geo_id'][0]][1][0]
		_label, _method = ds_lb.lower().replace(' ', '_'), method.lower().replace(' ', '_')
		sample_path, ge_path, dge_path, dge_cache_path = os.path.join(sample_dir, format, _label, 'samples'), os.path.join(ge_dir, _label), os.path.join(dge_dir, _method, _label), os.path.join(dge_dir, 'cache', _label)
		dge_filter_path, dge_cache_filter_path = os.path.join(dge_path, 'filtered'), os.path.join(dge_cache_path, 'filtered')
		fs.mkdir(dge_filter_path), fs.mkdir(os.path.join(dge_cache_filter_path, _method))
		io.inst_print('Calculating the gene expression for dataset %i ...' % lbid)
		helper._sgn2ge(presgn_dfs[-1], sample_path, ge_path, format=format)
		io.inst_print('Calculating the differential gene expression for dataset %i ...' % lbid)
		dge_dfs = helper._sgn2dge(presgn_dfs[-1], method, ge_path, dge_path, dge_cache_path)
		# Filter the pairs with low p-value
		io.inst_print('Filtering the signatures for dataset %i according to the p-value of differential gene expression ...' % lbid)
		pvalues = np.array([dge_df['pvalue'].min() for dge_df in dge_dfs])
		selection = pvalues < (sgn_cfg['pval_thrshd'] if sgn_cfg.has_key('pval_thrshd') and sgn_cfg['pval_thrshd'] is not None else 0.05)
		presgn_dfs[-1] = presgn_dfs[-1][selection]
		orig_ids = np.arange(pvalues.shape[0])[selection]
		orig_map = pd.DataFrame(orig_ids.reshape((-1,1)), index=presgn_dfs[-1].index, columns=['orig_idx'])
		io.write_df(orig_map, 'orig_map.npz', with_idx=True)
		# Set the index
		presgn_dfs[-1].index = ['%s:%i' % (spdr.LABEL2ID[ds_lb], x) for x in range(presgn_dfs[-1].shape[0])]
		for idx, orig_idx in enumerate(orig_ids):
			dge_src = os.path.join(dge_path, 'dge_%i.npz' % orig_idx)
			dge_dst = os.path.join(dge_filter_path, 'dge_%i.npz' % idx)
			if (not os.path.exists(dge_dst)):
				copyfile(dge_src, dge_dst)
			dge_cache_src = os.path.join(dge_cache_path, _method, '%i.npz' % orig_idx)
			dge_cache_dst = os.path.join(dge_cache_filter_path, _method, '%i.npz' % idx)
			if (not os.path.exists(dge_cache_dst)):
				copyfile(dge_cache_src, dge_cache_dst)
	## Annotating Signatures
	top_k = 3
	cache_path = os.path.join(spdr.GEO_PATH, 'annot')
	txt_fields = [['title', 'summary', 'keywords'], ['title', 'description', 'source', 'organism', 'treat_protocol', 'trait']]
	txtfield_importance = {'title':8, 'summary':4, 'keywords':7, 'description':4, 'source':5, 'organism':5, 'treat_protocol':9, 'trait':5}
	sgn_dfs, annot_lists, common_annots, annot_dicts = [[] for x in range(4)]
	for i, (X, Y, z, presgn_df) in enumerate(zip(Xs, Ys, labels, presgn_dfs)):
		lbid = i if (pid == -1) else pid
		sgn_fname = 'signature_%s.npz' % lbid
		cachef = os.path.join(opts.cache, sgn_fname)
		if (os.path.exists(cachef)):
			io.inst_print('Reading cache for annotated signatures of dataset %i ...' % lbid)
			annot_list = io.read_obj(os.path.join(opts.cache, 'annot_list_%i.pkl' % lbid))
			common_annot = io.read_obj(os.path.join(opts.cache, 'common_annot_%i.pkl' % lbid))
			annot_dict = io.read_obj(os.path.join(opts.cache, 'annot_dict_%i.pkl' % lbid))
			if (annot_list is not None and common_annot is not None and annot_dict is not None):
				annot_lists.append(annot_list)
				common_annots.append(common_annot)
				annot_dicts.append(annot_dict)
				sgn_dfs.append(io.read_df(cachef, with_idx=True))
				continue
		io.inst_print('Annotating the signatures for dataset %i ...' % lbid)
		common_annot_list, annot_list = [[] for x in range(2)]
		for gse_id, ctrl_str, pert_str in zip(presgn_df['geo_id'], presgn_df['ctrl_ids'], presgn_df['pert_ids']):
			gsm_annotres, annot_ents, annot_terms, annot_weights = [], {}, {}, {}
			gsm_list = ctrl_str.split('|') + pert_str.split('|')
			gse_doc, gsm_doc_list = gse_docs[gse_id][0], [gsm_docs[gsm_id][0] for gsm_id in gsm_list]
			txt_field_maps = [0] + [1] * len(gsm_doc_list)
			# Annotate the GSE document
			gse_annotres = helper._annot_sgn(gse_id, gse_doc, txt_fields[0], cache_path=cache_path)
			# Annotate the GSM document
			for geo_id, geo_doc in zip(gsm_list, gsm_doc_list):
				gsm_annotres.append(helper._annot_sgn(geo_id, geo_doc, txt_fields[1], cache_path=cache_path))
			annot_list.append([gse_annotres] + gsm_annotres)
			# Extract the annotated entities from the results, and classify them based on the annotation (modifier) type
			for annotres, tfmap in zip(annot_list[-1], txt_field_maps):
				for annot_gp, txt_field in zip(annotres, txt_fields[tfmap]):
					for annotype, entities in annot_gp.iteritems():
						annot_ent = [':'.join(entity['ids'] + [entity['word']]) for entity in entities]
						annot_ents.setdefault(annotype, []).extend(annot_ent)
						annot_weights.setdefault(annotype, []).extend([txtfield_importance[txt_field]] * len(annot_ent))
						annot_mdf = [entity['modifier'] for entity in entities if entity['modifier'] != '']
						annot_ents.setdefault('mdf_'+annotype, []).extend(annot_mdf)
						annot_weights.setdefault('mdf_'+annotype, []).extend([txtfield_importance[txt_field]] * len(annot_mdf))
			# Obtain the top (2) k most common entities for each annotation (modifier) type
			for annotype, entities in annot_ents.iteritems():
				if (len(entities) == 0): continue
				annot_weight = dstclc.normdist(np.array(annot_weights[annotype]))
				ent_array = np.array(entities)
				annot_count = func.sorted_tuples([(k, annot_weight[np.where(ent_array == k)[0]].sum()) for k in set(entities)], key_idx=1)[::-1]
				if (annotype.startswith('mdf_')):
					# annot_count = collections.Counter(entities).most_common(2)
					annot_count = annot_count[:2]
					if (len(annot_count) > 1 and annot_count[0][1] == annot_count[1][1]):
						annot_text = ' & '.join(sorted(zip(*annot_count[:2])[0]))
					elif len(annot_count) > 0:
						annot_text = annot_count[0][0]
					else:
						annot_text = ''
					annot_terms[annotype] = [annot_text]
				else:
					# annot_count = collections.Counter(entities).most_common(top_k)
					annot_count = annot_count[:top_k]
					annot_terms[annotype] = [x[0].split(':')[-1] for x in annot_count] if len(annot_count) > 0 else ['']
			if (len(annot_terms) == 0):
				print 'Unable to annotate signatures for GEO document %s !' % gse_id
			common_annot_list.append(annot_terms)
		annot_lists.append(annot_list)
		io.write_obj(annot_list, 'annot_list_%i.pkl' % lbid)
		common_annots.append(common_annot_list)
		io.write_obj(common_annot_list, 'common_annot_%i.pkl' % lbid)
		if (len(common_annot_list) == 0):
			print 'Unable to annotate signatures, please check your network!'
			continue
		print 'Generating the annotated signatures for dataset %i ...' % lbid
		# Combine all the annotation types
		annotypes = list(set(func.flatten_list([x.keys() for x in common_annot_list])))
		# Make a unified annotation dictionary
		annot_dict = dict([(annotype, []) for annotype in annotypes])
		for annotype in annotypes:
			annot_dict[annotype].extend([annot_terms.setdefault(annotype, [''])[0] for annot_terms in common_annot_list])
		annot_dicts.append(annot_dict)
		io.write_obj(annot_dict, 'annot_dict_%i.pkl' % lbid)
		annot_df = pd.DataFrame.from_items([(k, v) for k, v in annot_dict.iteritems() if len(v) == presgn_df.shape[0]])
		annot_df.index = presgn_df.index
		sgn_df = pd.concat([presgn_df, annot_df], axis=1, join_axes=[presgn_df.index], copy=False)
		io.write_df(sgn_df, 'signature_%s.npz' % lbid, with_idx=True)
		sgn_df.to_excel('signature_%s.xlsx' % lbid, encoding='utf8')
		sgn_dfs.append(sgn_df)
	## Annotating the Gene, Disease, and Drug ontology
	postsgn_dfs, ontoid_cols, ontolb_cols = [[] for x in range(3)]
	for i, (X, Y, z, annot_list, common_annot_list, sgn_df) in enumerate(zip(Xs, Ys, labels, annot_lists, common_annots, sgn_dfs)):
		lbid = i if (pid == -1) else pid
		sgn_fname = 'post_sgn_%s.npz' % lbid
		cachef = os.path.join(opts.cache, sgn_fname)
		if (os.path.exists(cachef)):
			io.inst_print('Reading cache for ontology-annotated signatures of dataset %i ...' % lbid)
			postsgn_dfs.append(io.read_df(cachef, with_idx=True))
			continue
		io.inst_print('Annotating the ontology for dataset %i ...' % lbid)
		# Read the ontology database
		ds_lb = gse_docs[sgn_df['geo_id'][0]][1][0]
		onto_lb, ontodb_name = spdr.LABEL2ONTO[ds_lb], spdr.LABEL2DB[ds_lb]
		onto_lang, idns, prdns, idprds, lbprds = spdr.DB2LANG[ontodb_name], getattr(ontology, spdr.DB2IDNS[ontodb_name]), [(ns.lower(), getattr(ontology, ns)) for ns in dict(spdr.DB2PRDS[ontodb_name]['idprd']).keys()], dict([((prdn[0].lower(), prdn[1]), '_'.join(prdn)) for prdn in spdr.DB2PRDS[ontodb_name]['idprd']]), dict([((prdn[0].lower(), prdn[1]), '_'.join(prdn)) for prdn in spdr.DB2PRDS[ontodb_name]['lbprds']])
		ontodb_path = os.path.join(spdr.ONTO_PATH, ontodb_name)
		# Get the ontology graph
		# ontog = ontology.get_db_graph(ontodb_path, db_name=ontodb_name, db_type='SQLAlchemy') # from rdflib db
		ontog = sparql.SPARQL('http://localhost:8890/%s/query' % ontodb_name, use_cache=common_cfg.setdefault('memcache', False)) # from Jena TDB
		ontoid_cols.append(spdr.DB2IDN[ontodb_name])
		ontolb_cols.append(spdr.DB2ONTON[ontodb_name])
		ontoids, onto_labels = [[] for x in range(2)]
		for gse_id, ctrl_str, pert_str, annotres_list, common_annot in zip(sgn_df['geo_id'], sgn_df['ctrl_ids'], sgn_df['pert_ids'], annot_list, common_annot_list):
			gsm_list = ctrl_str.split('|') + pert_str.split('|')
			gse_doc, gsm_doc_list = gse_docs[gse_id][0], [gsm_docs[gsm_id][0] for gsm_id in gsm_list]
			annot_tkns, optional_tkns = [], []
			txt_field_maps = [0] + [1] * len(gsm_doc_list)
			# Only consider the summarized text fields of each GEO document
			txt_lengths, txt_weights, opt_txt_lengths, opt_txt_weights = [[] for x in range(4)] # Record the boundary of different text fields and their weights
			for annotres, geo_doc, tfmap in zip(annotres_list, [gse_doc] + gsm_doc_list, txt_field_maps):
				for annot_gp, txt, txtfield in zip(annotres, [geo_doc[txt_field] for txt_field in txt_fields[tfmap]], txt_fields[tfmap]):
					if (txt.isspace()): continue
					txt_length = 0
					init_tokens, locs = nlp.tokenize(txt, model='word', ret_loc=True)
					if (locs is None or len(locs) == 0): continue
					tokens, locs = nlp.del_punct(init_tokens, location=locs)
					if (locs is None or len(locs) == 0): continue
					start_loc, end_loc = zip(*locs)
					entities = annot_gp.setdefault(onto_lb, [])
					if (len(entities) > 0):
						# entities = _filt_ent(entities, onto_lb)
						# Only consider the top k most common annotation
						# for entity in [x for x in entities if x['word'] in common_annot[onto_lb]]:
						for entity in [x for x in entities]:
							annot_tkns.append(entity['word'])
							txt_length += (len(entity['word']) + 1)
							# left_tkn_id = bisect.bisect_left(list(start_loc), int(entity['offset'])) - 1
							# right_tkn_id = bisect.bisect_left(list(start_loc), int(entity['offset']) + len(entity['word']))
							# print left_tkn_id, right_tkn_id, entity['offset'], locs
							# Also consider a sliding window of the annotation terms to avoid inaccuracy of the annotation tool
							# annot_tkns.extend([tokens[max(0, left_tkn_id)], entity['word'], entity['word'], entity['word'], tokens[min(len(tokens) - 1, right_tkn_id)]])
						txt_lengths.append(txt_length)
						txt_weights.append(txtfield_importance[txtfield])
					if (onto_lb == 'PRGE' or onto_lb == 'DISO'):
						optional_tkns.append(txt)
						opt_txt_lengths.append(len(txt) + 1)
						opt_txt_weights.append(txtfield_importance[txtfield])
			annot_txt = ' '.join(annot_tkns)
			if (annot_txt.isspace()):
				annot_txt = ' '.join(optional_tkns).strip()
				txt_lengths, txt_weights = opt_txt_lengths, opt_txt_weights
			if (annot_txt.isspace()):
				ontoids.append('')
				onto_labels.append('')
				continue
			#　Map the annotations to the ontology
			onto_annotres = annot.annotonto(nlp.clean_txt(annot_txt), ontog, lang=onto_lang, idns=idns, prdns=prdns, idprds=idprds, dominant=True, lbprds=lbprds)
			# Complementary of the ontology mapping using biological entities identification method
			if (len(onto_annotres) == 0):
				annot_txt = ' '.join(optional_tkns).strip()
				txt_lengths, txt_weights = opt_txt_lengths, opt_txt_weights
				if (onto_lb == 'PRGE'):
					hgnc_cachef = os.path.join(spdr.HGNC_PATH, '%s_hgnc.npz' % gse_id)
					if (os.path.exists(hgnc_cachef)):
						annot_df = io.read_df(hgnc_cachef)
					else:
						annot_df = hgnc.symbol_checker(annot_txt, synonyms=True).dropna()
						io.write_df(annot_df, hgnc_cachef, compress=True)
					onto_annotres = zip(annot_df['HGNC ID'], annot_df['Approved symbol'], annot_df['Input'], map(func.find_substr(annot_txt), annot_df['Input']))
					ontoid_cols[-1] = 'hgnc_id'
				if (onto_lb == 'DISO'):
					dnorm_cachef = os.path.join(spdr.DNORM_PATH, '%s_dnorm.npz' % gse_id)
					if (os.path.exists(dnorm_cachef)):
						annot_df = io.read_df(dnorm_cachef)
					else:
						annot_df = dnorm.annot_dss(nlp.clean_txt(annot_txt))
						io.write_df(annot_df, dnorm_cachef, compress=True)
					locations = zip(annot_df['start'], annot_df['end'])
					onto_annotres = zip(annot_df['cid'], annot_df['concept'], [annot_txt[start:end] for start, end in locations], locations)
					ontoid_cols[-1] = 'dnorm_id'
			onto_annot_res = zip(*onto_annotres)
			if (len(onto_annotres) == 0 or len(onto_annot_res) == 0 or len(onto_annot_res[0]) == 0):
				ontoids.append('')
				onto_labels.append('')
				continue
			ids, labels, tokens, locs = onto_annot_res
			annotres_dict = dict(zip(ids, labels))
			txt_bndry = np.cumsum(txt_lengths)
			txt_normw = dstclc.normdist(np.array(txt_weights, dtype='float32'))
			token_weights = np.array([txt_normw[txt_bndry.searchsorted(loc[0], side='right')] if loc[0] < len(annot_txt) else 0 for loc in locs])
			id_array = np.array(ids)
			annot_count = [(k, token_weights[np.where(id_array == k)[0]].sum()) for k in set(ids)]
			# There might be several annotations with the same number
			# annot_count = collections.Counter(ids).most_common(10)

			# Find out the annotations with the most number, then sort them in alphabet order and pick the first one
			max_count = max(map(operator.itemgetter(1), annot_count))
			annot_ids = sorted([x[0] for x in func.sorted_tuples(annot_count, key_idx=1)[::-1] if x[1] == max_count])
			annot_length = [(x, len(str(x))) for x in annot_ids]
			annot_id = sorted([x[0] for x in func.sorted_tuples(annot_length, key_idx=1)])[0]
			ontoids.append(annot_id)
			onto_labels.append(annotres_dict[annot_id])
		annot_df = pd.DataFrame.from_items([(ontoid_cols[-1], ontoids), (ontolb_cols[-1], onto_labels)])
		annot_df.index = sgn_df.index
		postsgn_df = pd.concat([sgn_df, annot_df], axis=1, join_axes=[sgn_df.index], copy=False)
		# postsgn_df.index.name = 'id'
		io.write_df(postsgn_df, 'post_sgn_%s.npz' % lbid, with_idx=True)
		postsgn_df.to_excel('post_sgn_%s.xlsx' % lbid, encoding='utf8')
		postsgn_dfs.append(postsgn_df)
	## Signature Filtering and Cleaning
	for i, postsgn_df in enumerate(postsgn_dfs):
		lbid = i if (pid == -1) else pid
		io.inst_print('Cleaning the signatures for dataset %i ...' % lbid)
		ds_lb = gse_docs[postsgn_df['geo_id'][0]][1][0]
		_label = ds_lb.lower().replace(' ', '_')
		cln_sgn_df = postsgn_df.drop(postsgn_df.index[np.where(postsgn_df[ontolb_cols[i]] == '')[0]], axis=0)
		# Create cell type column
		cln_sgn_df['ANAT'] = [' '.join([mdf, x]) if x.startswith('cell') else x for mdf, x in zip(map(nlp.clean_txt, cln_sgn_df['mdf_ANAT'].fillna('')), map(nlp.clean_txt, cln_sgn_df['ANAT'].fillna('')))]
		cln_sgn_df.rename(columns={'ANAT': 'cell_type'}, inplace=True)
		cln_sgn_df.drop('mdf_ANAT', axis=1, inplace=True)
		# Delete other useless columns
		threshold = 0.5 * cln_sgn_df.shape[0]
		del_cols = [col for col in cln_sgn_df.columns if np.where(cln_sgn_df[col] != '')[0].shape[0] < threshold]
		cln_sgn_df.drop(del_cols, axis=1, inplace=True)
		io.write_df(cln_sgn_df, '%s.npz' % _label, with_idx=True)
		cln_sgn_df.to_excel('%s.xlsx' % _label, encoding='utf8')

		
def tuning(type='gse'):
	if (type == 'gse'):
		tuning_gse()
	elif (type == 'gsm'):
		tuning_gsm()
	else:
		tuning_gse()
		tuning_gsm()

	
def tuning_gse():
	from sklearn.model_selection import KFold
	if (opts.mltl):
		pid = -1
	else:
		pid = opts.pid
	print 'Process ID: %s' % pid
	
	## Load data for GSE
	gse_X, gse_Y = load_data(type='gse', pid=pid, fmt=opts.fmt, spfmt=opts.spfmt)
	# gse_X = gse_X.as_matrix().astype('float32')
	gse_X = gse_X.as_matrix()
	if (opts.mltl):
		gse_Y = gse_Y.as_matrix()
		if (len(gse_Y.shape) == 1 or gse_Y.shape[1] == 1):
			gse_Y = gse_Y.reshape((gse_Y.shape[0],))
	else:
		gse_Y = gse_Y.as_matrix().reshape((gse_Y.shape[0],))
	
	## Parameter tuning for GSE
	print 'Parameter tuning for GSE is starting...'
	ext_params = dict(cv=KFold(n_splits=opts.kfold, shuffle=True, random_state=0))
	params_generator = gen_mdl_params(opts.rdtune) if opts.dend is None else gen_nnmdl_params(gse_X.shape[1], gse_Y.shape[1] if len(gse_Y.shape) > 1 else 1, opts.rdtune)
	for mdl_name, mdl, params in params_generator:
		params.update(ext_params)
		print 'Tuning hyperparameters for %s' % mdl_name
		pt_result = txtclf.tune_param(mdl_name, mdl, gse_X, gse_Y, opts.rdtune, params, mltl=opts.mltl, avg='micro' if opts.avg == 'all' else opts.avg, n_jobs=opts.np)
		io.write_npz(dict(zip(['best_params', 'best_score', 'score_avg_cube', 'score_std_cube', 'dim_names', 'dim_vals'], pt_result)), 'gse_%sparam_tuning_for_%s_%s' % ('rd_' if opts.rdtune else '', mdl_name.replace(' ', '_').lower(), pid))
		
		
def tuning_gsm():
	from sklearn.model_selection import KFold
	if (opts.mltl):
		pid = -1
	else:
		pid = opts.pid
	print 'Process ID: %s' % pid
	
	## Load data for GSM
	gsm_Xs, gsm_Ys = load_data(type='gsm', pid=pid, fmt=opts.fmt, spfmt=opts.spfmt)
	if (opts.mltl):
		gsm_Ys = [Y.as_matrix() if len(Y.shape) > 1 and Y.shape[1] > 1 else Y.as_matrix().reshape((Y.shape[0],)) for Y in gsm_Ys]
	else:
		gsm_Ys = [Y.as_matrix().reshape((Y.shape[0],)) for Y in gsm_Ys]
	
	## Parameter tuning for GSM
	print 'Parameter tuning for GSM is starting...'
	for i, (gsm_X, gsm_y) in enumerate(zip(gsm_Xs, gsm_Ys)):
		ext_params = dict(cv=KFold(n_splits=opts.kfold, shuffle=True, random_state=0))
		params_generator = gen_mdl_params(opts.rdtune) if opts.dend is None else gen_nnmdl_params(gsm_X.shape[1], gsm_y.shape[1] if len(gsm_y.shape) > 1 else 1, opts.rdtune)
		for mdl_name, mdl, params in params_generator:
			params.update(ext_params)
			print 'Tuning hyperparameters for %s in label %i' % (mdl_name, i)
			pt_result = txtclf.tune_param(mdl_name, mdl, gsm_X, gsm_y, opts.rdtune, params, mltl=opts.mltl, avg='micro' if opts.avg == 'all' else opts.avg, n_jobs=opts.np)
			io.write_npz(dict(zip(['best_params', 'best_score', 'score_avg_cube', 'score_std_cube', 'dim_names', 'dim_vals'], pt_result)), 'gsm_%sparam_tuning_for_%s_%s' % ('rd_' if opts.rdtune else '', mdl_name.replace(' ', '_').lower(), i if opts.mltl else '_'.join([int(pid / 2), int(pid % 2)])))


# def demo():
	# import urllib
	# global cfgr
	# if not os.path.exists('data'):
		# os.makedirs('data')
	# urllib.urlretrieve ('https://data.mendeley.com/datasets/s9m6tzcv9d/2/files/239690a1-3c24-45b8-96cf-ff32183d140f/udt_exp_X.npz', 'data/X.npz')
	# urllib.urlretrieve ('https://data.mendeley.com/datasets/s9m6tzcv9d/2/files/5ad8019b-4853-462f-8a42-d5eee1a311e3/Y.npz', 'data/Y.npz')
	# hoc.DATA_PATH = 'data'
	# X, Y = load_data(type='gse', pid=-1, fmt=opts.fmt, spfmt=opts.spfmt)
	# def gse_model_iter(tuned, glb_filtnames, glb_clfnames):
		# yield 'UDT-RF', Pipeline([('clf', OneVsRestClassifier(RandomForestClassifier(max_features=0.7, min_samples_leaf=1, n_estimators=200, class_weight='balanced'), n_jobs=opts.np))])
	# txtclf.cross_validate(X, Y, gse_model_iter, model_param=dict(tuned=False, glb_filtnames=[], glb_clfnames=[]), avg='micro', kfold=5, cfg_param=cfgr('bionlp.txtclf', 'cross_validate'), global_param=dict(comb=True, pl_names=[], pl_set=set([])), lbid=-1)


def main():
	if (opts.tune):
		tuning(opts.ftype)
		return
	if (opts.method == 'demo'):
		# demo()
		return
	elif (opts.method == 'gse_clf'):
		gse_clf()
		return
	elif (opts.method == 'gsm_clf'):
		gsm_clf()
		return
	elif (opts.method == 'gsm_clt'):
		gsm_clt()
		return
	elif (opts.method == 'gen_sgn'):
		gen_sgn()
		return
	all()


if __name__ == '__main__':
	# Parse commandline arguments
	op = OptionParser()
	op.add_option('-k', '--kfold', default=10, action='store', type='int', dest='kfold', help='indicate the K fold cross validation')
	op.add_option('-p', '--pid', default=-1, action='store', type='int', dest='pid', help='indicate the process ID')
	op.add_option('-n', '--np', default=-1, action='store', type='int', dest='np', help='indicate the number of processes used for calculation')
	op.add_option('-o', '--omp', action='store_true', dest='omp', default=False, help='use openmp multi-thread')
	op.add_option('-f', '--fmt', default='npz', help='data stored format: csv or npz [default: %default]')
	op.add_option('-s', '--spfmt', default='csr', help='sparse data stored format: csr or csc [default: %default]')
	op.add_option('-t', '--tune', action='store_true', dest='tune', default=False, help='firstly tune the hyperparameters')
	op.add_option('-r', '--rdtune', action='store_true', dest='rdtune', default=False, help='randomly tune the hyperparameters')
	op.add_option('-b', '--best', action='store_true', dest='best', default=False, help='use the tuned hyperparameters')
	op.add_option('-c', '--comb', action='store_true', dest='comb', default=False, help='run the combined methods')
	op.add_option('-l', '--mltl', action='store_true', dest='mltl', default=False, help='use multilabel strategy')
	op.add_option('-a', '--avg', default='micro', help='averaging strategy for performance metrics: micro or macro [default: %default]')
	op.add_option('-d', '--dend', dest='dend', help='deep learning backend: tf or th')
	op.add_option('-g', '--gpunum', default=1, action='store', type='int', dest='gpunum', help='indicate the gpu device number')
	op.add_option('-q', '--gpuq', dest='gpuq', help='prefered gpu device queue')
	op.add_option('-z', '--bsize', default=32, action='store', type='int', dest='bsize', help='indicate the batch size used in deep learning')
	op.add_option('-e', '--ftype', default='', type='str', dest='ftype', help='the document type used to generate data')
	op.add_option('-u', '--fuzzy', action='store_true', dest='fuzzy', default=False, help='use fuzzy clustering')
	op.add_option('-j', '--thrshd', default='mean', type='str', dest='thrshd', help='threshold value')
	op.add_option('-w', '--cache', default='.cache', help='the location of cache files')
	op.add_option('-i', '--input', default='gsc', help='input source: gsc or geo [default: %default]')
	op.add_option('-m', '--method', help='main method to run')
	op.add_option('-v', '--verbose', action='store_true', dest='verbose', default=False, help='display detailed information')

	(opts, args) = op.parse_args()
	if len(args) > 0:
		op.print_help()
		op.error('Please input options instead of arguments.')
		exit(1)
	# Option Correcting
	if (opts.spfmt.lower() in ['', ' ', 'none']): opts.spfmt = None
	# Logging setting
	logging.basicConfig(level=logging.INFO if opts.verbose else logging.ERROR, format='%(asctime)s %(levelname)s %(message)s')
	# Data Source
	spdr = SPDR_MAP[opts.input]
	# Parse config file
	if (os.path.exists(CONFIG_FILE)):
		cfgr = io.cfg_reader(CONFIG_FILE)
		spdr_cfg = cfgr('bionlp.spider.%s' % opts.input, 'init')
		if (len(spdr_cfg) > 0):
			if (spdr_cfg['DATA_PATH'] is not None and os.path.exists(spdr_cfg['DATA_PATH'])):
				spdr.DATA_PATH = spdr_cfg['DATA_PATH']
			if (spdr_cfg['GEO_PATH'] is not None and os.path.exists(spdr_cfg['GEO_PATH'])):
				spdr.GEO_PATH = spdr_cfg['GEO_PATH']
			if (spdr_cfg['ONTO_PATH'] is not None and os.path.exists(spdr_cfg['ONTO_PATH'])):
				spdr.ONTO_PATH = spdr_cfg['ONTO_PATH']
			if (spdr_cfg['HGNC_PATH'] is not None and os.path.exists(spdr_cfg['HGNC_PATH'])):
				spdr.HGNC_PATH = spdr_cfg['HGNC_PATH']
			if (spdr_cfg['DNORM_PATH'] is not None and os.path.exists(spdr_cfg['DNORM_PATH'])):
				spdr.DNORM_PATH = spdr_cfg['DNORM_PATH']
			if (spdr_cfg['RXNAV_PATH'] is not None and os.path.exists(spdr_cfg['RXNAV_PATH'])):
				spdr.RXNAV_PATH = spdr_cfg['RXNAV_PATH']
		hgnc_cfg = cfgr('bionlp.spider.hgnc', 'init')	
		if (len(hgnc_cfg) > 0):
			if (hgnc_cfg['MAX_TRIAL'] is not None and hgnc_cfg['MAX_TRIAL'] > 0):
				hgnc.MAX_TRIAL = hgnc_cfg['MAX_TRIAL']
		plot_cfg = cfgr('bionlp.util.plot', 'init')
		plot_common = cfgr('bionlp.util.plot', 'common')
		txtclf.init(plot_cfg=plot_cfg, plot_common=plot_common)
		txtclt.init(plot_cfg=plot_cfg, plot_common=plot_common)
		
	if (opts.dend is not None):
		if (opts.dend == 'th' and opts.gpunum == 0 and opts.omp):
			from multiprocessing import cpu_count
			os.environ['OMP_NUM_THREADS'] = '4' if opts.tune else str(int(1.5 * cpu_count() / opts.np))
		if (opts.gpuq is not None):
			gpuq = [int(x) for x in opts.gpuq.split(',')]
			dev_id = gpuq[opts.pid % len(gpuq)]
		else:
			dev_id = opts.pid % opts.gpunum if opts.gpunum > 0 else 0
		kerasext.init(dev_id, opts.gpunum, opts.dend, opts.np, opts.omp)
		
	annot.init()
	
	main()