#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2016 by Caspar. All rights reserved.
# File Name: gsx_helper.py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2016-04-10 14:55:16
###########################################################################
#

import os
import sys
import ast
import json
import glob
import logging
import operator
import itertools
import cStringIO
import collections
from optparse import OptionParser

import numpy as np
import scipy as sp
import pandas as pd

from bionlp.spider import annot, sparql
from bionlp.util import fs, io, func, plot, ontology
from bionlp import dstclc, nlp, metric
# from bionlp import txtclt

import gsc

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
PAR_DIR = os.path.abspath(os.path.join(FILE_DIR, os.path.pardir))
CONFIG_FILE = os.path.join(PAR_DIR, 'etc', 'config.yaml')
SC=';;'

opts, args = {}, []
common_cfg = {}


def init_plot(plot_cfg={}, plot_common={}):
	if (len(plot_cfg) > 0 and plot_cfg['MON'] is not None):
		plot.MON = plot_cfg['MON']
	global common_cfg
	if (len(plot_common) > 0):
		common_cfg = plot_common


def nt2db():
	import bionlp.util.ontology as ontology
	kwargs = {} if opts.cfg is None else ast.literal_eval(opts.cfg)
	ontology.files2db(opts.loc, fmt='nt', saved_path=os.path.splitext(opts.loc)[0] if opts.output is None else opts.output, **kwargs)


def xml2db():
	import bionlp.util.ontology as ontology
	kwargs = {} if opts.cfg is None else ast.literal_eval(opts.cfg)
	ontology.files2db(opts.loc, saved_path=os.path.splitext(opts.loc)[0] if opts.output is None else opts.output, **kwargs)
	
	
def xml2dbs():
	import bionlp.util.ontology as ontology
	kwargs = {} if opts.cfg is None else ast.literal_eval(opts.cfg)
	ontology.files2dbs(opts.loc, saved_path=os.path.splitext(opts.loc)[0] if opts.output is None else opts.output, merge=True, merged_dbname=os.path.splitext(os.path.basename(opts.loc))[0], cache=True, **kwargs)
	# ontology.files2dbs([os.path.join(opts.loc, x) for x in ['dron-full.owl', 'dron-ingredient.owl', 'dron-hand.owl', 'dron-ndc.owl', 'dron-chebi.owl', 'dron-pro.owl', 'dron-upper.owl', 'dron-rxnorm.owl']], saved_path=os.path.splitext(opts.loc)[0] if opts.output is None else opts.output, merge=True, merged_dbname=os.path.splitext(os.path.basename(opts.loc))[0], cache=True, **kwargs)
	
	
def dbcsv2nt():
	excel_df = pd.read_csv(opts.loc, encoding='utf-8').fillna('')
	dbid_tmplt, name_tmplt = u'<http://www.drugbank.ca/drugbank-id/%s>', u'"%s"@en'
	vcb_cmname, vcb_term = u'<http://www.drugbank.ca/vocab#common-name>', u'<http://www.drugbank.ca/vocab#term>'
	dbids, cmnames, synm_strs = excel_df['DrugBank ID'].tolist(), excel_df['Common name'].tolist(), excel_df['Synonyms'].tolist()
	triples = [(dbid_tmplt % dbid, vcb_cmname, name_tmplt % cmname) for dbid, cmname in zip(dbids, cmnames)]
	synonyms = [list(set([y.strip() for y in unicode(x).split('|')])) for x in synm_strs]
	triples.extend([(dbid_tmplt % dbid, vcb_term, name_tmplt % synm.replace('"', '\\"')) for dbid, synms in zip(dbids, synonyms) for synm in synms if synm != u''])
	triples = dict.fromkeys(triples).keys()
	fpath = opts.output if opts.output is not None else os.path.splitext(opts.loc)[0] + '.nt'
	fs.write_file(' .\n'.join([' '.join(x) for x in triples]) + ' .', fpath, code='utf-8')
	
	
def dgcsv2nt():
	excel_df = pd.read_csv(opts.loc, encoding='utf-8').fillna('')
	gene_tmplt, intype_tmplt, drug_tmplt = u'<http://dgidb.genome.wustl.edu/gene/%s>', u'<http://dgidb.genome.wustl.edu/vocab#%s>', u'<http://dgidb.genome.wustl.edu/drug/%s>'
	# vcb_interact = u'<http://dgidb.genome.wustl.edu/vocab#interact>'
	gene_name, intype, drug_name = excel_df['gene_long_name'].tolist(), excel_df['interaction_types'].tolist(), excel_df['drug_primary_name'].tolist()
	triples = [(gene_tmplt % gn.replace(' ', '_'), intype_tmplt % it.replace('n/a', 'unknown').replace(' ', '_'), drug_tmplt % dn.replace(' ', '_')) for gn, it, dn in zip(gene_name, intype, drug_name)]
	triples = dict.fromkeys(triples).keys()
	fpath = opts.output if opts.output is not None else os.path.splitext(opts.loc)[0] + '.nt'
	fs.write_file(' .\n'.join([' '.join(x) for x in triples]) + ' .', fpath, code='utf-8')


def download():
	excel_df = pd.read_csv(opts.loc)
	par_dir, basename = os.path.abspath(os.path.join(opts.loc, os.path.pardir)), os.path.splitext(os.path.basename(opts.loc))[0]
	if (opts.unified):
		saved_path = os.path.join(par_dir, opts.type) if opts.output is None else os.path.join(opts.output, opts.type)
	else:
		saved_path = os.path.join(par_dir, basename) if opts.output is None else os.path.join(opts.output, basename)
	kwargs = {} if opts.cfg is None else ast.literal_eval(opts.cfg)
	if (opts.type == 'soft'):
		from bionlp.spider import geo
		for geo_data in geo.fetch_geo(list(excel_df['geo_id']), saved_path=saved_path, skip_cached=True, **kwargs):
			del geo_data
	else:
		from bionlp.spider import geoxml as geo
		geo_strs = geo.fetch_geo(list(excel_df['geo_id']), saved_path=saved_path, **kwargs)
		geo_strios = [cStringIO.StringIO(str) for str in geo_strs]
		geo_docs = geo.parse_geos(geo_strios)
		samples = [sample for geo_doc in geo_docs for sample in geo_doc['samples']]
		sample_strs = geo.fetch_geo(samples, saved_path=os.path.join(saved_path, 'samples'), **kwargs)

def slct_featset():
	kwargs = {} if opts.cfg is None else ast.literal_eval(opts.cfg)
	X = io.read_df(opts.loc, with_idx=True, sparse_fmt=opts.spfmt)
	featset = kwargs.setdefault('featset', ','.join(X.columns)).split(',')
	featset = [col for col in X.columns for fs in featset if col.startswith(fs)]
	new_X = X[featset]
	io.write_df(new_X, os.path.abspath(os.path.join(opts.loc, os.path.pardir, 'new_%s.npz' % os.path.splitext(os.path.basename(opts.loc))[0])), with_idx=True, sparse_fmt=opts.spfmt, compress=True)


def gsm2gse():
	geo_docs = gsc.get_geos(type='gse', fmt='xml')
	gsm_Xs, _, _ = gsc.get_mltl_npz(type='gsm', lbs=['0_0'], mltlx=False, spfmt=opts.spfmt)
	gsm_X = gsm_Xs[0]
	gse_lbs = ['' for i in range(gsm_X.shape[0])]
	for geo_id, geo_data in geo_docs.iteritems():
		for sample in geo_data[0]['samples']:
			gse_lbs[gsm_X.index.get_loc(sample)] = geo_id
	y = pd.DataFrame(gse_lbs, index=gsm_X.index, columns=['gse_id'])
	io.write_df(y, os.path.join(gsc.DATA_PATH, 'gsm2gse.npz'), with_idx=True)

	
def _cltpred2df(Xs, Ys, labels, lbids, predf_patn):
	pred_dfs = []
	for X, Y, z, i in zip(Xs, Ys, labels, lbids):
		predf = predf_patn.replace('#LB', str(i))
		predf_list = []
		for fpath in fs.listf(opts.loc, pattern=predf, full_path=True):
			# Read the cluster prediction file
			try:
				pred = io.read_npz(fpath)['pred_lb']
			except:
				print 'Unable to read the prediction file: %s !' % fpath
				continue
			is_fuzzy = len(pred.shape) > 1 and pred.shape[1] > 1
			pred_df = pd.DataFrame(pred, index=z.index, columns=['clt_%i' % x for x in range(pred.shape[1])] if is_fuzzy else ['cltlb'])
			fpath = fpath.replace('pred', 'lb')
			io.write_df(pred_df, fpath, with_idx=True, sparse_fmt=opts.spfmt)
			predf_list.append((os.path.splitext(os.path.basename(fpath))[0], pred_df))
		pred_dfs.append(predf_list)
	return pred_dfs
	
	
def cltpred2df():
	kwargs = {} if opts.cfg is None else ast.literal_eval(opts.cfg)
	predf_patn = kwargs.setdefault('predf', 'clt_pred_.*_#LB')
	Xs, Ys, labels = gsc.get_data(None, type='gsm', from_file=True, fmt=opts.fmt, spfmt=opts.spfmt)
	_cltpred2df(Xs, Ys, labels, range(len(Xs)), predf_patn)
	

# For hard clustering method
def gen_gsmclt_pair():
	kwargs = {} if opts.cfg is None else ast.literal_eval(opts.cfg)
	predf_patn, method, threshold = kwargs.setdefault('predf', 'clt_pred_#MDL_#LB'), kwargs.setdefault('filtclt', 'std'), kwargs.setdefault('threshold', 0.5)
	Xs, Ys, labels = gsc.get_data(None, type='gsm', from_file=True, fmt=opts.fmt, spfmt=opts.spfmt)
	# Read the GSM-GSE association file
	m2e = io.read_df(os.path.join(gsc.DATA_PATH, 'gsm2gse.npz'), with_idx=True)
	gsmclt_pairs = []
	for i, (X, Y, label) in enumerate(zip(Xs, Ys, labels)):
		predf = predf_patn.replace('#LB', str(i))
		# Read the cluster prediction file
		pred = io.read_npz(predf)['pred_lb']
		# Filter the clusters
		filt_pred = txtclt.filt_clt(X.as_matrix(), pred, method=method, threshold=threshold)
		clts = set(filt_pred)
		# clts = set(label['index'].tolist())
		clts.discard(-1)
		clt_glb = [0 for j in range(len(clts))]
		clt_groups = {}
		for j, clt_lb in enumerate(clts):
			clt_idx = np.where(filt_pred==clt_lb)[0]
			# Determine whether the cluster is a control or perturbation group
			votes = Y.as_matrix()[clt_idx,:].sum(axis=0)
			clt_glb[j] = votes.argmax()
			# Make sure that all the samples in every cluster are within the same study
			gse_dict = dict([(m2e.ix[X.index[x], 'gse_id'], x) for x in clt_idx])
			if (len(gse_dict) == 1):
				clt_groups.setdefault(gse_dict.keys()[0], [[] for x in range(Y.shape[1])])[clt_glb[j]].append('|'.join(X.index[clt_idx]))
		# Enumerate every pair of control and perturbation groups within each study
		gsmclt_pair = []
		for gse_id, gsm_group in clt_groups.iteritems():
			gsm_group = [x for x in gsm_group if len(x) > 0]
			if len(gsm_group) < 2: continue
			for group_pair in itertools.product(gsm_group[0], gsm_group[1]):
				gsmclt_pair.append({k:v for k, v in zip(['geo_id'] + [y + '_ids' for y in Y.columns], [gse_id] + list(group_pair))})
		gp_df = pd.DataFrame(gsmclt_pair)
		io.write_df(gp_df, 'gsmclt_pair_%i.npz' % i, with_idx=True)
		gp_df.to_excel('gsmclt_pair_%i.xlsx' % i)
		gsmclt_pairs.append(gsmclt_pair)
	return gsmclt_pairs


def _gsmclt_pair(X, Y, z, gsm2gse, lbid, thrshd=0.5, cache_path='.cache', fname=None):
	fname = 'gsmclt_pair_%s' % lbid if (fname is None) else fname
	cachef = os.path.join(cache_path, fname + '.npz')
	if (os.path.exists(cachef)):
		return io.read_df(cachef, with_idx=True)
	is_fuzzy = len(z.shape) > 1
	# Construct clusters
	if (is_fuzzy):
		clts = [np.where(z.iloc[:,j] >= (thrshd if type(thrshd) is float else getattr(z.iloc[:,j], thrshd)()))[0] for j in range(z.shape[1])]
	else:
		clts = [np.where(z == j)[0] for j in np.unique(z)]
	clts = [clt for clt in clts if clt.shape[0] > 0]
	if (len(clts) == 0):
		print 'No cluster found in Data Set %i! Please input a smaller threshold!' % i
		return None
	clusters = dict(zip([SC.join(clt.astype('str')) for clt in clts], [z.index[clt].tolist() for clt in clts])).values()
	print 'Generated Cluster Number:%s' % len(clusters)
	# print clusters
	# Refine clusters
	ctrl_pert_clts = {}
	for clt in clusters:
		# Must be within the same GEO document
		gse_ids = gsm2gse.ix[clt]
		for gse_id in gse_ids.gse_id.value_counts().index:
			common_gseids = gse_ids[gse_ids == gse_id]
			if (common_gseids.shape[0] < gse_ids.shape[0]):
				print 'Impure cluster with %.2f%% extra GEO studies:' % (1 - 1.0 * common_gseids.shape[0] / gse_ids.shape[0])
				break
			cln_clt = common_gseids.index.tolist()
			# Control group or perturbation group
			clt_y = Y.ix[cln_clt]
			clty_sum = clt_y.sum(axis=1)
			clt_y = clt_y.iloc[np.where(np.logical_and(clty_sum != 0, clty_sum != 2))[0]]
			pure_clty = clt_y.iloc[np.where(clt_y.sum(axis=1) == 1)[0]]
			# print clt_y.shape, pure_clty.shape
			if (pure_clty.empty or pure_clty.shape[0] == 0): continue
			if (pure_clty.shape[0] < clt_y.shape[0]):
				# print clt_y.pert
				# print clt_y.ctrl
				print 'Impure cluster with %.2f%% ctrls, %.2f%% perts, and %.2f%% mixtures:' % (1.0 * np.where(clt_y.ctrl == 1)[0].shape[0] / clt_y.shape[0], 1.0 * np.where(clt_y.pert == 1)[0].shape[0] / clt_y.shape[0], 1.0 * np.where(clt_y.sum(axis=1) == 2)[0].shape[0] / clt_y.shape[0])
			cln_clt = pure_clty.index.tolist()
			ctrl_pert_clts.setdefault(gse_id, [[], []])[Y.ix[cln_clt[0]].pert].append(cln_clt)
	# print 'Refined Cluster Number:%s' % dict([(gse_id, [len(clts[0]), len(clts[1])]) for gse_id, clts in ctrl_pert_clts.iteritems()])
	# print ctrl_pert_clts
	# Enumerate every pair of control and perturbation group
	geo_ids, ctrl_ids, pert_ids, tissues, organisms, platforms = [[] for x in range(6)]
	for gse_id, cpclts in ctrl_pert_clts.iteritems():
		# print gse_id, len(cpclts[0]), len(cpclts[1])
		for ctrl, pert in itertools.product(cpclts[0], cpclts[1]):
			geo_ids.append(gse_id)
			ctrl_ids.append('|'.join(sorted(ctrl)))
			pert_ids.append('|'.join(sorted(pert)))
	# Write to file
	pair_df = pd.DataFrame.from_items([('geo_ids', geo_ids), ('ctrl_ids', ctrl_ids), ('pert_ids', pert_ids)])
	print 'Generated %i signatures' % pair_df.shape[0]
	io.write_df(pair_df, cachef, with_idx=True)
	pair_df.to_excel(fname + '.xlsx', encoding='utf8')
	return pair_df
	

# For soft and hard clustering method
def gen_gsmfzclt_pair():
	kwargs = {} if opts.cfg is None else ast.literal_eval(opts.cfg)
	predf_patn, threshold = kwargs.setdefault('predf', 'clt_pred_#MDL_#LB'), kwargs.setdefault('threshold', 0.5)
	if (opts.pid == -1):
		Xs, Ys, labels = gsc.get_data(None, type='gsm', from_file=True, fmt=opts.fmt, spfmt=opts.spfmt)
	else:
		Xs, Ys, labels = gsc.get_mltl_npz(type='gsm', lbs=[opts.pid], spfmt=opts.spfmt)
	m2e = io.read_df(os.path.join(gsc.DATA_PATH, 'gsm2gse.npz'), with_idx=True)
	lbids = range(len(Xs)) if (opts.pid == -1) else [opts.pid]
	pred_dfs = _cltpred2df(Xs, Ys, labels, lbids, predf_patn)
	for i, (X, Y, predfs) in enumerate(zip(Xs, Ys, pred_dfs)):
		lbid = i if (opts.pid == -1) else opts.pid
		for fname, pred_df in predfs:
			# Generate GSM cluster pairs
			pair_df = _gsmclt_pair(X, Y, pred_df, m2e, lbid, thrshd=threshold, cache_path=opts.cache)
			
			
def _annot_sgn(geo_id, geo_doc, txt_fields, cache_path='.cache'):
	fs.mkdir(cache_path)
	# Obtain the annotation results that correspond to each field
	geo_cncpt_list, annot_res = [[] for i in range(2)]
	doc_cachef = os.path.join(cache_path, '%s.pkl' % geo_id)
	if (os.path.exists(doc_cachef)):
		cached_res = io.read_obj(doc_cachef)
		if (len(cached_res) != 0): 
			return cached_res
	for field in txt_fields:
		# Retrieve the annotated data in dict format
		field_cachef = os.path.join(cache_path, '%s_%s.json' % (geo_id, field))
		if (os.path.exists(field_cachef)):
			json_str = '\n'.join(fs.read_file(field_cachef, code='utf8'))
			ret_dict = json.loads(json_str)
		else:
			try:
				ret_dict = annot.annotxt(geo_doc[field], retype='dict')
			except Exception as e:
				print 'Unable to annotate %s in the %s field!' % (geo_id, field)
				print e
				continue
			json_str = json.dumps(ret_dict)
			fs.write_file(json_str, field_cachef, code='utf8')
		# Transform the data into groups
		ret_dict['text'] = geo_doc[field]
		annot_res.append(annot.annotxt(ret_dict, retype='group', with_mdf=True if (field=='source') else False))
	io.write_obj(annot_res, doc_cachef)
	# Return the groups for each text fields
	return annot_res
	
	
def annot_sgn():
	cache_path = os.path.join(gsc.GEO_PATH, 'annot')
	txt_field_set = [['title', 'summary', 'keywords'], ['title', 'description', 'source', 'trait']]
	geo_doc_set = gsc.get_geos(type='gse', fmt='xml'), gsc.get_geos(type='gsm', fmt='xml')
	annot_res_set = [{} for i in range(2)]
	for geo_docs, annot_res, txt_field in zip(geo_doc_set, annot_res_set, txt_field_set):
		annot_res.update(dict([(geo_id, _annot_sgn(geo_id, geo_data[0], txt_field, cache_path=cache_path)) for geo_id, geo_data in geo_docs.iteritems()]))
	io.write_obj(annot_res_set[0], os.path.join(cache_path, 'gse_annot.pkl'))
	io.write_obj(annot_res_set[1], os.path.join(cache_path, 'gsm_annot.pkl'))
	
	
def sgn2ge():
	input_ext = os.path.splitext(opts.loc)[1]
	if (input_ext == '.xlsx' or input_ext == '.xls'):
		excel_df = pd.read_excel(opts.loc)
	elif (input_ext == '.csv'):
		excel_df = pd.read_csv(opts.loc)
	elif (input_ext == '.npz'):
		excel_df = io.read_df(opts.loc)
	par_dir, basename = os.path.abspath(os.path.join(opts.loc, os.path.pardir)), os.path.splitext(os.path.basename(opts.loc))[0]
	sample_path = os.path.join(gsc.GEO_PATH, opts.type, basename, 'samples')
	saved_path = os.path.join(par_dir, 'gedata', basename) if opts.output is None else os.path.join(opts.output, 'gedata', basename)
	# Find the control group and perturbation group for every signature
	for i, (ctrl_str, pert_str) in enumerate(zip(excel_df['ctrl_ids'], excel_df['pert_ids'])):
		ctrl_file, pert_file = os.path.join(saved_path, 'ctrl_%i.npz' % i), os.path.join(saved_path, 'pert_%i.npz' % i)
		if (os.path.exists(ctrl_file) and os.path.exists(pert_file)): continue
		ctrl_ids, pert_ids = ctrl_str.split('|'), pert_str.split('|')
		# Obtain the geo files for each sample
		ctrl_geo_docs, pert_geo_docs = geo.parse_geos([os.path.join(sample_path, '.'.join([ctrl_id, opts.type])) for ctrl_id in ctrl_ids], view='full', type='gsm', fmt=opts.type), geo.parse_geos([os.path.join(sample_path, '.'.join([pert_id, opts.type])) for pert_id in pert_ids], view='full', type='gsm', fmt=opts.type)
		# Extract the gene expression data from the geo files for each sample, and combine the data within the same group
		ctrl_ge_dfs, pert_ge_dfs = [geo_doc['data']['VALUE'] for geo_doc in ctrl_geo_docs], [geo_doc['data']['VALUE'] for geo_doc in pert_geo_docs]
		ctrl_df, pert_df = pd.concat(ctrl_ge_dfs, axis=1, join='inner'), pd.concat(pert_ge_dfs, axis=1, join='inner')
		io.write_df(ctrl_df, ctrl_file, with_col=False, with_idx=True)
		io.write_df(pert_df, pert_file, with_col=False, with_idx=True)
		
		
def sgn2deg():
	from bioinfo.ext.chdir import chdir as deg_chdir
	input_ext = os.path.splitext(opts.loc)[1]
	if (input_ext == '.xlsx' or input_ext == '.xls'):
		excel_df = pd.read_excel(opts.loc)
	elif (input_ext == '.csv'):
		excel_df = pd.read_csv(opts.loc)
	elif (input_ext == '.npz'):
		excel_df = io.read_df(opts.loc)
	else:
		print 'Unsupported input file extension %s, please use csv or npz file!' % input_ext
		exit(1)
	kwargs = {} if opts.cfg is None else ast.literal_eval(opts.cfg)
	par_dir, basename = os.path.abspath(os.path.join(opts.loc, os.path.pardir)), os.path.splitext(os.path.basename(opts.loc))[0]
	ge_path = os.path.join(kwargs.setdefault('ge_dir', gsc.GEO_PATH), 'gedata', basename)
	saved_path = os.path.join(par_dir, 'deg', basename) if opts.output is None else os.path.join(opts.output, 'deg', basename)
	# Extract the control group and perturbation group of each gene expression signature
	ids = excel_df['id'] if hasattr(excel_df, 'id') else excel_df.index
	for i, id in enumerate(ids):
		deg_file = os.path.join(saved_path, 'deg_%i.npz' % i)
		if (os.path.exists(deg_file)): continue
		ctrl_file, pert_file = os.path.join(ge_path, 'ctrl_%i.npz' % i), os.path.join(ge_path, 'pert_%i.npz' % i)
		ctrl_df, pert_df = io.read_df(ctrl_file, with_col=False, with_idx=True), io.read_df(pert_file, with_col=False, with_idx=True)
		# Find the gene sets that are both in control group and perturbation group
		join_df = pd.concat([ctrl_df, pert_df], axis=1, join='inner')
		print 'Start CD algorithm for No.%i %s...%s, %s, %s' % (i, id, ctrl_df.shape, pert_df.shape, join_df.shape)
		# Calculate the differentially expressed genes vector
		deg_vec = deg_chdir(join_df.iloc[:,:ctrl_df.shape[1]].as_matrix(), join_df.iloc[:,ctrl_df.shape[1]:].as_matrix(), 1).reshape((-1,))
		deg_df = pd.DataFrame(deg_vec, index=join_df.index, columns=[id])
		io.write_df(deg_df, deg_file, with_idx=True)
		

def _ji(a, b):
	if (len(a) == 0 and len(b) == 0): return 0
	return 1.0 * len(a & b) / (len(a) + len(b) - len(a & b))
	
def _sji(a, b):
	return (_ji(a[0], b[0]) + _ji(a[1], b[1]) - _ji(a[0], b[1]) - _ji(a[1], b[0])) / 2
	
def _sjiv(X, Y):
	def _ji(a, b):
		if (len(a) == 0 and len(b) == 0): return 0
		return 1.0 * len(a & b) / (len(a) + len(b) - len(a & b))
	def _sji(a, b):
		return (_ji(a[0], b[0]) + _ji(a[1], b[1]) - _ji(a[0], b[1]) - _ji(a[1], b[0])) / 2
	import numpy as np
	import itertools
	shape = len(X), len(Y)
	simmt = np.ones(shape)
	for i, j in itertools.product(range(shape[0]), range(shape[1])):
		simmt[i, j] = _sji(X[i], Y[j])
	return simmt
	
def _sjim(X, Y):
	Y_T = Y.T
	interaction = X.dot(Y_T) # XY' shape of (2m, 2n)
	# union = X.sum(axis=1).reshape((-1, 1)).repeat(Y.shape[0], axis=1) + Y_T.sum(axis=0).reshape((1, -1)).repeat(X.shape[0], axis=0) - interaction
	union = X.dot(np.ones((X.shape[1]), dtype='int8')).reshape((-1, 1)).repeat(Y.shape[0], axis=1) + np.ones((Y_T.shape[0]), dtype='int8').dot(Y_T).reshape((1, -1)).repeat(X.shape[0], axis=0) - interaction # XI+IY'-XY', dot can be parallelized but not sum
	r = 1.0 * interaction / union
	r = r.reshape((r.shape[0]/2, 2, r.shape[1]/2, 2))
	s = np.tensordot(np.tensordot(np.array([[1,-1]]), r, axes=[[1],[1]]).reshape((r.shape[0],)+r.shape[2:]), np.array([[1],[-1]]), axes=[[2],[0]]).reshape((X.shape[0]/2, Y.shape[0]/2)) # sum reduction
	return s
	
def _sjic(X, Y):
	Y_T = Y.T
	interaction = np.tensordot(X, Y, axes=[[-1],[-1]]).transpose(range(len(X.shape)-1)+range(len(X.shape)-1, len(X.shape)+len(Y.shape)-2)[::-1]) # XY' shape of (m, 2, 2, n)
	union = np.tensordot(X, np.ones((X.shape[-1], X.shape[-2])), axes=1).reshape(X.shape[:-1] + (X.shape[-2], 1)).repeat(Y.shape[0], axis=-1) + np.tensordot(Y, np.ones((Y.shape[-1], Y.shape[-2])), axes=1).reshape(Y.shape[:-1] + (Y.shape[-2], 1)).repeat(X.shape[0], axis=-1).T - interaction # XI+IY'-XY'
	r = 1.0 * interaction / union
	s = np.tensordot(np.tensordot(np.array([[1,-1]]), r, axes=[[1],[1]]).reshape((r.shape[0],)+r.shape[2:]), np.array([[1],[-1]]), axes=[[1],[0]]).reshape((X.shape[0], Y.shape[0])) # sum reduction
	return s
		
def deg2simmt():
	# from sklearn.externals.joblib import Parallel, delayed
	from sklearn.metrics import pairwise
	from sklearn.preprocessing import MultiLabelBinarizer
	locs = opts.loc.split(',')
	kwargs = {} if opts.cfg is None else ast.literal_eval(opts.cfg)
	basenames, input_exts = zip(*[os.path.splitext(os.path.basename(loc)) for loc in locs])
	if (input_exts[0] == '.csv'):
		excel_dfs = [pd.read_csv(loc) for loc in locs]
	elif (input_exts[0] == '.npz'):
		excel_dfs = [io.read_df(loc) for loc in locs]
	ge_dir = kwargs.setdefault('ge_dir', gsc.GEO_PATH)
	simmt_file = os.path.join(ge_dir, 'simmt.npz')
	idx_cols = kwargs.setdefault('idx_cols', 'hs_gene_symbol,drug_name,disease_name').split(',')
	cache_f = os.path.join(opts.cache, 'udgene.pkl')
	# Read the data

	if (os.path.exists(cache_f)):
		print 'Reading cache...'
		sys.stdout.flush()
		ids, id_bndry = io.read_obj(cache_f)
		if (not os.path.exists(simmt_file)):
			udgene_spmt = io.read_spmt(os.path.splitext(cache_f)[0]+'.npz')
	else:
		print 'Preparing data...'
		sys.stdout.flush()
		ids, id_bndry, udgene = [[] for i in range(3)]
		# Read all the differentially expressed genes vector of each collection
		for basename, excel_df, idx_col in zip(basenames, excel_dfs, idx_cols):
			deg_path = os.path.join(ge_dir, 'deg', basename)
			sgn_ids = excel_df['id'].tolist()
			# sgn_ids = excel_df[idx_col].tolist() # customized identity for each signature
			ids.extend(sgn_ids)
			id_bndry.append(len(sgn_ids)) # append number of signatures to form the boundaries
			for i in xrange(len(sgn_ids)):
				deg_df = io.read_df(os.path.join(deg_path, 'deg_%i.npz' % i), with_idx=True)
				udgene.append((set(deg_df.index[np.where(deg_df.iloc[:,0] > 0)[0]]), set(deg_df.index[np.where(deg_df.iloc[:,0] < 0)[0]])))
		unrolled_udgene = func.flatten_list(udgene)
		# Transform the up-down regulate gene expression data into binary matrix
		mlb = MultiLabelBinarizer(sparse_output=True)
		udgene_spmt = mlb.fit_transform(unrolled_udgene).astype('int8')
		io.write_spmt(udgene_spmt, os.path.splitext(cache_f)[0]+'.npz', sparse_fmt='csr', compress=True)
		id_bndry = np.cumsum([0] + id_bndry).tolist()
		io.write_obj([ids, id_bndry], cache_f)
	if (os.path.exists(simmt_file)):
		print 'Reading similarity matrix...'
		sys.stdout.flush()
		simmt = io.read_df(simmt_file, with_idx=True, sparse_fmt=opts.spfmt)
	else:
		# Calculate the global similarity matrix across all the collections
		print 'Calculating the similarity matrix...'
		sys.stdout.flush()
		# Serial method
		# simmt = pd.DataFrame(np.ones((len(ids), len(ids))), index=ids, columns=ids)
		# for i, j in itertools.combinations(range(len(ids)), 2):
			# similarity = _sji(udgene[i], udgene[j])
			# simmt.iloc[i, j] = similarity
			# simmt.iloc[j, i] = similarity
		udgene_spmt = udgene_spmt.astype('float32') # Numpy only support parallelism for float32/64
		udgene_mt = udgene_spmt.toarray()
		del udgene_spmt
		# Parallel method
		# similarity = dstclc.parallel_pairwise(udgene_mt, None, _sjim, n_jobs=opts.np, min_chunksize=2)
		similarity = _sjim(udgene_mt, udgene_mt)
		# Tensor data structure
		# udgene_cube = udgene_mt.reshape((-1, 2, udgene_mt.shape[1]))
		# similarity = _sjic(udgene_cube, udgene_cube)
		simmt = pd.DataFrame(similarity, index=ids, columns=ids, dtype=similarity.dtype)
		io.write_df(simmt, simmt_file, with_idx=True, sparse_fmt=opts.spfmt, compress=True)
	# Calculate the similarity matrix within each collection
	print 'Splitting the similarity matrix...'
	sys.stdout.flush()
	for k in xrange(len(excel_dfs)):
		idx_pair = (id_bndry[k], id_bndry[k + 1])
		sub_simmt = simmt.iloc[idx_pair[0]:idx_pair[1],idx_pair[0]:idx_pair[1]]
		fpath = os.path.splitext(simmt_file)
		io.write_df(sub_simmt, fpath[0] + '_%i' % k + fpath[1], with_idx=True, sparse_fmt=opts.spfmt, compress=True)

		
def simhrc():
	sys.setrecursionlimit(10000)
	simmt = io.read_df(opts.loc, with_idx=True)
	plot.plot_clt_hrc(simmt.as_matrix(), dist_metric='precomputed', fname=os.path.splitext(os.path.basename(opts.loc))[0])
	
	
def onto2simmt():
	from scipy.sparse import coo_matrix
	def filter(txt_list):
		new_list = []
		for txt in txt_list:
			if (len(txt) <= 30):
				new_list.append(txt)
		return set(new_list)
	excel_df = pd.read_csv(opts.loc)
	kwargs = {} if opts.cfg is None else ast.literal_eval(opts.cfg)
	col_name, db_name = kwargs['col_name'], kwargs['db_name']
	ontog = sparql.SPARQL('http://localhost:8890/%s/query' % db_name)
	fn_func = ontology.define_obo_fn(ontog, type='exact', prdns=[('obowl', ontology.OBOWL)], eqprds={})
	distmt, vname = ontology.transitive_closure_dsg(ontog, excel_df[col_name].tolist(), find_neighbors=fn_func, filter=filter)
	if (distmt.shape[1] == 0):
		print 'Could not find any neighbors using exact matching. Using fuzzy matching instead...'
		fn_func = ontology.define_obo_fn(ontog, type='fuzzy', prdns=[('obowl', ontology.OBOWL)], eqprds={})
		distmt, vname = ontology.transitive_closure_dsg(ontog, excel_df[col_name].tolist(), find_neighbors=fn_func, filter=filter)
	simmt = coo_matrix((1-dstclc.normdist(distmt.data.astype('float32')), (distmt.row, distmt.col)), shape=distmt.shape)
	sim_df = pd.DataFrame(simmt.toarray(), index=vname, columns=vname)
	io.write_df(sim_df, 'simmt_%s_%s.npz' % (col_name, db_name), with_idx=True, sparse_fmt=opts.spfmt, compress=True)
	
	
def ddi2simmt():
	from bionlp.spider import rxnav
	from scipy.sparse import coo_matrix
	kwargs = {} if opts.cfg is None else ast.literal_eval(opts.cfg)
	col_name = kwargs['col_name']
	cache_path = os.path.join(opts.cache, 'drug_intrcts.pkl')
	drug_cache_path = os.path.join(gsc.RXNAV_PATH, 'drug')
	intr_cache_path = os.path.join(gsc.RXNAV_PATH, 'interaction')
	excel_df = pd.read_csv(opts.loc)
	drug_list = excel_df[col_name].tolist()
	if (os.path.exists(cache_path)):
		interactions = io.read_obj(cache_path)
	else:
		# RxNav API needs rxcui as the identity of drugs so we need to obtain the rxcuis
		drug_client = rxnav.RxNavAPI(function='drugs')
		rxcuis = []
		for drug in drug_list:
			# If the cache exists then just read it, otherwise call the API to retrieve it
			cache_f = os.path.join(drug_cache_path, '%s.pkl' % nlp.clean_text(str(drug)).replace(' ', '_'))
			if (os.path.exists(cache_f)):
				res = io.read_obj(cache_f)
			else:
				res = drug_client.call(name=drug)
				io.write_obj(res, cache_f)
			# Process the results
			drug_rxcuis = []
			for cg in res['concept_group']:
				if (not cg.has_key('property')):
					continue
				drug_rxcuis.extend([prop['rxcui'] for prop in cg['property']])
			rxcuis.append(drug_rxcuis)
			del res, drug_rxcuis
		# Retrieve the interaction
		intrct_client = rxnav.RxNavAPI(function='interaction')
		interactions = []
		for drug, drug_rxcuis in zip(drug_list, rxcuis):
			# Retrieve interaction list for all the rxcuis of each drug
			intrct_concepts, synonymous_ic = [[] for x in range(len(drug_rxcuis))], [[] for x in range(len(drug_rxcuis))]
			for i, cui in enumerate(drug_rxcuis):
				# Check the cache
				cache_f = os.path.join(intr_cache_path, '%s.pkl' % cui)
				if (os.path.exists(cache_f)):
					res = io.read_obj(cache_f)
				else:
					res = intrct_client.call(rxcui=cui)
					io.write_obj(res, cache_f)
				# Process the results
				if (not res.has_key('interactionTypeGroup')):
					continue
				for tg in res['interactionTypeGroup']:
					for itype in tg['interactionType']:
						for ipair in itype['interactionPair']:
							iconcept = ipair['interactionConcept'][0]['sourceConceptItem']['name'], ipair['interactionConcept'][1]['sourceConceptItem']['name']
							if (func.strsim(iconcept[0], drug) > 0.8):
								intrct_concepts[i].append(iconcept[1])
							elif (func.strsim(iconcept[1], drug) > 0.8):
								intrct_concepts[i].append(iconcept[0])
							else:
								synonymous_ic[i].append(iconcept)
				del res
			# Filter out the empty list
			intrct_concepts = [x for x in intrct_concepts if len(x) > 0]
			# Only synonymous concepts are found
			if (len(intrct_concepts) == 0):
				intrct_cpairs = [x for x in synonymous_ic if len(x) > 0]
				# Nothing is found
				if (len(intrct_cpairs) == 0):
					interactions.append([])
					print 'No interaction of Drug [%s] is found with these rxcuis: %s' % (drug, drug_rxcuis)
					continue
				intrct_concepts = []
				# For each interaction of corresponding rxcui
				for cpairs in intrct_cpairs:
					# Find out the common symbol, namely the synonym
					cpair_sets = [set(x) for x in cpairs]
					synonym = set.intersection(*cpair_sets)
					# Subtract the synonym
					ic_list = [(x - synonym) for x in cpair_sets]
					# Filter out the empty ones and transform to singular value
					intrct_concepts.append([x.pop() for x in ic_list if len(x) > 0])
			# Find the consensus of the interaction list across all the rxcuis
			iconcept_lengths = [len(x) for x in intrct_concepts]
			intrct_length = min([len(x) for x in intrct_concepts])
			interactions.append([collections.Counter([x[i] for x in intrct_concepts]).most_common(1)[0][0] for i in range(intrct_length)])
			del intrct_concepts
		io.write_obj(interactions, cache_path)
	drugs = list(set(drug_list + func.flatten_list(interactions)))
	drug_idx = dict([(s, i) for i, s in enumerate(drugs)])
	rows, cols, data = [[] for x in range(3)]
	for drug, interaction in zip(drug_list, interactions):
		row, col, val = [drug_idx[drug]] * len(interaction), [drug_idx[x] for x in interaction], [1] * len(interaction)
		rows.extend(row + col)
		cols.extend(col + row)
		data.extend(val + val)
	simmt = coo_matrix((data, (rows, cols)), shape=(len(drugs), len(drugs)), dtype='int8')
	sim_df = pd.DataFrame(simmt.toarray(), index=drugs, columns=drugs)
	io.write_df(sim_df, 'simmt_drug_%s.npz' % col_name, with_idx=True, sparse_fmt=opts.spfmt, compress=True)

	
def ppi2simmt():
	from bionlp.spider import biogrid
	from scipy.sparse import coo_matrix
	kwargs = {} if opts.cfg is None else ast.literal_eval(opts.cfg)
	col_name = kwargs['col_name']
	cache_path = os.path.join(opts.cache, 'gene_intrcts.pkl')
	intr_cache_path = os.path.join(gsc.BIOGRID_PATH, 'interaction')
	excel_df = pd.read_csv(opts.loc)
	gene_list = excel_df[col_name].tolist()
	if (os.path.exists(cache_path)):
		interactions = io.read_obj(cache_path)
	else:
		interactions = []
		ppi_client = biogrid.BioGRIDAPI(function='interaction', api_key=kwargs['api_key'])
		for gene in gene_list:
			# Check the cache
			cache_f = os.path.join(intr_cache_path, '%s.pkl' % nlp.clean_text(str(gene)).replace(' ', '_'))
			if (os.path.exists(cache_f)):
				res = io.read_obj(cache_f)
			else:
				res = ppi_client.call(geneList=gene)
				io.write_obj(res, cache_f)
			# Process the results
			intrct_concepts = []
			if (type(res) is not dict):
				interactions.append([])
				print 'No interaction of Gene [%s] is found in: %s' % (gene, res)
				continue
			for k in sorted([int(x) for x in res.keys()]):
				ipair = res[str(k)]
				isymbol = set([ipair['OFFICIAL_SYMBOL_A'].lower()] + ipair['SYNONYMS_A'].lower().split('|')), set([ipair['OFFICIAL_SYMBOL_B'].lower()] + ipair['SYNONYMS_B'].lower().split('|'))
				gene = gene.lower()
				if (gene in isymbol[0]):
					intrct_concepts.append(ipair['OFFICIAL_SYMBOL_B'])
				elif (gene in isymbol[1]):
					intrct_concepts.append(ipair['OFFICIAL_SYMBOL_A'])
			interactions.append(intrct_concepts)
			del res, intrct_concepts
		io.write_obj(interactions, cache_path)
	genes = list(set(gene_list + func.flatten_list(interactions)))
	gene_idx = dict([(s, i) for i, s in enumerate(genes)])
	rows, cols, data = [[] for x in range(3)]
	for gene, interaction in zip(gene_list, interactions):
		row, col, val = [gene_idx[gene]] * len(interaction), [gene_idx[x] for x in interaction], [1] * len(interaction)
		rows.extend(row + col)
		cols.extend(col + row)
		data.extend(val + val)
	simmt = coo_matrix((data, (rows, cols)), shape=(len(genes), len(genes)), dtype='int8')
	sim_df = pd.DataFrame(simmt.toarray(), index=genes, columns=genes)
	io.write_df(sim_df, 'simmt_gene_%s.npz' % col_name, with_idx=True, sparse_fmt=opts.spfmt, compress=True)
	
	
def sgn_eval():
	kwargs = {} if opts.cfg is None else ast.literal_eval(opts.cfg)
	excel_df, sgn_simmt, true_simmt = pd.read_csv(os.path.join(opts.loc, kwargs['sgn'])), io.read_df(os.path.join(opts.loc, kwargs['sgnsim']), with_idx=True, sparse_fmt=opts.spfmt), io.read_df(os.path.join(opts.loc, kwargs['truesim']), with_idx=True, sparse_fmt=opts.spfmt)
	col_name, sgnsim_lb, truesim_lb = kwargs['col_name'], kwargs['sgnsim_lb'], kwargs['truesim_lb']
	overlaps = set([str(x).lower() for x in excel_df[col_name]]) & set([str(x).lower() for x in true_simmt.index])
	sgn_simmt = dstclc.normdist(sgn_simmt)
	unique_idx = {}
	for i, symbol in enumerate(excel_df[col_name]):
		unique_idx.setdefault('symbol', []).append(i)
	duplicate_idx = [(k, v) for k, v in unique_idx.iteritems() if len(v) > 1]
	for k, v in duplicate_idx:
		combined_row = sgn_simmt.iloc[v,:].max(axis=0)
		sgn_simmt.drop(k, axis=0, inplace=True)
		pd.concat(sgn_simmt, combined_row)
	
	
def cmp2sim():
	kwargs = {} if opts.cfg is None else ast.literal_eval(opts.cfg)
	sim0_df, sim1_df = io.read_df(os.path.join(opts.loc, kwargs['sim0']), with_idx=True, sparse_fmt=opts.spfmt), io.read_df(os.path.join(opts.loc, kwargs['sim1']), with_idx=True, sparse_fmt=opts.spfmt)
	sim0_lb, sim1_lb = kwargs['sim0_lb'], kwargs['sim1_lb']
	overlaps = set([str(x).lower() for x in sim0_df.index]) & set([str(x).lower() for x in sim1_df.index])
	y_true, y_pred = [], []
	for ol in overlaps:
		y_true.append(sim0_df.columns[sim0_df.ix[ol,:].argsort()[::-1]].tolist())
		y_pred.append(sim1_df.columns[sim1_df.ix[ol,:].argsort()[::-1]].tolist())
	fpr, tpr, roc_auc, thrshd = metric.list_roc(y_true, y_pred, average=opts.avg)
	roc_labels = ['%s (AUC=%0.2f)' % (sim1_lb, roc_auc)]
	roc_data = [[mean_fpr, mean_tpr]]
	plot.plot_roc(roc_data, roc_labels, fname='roc_%s_' % (sim0_lb, sim1_l), plot_cfg=common_cfg)
	
	
def cmp_sim_list():
	kwargs = {} if opts.cfg is None else ast.literal_eval(opts.cfg)
	excel_path, col_name, db_name, sim_lb, rankl_lb = kwargs['excel_path'], kwargs['col_name'], kwargs['db_name'], kwargs['sim_lb'], kwargs['rankl_lb']
	excel_df, sim_df, rank_list = pd.read_csv(excel_path), io.read_df(os.path.join(opts.loc, kwargs['sim']), with_idx=True, sparse_fmt=opts.spfmt), io.read_obj(os.path.join(opts.loc, kwargs['rankl']))
	overlaps = set([str(x).lower() for x in sim_df.index]) & set([str(x).lower() for x in excel_df[col_name]])
	y_true, y_pred = [], []
	for ol in overlaps:
		rankls = [rank_list[x] for x in np.where(excel_df[col_name] == ol)[0] if len(rank_list[x]) > 0]
		if (not rankls): continue
		print ol
		max_length = max([len(x) for x in rankls])
		rankl = [collections.Counter([x[l] for x in rankls if len(x) > l]).most_common(1)[0][0] for l in range(max_length)]
		y_true.append(rankl)
		y_pred.append(sim_df.columns[sim_df.ix[ol,:].argsort()[::-1]].tolist())
	fpr, tpr, roc_auc, thrshd = metric.list_roc(y_true, y_pred, average=opts.avg)
	roc_labels = ['%s (AUC=%0.2f)' % (sim_lb, roc_auc)]
	roc_data = [[fpr, tpr]]
	plot.plot_roc(roc_data, roc_labels, fname='roc_%s_%s' % (sim_lb, rankl_lb), plot_cfg=common_cfg)
	
	
def plot_clt(is_fuzzy=False, threshold='0.5'):
	Xs, Ys, labels = gsc.get_data(None, type='gsm', from_file=True, fmt=opts.fmt, spfmt=opts.spfmt)
	for i, (X, Y, label) in enumerate(zip(Xs, Ys, labels)):
		if (is_fuzzy):
			if (threshold != 'mean' and threshold != 'min'):
				thrshd = float(ast.literal_eval(threshold))
			else:
				thrshd = getattr(label, threshold)()
			label[label >= thrshd] = 1
			label[label < thrshd] = 0
			plot.plot_fzyclt(X.as_matrix(), label.as_matrix(), fname='clustering_%i' % i)
		else:
			plot.plot_clt(X.as_matrix(), label.as_matrix().reshape((label.shape[0],)), fname='clustering_%i' % i)
			
			
def plot_sampclt(with_cns=False):
	from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
	import networkx as nx
	import matplotlib.pyplot as plt
	cluster_sets = {
		0:[
		# 'GSM155564|GSM155565|GSM155566|GSM155567',
		# 'GSM155568|GSM155569|GSM155570|GSM155571|GSM155572|GSM155573|GSM155574',
		# 'GSM155592|GSM155593|GSM155594|GSM155595',
		'GSM684691|GSM684692|GSM684693',
		'GSM684688|GSM684689|GSM684690',
		'GSM684685|GSM684686|GSM684687',
		'GSM684682|GSM684683|GSM684684'
		],
		1:[
		# 'GSM271362|GSM271365|GSM271367|GSM271369', 
		# 'GSM271386|GSM271387|GSM271388|GSM271389',
		# 'GSM271382|GSM271383|GSM271384|GSM271385',
		'GSM487691|GSM487692|GSM487693|GSM487694',
		'GSM487687|GSM487688|GSM487689|GSM487690',
		'GSM487683|GSM487684|GSM487685|GSM487686'], 
		2:[
		'GSM272915|GSM272917|GSM272919|GSM272921',
		'GSM272914|GSM272916|GSM272918|GSM272920',
		'GSM272919|GSM272921',
		'GSM272918|GSM272920',
		'GSM272915|GSM272917',
		'GSM272914|GSM272916',
		# 'GSM312684|GSM312685|GSM312686',
		# 'GSM312687|GSM312688|GSM312689|GSM312690|GSM312691|GSM312692',
		# 'GSM312708|GSM312709|GSM312710',
		# 'GSM312727|GSM312728|GSM312729|GSM312730',
		# 'GSM312702|GSM312703|GSM312704',
		# 'GSM312696|GSM312697|GSM312698',
		# 'GSM312715|GSM312716|GSM312717|GSM312718',
		# 'GSM312723|GSM312724|GSM312725|GSM312726',
		# 'GSM312711|GSM312712|GSM312713|GSM312714',
		# 'GSM312690|GSM312691|GSM312692',
		# 'GSM312687|GSM312688|GSM312689',	'GSM312684|GSM312685|GSM312686|GSM312693|GSM312694|GSM312695|GSM312699|GSM312700|GSM312701|GSM312705|GSM312706|GSM312707',	'GSM312687|GSM312688|GSM312689|GSM312690|GSM312691|GSM312692|GSM312696|GSM312697|GSM312698|GSM312702|GSM312703|GSM312704',
		# 'GSM312708|GSM312709|GSM312710|GSM312719|GSM312720|GSM312721|GSM312722|GSM312731|GSM312732|GSM312733|GSM312734',
		# 'GSM312711|GSM312712|GSM312713|GSM312714|GSM312715|GSM312716|GSM312717|GSM312718|GSM312723|GSM312724|GSM312725|GSM312726|GSM312727|GSM312728|GSM312729|GSM312730'
		]}
	kwargs = {} if opts.cfg is None else ast.literal_eval(opts.cfg)
	metric = kwargs.setdefault('metric', 'euclidean')
	a = kwargs.setdefault('a', 0.5)
	for cltid, clusters in cluster_sets.iteritems():
		if (opts.pid != -1 and opts.pid != cltid):
			continue
		clts = [clt.split('|') for clt in clusters]
		## Extract all the vertices and the index mapping
		vertices = np.array(list(set(func.flatten_list(clts))))
		v_map = dict([(v, i) for i, v in enumerate(vertices)])
		## Retrieve the data
		Xs, Ys, labels = gsc.get_mltl_npz(type='gsm', lbs=[cltid], spfmt=opts.spfmt)
		X, Y, z = Xs[0].loc[vertices], Ys[0].loc[vertices], labels[0].loc[vertices]
		io.write_df(X, os.path.join(gsc.DATA_PATH, 'samp_gsm_X_%s' % cltid), with_idx=True, sparse_fmt=opts.spfmt, compress=True)
		io.write_df(Y, os.path.join(gsc.DATA_PATH, 'samp_gsm_y_%s' % cltid), with_idx=True)
		io.write_df(z, os.path.join(gsc.DATA_PATH, 'samp_gsm_lb_%s' % cltid), with_idx=True, sparse_fmt=opts.spfmt, compress=True)
		constraint = Y.as_matrix() if with_cns else None
		## Calculate the distance
		D = dstclc.cns_dist(X.as_matrix(), C=constraint, metric=metric, a=a, n_jobs=opts.np)
		## Construct a KNN graph
		# KNNG = kneighbors_graph(D, 4, mode='distance', metric='precomputed', n_jobs=opts.np).tocoo()
		KNNG = radius_neighbors_graph(D, 0.5, mode='distance', metric='precomputed', n_jobs=opts.np).tocoo()
		## Construct the graph
		G = nx.Graph()
		G.add_nodes_from([(idx, dict(id=vid)) for idx, vid in enumerate(vertices)])
		# G.add_weighted_edges_from([(i, j, D[i,j]) for i, j in itertools.permutations(range(D.shape[0]), 2)])
		edges = list(zip(KNNG.row, KNNG.col))
		G.add_weighted_edges_from([(i, j, KNNG.data[k]) for k, (i, j) in enumerate(edges)])
		## Remove the edge that has the weight greater than threshold
		# G.remove_edges_from([(u, v) for u, v, w in G.edges(data='weight') if w > 0.5])
		# G.remove_edges_from([(u, v) for u, v, d in G.edges(data=True) if d['weight'] > 0.5])
		## Save the graph
		nx.write_gml(G, 'sampclt_%i.gml' % cltid)
		## Construct the MST graph
		G = nx.minimum_spanning_tree(G)
		edges = G.edges()
		## Set the layout of the graph
		pos = nx.nx_pydot.graphviz_layout(G)
		# pos = nx.nx_pydot.pydot_layout(G)
		fig = plt.figure(figsize=(8, 6))
		# fig = plt.figure(figsize=(20, 15))
		# fig = plt.figure(figsize=(24, 18))
		## Draw the nodes
		ctrl_nodes = [v_map[v] for v in Y.index[Y.ctrl == 1]]
		pert_nodes = [v_map[v] for v in Y.index[Y.pert == 1]]
		nx.draw_networkx_nodes(G, pos, nodelist=ctrl_nodes, node_color='g', node_shape='o', node_size=500, alpha=1)
		nx.draw_networkx_nodes(G, pos, nodelist=pert_nodes, node_color='r', node_shape='h', node_size=500, alpha=1)
		nx.draw_networkx_labels(G, pos, labels=dict([(i, v[-3:]) for i, v in enumerate(vertices)]), font_size='10')
		## Draw the edges
		# elarge=[(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= 0.3]
		# esmall=[(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 0.3]
		# nx.draw_networkx_edges(G, pos, edgelist=elarge, width=3, alpha=0.8)
		# nx.draw_networkx_edges(G, pos, edgelist=esmall, width=1, alpha=0.5, edge_color='grey', style='dashed')
		# nx.draw_networkx_edge_labels(G, pos, edge_labels=dict([(e, '%.3f' % G.edge[e[0]][e[1]]['weight']) for e in elarge]), bbox=dict(boxstyle='round,pad=0.,rounding_size=0.2', fc='w', ec='w', alpha=1))
		nx.draw_networkx_edges(G, pos, edgelist=edges, width=2, alpha=0.7)
		nx.draw_networkx_edge_labels(G, pos, edge_labels=dict([(e, '%.3f' % G.edge[e[0]][e[1]]['weight']) for e in edges]), bbox=dict(boxstyle='round,pad=0.,rounding_size=0.2', fc='w', ec='w', alpha=1))
		plt.axis('off')
		if (plot.MON):
			plt.show()
		else:
			plt.savefig('sampclt_%i' % cltid)
		plt.close()


def main():
	
	if (opts.method is None):
		return
	elif (opts.method == 'n2d'):
		nt2db()
	elif (opts.method == 'x2d'):
		xml2db()
	elif (opts.method == 'x2ds'):
		xml2dbs()
	elif (opts.method == 'db2nt'):
		dbcsv2nt()
	elif (opts.method == 'dg2nt'):
		dgcsv2nt()
	elif (opts.method == 'download'):
		download()
	elif (opts.method == 'slctfs'):
		slct_featset()
	elif (opts.method == 'm2e'):
		gsm2gse()
	elif (opts.method == 'p2d'):
		cltpred2df()
	elif (opts.method == 'ggp'):
		gen_gsmclt_pair()
	elif (opts.method == 'ggfp'):
		gen_gsmfzclt_pair()
	elif (opts.method == 'asgn'):
		annot_sgn()
	elif (opts.method == 's2g'):
		sgn2ge()
	elif (opts.method == 's2d'):
		sgn2deg()
	elif (opts.method == 'd2s'):
		deg2simmt()
	elif (opts.method == 'simhrc'):
		simhrc()
	elif (opts.method == 'o2s'):
		onto2simmt()
	elif (opts.method == 'ddis'):
		ddi2simmt()
	elif (opts.method == 'ppis'):
		ppi2simmt()
	elif (opts.method == 'c2s'):
		cmp2sim()
	elif (opts.method == 'csl'):
		cmp_sim_list()
	elif (opts.method == 'pltclt'):
		plot_clt(opts.fuzzy, threshold=opts.thrshd)
	elif (opts.method == 'smpclt'):
		plot_sampclt(with_cns=opts.cns)


if __name__ == '__main__':
	# Parse commandline arguments
	op = OptionParser()
	op.add_option('-p', '--pid', default=-1, action='store', type='int', dest='pid', help='indicate the process ID')
	op.add_option('-n', '--np', default=-1, action='store', type='int', dest='np', help='indicate the number of processes used for calculation')
	op.add_option('-f', '--fmt', default='npz', help='data stored format: csv or npz [default: %default]')
	op.add_option('-s', '--spfmt', default='csr', help='sparse data stored format: csr or csc [default: %default]')
	op.add_option('-t', '--type', default='soft', help='file type: soft, xml, txt [default: %default]')
	op.add_option('-c', '--cfg', help='config string used in the utility functions, format: {\'param_name1\':param_value1[, \'param_name1\':param_value1]}')
	op.add_option('-a', '--avg', default='micro', help='averaging strategy for performance metrics: micro or macro [default: %default]')
	op.add_option('-i', '--unified', action='store_true', dest='unified', default=True, help='store the data in the same folder')
	op.add_option('-l', '--loc', default='.', help='the files in which location to be process')
	op.add_option('-o', '--output', help='the path to store the data')
	op.add_option('-u', '--fuzzy', action='store_true', dest='fuzzy', default=False, help='use fuzzy clustering')
	op.add_option('-d', '--cns', action='store_true', dest='cns', default=False, help='use constraint clustering')
	op.add_option('-r', '--thrshd', default='mean', type='str', dest='thrshd', help='threshold value')
	op.add_option('-w', '--cache', default='.cache', help='the location of cache files')
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
	if (os.path.exists(CONFIG_FILE)):
		cfgr = io.cfg_reader(CONFIG_FILE)
		gsc_cfg = cfgr('bionlp.spider.gsc', 'init')
		if (len(gsc_cfg) > 0):
			if (gsc_cfg['DATA_PATH'] is not None and os.path.exists(gsc_cfg['DATA_PATH'])):
				gsc.DATA_PATH = gsc_cfg['DATA_PATH']
			if (gsc_cfg['GEO_PATH'] is not None and os.path.exists(gsc_cfg['GEO_PATH'])):
				gsc.GEO_PATH = gsc_cfg['GEO_PATH']
			if (gsc_cfg['RXNAV_PATH'] is not None and os.path.exists(gsc_cfg['RXNAV_PATH'])):
				gsc.RXNAV_PATH = gsc_cfg['RXNAV_PATH']
			if (gsc_cfg['BIOGRID_PATH'] is not None and os.path.exists(gsc_cfg['BIOGRID_PATH'])):
				gsc.BIOGRID_PATH = gsc_cfg['BIOGRID_PATH']
		plot_cfg = cfgr('bionlp.util.plot', 'init')
		plot_common = cfgr('bionlp.util.plot', 'common')
		init_plot(plot_cfg=plot_cfg, plot_common=plot_common)

	annot.init()
			
	main()