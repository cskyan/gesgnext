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
import psutil
import logging
import operator
import itertools
import cStringIO
import collections
from optparse import OptionParser

import numpy as np
import scipy as sp
from scipy import misc
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import pairwise_distances as pdist

from bioinfo.spider import nihnuccore
from bionlp.spider import annot, sparql, nihgene
from bionlp.util import fs, io, func, plot, ontology, shell, njobs, sampling
from bionlp import dstclc, nlp, metric
# from bionlp import txtclt

import gsc

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
PAR_DIR = os.path.abspath(os.path.join(FILE_DIR, os.path.pardir))
CONFIG_FILE = os.path.join(PAR_DIR, 'etc', 'config.yaml')
RAMSIZE = 4
SC=';;'

opts, args = {}, []
cfgr, plot_common_cfg = None, {}


def init_plot(plot_cfg={}, plot_common={}):
	if (len(plot_cfg) > 0 and plot_cfg['MON'] is not None):
		plot.MON = plot_cfg['MON']
	global plot_common_cfg
	if (len(plot_common) > 0):
		plot_common_cfg = plot_common

		
def fuseki(sh='bash'):
	common_cfg = cfgr('gsx_helper', 'common')
	proc_cmd = ' && '.join([common_cfg['FUSEKI_ENV'], '%s --port=8890 --config=$FUSEKI_HOME/run/config.ttl >/var/log/fuseki.log 2>&1 &' % 'fuseki-server' if common_cfg['FUSEKI_PATH'] is None or common_cfg['FUSEKI_PATH'].isspace() else os.path.join(common_cfg['FUSEKI_PATH'], 'fuseki-server')])
	cmd = proc_cmd if sh == 'sh' else '%s -c "%s"' % (sh, proc_cmd)
	shell.daemon(cmd, 'fuseki-server')
	
	
def npzs2yaml(dir_path='.', mdl_t='Classifier'):
	pw = io.param_writer(os.path.join(dir_path, 'mdlcfg'))
	for file in fs.listf(dir_path):
		if file.endswith(".npz"):
			fpath = os.path.join(dir_path, file)
			params = io.read_npz(fpath)['best_params'].tolist()
			for k in params.keys():
				if (type(params[k]) == np.ndarray):
					params[k] == params[k].tolist()
				if (isinstance(params[k], np.generic)):
					params[k] = np.asscalar(params[k])
			pw(mdl_t, file, params)
	pw(None, None, None, True)


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
	data_df = pd.read_csv(opts.loc, encoding='utf-8').fillna('')
	dbid_tmplt, name_tmplt = u'<http://www.drugbank.ca/drugbank-id/%s>', u'"%s"@en'
	vcb_cmname, vcb_term = u'<http://www.drugbank.ca/vocab#common-name>', u'<http://www.drugbank.ca/vocab#term>'
	dbids, cmnames, synm_strs = data_df['DrugBank ID'].tolist(), data_df['Common name'].tolist(), data_df['Synonyms'].tolist()
	triples = [(dbid_tmplt % dbid, vcb_cmname, name_tmplt % cmname) for dbid, cmname in zip(dbids, cmnames)]
	synonyms = [list(set([y.strip() for y in unicode(x).split('|')])) for x in synm_strs]
	triples.extend([(dbid_tmplt % dbid, vcb_term, name_tmplt % synm.replace('"', '\\"')) for dbid, synms in zip(dbids, synonyms) for synm in synms if synm != u''])
	triples = dict.fromkeys(triples).keys()
	fpath = opts.output if opts.output is not None else os.path.splitext(opts.loc)[0] + '.nt'
	fs.write_file(' .\n'.join([' '.join(x) for x in triples]) + ' .', fpath, code='utf-8')
	
	
def dgcsv2nt():
	data_df = pd.read_csv(opts.loc, encoding='utf-8').fillna('')
	gene_tmplt, intype_tmplt, drug_tmplt = u'<http://dgidb.genome.wustl.edu/gene/%s>', u'<http://dgidb.genome.wustl.edu/vocab#%s>', u'<http://dgidb.genome.wustl.edu/drug/%s>'
	# vcb_interact = u'<http://dgidb.genome.wustl.edu/vocab#interact>'
	gene_name, intype, drug_name = data_df['entrez_gene_symbol'].tolist(), data_df['interaction_types'].tolist(), data_df['drug_primary_name'].tolist()
	triples = [(gene_tmplt % gn.replace(' ', '_'), intype_tmplt % it.replace('n/a', 'unknown').replace(' ', '_'), drug_tmplt % dn.replace(' ', '_')) for gn, it, dn in zip(gene_name, intype, drug_name)]
	triples = dict.fromkeys(triples).keys()
	fpath = opts.output if opts.output is not None else os.path.splitext(opts.loc)[0] + '.nt'
	fs.write_file(' .\n'.join([' '.join(x) for x in triples]) + ' .', fpath, code='utf-8')


def download():
	sgn_df = pd.read_csv(opts.loc)
	par_dir, basename = os.path.abspath(os.path.join(opts.loc, os.path.pardir)), os.path.splitext(os.path.basename(opts.loc))[0]
	if (opts.unified):
		saved_path = os.path.join(par_dir, opts.type) if opts.output is None else os.path.join(opts.output, opts.type)
	else:
		saved_path = os.path.join(par_dir, opts.type, basename) if opts.output is None else os.path.join(opts.output, opts.type, basename)
	kwargs = {} if opts.cfg is None else ast.literal_eval(opts.cfg)
	if (opts.type == 'soft'):
		from bionlp.spider import geo
		for geo_data in geo.fetch_geo(list(sgn_df['geo_id']), saved_path=saved_path, skip_cached=True, **kwargs):
			del geo_data
	else:
		from bionlp.spider import geoxml as geo
		# Download GSE data
		geo_strs = list(geo.fetch_geo(list(sgn_df['geo_id']), saved_path=saved_path, **kwargs))
		geo_strios = [cStringIO.StringIO(geo_str) for geo_str in geo_strs]
		# Parse GSE data
		geo_docs = geo.parse_geos(geo_strios)
		# Download GSM data
		samples = [sample for geo_doc in geo_docs for sample in geo_doc['samples']]
		for sample_str in geo.fetch_geo(samples, saved_path=os.path.join(saved_path, 'samples'), **kwargs):
			del sample_str
		# Download GPL data
		for gpl_str in geo.fetch_geo(list(sgn_df['platform']), saved_path=os.path.join(saved_path, 'platforms'), view='full', **kwargs):
			del gpl_str

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


def _gpl2map(gpl_fpaths, fmt='xml'):
	if (fmt == 'soft'):
		from bionlp.spider import geo
	else:
		from bionlp.spider import geoxml as geo
	# Pre-defined symbol column
	probe_gene, symbol_kw = [], ['GENE SYMBOL', 'GENE_SYMBOL', 'GENESYMBOL', 'GENE NAME', 'GENE_NAME', 'GENENAME', 'SYMBOL']
	for doc in geo.parse_geos(gpl_fpaths, view='full', type='gpl', fmt=fmt):
		genes, empty = pd.Series([]), np.array([True]*doc['data'].shape[0])
		# Map the gene id to symbol given in the dataset
		for kw in symbol_kw:
			if (not kw in doc['data'].columns): continue
			if (genes.empty):
				genes = doc['data'][kw]
			else:
				genes.loc[empty] = doc['data'][kw].loc[empty]
			empty[empty] = [not x or str(x).isspace() for x in genes[empty]]
		if (any(empty)):
			# Some dataset has ambiguous column name
			if ('GENE' in doc['data'].columns and any([kw in doc['col_desc'][doc['data'].columns.tolist().index('GENE') + 1].upper() for kw in ['GENE NAME', 'GENE SYMBOL']])):
				if (genes.empty):
					genes = doc['data']['GENE']
				else:
					genes.loc[empty] = doc['data']['GENE'].loc[empty]
				empty[empty] = [not x or str(x).isspace() for x in genes[empty]]
		# Query the NIH gene database to map the gene id to symbol
		has_cols = dict([(col, col in doc['data'].columns # whether has this column
				and any([kw in doc['col_desc'][doc['data'].columns.tolist().index(col) + 1].upper() for kw in ['ACCESSION', 'GENEBANK']]) # further confirm the content of this column
				) for col in ['GB_LIST', 'GB_ACC']] # Gene Bank Accession Number
			+ [(col, col in doc['data'].columns # whether has this column
				and any([kw in doc['col_desc'][doc['data'].columns.tolist().index(col) + 1].upper() for kw in ['ENTREZ GENE']]) # further confirm the content of this column
				) for col in ['GENE', 'ENTREZ_GENE_ID']]) # ENTREZ GENE ID
		if (any(empty) and (has_cols['GB_LIST'] or has_cols['GB_ACC'])): # first priority
			col = 'GB_LIST' if has_cols['GB_LIST'] else 'GB_ACC'
			gene_ids = doc['data'][col].apply(lambda x: str(x).strip(';,| ')).apply(lambda x: x.split(';') if x else []).apply(lambda x: func.flatten_list([dx.split(',') for dx in x if dx and not dx.isspace()])).apply(lambda x: func.flatten_list([dx.split('|') for dx in x if dx and not dx.isspace()])).apply(lambda x: func.flatten_list([dx.split(' ') for dx in x if dx and not dx.isspace()]))
			print 'Converting the GenBank Accession Numbers %s of %s... into Gene Symbol...' % (','.join(func.flatten_list(gene_ids.iloc[empty][:5].tolist())), doc['id'])
			# query_res = pd.Series([ for gene_doc in nihnuccore.parse_genes(nihnuccore.fetch_gene(gene_ids, ret_strio=True))], index=doc['data'].index)
			query_res = []
			for gene_doc in nihnuccore.parse_genes(nihnuccore.fetch_gene(gene_ids.tolist(), ret_strio=True)):
				if (type(gene_doc) is list):
					query_res.append(' /// '.join([gd['symbol'] for gd in gene_doc if gd.has_key('symbol') and gd['symbol'] and not gd['symbol'].isspace()]))
				else:
					query_res.append(gene_doc['symbol'] if gene_doc.has_key('symbol') and gene_doc['symbol'] and not gene_doc['symbol'].isspace() else '')
			query_res = pd.Series(query_res, index=doc['data'].index)
			if (genes.empty):
				genes = query_res
			else:
				genes.loc[empty] = query_res.loc[empty]
			empty[empty] = [not x or str(x).isspace() for x in genes[empty]]
		if (any(empty) and (has_cols['GENE'] or has_cols['ENTREZ_GENE_ID'])): # second priority
			col = 'GENE' if has_cols['GENE'] else 'ENTREZ_GENE_ID'
			gene_ids = doc['data'][col].apply(lambda x: str(x).strip(';,| ')).apply(lambda x: x.split(';') if x else []).apply(lambda x: func.flatten_list([dx.split(',') for dx in x if dx and not dx.isspace()])).apply(lambda x: func.flatten_list([dx.split('|') for dx in x if dx and not dx.isspace()])).apply(lambda x: func.flatten_list([dx.split(' ') for dx in x if dx and not dx.isspace()]))
			print 'Converting the Entrez GENE IDs %s of %s... into Gene Symbol...' % (','.join(func.flatten_list(gene_ids.iloc[empty][:5].tolist())), doc['id'])
			# query_res = pd.Series([gene_doc['symbol'] if gene_doc.has_key('symbol') and gene_doc['symbol'] else '' for gene_doc in nihgene.parse_genes(nihgene.fetch_gene(gene_ids, ret_strio=True))], index=doc['data'].index)
			query_res = []
			for gene_doc in nihgene.parse_genes(nihgene.fetch_gene(gene_ids.tolist(), ret_strio=True)):
				if (type(gene_doc) is list):
					query_res.append(' /// '.join([gd['symbol'] for gd in gene_doc if gd.has_key('symbol') and gd['symbol'] and not gd['symbol'].isspace()]))
				else:
					query_res.append(gene_doc['symbol'] if gene_doc.has_key('symbol') and gene_doc['symbol'] and not gene_doc['symbol'].isspace() else '')
			query_res = pd.Series(query_res, index=doc['data'].index)
			if (genes.empty):
				genes = query_res
			else:
				genes.loc[empty] = query_res.loc[empty]
			empty[empty] = [not x or str(x).isspace() for x in genes[empty]]
		# Cannot find the symbol column or the column is empty
		if (genes.empty or ' '.join(map(str, genes.tolist())).isspace()): continue
		probe_gene.append((doc['id'], genes))
	return probe_gene

	
def gpl2map(fmt='xml'):
	labels = gsc.LABEL2ID.keys()
	if (opts.pid != -1):
		labels = [labels[opts.pid % len(labels)]]
	for label in labels:
		# Read the corresponding platform metadata
		platform_path = os.path.join(gsc.GEO_PATH, opts.type, label.replace(' ', '_'), 'platforms')
		gpl_fpaths = [fpath for fpath in fs.listf(platform_path, pattern='GPL.*\.%s'%opts.type, full_path=True)]
		if (opts.np == 1):
			probe_gene = _gpl2map(gpl_fpaths, fmt=fmt)
		else:
			task_bnd = njobs.split_1d(len(gpl_fpaths), split_num=opts.np, ret_idx=True)
			pgs = njobs.run_pool(_gpl2map, n_jobs=opts.np, dist_param=['gpl_fpaths'], gpl_fpaths=[gpl_fpaths[task_bnd[i]:task_bnd[i+1]] for i in range(opts.np)], fmt=fmt)
			probe_gene = func.flatten_list(pgs)
		io.inst_print('Finish mapping probe to gene in dataset %s!' % label)
		io.inst_print('Saving results...')
		probe_gene_map = dict(probe_gene)
		pgm_df = pd.DataFrame(pd.concat(probe_gene_map.values()))
		io.write_df(pgm_df, os.path.join(platform_path, 'probe_gene_map.npz'))
		io.write_obj(probe_gene_map, os.path.join(platform_path, 'probe_gene_map.pkl'))
		io.inst_print('Finish saving to disk!')

	
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


def _gsmclt_pair(X, Y, z, gsm2gse, lbid, thrshd=0.5, iterative=False, cache_path='.cache', n_jobs=1, fname=None):
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
		gse_ids = gsm2gse.loc[clt]
		for gse_id in gse_ids.gse_id.value_counts().index:
			common_gseids = gse_ids[gse_ids.gse_id == gse_id]
			if (common_gseids.shape[0] < gse_ids.shape[0]):
				print 'Impure cluster with %.2f%% extra GEO studies:' % (1 - 1.0 * common_gseids.shape[0] / gse_ids.shape[0])
				break
			cln_clt = common_gseids.index.tolist()
			# Control group or perturbation group (It will ignore the samples predicted as both control and perturbation, or neither)
			# clt_y = Y.ix[cln_clt]
			# clty_sum = clt_y.sum(axis=1)
			# clt_y = clt_y.iloc[np.where(np.logical_and(clty_sum != 0, clty_sum != 2))[0]]
			# pure_clty = clt_y.iloc[np.where(clt_y.sum(axis=1) == 1)[0]]
			# print clt_y.shape, pure_clty.shape
			# if (pure_clty.empty or pure_clty.shape[0] == 0): continue
			# if (pure_clty.shape[0] < clt_y.shape[0]):
				# print clt_y.pert
				# print clt_y.ctrl
				# print 'Impure cluster with %.2f%% ctrls, %.2f%% perts, and %.2f%% mixtures:' % (1.0 * np.where(clt_y.ctrl == 1)[0].shape[0] / clt_y.shape[0], 1.0 * np.where(clt_y.pert == 1)[0].shape[0] / clt_y.shape[0], 1.0 * np.where(clt_y.sum(axis=1) == 2)[0].shape[0] / clt_y.shape[0])
			# cln_clt = pure_clty.index.tolist()
			ctrl_pert_clts.setdefault(gse_id, [[], []])[Y.ix[cln_clt[0]].pert].append(cln_clt)
	# print 'Refined Cluster Number:%s' % dict([(gse_id, [len(clts[0]), len(clts[1])]) for gse_id, clts in ctrl_pert_clts.iteritems()])
	# print ctrl_pert_clts
	# Enumerate every pair of control and perturbation group
	geo_ids, ctrl_ids, pert_ids, tissues, organisms, platforms = [[] for x in range(6)]
	first_pass = True
	for gse_id, cpclts in ctrl_pert_clts.iteritems():
		## Construct the GSM graph
		# Extract all the GSM
		gsms = list(set(func.flatten_list(cpclts)))
		gsm_id_map = dict(zip(gsms, range(len(gsms))))
		# Retrieve the data for each GSM
		gsm_X = X.loc[gsms]
		# Calculate the pairwise distance
		dist_mt = pdist(gsm_X, metric='euclidean', n_jobs=n_jobs)
		pw_dist, cnd_pw = [], []
		for ctrl, pert in itertools.product(cpclts[0], cpclts[1]):
			# Filter the uninterpretable pairs
			ctrl_idx, pert_idx = [gsm_id_map[x] for x in ctrl], [gsm_id_map[x] for x in pert]
			# Obtain the distance matrix of those GSMs
			# pw_dist.append(dist_mt[ctrl_idx,:][:,pert_idx].mean())
			# Use Ward's Method to measure the cluster distance
			num_ctrl, num_pert = len(ctrl), len(pert)
			pw_dist.append(1.0 * (num_ctrl * num_pert) / (num_ctrl + num_pert) * (np.linalg.norm(gsm_X.loc[ctrl].mean(axis=0) - gsm_X.loc[pert].mean(axis=0))))
			cnd_pw.append((ctrl, pert))
		# Find a cut value for filtering
		hist, bin_edges = np.histogram(pw_dist)
		weird_val_idx = len(hist) - 1 - np.abs(hist[-1:0:-1] - hist[-2::-1]).argmax()
		cut_val = (bin_edges[weird_val_idx] + bin_edges[weird_val_idx + 1]) / 2
		
		if (iterative):
			geo_ids, ctrl_ids, pert_ids = [[] for x in range(3)]
			for ctrl, pert in itertools.product(cpclts[0], cpclts[1]):
				geo_ids.append(gse_id)
				ctrl_ids.append('|'.join(sorted(ctrl)))
				pert_ids.append('|'.join(sorted(pert)))
			# Iterative write to file
			pair_df = pd.DataFrame.from_items([('geo_id', geo_ids), ('ctrl_ids', ctrl_ids), ('pert_ids', pert_ids)])
			print 'Generated %i signatures in %s' % (pair_df.shape[0], gse_id)
			pair_df.to_csv(fname + '.csv', header=first_pass, index=False, mode='a', encoding='utf8')
			first_pass = False
			del geo_ids, ctrl_ids, pert_ids
		else:
			for dist, pw in zip(pw_dist, cnd_pw):
				if (dist > cut_val): continue
				geo_ids.append(gse_id)
				ctrl, pert = pw
				ctrl_ids.append('|'.join(sorted(ctrl)))
				pert_ids.append('|'.join(sorted(pert)))
		del pw_dist, cnd_pw

	if (iterative):
		return None
	else:
		# Write to file
		pair_df = pd.DataFrame.from_items([('geo_id', geo_ids), ('ctrl_ids', ctrl_ids), ('pert_ids', pert_ids)])
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
	gsmclt_pairs = {}
	m2e = io.read_df(os.path.join(gsc.DATA_PATH, 'gsm2gse.npz'), with_idx=True)
	lbids = range(len(Xs)) if (opts.pid == -1) else [opts.pid]
	pred_dfs = _cltpred2df(Xs, Ys, labels, lbids, predf_patn)
	for i, (X, Y, predfs) in enumerate(zip(Xs, Ys, pred_dfs)):
		lbid = i if (opts.pid == -1) else opts.pid
		for fname, pred_df in predfs:
			# Generate GSM cluster pairs
			gsmclt_pair = _gsmclt_pair(X, Y, pred_df, m2e, lbid, thrshd=threshold, cache_path=opts.cache, n_jobs=opts.np)
			gsmclt_pairs.setdefault(fname.split('_')[2], []).append(gsmclt_pair)
	return gsmclt_pairs
			
			
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
				ret_dict = annot.annotext(geo_doc[field], retype='dict')
			except Exception as e:
				print 'Unable to annotate %s in the %s field!' % (geo_id, field)
				print e
				continue
			json_str = json.dumps(ret_dict)
			fs.write_file(json_str, field_cachef, code='utf8')
		# Transform the data into groups
		ret_dict['text'] = geo_doc[field]
		annot_res.append(annot.annotext(ret_dict, retype='group', with_mdf=True if (field=='source') else False))
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

	
def _sgn2ge(sgn_df, sample_path, saved_path, fmt='xml'):
	if (fmt == 'soft'):
		from bionlp.spider import geo
	else:
		from bionlp.spider import geoxml as geo
	ids = sgn_df['id'] if hasattr(sgn_df, 'id') else sgn_df.index
	for sid, ctrl_str, pert_str in zip(ids, sgn_df['ctrl_ids'], sgn_df['pert_ids']):
		ctrl_file, pert_file = os.path.join(saved_path, 'ctrl_%s.npz' % sid), os.path.join(saved_path, 'pert_%s.npz' % sid)
		if (os.path.exists(ctrl_file) and os.path.exists(pert_file)): continue
		ctrl_ids, pert_ids = ctrl_str.split('|'), pert_str.split('|')
		# Obtain the geo files for each sample
		ctrl_geo_docs, pert_geo_docs = geo.parse_geos([os.path.join(sample_path, '.'.join([ctrl_id, fmt])) for ctrl_id in ctrl_ids], view='full', type='gsm', fmt=fmt), geo.parse_geos([os.path.join(sample_path, '.'.join([pert_id, fmt])) for pert_id in pert_ids], view='full', type='gsm', fmt=fmt)
		# Extract the gene expression data from the geo files for each sample, and combine the data within the same group
		ctrl_ge_dfs, pert_ge_dfs = [geo_doc['data']['VALUE'] for geo_doc in ctrl_geo_docs], [geo_doc['data']['VALUE'] for geo_doc in pert_geo_docs]
		ctrl_df, pert_df = pd.concat(ctrl_ge_dfs, axis=1, join='inner').astype('float32'), pd.concat(pert_ge_dfs, axis=1, join='inner').astype('float32')
		io.write_df(ctrl_df, ctrl_file, with_col=False, with_idx=True)
		io.write_df(pert_df, pert_file, with_col=False, with_idx=True)
	
	
def sgn2ge():
	input_ext = os.path.splitext(opts.loc)[1]
	if (input_ext == '.xlsx' or input_ext == '.xls'):
		sgn_df = pd.read_excel(opts.loc)
	elif (input_ext == '.csv'):
		sgn_df = pd.read_csv(opts.loc)
	elif (input_ext == '.npz'):
		sgn_df = io.read_df(opts.loc)
	par_dir, basename = os.path.abspath(os.path.join(opts.loc, os.path.pardir)), os.path.splitext(os.path.basename(opts.loc))[0]
	sample_path = os.path.join(gsc.GEO_PATH, opts.type, basename, 'samples')
	saved_path = os.path.join(par_dir, 'gedata', basename) if opts.output is None else os.path.join(opts.output, basename)
	# Find the control group and perturbation group for every signature
	if (opts.np == 1):
		_sgn2ge(sgn_df, sample_path, saved_path, fmt=opts.type)
	else:
		task_bnd = njobs.split_1d(sgn_df.shape[0], split_num=opts.np, ret_idx=True)
		_ = njobs.run_pool(_sgn2ge, n_jobs=opts.np, dist_param=['sgn_df'], sgn_df=[sgn_df.iloc[task_bnd[i]:task_bnd[i+1]] for i in range(opts.np)], sample_path=sample_path, saved_path=saved_path, fmt=opts.type)


def _sgn2dge(sgn_df, method, ge_path, saved_path, cache_path):
	from bioinfo.ext import chdir, limma
	_method = method.lower()
	sids = sgn_df['id'] if hasattr(sgn_df, 'id') else sgn_df.index
	dge_dfs = []
	for sid in sids:
		dge_file = os.path.join(saved_path, 'dge_%s.npz' % sid)
		if (os.path.exists(dge_file)):
			dge_df = io.read_df(dge_file, with_idx=True)
			dge_dfs.append(dge_df)
			continue
		ctrl_file, pert_file = os.path.join(ge_path, 'ctrl_%s.npz' % sid), os.path.join(ge_path, 'pert_%s.npz' % sid)
		ctrl_df, pert_df = io.read_df(ctrl_file, with_col=False, with_idx=True), io.read_df(pert_file, with_col=False, with_idx=True)
		# Find the gene sets that are both in control group and perturbation group
		join_df = pd.concat([ctrl_df, pert_df], axis=1, join='inner')
		print 'Start %s algorithm for %s...%s, %s, %s' % (method.upper(), sid, ctrl_df.shape, pert_df.shape, join_df.shape)
		# Calculate the differential gene expression vector
		if (ctrl_df.shape[0] == 0 or pert_df.shape[0] == 0):
			dge_vec, pval_vec = [], []
		elif (_method == 'cd'):
			dge_vec = chdir.chdir(join_df.iloc[:,:ctrl_df.shape[1]].as_matrix(), join_df.iloc[:,ctrl_df.shape[1]:].as_matrix(), 1).reshape((-1,))
			pval_vec = 0.01 * np.ones_like(dge_vec, dtype='float16')
		elif (_method.startswith('limma')):
			join_df.index = map(str, join_df.index) # It may cause incorrect convert from Python to R if the index is a large number
			if (_method == 'limma'):
				metric, adjust = 't', None
			elif (_method == 'limma-fdr'):
				metric, adjust = 't', 'BH'
			elif (_method == 'limma-bonferroni'):
				metric, adjust = 't', 'bonferroni'
			elif (_method == 'limma-logfc'):
				metric, adjust = 'logFC', None
			elif (_method == 'limma-logodd'):
				metric, adjust = 'B', None
			if (adjust is None):
				cachef = os.path.join(cache_path, 'limma', '%s.npz' % sid)
			else:
				cachef = os.path.join(cache_path, 'limma-%s' % adjust, '%s.npz' % sid)
			cache, df = False, join_df
			if (os.path.exists(cachef)):
				df = io.read_df(cachef, with_idx=True)
				cache = True
			dge_vec, pval_vec, dt = limma.dge(df, mask=[0]*ctrl_df.shape[1] + [1]*pert_df.shape[1], metric=metric, adjust=adjust, cache=cache)
			if (not cache):
				io.write_df(dt, cachef, with_idx=True, compress=True)
		else:
			print '%s is not implemented!' % method.upper()
			sys.exit(1)
		dge_df = pd.DataFrame(np.stack((dge_vec, pval_vec), axis=-1), index=join_df.index, columns=['statistic', 'pvalue'], dtype='float16')
		io.write_df(dge_df, dge_file, with_idx=True, compress=True)
		dge_dfs.append(dge_df)
	return dge_dfs
		
		
def sgn2dge():
	input_ext = os.path.splitext(opts.loc)[1]
	if (input_ext == '.xlsx' or input_ext == '.xls'):
		sgn_df = pd.read_excel(opts.loc)
	elif (input_ext == '.csv'):
		sgn_df = pd.read_csv(opts.loc)
	elif (input_ext == '.npz'):
		sgn_df = io.read_df(opts.loc)
	else:
		print 'Unsupported input file extension %s, please use csv or npz file!' % input_ext
		sys.exit(1)
	kwargs = {} if opts.cfg is None else ast.literal_eval(opts.cfg)
	method = kwargs.setdefault('method', 'cd')
	par_dir, basename = os.path.abspath(os.path.join(opts.loc, os.path.pardir)), os.path.splitext(os.path.basename(opts.loc))[0]
	ge_path = os.path.join(kwargs.setdefault('ge_dir', os.path.join(gsc.GEO_PATH, 'gedata')), basename)
	saved_path = os.path.join(par_dir, 'dge', method.lower(), basename) if opts.output is None else os.path.join(opts.output, method.lower(), basename)
	cache_path = os.path.join(par_dir, 'dge', 'cache', basename) if opts.output is None else os.path.join(opts.output, 'cache', basename)
	if (os.path.isdir(os.path.join(saved_path, 'filtered'))):
		print 'Filtered data exists in save path: %s\nPlease move them to the original folder!' % saved_path
		sys.exit(-1)
	elif (os.path.isdir(os.path.join(cache_path, 'filtered'))):
		print 'Filtered data exists in cache path: %s\nPlease move them to the original folder!' % cache_path
		sys.exit(-1)
	# Extract the control group and perturbation group of each gene expression signature
	if (opts.np == 1):
		_sgn2dge(sgn_df, method, ge_path, saved_path, cache_path)
	else:
		task_bnd = njobs.split_1d(sgn_df.shape[0], split_num=opts.np, ret_idx=True)
		_ = njobs.run_pool(_sgn2dge, n_jobs=opts.np, dist_param=['sgn_df'], sgn_df=[sgn_df.iloc[task_bnd[i]:task_bnd[i+1]] for i in range(opts.np)], method=method, ge_path=ge_path, saved_path=saved_path, cache_path=cache_path)
	
	
def plot_dgepval():
	kwargs = {} if opts.cfg is None else ast.literal_eval(opts.cfg)
	dge_dirs, labels, numsamp = kwargs['dges'].split(SC), kwargs['labels'].split(SC), kwargs.setdefault('numsamp', 10)
	groups = [[int(x) for x in grp.split(',')] for grp in kwargs['groups'].split(SC)] if kwargs.has_key('groups') else None
	group_labels = [x for grp in kwargs['group_labels'].split(SC) for x in grp.split(',')]
	data = []
	for dge_dir, label in zip(dge_dirs, labels):
		pvalues = []
		for fpath in fs.listf(dge_dir, pattern='dge_.*.npz', full_path=True):
			selected_pvalue = np.random.choice(io.read_df(fpath, with_idx=True)['pvalue'].tolist(), numsamp, replace=False)
			selected_pvalue[selected_pvalue <= 0] = 2
			selected_pvalue[selected_pvalue == 2] = selected_pvalue.min()
			selected_pvalue[selected_pvalue >= 1] = -2
			selected_pvalue[selected_pvalue == -2] = selected_pvalue.max()
			pvalues.append(selected_pvalue)
		data_col = np.concatenate(pvalues)
		label_col = np.repeat([label], data_col.shape[0])
		data.append(np.stack([label_col, data_col], axis=1))
	plot.plot_violin(data, xlabel='GEO Collection', ylabel='-log10(P-Value)', labels=group_labels, groups=groups, ref_lines={'y':[-np.log10(0.05)]}, plot_cfg=plot_common_cfg, log=-1, sns_inner='box', sns_bw=.3)


def _dge2udrg(sgn_dge_fpaths, sgn_df, probe_gene_map, keep_unkown_probe=False, hist_bnd=(-2, 1)):
    udr_genes = []
    for sgn_dge_fpath in sgn_dge_fpaths:
        sgn_dge = io.read_df(sgn_dge_fpath, with_idx=True)
        if (np.all(pd.isnull(sgn_dge))): continue
        # Filter out the probes that cannot be converted to gene symbols
        plfm = sgn_df['platform'].loc[sgn_dge.index[0]]
        has_plfm = probe_gene_map.has_key(plfm) and not probe_gene_map[plfm].empty
        if (has_plfm and not keep_unkown_probe):
            pgmap = probe_gene_map[plfm]
            columns = [col for col in sgn_dge.columns if col in pgmap.index and pgmap.loc[col] and not pgmap.loc[col].isspace()]
            sgn_dge = sgn_dge[columns]
        
        hist, bin_edges = zip(*[np.histogram(sgn_dge.iloc[i]) for i in range(sgn_dge.shape[0])])
        uprg = [sgn_dge.iloc[i, np.where(sgn_dge.iloc[i] >= bin_edges[i][hist_bnd[0]])[0]].sort_values(ascending=False) for i in range(sgn_dge.shape[0])]
        dwrg = [sgn_dge.iloc[i, np.where(sgn_dge.iloc[i] <= bin_edges[i][hist_bnd[1]])[0]].sort_values(ascending=True) for i in range(sgn_dge.shape[0])]
        upr_genes, dwr_genes = [x.index.tolist() for x in uprg], [x.index.tolist() for x in dwrg]
        upr_dges, dwr_dges = [x.tolist() for x in uprg], [x.tolist() for x in dwrg]

        # Map to Gene Symbol
        if (has_plfm):
            pgmap = probe_gene_map[plfm]
            upr_genes = [[[x.strip() for x in pgmap.loc[probe].split('///')] if (probe in pgmap.index) else [probe] for probe in probes] for probes in upr_genes]
            uprg_lens = [[len(x) for x in genes] for genes in upr_genes]
            upr_dges = [[[dge] * length for dge, length in zip(dges, lens)] for dges, lens in zip(upr_dges, uprg_lens)]
            upr_genes = [func.flatten_list(probes) for probes in upr_genes]
            upr_dges = [func.flatten_list(dges) for dges in upr_dges]
            dwr_genes = [[[x.strip() for x in pgmap.loc[probe].split('///')] if (probe in pgmap.index) else [probe] for probe in probes] for probes in dwr_genes]
            dwrg_lens = [[len(x) for x in genes] for genes in dwr_genes]
            dwr_dges = [[[dge] * length for dge, length in zip(dges, lens)] for dges, lens in zip(dwr_dges, dwrg_lens)]
            dwr_genes = [func.flatten_list(probes) for probes in dwr_genes]
            dwr_dges = [func.flatten_list(dges) for dges in dwr_dges]
        udr_genes.append(pd.DataFrame(OrderedDict([('Up-regulated Genes', ['|'.join(map(str, x)) for x in upr_genes]), ('Down-regulated Genes', ['|'.join(map(str, x)) for x in dwr_genes]), ('Up-regulated DGEs', ['|'.join(map(str, x)) for x in upr_dges]), ('Down-regulated DGEs', ['|'.join(map(str, x)) for x in dwr_dges])]), index=sgn_dge.index))
    return pd.concat(udr_genes, axis=0, join='inner')
		

def dge2udrg():
	input_ext = os.path.splitext(opts.loc)[1]
	if (input_ext == '.xlsx' or input_ext == '.xls'):
		sgn_df = pd.read_excel(opts.loc)
	elif (input_ext == '.csv'):
		sgn_df = pd.read_csv(opts.loc)
	elif (input_ext == '.npz'):
		sgn_df = io.read_df(opts.loc)
	else:
		print 'Unsupported input file extension %s, please use csv or npz file!' % input_ext
		sys.exit(1)
	kwargs = {} if opts.cfg is None else ast.literal_eval(opts.cfg)
	dge_dir = kwargs['dge_dir']
	pgmap_dir = kwargs['pgmap_dir']
	keep_unkown_probe = kwargs.setdefault('kup', False)
	hist_bnd = kwargs.setdefault('hist_bnd', (-2, 1))
	idx_sgn_df = sgn_df.set_index('id')
	probe_gene_map = io.read_obj(os.path.join(pgmap_dir, 'probe_gene_map.pkl'))
	sgn_dge_fpaths = fs.listf(dge_dir, pattern='dge_X_.*\.npz', full_path=True)
	
	task_bnd = njobs.split_1d(len(sgn_dge_fpaths), split_num=opts.np, ret_idx=True)
	udr_genes = njobs.run_pool(_dge2udrg, n_jobs=opts.np, dist_param=['sgn_dge_fpaths'], sgn_dge_fpaths=[sgn_dge_fpaths[task_bnd[i]:task_bnd[i+1]] for i in range(opts.np)], sgn_df=idx_sgn_df, probe_gene_map=probe_gene_map, keep_unkown_probe=keep_unkown_probe)
	new_sgn_df = pd.concat([idx_sgn_df, pd.concat(udr_genes, axis=0, join='inner')], axis=1, join_axes=[idx_sgn_df.index])
	new_sgn_fpath = '%s_udrg' % os.path.splitext(opts.loc)[0]
	new_sgn_fpath = os.path.join(opts.output, os.path.basename(new_sgn_fpath)) if os.path.exists(opts.output) else new_sgn_fpath
	io.write_df(new_sgn_df, new_sgn_fpath, with_idx=True)
	new_sgn_df.to_excel(new_sgn_fpath, encoding='utf8')


# Binary Jaccard index
def _ji(a, b):
	a_sum, b_sum = a.sum(), b.sum()
	if (a_sum == 0 and b_sum == 0): return 1
	ab_sum = (a & b).sum()
	return 1.0 * ab_sum / (a_sum + b_sum - ab_sum)
	
# Weighted Jaccard index
def _wji(a, b):
	max_sum = np.max((a,b), axis=0).sum()
	if (max_sum == 0): return 1
	min_sum = np.min((a,b), axis=0).sum()
	return 1.0 * min_sum / max_sum

# Binary Spearman's Footrule
def _sf(a, b):
	return 1 - np.abs(a - b).sum() / np.abs(2 * a - len(a) - 1).sum()
	
# Binary Kendall's Tau
def _kt(a, b):
	return 1 - np.logical_xor([m < n for m, n in itertools.combinations(a, 2)], [m < n for m, n in itertools.combinations(b, 2)]).astype('int8').sum() / sp.misc.comb(len(a), 2)

_wfunc = {'iota': np.frompyfunc(lambda x: 1, 1, 1), 'dcgw': np.frompyfunc(lambda x: np.log10(1+x)/np.power(2,x), 1, 1)}
# Weighted Spearman's Footrule
def _wsf(a, b, wfunc='iota'):
	mask = np.logical_or(a != 0, b != 0).astype('int8')
	a = len(a) - np.argsort(a).argsort()
	b = len(b) - np.argsort(b).argsort()
	return 1 - 2.0 * np.abs(a - b).sum() / (mask * _wfunc[wfunc](range(1, len(a) + 1)) * np.abs(2 * a - len(a) - 1)).sum()

# Weighted Kendall's Tau
def _wkt(a, b, wfunc='iota'):
	mask = np.logical_xor([m < n for m, n in itertools.combinations(a, 2)], [m < n for m, n in itertools.combinations(b, 2)])
	union_mask = np.array([np.logical_and((a[m] != 0 or b[m] != 0), a[n] != 0 or b[n] != 0) for m, n in itertools.combinations(range(len(a)), 2)])
	pair_idx = np.array([(m, n) for m, n in itertools.combinations(range(1, len(a) + 1), 2)])
	return 1 - 1.0 * _wfunc[wfunc](pair_idx[mask]).sum() / _wfunc[wfunc](pair_idx[union_mask]).sum()

# Signed Binary Jaccard index
def _sji(a, b):
	return (_ji(a[0], b[0]) + _ji(a[1], b[1]) - _ji(a[0], b[1]) - _ji(a[1], b[0])) / 2

# Signed Binary Spearman's Footrule
def _ssf(a, b):
	return (_sf(a[0], b[0]) + _sf(a[1], b[1]) - _sf(a[0], b[1]) - _sf(a[1], b[0])) / 2

# Signed weighted Jaccard index
def _swji(a, b):
	return (_wji(a[0], b[0]) + _wji(a[1], b[1]) - _wji(a[0], b[1]) - _wji(a[1], b[0])) / 2
	
# Signed Weighted Spearman's Footrule
def _swsf(a, b, wfunc='iota'):
	return (_wsf(a[0], b[0], wfunc=wfunc) + _wsf(a[1], b[1], wfunc=wfunc) - _wsf(a[0], b[1], wfunc=wfunc) - _wsf(a[1], b[0], wfunc=wfunc)) / 2
	
# Signed Weighted Kendall's Tau
def _swkt(a, b, wfunc='iota'):
	return (_wkt(a[0], b[0], wfunc=wfunc) + _wkt(a[1], b[1], wfunc=wfunc) - _wkt(a[0], b[1], wfunc=wfunc) - _wkt(a[1], b[0], wfunc=wfunc)) / 2

# Signed Binary Jaccard index vector-mode
def _sjiv(X, Y):
	def _ji(a, b):
		a_sum, b_sum = a.sum(), b.sum()
		if (a_sum == 0 and b_sum == 0): return 0
		ab_sum = (a & b).sum()
		return 1.0 * ab_sum / (a_sum + b_sum - ab_sum)
	def _sji(a, b):
		return (_ji(a[0], b[0]) + _ji(a[1], b[1]) - _ji(a[0], b[1]) - _ji(a[1], b[0])) / 2
	import numpy as np
	import itertools
	shape = len(X), len(Y)
	simmt = np.ones(shape)
	for i, j in itertools.product(range(shape[0]), range(shape[1])):
		simmt[i, j] = _sji(X[i], Y[j])
	return simmt

# Signed Binary Jaccard index cube-mode
def _sjic(X, Y):
	import numpy as np
	interaction = np.tensordot(X, Y, axes=[[-1],[-1]]).transpose(range(len(X.shape)-1)+range(len(X.shape)-1, len(X.shape)+len(Y.shape)-2)[::-1]) # XY' shape of (m, 2, 2, n)
	union = np.tensordot(X, np.ones((X.shape[-1], X.shape[-2])), axes=1).reshape(X.shape[:-1] + (X.shape[-2], 1)).repeat(Y.shape[0], axis=-1) + np.tensordot(Y, np.ones((Y.shape[-1], Y.shape[-2])), axes=1).reshape(Y.shape[:-1] + (Y.shape[-2], 1)).repeat(X.shape[0], axis=-1).T - interaction # XI+IY'-XY'
	r = 1.0 * interaction / union
	s = np.tensordot(np.tensordot(np.array([[1,-1]]), r, axes=[[1],[1]]).reshape((r.shape[0],)+r.shape[2:]), np.array([[1],[-1]]), axes=[[1],[0]]).reshape((X.shape[0], Y.shape[0])) # sum reduction
	return s / 2.0

# Binary Jaccard index matrix-mode
def _jim(X, Y, signed=True):
	import numpy as np
	Y_T = Y.T
	interaction = X.dot(Y_T) # XY' shape of (2m, 2n)
	# union = X.sum(axis=1).reshape((-1, 1)).repeat(Y.shape[0], axis=1) + Y_T.sum(axis=0).reshape((1, -1)).repeat(X.shape[0], axis=0) - interaction
	union = X.dot(np.ones((X.shape[1]), dtype='int8')).reshape((-1, 1)).repeat(Y.shape[0], axis=1) + np.ones((Y_T.shape[0]), dtype='int8').dot(Y_T).reshape((1, -1)).repeat(X.shape[0], axis=0) - interaction # XI+IY'-XY', dot can be parallelized but not sum
	r = 1.0 * interaction / union
	r = r.reshape((r.shape[0]/2, 2, r.shape[1]/2, 2))
	# Sum reduction
	if (signed):
		s = np.tensordot(np.tensordot(np.array([[1,-1]]), r, axes=[[1],[1]]).reshape((r.shape[0],)+r.shape[2:]), np.array([[1],[-1]]), axes=[[2],[0]]).reshape((X.shape[0]/2, Y.shape[0]/2))
	else:
		s = np.tensordot(np.tensordot(np.array([[1,0]]), r, axes=[[1],[1]]).reshape((r.shape[0],)+r.shape[2:]), np.array([[1],[0]]), axes=[[2],[0]]).reshape((X.shape[0]/2, Y.shape[0]/2))
	return s / 2.0

# Binary Jaccard index matrix-mode true reference
def _jim_t(X, Y, signed=True):
	shape = X.shape[0] / 2, Y.shape[0] / 2
	simmt = np.ones(shape)
	for i, j in itertools.product(range(shape[0]), range(shape[1])):
		simmt[i, j] = _sji([X[2*i], X[2*i+1]], [Y[2*j], Y[2*j+1]])
	return simmt

# Weighted Jaccard index matrix-mode
def _wjim(X, Y, signed=True):
	import numpy as np
	_X = X.reshape((X.shape[0], 1, X.shape[1])).repeat(Y.shape[0], axis=1)
	_Y = Y.reshape((1, Y.shape[0], Y.shape[1])).repeat(X.shape[0], axis=0)
	min_tensor = _X * (_X <= _Y).astype('int8') + _Y * (_Y < _X).astype('int8')
	interaction = np.tensordot(min_tensor, np.ones(X.shape[1]), axes=[[2],[0]]).reshape((X.shape[0], Y.shape[0])) # SUM<0:k>[min(X_i,Y_i)]
	del min_tensor
	max_tensor = _X * (_X >= _Y).astype('int8') + _Y * (_Y > _X).astype('int8')
	del _X, _Y
	union = np.tensordot(max_tensor, np.ones(X.shape[1]), axes=[[2],[0]]).reshape((X.shape[0], Y.shape[0])) # SUM<0:k>[max(X_i,Y_i)]
	del max_tensor
	r = 1.0 * interaction / union
	r = r.reshape((r.shape[0]/2, 2, r.shape[1]/2, 2))
	del interaction, union
	# Sum reduction
	if (signed):
		s = np.tensordot(np.tensordot(np.array([[1,-1]]), r, axes=[[1],[1]]).reshape((r.shape[0],)+r.shape[2:]), np.array([[1],[-1]]), axes=[[2],[0]]).reshape((X.shape[0]/2, Y.shape[0]/2))
	else:
		s = np.tensordot(np.tensordot(np.array([[1,0]]), r, axes=[[1],[1]]).reshape((r.shape[0],)+r.shape[2:]), np.array([[1],[0]]), axes=[[2],[0]]).reshape((X.shape[0]/2, Y.shape[0]/2))
	return s / 2.0

# Weighted Jaccard index matrix-mode true reference
def _wjim_t(X, Y, signed=True):
	shape = X.shape[0] / 2, Y.shape[0] / 2
	simmt = np.ones(shape)
	for i, j in itertools.product(range(shape[0]), range(shape[1])):
		simmt[i, j] = _swji([X[2*i], X[2*i+1]], [Y[2*j], Y[2*j+1]])
	return simmt
	
# Weighted Spearman's Footrule matrix-mode
def _wsfm(X, Y, signed=True, wfunc='iota'):
	import numpy as np
	_wfunc = {'iota': np.frompyfunc(lambda x: 1, 1, 1), 'dcgw': np.frompyfunc(lambda x: np.log10(1+x)/np.power(2,x), 1, 1)}
	X_mask = (X != 0)
	Y_mask = (Y != 0)
	X = X.shape[1] - X.argsort(axis=1).argsort(axis=1)
	Y = Y.shape[1] - Y.argsort(axis=1).argsort(axis=1)
	_X = X.reshape((X.shape[0], 1, X.shape[1])).repeat(Y.shape[0], axis=1)
	_Y = Y.reshape((1, Y.shape[0], Y.shape[1])).repeat(X.shape[0], axis=0)
	_X_mask = X_mask.reshape((X_mask.shape[0], 1, X_mask.shape[1])).repeat(Y_mask.shape[0], axis=1)
	_Y_mask = Y_mask.reshape((1, Y_mask.shape[0], Y_mask.shape[1])).repeat(X_mask.shape[0], axis=0)
	union_mask = np.logical_or(_X_mask != 0, _Y_mask != 0).astype('int8')
	del X_mask, Y_mask, _X_mask, _Y_mask
	_order_idx = np.arange(1, X.shape[1] + 1).reshape((1, 1, -1)).repeat(X.shape[0], axis=0).repeat(Y.shape[0], axis=1)
	weights = _wfunc[wfunc](_order_idx)
	sf = np.tensordot(union_mask * weights * np.abs(_X - _Y), np.ones(X.shape[1]), axes=[[2],[0]]).reshape((X.shape[0], Y.shape[0]))
	del _X, _Y
	norm = np.tensordot(union_mask * weights * np.abs(2 * _order_idx - X.shape[1] - 1), np.ones(X.shape[1]), axes=[[2],[0]]).reshape((X.shape[0], Y.shape[0]))
	del union_mask, _order_idx, weights
	no_overlap = (norm == 0)
	sf[no_overlap] = 1.0
	norm[no_overlap] = 2.0
	r = (1.0 - sf / norm).astype('float32')
	r = r.reshape((r.shape[0]/2, 2, r.shape[1]/2, 2))
	del sf, norm, no_overlap
	# Sum reduction
	if (signed):
		s = np.tensordot(np.tensordot(np.array([[1,-1]]), r, axes=[[1],[1]]).reshape((r.shape[0],)+r.shape[2:]), np.array([[1],[-1]]), axes=[[2],[0]]).reshape((X.shape[0]/2, Y.shape[0]/2))
	else:
		s = np.tensordot(np.tensordot(np.array([[1,0]]), r, axes=[[1],[1]]).reshape((r.shape[0],)+r.shape[2:]), np.array([[1],[0]]), axes=[[2],[0]]).reshape((X.shape[0]/2, Y.shape[0]/2))
	return s / 2.0

# Weighted Spearman's Footrule matrix-mode true reference
def _wsfm_t(X, Y, signed=True, wfunc='iota'):
	shape = X.shape[0] / 2, Y.shape[0] / 2
	simmt = np.ones(shape)
	for i, j in itertools.product(range(shape[0]), range(shape[1])):
		simmt[i, j] = _swsf([X[2*i], X[2*i+1]], [Y[2*j], Y[2*j+1]], wfunc=wfunc)
	return simmt
	
# Weighted Kendall's Tau matrix-mode
def _wktm(X, Y, signed=True, wfunc='iota'):
	import numpy as np
	_wfunc = {'iota': np.frompyfunc(lambda x: 1, 1, 1), 'dcgw': np.frompyfunc(lambda x: np.log10(1+x)/np.power(2,x), 1, 1)}
	def cmp_comb(x):
		return [m < n for m, n in itertools.combinations(x, 2)]
	X_cmpcomb = np.apply_along_axis(cmp_comb, 1, X)
	Y_cmpcomb = np.apply_along_axis(cmp_comb, 1, Y)
	_X_cmpcomb = X_cmpcomb.reshape((X_cmpcomb.shape[0], 1, X_cmpcomb.shape[1])).repeat(Y_cmpcomb.shape[0], axis=1)
	_Y_cmpcomb = Y_cmpcomb.reshape((1, Y_cmpcomb.shape[0], Y_cmpcomb.shape[1])).repeat(X_cmpcomb.shape[0], axis=0)
	_mask = np.logical_xor(_X_cmpcomb, _Y_cmpcomb)
	mask = _mask.reshape(_mask.shape+(1,)).repeat(2, axis=-1).astype('int8')
	del X_cmpcomb, Y_cmpcomb, _X_cmpcomb, _Y_cmpcomb, _mask
	X_mask = (X != 0)
	Y_mask = (Y != 0)
	_X_mask = X_mask.reshape((X_mask.shape[0], 1, X_mask.shape[1])).repeat(Y_mask.shape[0], axis=1)
	_Y_mask = Y_mask.reshape((1, Y_mask.shape[0], Y_mask.shape[1])).repeat(X_mask.shape[0], axis=0)
	__union_mask = np.logical_or(_X_mask != 0, _Y_mask != 0)
	del X_mask, Y_mask, _X_mask, _Y_mask
	def union_comb(x):
		return [m and n for m, n in itertools.combinations(x, 2)]
	_union_mask = np.apply_along_axis(union_comb, 2, __union_mask)
	union_mask = _union_mask.reshape(_union_mask.shape+(1,)).repeat(2, axis=-1).astype('int8')
	del __union_mask, _union_mask
	pair_idx = np.array([(m, n) for m, n in itertools.combinations(range(1, X.shape[1] + 1), 2)]).reshape((1, 1, -1, 2)).repeat(X.shape[0], axis=0).repeat(Y.shape[0], axis=1)
	weights = _wfunc[wfunc](pair_idx)
	del pair_idx
	r = (1 - 1.0 * np.tensordot(mask * weights, np.ones((union_mask.shape[2],2)), axes=[[2,3],[0,1]]) / np.tensordot(union_mask * weights, np.ones((union_mask.shape[2],2)), axes=[[2,3],[0,1]])).astype('float32')
	r = r.reshape((r.shape[0]/2, 2, r.shape[1]/2, 2))
	del mask, weights, union_mask
	# Sum reduction
	if (signed):
		s = np.tensordot(np.tensordot(np.array([[1,-1]]), r, axes=[[1],[1]]).reshape((r.shape[0],)+r.shape[2:]), np.array([[1],[-1]]), axes=[[2],[0]]).reshape((X.shape[0]/2, Y.shape[0]/2))
	else:
		s = np.tensordot(np.tensordot(np.array([[1,0]]), r, axes=[[1],[1]]).reshape((r.shape[0],)+r.shape[2:]), np.array([[1],[0]]), axes=[[2],[0]]).reshape((X.shape[0]/2, Y.shape[0]/2))
	return s / 2.0

# Weighted Kendall's Tau matrix-mode true reference
def _wktm_t(X, Y, signed=True, wfunc='iota'):
	shape = X.shape[0] / 2, Y.shape[0] / 2
	simmt = np.ones(shape)
	for i, j in itertools.product(range(shape[0]), range(shape[1])):
		simmt[i, j] = _swkt([X[2*i], X[2*i+1]], [Y[2*j], Y[2*j+1]], wfunc=wfunc)
	return simmt

# 1D parallelism
def _iter_1d(X, Y, method, signed=True, ramsize=1, task_size_f=lambda X: 1024**2, **kwargs):
	transposed = X.shape[0] < Y.shape[0]
	X, Y = (Y, X) if transposed else (X, Y)
	splitted_tasks = njobs.split_1d(Y.shape[0]/2, task_size=task_size, split_size=ramsize*1024**3)
	st_str = '[%s]' % ', '.join(['*'.join(map(str, tpl)) for tpl in func.sorted_dict(dict(zip(*np.unique(splitted_tasks, return_counts=True))))[::-1]])
	io.inst_print('The tasks are divided into the grid of size: %s' % st_str)
	task_results, task_idx = [], np.cumsum([0]+splitted_tasks)
	for i in range(len(splitted_tasks)):
		sub_Y = Y[2*task_idx[i]:2*task_idx[i+1]]
		task_results.append(method(X, sub_Y, signed=signed, **kwargs))
	s = np.concatenate(task_results, axis=-1)
	s = s.T if transposed else s
	return s

# 2D parallelism
def _iter_2d(X, Y, method, signed=True, ramsize=1, task_size_f=lambda X: 1024**2, cache_path=None, **kwargs):
	shape = (X.shape[0]/2, Y.shape[0]/2)
	splitted_tasks = njobs.split_2d(shape, task_size=task_size, split_size=ramsize*1024**3)
	st_str = '[[%s], [%s]]' % (', '.join(['*'.join(map(str, tpl)) for tpl in func.sorted_dict(dict(zip(*np.unique(splitted_tasks[0], return_counts=True))))[::-1]]), ', '.join(['*'.join(map(str, tpl)) for tpl in func.sorted_dict(dict(zip(*np.unique(splitted_tasks[0], return_counts=True))))[::-1]]))
	io.inst_print('The tasks are divided into the grid of size: %s' % st_str)
	task_results, task_idx = [], [np.cumsum([0]+splitted_tasks[0]), np.cumsum([0]+splitted_tasks[1])]
	for i in range(len(splitted_tasks[0])):
		sub_X = X[2*task_idx[0][i]:2*task_idx[0][i+1]]
		subtask_results = []
		for j in range(len(splitted_tasks[1])):
			sub_Y = Y[2*task_idx[1][j]:2*task_idx[1][j+1]]
			subtask_results.append(method(sub_X, sub_Y, signed=signed, **kwargs))
		task_results.append(np.concatenate(subtask_results, axis=-1))
	return np.concatenate(task_results, axis=0).reshape(shape)

# 2D parallelism with multi-processing
def _iter_2d_mltp(X, Y, method, signed=True, ramsize=1, task_size_f=lambda X: 1024**2, n_jobs=1, ipp_profile='', cache_path='', **kwargs):
	shape = (X.shape[0]/2, Y.shape[0]/2)
	splitted_tasks = njobs.split_2d(shape, task_size=task_size_f(X), split_size=ramsize/n_jobs*1024**3)
	st_str = '[[%s], [%s]]' % (', '.join(['*'.join(map(str, tpl)) for tpl in func.sorted_dict(dict(zip(*np.unique(splitted_tasks[0], return_counts=True))))[::-1]]), ', '.join(['*'.join(map(str, tpl)) for tpl in func.sorted_dict(dict(zip(*np.unique(splitted_tasks[0], return_counts=True))))[::-1]]))
	io.inst_print('The tasks are divided into the grid of size: %s' % st_str)
	pool, ipp_client, use_cache, use_ipp, task_results, task_idx = None, ipp_profile, not cache_path.isspace() and os.path.isdir(cache_path), ipp_profile and not ipp_profile.isspace(), [], [np.cumsum([0]+splitted_tasks[0]), np.cumsum([0]+splitted_tasks[1])]
	for i in range(len(splitted_tasks[0])):
		cache_f = os.path.join(cache_path, 'task_%i.npz' % i)
		if (use_cache and os.path.exists(cache_f)):
			task_results.append(io.read_spmt(cache_f, sparse_fmt='csr'))
		else:
			sub_X = X[2*task_idx[0][i]:2*task_idx[0][i+1]]
			if (use_ipp):
				subtask_results, ipp_client = njobs.run_ipp(method, n_jobs=n_jobs, client=ipp_client, ret_client=True, dist_param=['Y'], X=sub_X, Y=[Y[2*task_idx[1][j]:2*task_idx[1][j+1]] for j in range(len(splitted_tasks[1]))], signed=signed, **kwargs)
			else:
				subtask_results, pool = njobs.run_pool(method, n_jobs=n_jobs, pool=pool, ret_pool=True, dist_param=['Y'], X=sub_X, Y=[Y[2*task_idx[1][j]:2*task_idx[1][j+1]] for j in range(len(splitted_tasks[1]))], signed=signed, **kwargs)
			res_spmt = sp.sparse.csr_matrix(np.concatenate(subtask_results, axis=-1))
			if (use_cache): io.write_spmt(res_spmt, cache_f, sparse_fmt='csr', compress=True)
			task_results.append(res_spmt)
			del sub_X, subtask_results, res_spmt
		io.inst_print('Completed %.1f%% of the tasks' % (100.0*(i+1)/len(splitted_tasks[0])))
	if (use_ipp):
		njobs.run_ipp(None, client=ipp_client, ret_client=False)
	else:
		njobs.run_pool(None, pool=pool, ret_pool=False)
	return sp.sparse.vstack(task_results).toarray().reshape(shape)
	
_sim_method = {'ji':_jim, 'wji':_wjim, 'wsf':_wsfm, 'wkt':_wktm}
BYTE_OF_FLOAT32, DIM_OF_SGN = 4, 2
_task_size_1d = {'ji':lambda X: BYTE_OF_FLOAT32*(3*DIM_OF_SGN*X.shape[0]*X.shape[1]), 'wji':lambda X: BYTE_OF_FLOAT32*(6*DIM_OF_SGN*X.shape[0]*X.shape[1]), 'wsf':lambda X: BYTE_OF_FLOAT32*(12*DIM_OF_SGN*X.shape[0]*X.shape[1]), 'wkt':lambda X: BYTE_OF_FLOAT32*(12*DIM_OF_SGN*X.shape[0]*X.shape[1]+12*DIM_OF_SGN*X.shape[0]*sp.misc.comb(X.shape[1], 2))}
_task_size_2d = {'ji':lambda X: BYTE_OF_FLOAT32*(3*DIM_OF_SGN*DIM_OF_SGN*X.shape[1]), 'wji':lambda X: BYTE_OF_FLOAT32*(6*DIM_OF_SGN*DIM_OF_SGN*X.shape[1]), 'wsf':lambda X: BYTE_OF_FLOAT32*(12*DIM_OF_SGN*DIM_OF_SGN*X.shape[1]), 'wkt':lambda X: BYTE_OF_FLOAT32*(12*DIM_OF_SGN*DIM_OF_SGN*X.shape[1]+12*DIM_OF_SGN*DIM_OF_SGN*sp.misc.comb(X.shape[1], 2))}

def dge2simmt(**kw_args):
	# from sklearn.externals.joblib import Parallel, delayed
	from sklearn.metrics import pairwise
	from sklearn.preprocessing import MultiLabelBinarizer
	kwargs = kw_args if len(kw_args) > 0 else ({} if opts.cfg is None else ast.literal_eval(opts.cfg))
	if (type(kwargs) is str): kwargs = ast.literal_eval(kwargs)
	locs = (kwargs['loc'] if (kwargs.has_key('loc')) else opts.loc).split(SC)
	sim_method = kwargs.setdefault('sim_method', 'ji')
	method = kwargs.setdefault('method', 'cd')
	_method = method.lower()
	basenames, input_exts = zip(*[os.path.splitext(os.path.basename(loc)) for loc in locs])
	if (input_exts[0] == '.xlsx'):
		sgn_dfs = [pd.read_excel(loc) for loc in locs]
	elif (input_exts[0] == '.csv'):
		sgn_dfs = [pd.read_csv(loc) for loc in locs]
	elif (input_exts[0] == '.npz'):
		sgn_dfs = [io.read_df(loc) for loc in locs]
	dge_dir = kwargs.setdefault('dge_dir', os.path.join(gsc.GEO_PATH, 'dge'))
	output_dir = kwargs['output'] if (kwargs.has_key('output')) else opts.output
	simmt_file = os.path.join(opts.cache if output_dir is None else output_dir, 'simmt.npz')
	idx_cols = kwargs.setdefault('idx_cols', 'disease_name;;drug_name;;gene_symbol').split(SC)
	cache_f = os.path.join(opts.cache, 'udgene.pkl')
	signed = True if (int(kwargs.setdefault('signed', 1)) == 1) else False
	weighted = True if (int(kwargs.setdefault('weighted', 1)) == 1) else False
	if (weighted): assert(sim_method.startswith('w'))
	sim_kwargs = {}
	if ('sf' in sim_method): 
		sim_kwargs['wfunc'] = kwargs.setdefault('wfunc', 'iota')
	# Read the data
	print 'Differentially expressed genes are calculated by %s...' % method
	if (os.path.exists(cache_f)):
		io.inst_print('Reading cache...')
		ids, id_bndry = io.read_obj(cache_f)
		if (not os.path.exists(simmt_file)):
			udgene_spmt = io.read_spmt(os.path.splitext(cache_f)[0]+'.npz')
			if (weighted):
				pvalue_spmt = io.read_spmt(os.path.splitext(cache_f)[0]+'_pval.npz')
	else:
		io.inst_print('Preparing data...')
		ids, id_bndry, udgene, pvaldict = [[] for i in range(4)]
		# Read all the differentially expressed genes vector of each collection
		for basename, sgn_df, idx_col in zip(basenames, sgn_dfs, idx_cols):
			dge_path = os.path.join(dge_dir, _method, basename)
			if (os.path.isdir(os.path.join(dge_path, 'filtered'))):
				print 'Filtered data exists in dge path: %s\nPlease move them to the original folder!' % dge_path
				exists(-1)
			sgn_ids = sgn_df['id'].tolist()
			# sgn_ids = sgn_df[idx_col].tolist() # customized identity for each signature
			ids.extend(sgn_ids)
			id_bndry.append(len(sgn_ids)) # append number of signatures to form the boundaries
			for i in xrange(len(sgn_ids)):
				dge_df = io.read_df(os.path.join(dge_path, 'dge_%i.npz' % i), with_idx=True)
				if (hasattr(dge_df, 'pvalue')):
					thrshd = kwargs.setdefault('thrshd', opts.thrshd)
					dge_df.drop(dge_df.index[np.where(dge_df['pvalue'] > (thrshd if (type(thrshd) is float) else 0.05))[0]], axis=0, inplace=True)
				else:
					dge_df['pvalue'] = pd.Series(0.05 * np.ones(dge_df.shape[0]), index=dge_df.index)
				udgene.append((set(dge_df.index[np.where(dge_df.iloc[:,0] > 0)[0]]), set(dge_df.index[np.where(dge_df.iloc[:,0] < 0)[0]])))
				if (weighted):
					pvaldict.append((dict([(gene, dge_df['pvalue'][gene]) for gene in udgene[-1][0]]), dict([(gene, dge_df['pvalue'][gene]) for gene in udgene[-1][1]])))
		unrolled_udgene = func.flatten_list(udgene)
		if (weighted):
			unrolled_pvaldict = func.flatten_list(pvaldict)
		# Transform the up-down regulate gene expression data into binary matrix
		mlb = MultiLabelBinarizer(sparse_output=True)
		udgene_spmt = mlb.fit_transform(unrolled_udgene)
		if (not sp.sparse.isspmatrix_csr(udgene_spmt)):
			udgene_spmt = udgene_spmt.tocsr()
		# Construct the corresponding pvalue matrix
		if (weighted):
			pval_data = []
			for i, offset in enumerate(udgene_spmt.indptr[:-1]):
				cum_num = udgene_spmt.indptr[i + 1]
				pval_data.extend([unrolled_pvaldict[i][mlb.classes_[j]] for j in udgene_spmt.indices[offset:cum_num]])
			pvalue_spmt = sp.sparse.csr_matrix((pval_data, udgene_spmt.indices, udgene_spmt.indptr), shape=udgene_spmt.shape)
			io.write_spmt(pvalue_spmt, os.path.splitext(cache_f)[0]+'_pval.npz', sparse_fmt='csr', compress=True)
		io.write_spmt(udgene_spmt, os.path.splitext(cache_f)[0]+'.npz', sparse_fmt='csr', compress=True)
		id_bndry = np.cumsum([0] + id_bndry).tolist()
		io.write_obj([ids, id_bndry], cache_f)
	if (os.path.exists(simmt_file)):
		io.inst_print('Reading similarity matrix...')
		simmt = io.read_df(simmt_file, with_idx=True, sparse_fmt=opts.spfmt)
	else:
		# Calculate the global similarity matrix across all the collections
		io.inst_print('Calculating the %s similarity matrix using %s...' % ('signed' if signed else 'unsigned', sim_method))
		# Serial method
		# simmt = pd.DataFrame(np.ones((len(ids), len(ids))), index=ids, columns=ids)
		# for i, j in itertools.combinations(range(len(ids)), 2):
			# similarity = _sji(udgene[i], udgene[j])
			# simmt.iloc[i, j] = similarity
			# simmt.iloc[j, i] = similarity
		
		# Deal with weighted data
		if (weighted):
			udgene_mt = sp.sparse.csr_matrix(((1 - pvalue_spmt.data) * udgene_spmt.data, udgene_spmt.indices, udgene_spmt.indptr), shape=udgene_spmt.shape).astype('float32').toarray()
			del pvalue_spmt
		else:
			udgene_spmt = udgene_spmt.astype('float32') # Numpy only support parallelism for float32/64
			udgene_mt = udgene_spmt.toarray()
		del udgene_spmt
		
		if (opts.np > 1):
			# Multi-processing method
			# similarity = dstclc.parallel_pairwise(udgene_mt, None, _sim_method[sim_method], n_jobs=opts.np, min_chunksize=2)
			similarity = _iter_2d_mltp(udgene_mt, udgene_mt, _sim_method[sim_method], signed=signed, ramsize=RAMSIZE, task_size_f=_task_size_2d[sim_method], n_jobs=opts.np, ipp_profile=kwargs.setdefault('ipp', opts.ipp), cache_path=opts.cache, **sim_kwargs)
		else:
			# Non-multi-processing method
			similarity = _sim_method[sim_method](udgene_mt, udgene_mt, signed=signed, **sim_kwargs)

		# Tensor data structure
		# udgene_cube = udgene_mt.reshape((-1, 2, udgene_mt.shape[1]))
		# similarity = _sjic(udgene_cube, udgene_cube)
		np.fill_diagonal(similarity, 1)
		simmt = pd.DataFrame(similarity, index=ids, columns=ids, dtype=similarity.dtype)
		io.write_df(simmt, simmt_file, with_idx=True, sparse_fmt=opts.spfmt, compress=True)
	# Calculate the similarity matrix within each collection
	io.inst_print('Splitting the similarity matrix...')
	for k in xrange(len(sgn_dfs)):
		idx_pair = (id_bndry[k], id_bndry[k + 1])
		sub_simmt = simmt.iloc[idx_pair[0]:idx_pair[1],idx_pair[0]:idx_pair[1]]
		fpath = os.path.splitext(simmt_file)
		io.write_df(sub_simmt, fpath[0] + '_%i' % k + fpath[1], with_idx=True, sparse_fmt=opts.spfmt, compress=True)

		
def simhrc():
	sys.setrecursionlimit(10000)
	simmt = io.read_df(opts.loc, with_idx=True)
	plot.plot_clt_hrc(simmt.as_matrix(), dist_metric='precomputed', fname=os.path.splitext(os.path.basename(opts.loc))[0], plot_cfg=plot_common_cfg)
	
## Single collection of signatures to similarity matrix
def onto2simmt():
	from scipy.sparse import coo_matrix
	def filter(txt_list):
		new_list = []
		for txt in txt_list:
			if (len(txt) <= 30):
				new_list.append(txt)
		return set(new_list)
	input_ext = os.path.splitext(opts.loc)[1]
	if (input_ext == '.xlsx' or input_ext == '.xls'):
		sgn_df = pd.read_excel(opts.loc)
	elif (input_ext == '.csv'):
		sgn_df = pd.read_csv(opts.loc)
	elif (input_ext == '.npz'):
		sgn_df = io.read_df(opts.loc)
	else:
		print 'Unsupported input file extension %s, please use csv or npz file!' % input_ext
		sys.exit(1)
	kwargs = {} if opts.cfg is None else ast.literal_eval(opts.cfg)
	col_name, db_name, closure_kwargs = kwargs['col_name'], kwargs['db_name'], dict([('max_length', kwargs.setdefault('max_length', 3))])
	noid, lang = gsc.DB2ATTR[db_name]['noid'], gsc.DB2LANG[db_name]
	ontog = sparql.SPARQL('http://localhost:8890/%s/query' % db_name, use_cache=common_cfg.setdefault('memcache', False), timeout=opts.timeout)
	fn_func = ontology.define_fn(ontog, type='exact', has_id=not noid, lang=lang, prdns=[(k, getattr(ontology, v)) for k, v in gsc.DB2INTPRDS[db_name]], eqprds={})
	distmt, vname = ontology.transitive_closure_dsg(ontog, sgn_df[col_name].tolist(), find_neighbors=fn_func, filter=filter, **closure_kwargs)
	if (distmt.shape[1] == 0):
		print 'Could not find any neighbors using exact matching. Using fuzzy matching instead...'
		fn_func = ontology.define_fn(ontog, type='fuzzy', has_id=not noid, lang=lang, prdns=[(k, getattr(ontology, v)) for k, v in gsc.DB2INTPRDS[db_name]], eqprds={})
		distmt, vname = ontology.transitive_closure_dsg(ontog, sgn_df[col_name].tolist(), find_neighbors=fn_func, filter=filter, **closure_kwargs)
	simmt = coo_matrix((1-dstclc.normdist(distmt.data.astype('float32')), (distmt.row, distmt.col)), shape=distmt.shape)
	simmt.setdiag(1)
	sim_df = pd.DataFrame(simmt.toarray(), index=vname, columns=vname)
	io.write_df(sim_df, 'simmt_%s_%s.npz' % (col_name, db_name), with_idx=True, sparse_fmt=opts.spfmt, compress=True)


## Two collections of signatures within one ontology database to similarity matrix
def onto22simmt():
	from scipy.sparse import coo_matrix
	kwargs = {} if opts.cfg is None else ast.literal_eval(opts.cfg)
	fnames = kwargs['fnames'].split(SC)
	input_ext = os.path.splitext(fnames[0])[1]
	if (input_ext == '.xlsx' or input_ext == '.xls'):
		sgn_dfs = [pd.read_excel(os.path.join(opts.loc, fname)) for fname in fnames]
	elif (input_ext == '.csv'):
		sgn_dfs = [pd.read_csv(os.path.join(opts.loc, fname)) for fname in fnames]
	elif (input_ext == '.npz'):
		sgn_dfs = [io.read_df(os.path.join(opts.loc, fname)) for fname in fnames]
	else:
		print 'Unsupported input file extension %s, please use csv or npz file!' % input_ext
		sys.exit(1)
	col_names, db_name, closure_kwargs = kwargs['col_names'].split(SC), kwargs['db_name'], dict([('max_length', kwargs.setdefault('max_length', 3))])
	# Make sure that the column name in the dataframe contains the keywords of the identifier name space
	idnss = [[(idns[0], getattr(ontology, idns[1])) for k, v in gsc.DB2IDNS[db_name].iteritems() if k in col_name for idns in v ][0] for col_name in col_names]
	noid, lang, filt_func, clean_func = gsc.DB2ATTR[db_name]['noid'], gsc.DB2LANG[db_name], ontology.filter_result(db_name), ontology.clean_result(db_name)
	full_items = [['%s%s'%(idns[1],ontology.replace_invalid_sparql_str(item)) for item in df[col_name]] for df, col_name, idns in zip(sgn_dfs, col_names, idnss)] if noid else [df[col_name].tolist() for df, col_name in zip(sgn_dfs, col_names)]
	ontog = sparql.SPARQL('http://localhost:8890/%s/query' % db_name, use_cache=common_cfg.setdefault('memcache', False), timeout=opts.timeout)
	fn_func = ontology.define_fn(ontog, type='exact', has_id=not noid, lang=lang, idns=idnss, prdns=[(k, getattr(ontology, v)) for k, v in gsc.DB2INTPRDS[db_name]], eqprds={})
	distmt, vname = ontology.transitive_closure_dsg(ontog, full_items[0]+full_items[1], find_neighbors=fn_func, filter=filt_func, cleaner=clean_func, **closure_kwargs)
	if (distmt.shape[1] == 0):
		print 'Could not find any neighbors using exact matching. Using fuzzy matching instead...'
		fn_func = ontology.define_fn(ontog, type='fuzzy', has_id=not noid, lang=lang, idns=idnss, prdns=[(k, getattr(ontology, v)) for k, v in gsc.DB2INTPRDS[db_name]], eqprds={})
		distmt, vname = ontology.transitive_closure_dsg(ontog, full_items[0]+full_items[1], find_neighbors=fn_func, filter=filt_func, cleaner=clean_func, **closure_kwargs)
	simmt = coo_matrix((1-dstclc.normdist(distmt.data.astype('float32')), (distmt.row, distmt.col)), shape=distmt.shape)
	simmt.setdiag(1)
	vname = [x.lstrip(idnss[0][1]).lstrip(idnss[1][1]) for x in vname] if noid else vname
	sim_df = dict(values=simmt, shape=simmt.shape, index=vname, columns=vname)
	io.write_spdf(sim_df, 'simmt_%s_%s.npz' % ('-'.join(col_names), db_name), with_idx=True, sparse_fmt=opts.spfmt, compress=True)


## Two collections of signatures across two ontology databases to similarity matrix
def ontoc2simmt():
	locs = opts.loc.split(SC)
	basenames, input_exts = zip(*[os.path.splitext(os.path.basename(loc)) for loc in locs])
	simmts = [io.read_spdf(loc, with_idx=True, sparse_fmt=opts.spfmt) for loc in locs]
	columns = [simmt['columns'] for simmt in simmts]
	olcol = list(set.intersection(*[set(col) for col in columns]))
	olcol_idx = [pd.Series(range(len(col)), index=col).loc[olcol].tolist() for col in columns]
	olsimmts = [simmt['values'][:,ol_idx] for simmt, ol_idx in zip(simmts, olcol_idx)]
	infer_simmt = np.dot(olsimmts[0].toarray(), olsimmts[1].toarray().T)
	infer_simmt[infer_simmt > 1] = 1
	infer_simmt[infer_simmt < -1] = -1
	sim_df = dict(values=infer_simmt, shape=infer_simmt.shape, index=simmts[0]['index'], columns=simmts[1]['index'])
	fname = 'simmt_%s.npz' % '-'.join([bn.split('-')[0].strip('simmt_') for bn in basenames])
	io.write_spdf(sim_df, fname, with_idx=True, sparse_fmt=opts.spfmt, compress=True)
	

def ddi2simmt():
	from bionlp.spider import rxnav
	from scipy.sparse import coo_matrix
	kwargs = {} if opts.cfg is None else ast.literal_eval(opts.cfg)
	col_name = kwargs['col_name']
	cache_path = os.path.join(opts.cache, 'drug_intrcts.pkl')
	drug_cache_path = os.path.join(gsc.RXNAV_PATH, 'drug')
	intr_cache_path = os.path.join(gsc.RXNAV_PATH, 'interaction')
	input_ext = os.path.splitext(opts.loc)[1]
	if (input_ext == '.xlsx' or input_ext == '.xls'):
		sgn_df = pd.read_excel(opts.loc)
	elif (input_ext == '.csv'):
		sgn_df = pd.read_csv(opts.loc)
	elif (input_ext == '.npz'):
		sgn_df = io.read_df(opts.loc)
	else:
		print 'Unsupported input file extension %s, please use csv or npz file!' % input_ext
		sys.exit(1)
	drug_list = sgn_df[col_name].tolist()
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
	simmt.setdiag(1)
	sim_df = pd.DataFrame(simmt.toarray(), index=drugs, columns=drugs)
	io.write_df(sim_df, 'simmt_drug_%s.npz' % col_name, with_idx=True, sparse_fmt=opts.spfmt, compress=True)

	
def ppi2simmt():
	from bionlp.spider import biogrid
	from scipy.sparse import coo_matrix
	kwargs = {} if opts.cfg is None else ast.literal_eval(opts.cfg)
	col_name = kwargs['col_name']
	cache_path = os.path.join(opts.cache, 'gene_intrcts.pkl'), os.path.join(opts.cache, 'opt_gene_intrcts.pkl')
	intr_cache_path = os.path.join(gsc.BIOGRID_PATH, 'interaction')
	input_ext = os.path.splitext(opts.loc)[1]
	if (input_ext == '.xlsx' or input_ext == '.xls'):
		sgn_df = pd.read_excel(opts.loc)
	elif (input_ext == '.csv'):
		sgn_df = pd.read_csv(opts.loc)
	elif (input_ext == '.npz'):
		sgn_df = io.read_df(opts.loc)
	else:
		print 'Unsupported input file extension %s, please use csv or npz file!' % input_ext
		sys.exit(1)
	gene_list = sgn_df[col_name].tolist()
	if (os.path.exists(cache_path[0])):
		interactions = io.read_obj(cache_path[0])
		if (os.path.exists(cache_path[1])):
			opt_intrct = io.read_obj(cache_path[1])
		else:
			opt_intrct = []
	else:
		interactions, opt_intrct = [], []
		# Construct the client
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
				isymbol = list(set([ipair['OFFICIAL_SYMBOL_A'].lower()] + ipair['SYNONYMS_A'].lower().split('|'))), list(set([ipair['OFFICIAL_SYMBOL_B'].lower()] + ipair['SYNONYMS_B'].lower().split('|')))
				l_gene = gene.lower()
				# If the gene is an official symbol then directly add the interaction tems, otherwise add the relation between this synonym and the official symbol
				try:
					idx = isymbol[0].index(l_gene)
					if (idx != 0):
						opt_intrct.append((gene, ipair['OFFICIAL_SYMBOL_A']))
					intrct_concepts.append(ipair['OFFICIAL_SYMBOL_B'])
				except ValueError as e:
					try:
						idx = isymbol[1].index(l_gene)
						if (idx != 0):
							opt_intrct.append((gene, ipair['OFFICIAL_SYMBOL_B']))
						intrct_concepts.append(ipair['OFFICIAL_SYMBOL_A'])
					except Exception as e:
						pass
				# if (l_gene in isymbol[0]):
					# intrct_concepts.append(ipair['OFFICIAL_SYMBOL_B'])
				# elif (l_gene in isymbol[1]):
					# intrct_concepts.append(ipair['OFFICIAL_SYMBOL_A'])
			interactions.append(intrct_concepts)
			del res, intrct_concepts
		io.write_obj(interactions, cache_path[0])
		io.write_obj(opt_intrct, cache_path[1])
	# Construct the adjacent matrix of the genes
	genes = list(set(gene_list + func.flatten_list(interactions) + func.flatten_list(opt_intrct)))
	gene_idx = dict([(s, i) for i, s in enumerate(genes)])
	rows, cols, data = [[] for x in range(3)]
	for gene, interaction in zip(gene_list, interactions):
		row, col, val = [gene_idx[gene]] * len(interaction), [gene_idx[x] for x in interaction], [1] * len(interaction)
		rows.extend(row + col)
		cols.extend(col + row)
		data.extend(val + val)
	for gene, o_symbol in opt_intrct:
		row, col, val = gene_idx[gene] , gene_idx[o_symbol] , 2
		rows.extend([row, col])
		cols.extend([col, row])
		data.extend([val, val])
	simmt = coo_matrix((data, (rows, cols)), shape=(len(genes), len(genes)), dtype='int8')
	simmt.setdiag(1)
	sim_df = pd.DataFrame(simmt.toarray(), index=genes, columns=genes)
	# Calculate the transitive path through the synonym symbol
	# for m, k in set([tuple(sorted(x)) for x in zip(*np.where(sim_df == 2))]):
	for m, k in zip(*np.where(sim_df == 2)):
		for n in np.where(sim_df.iloc[:,k] == 1)[0]:
			sim_df.iloc[m, n] = 1
	# Reset the relation between the synonym and the official symbol to 1
	sim_df[sim_df==2] = 1
	io.write_df(sim_df, 'simmt_gene_%s.npz' % col_name, with_idx=True, sparse_fmt=opts.spfmt, compress=True)
	
	
def sgn_overlap():
	kwargs = {} if opts.cfg is None else ast.literal_eval(opts.cfg)
	sgn_files, cmp_sgn_files = kwargs['sgns'].split(SC), kwargs['cmp_sgn'].split(SC)
	for i, (sgn_f, cmp_sgn_f) in enumerate(zip(sgn_files, cmp_sgn_files)):
		sgn_df = pd.read_csv(os.path.join(opts.loc, sgn_f))
		cmp_sgn_df = pd.read_csv(os.path.join(opts.loc, cmp_sgn_f))
		sgn_set = set(zip(sgn_df['ctrl_ids'], sgn_df['pert_ids']))
		cmp_sgn_set = set(zip(cmp_sgn_df['ctrl_ids'], cmp_sgn_df['pert_ids']))
		num_sgn, num_cmp_sgn, num_intrct = len(sgn_set), len(cmp_sgn_set), len(sgn_set & cmp_sgn_set)
		print 'Signature group %i: our signatures: %i (sole %i), compared signatures: %i (sole %i), intersection: %s' % (i, num_sgn, num_sgn-num_intrct, num_cmp_sgn, num_cmp_sgn-num_intrct, num_intrct)


def sgn_eval():
	kwargs = {} if opts.cfg is None else ast.literal_eval(opts.cfg)
	sgn_df, true_simmt = pd.read_csv(kwargs['sgn']), io.read_df(kwargs['truesim'], with_idx=True, sparse_fmt=opts.spfmt)
	col_name, sgn_simdfs, sgnsim_lbs, truesim_lb = kwargs['col_name'], kwargs['sgnsims'].split(SC), kwargs['sgnsim_lbs'].split(SC), kwargs['truesim_lb']
	# Only compare the overlap symbols
	sgn_symbols, gt_symbols = [str(x).lower() for x in sgn_df[col_name]], [str(x).lower() for x in true_simmt.index]
	true_simmt.index, true_simmt.columns = gt_symbols, gt_symbols
	true_simmt = func.unique_rowcol_df(true_simmt, merge='sum')
	true_simmt[true_simmt >= 0.5] = 1
	true_simmt[true_simmt < 0.5] = 0
	gt_symbols = [str(x).lower() for x in true_simmt.index]
	overlaps = set(sgn_symbols) & set(gt_symbols)
	olgt_symbols = [x for x in gt_symbols if x in overlaps]
	olgt_simmt = true_simmt.loc[olgt_symbols, olgt_symbols]
	roc_data, roc_labels = [], []
	for sgn_simdf_f, sgnsim_lb in zip(sgn_simdfs, sgnsim_lbs):
		sgn_simdf = io.read_df(sgn_simdf_f, with_idx=True, sparse_fmt=opts.spfmt).fillna(value=0).abs()
		print 'Signature and ground truth similarity matrix size: %s, %s' % (sgn_simdf.shape, true_simmt.shape)
		# Record the id map
		sgn_simmt = sgn_simdf.as_matrix()
		id_map = dict(zip(sgn_simdf.index, range(sgn_simdf.shape[0])))
		# Find out the unique symbol and their signatures
		unique_idx = {}
		for id, symbol in zip(sgn_df['id'], sgn_df[col_name]):
			unique_idx.setdefault(str(symbol).lower(), []).append(id_map[id])
		for k, v in unique_idx.iteritems():
			unique_idx[k] = np.array(v)
		# Construct the predicted similarity matrix
		pred_simmt = np.eye(len(olgt_symbols), dtype='float32')
		for x, y in itertools.combinations(range(len(olgt_symbols)), 2):
			# print olgt_symbols[x], olgt_symbols[y]
			# print unique_idx[olgt_symbols[x]], unique_idx[olgt_symbols[y]]
			# print sgn_simmt[unique_idx[olgt_symbols[x]],:][:,unique_idx[olgt_symbols[y]]]
			pred_simmt[x, y] = pred_simmt[y, x] = min(1, sgn_simmt[unique_idx[olgt_symbols[x]],:][:,unique_idx[olgt_symbols[y]]].max())
		print 'Ground truth and prediction size: %s, %s' % (olgt_simmt.shape, pred_simmt.shape)
		io.write_spmt(olgt_simmt, 'truth_mt_%s.npz'%sgnsim_lb.replace(' ', '_').lower(), sparse_fmt=opts.spfmt, compress=True)
		io.write_spmt(pred_simmt, 'pred_mt_%s.npz'%sgnsim_lb.replace(' ', '_').lower(), sparse_fmt=opts.spfmt, compress=True)
		# Calculate the metrics
		fpr, tpr, roc_auc, thrshd = metric.mltl_roc(olgt_simmt.as_matrix(), pred_simmt, average=opts.avg)
		roc_data.append([fpr, tpr])
		roc_labels.append('%s (AUC=%0.2f)' % (sgnsim_lb, roc_auc))
	# Plot the figures
	plot.plot_roc(roc_data, roc_labels, groups=[(x, x+1) for x in range(0, len(roc_data), 2)], mltl_ls=True, fname='roc_%s' % truesim_lb.lower().replace(' ', '_'), plot_cfg=plot_common_cfg)
	# plot.plot_prc(prc_data, prc_labels, groups=[(x, x+1) for x in range(0, len(roc_data), 2)], mltl_ls=True, fname='prc_%s' % truesim_lb.lower().replace(' ', '_'), plot_cfg=plot_common_cfg)

	
def cross_sgn_eval():
	kwargs = {} if opts.cfg is None else ast.literal_eval(opts.cfg)
	sgn_dfs, true_simmt = [pd.read_csv(os.path.join(opts.loc, fname)) for fname in kwargs['sgns'].split(SC)], io.read_spdf(kwargs['truesim'], with_idx=True, sparse_fmt=opts.spfmt)
	col_names, sgn_simdfs, sgnsim_lbs, truesim_lb = kwargs['col_names'].split(SC), kwargs['sgnsims'].split(SC), kwargs['sgnsim_lbs'].split(SC), kwargs['truesim_lb']
	# Only compare the overlap symbols
	sgn_symbols, gt_symbols = [[unicode(x).lower() for x in sgn_df[col_name]] for sgn_df, col_name in zip(sgn_dfs, col_names)], [[unicode(x).lower() for x in symbs] for symbs in [true_simmt['index'], true_simmt['columns']]]
	true_simmt['index'], true_simmt['columns'] = gt_symbols[0], gt_symbols[1]
	true_simmt['values'], true_simmt['shape'], true_simmt['index'], true_simmt['columns'] = func.unique_rowcol(sp.sparse.lil_matrix(true_simmt['values']), true_simmt['index'], true_simmt['columns'], merge='sum')
	true_simmt['values'] = sp.sparse.csr_matrix(true_simmt['values'])
	true_simmt['values'].data[true_simmt['values'].data >= 0.5] = 1
	true_simmt['values'].data[true_simmt['values'].data < 0.5] = 0
	gt_symbols = [[unicode(x).lower() for x in symbs] for symbs in [true_simmt['index'], true_simmt['columns']]]
	overlaps = [set(sgn_symbol) & set(gt_symbol) for sgn_symbol, gt_symbol in zip(sgn_symbols, gt_symbols)]
	olgt_symbols = [[x for x in gt_symbol if x in overlap] for overlap, gt_symbol in zip(overlaps, gt_symbols)]
	olgt_simmt = true_simmt['values'][pd.Series(range(len(true_simmt['index'])), index=true_simmt['index']).loc[olgt_symbols[0]].tolist(),:][:,pd.Series(range(len(true_simmt['columns'])), index=true_simmt['columns']).loc[olgt_symbols[1]].tolist()]
	roc_data, roc_labels = [], []
	for sgn_simdf_f, sgnsim_lb in zip(sgn_simdfs, sgnsim_lbs):
		sgn_simdf = io.read_df(sgn_simdf_f, with_idx=True, sparse_fmt=opts.spfmt).fillna(value=0).abs()
		print 'Signature and ground truth similarity matrix size: %s, %s' % (sgn_simdf.shape, true_simmt['shape'])
		# Record the id map
		sgn_simmt = sgn_simdf.as_matrix()
		id_map = dict(zip(sgn_simdf.index, range(sgn_simdf.shape[0])))
		# Find out the unique symbol and their signatures
		unique_idx = {}
		for sgn_df, col_name in zip(sgn_dfs, col_names):
			for id, symbol in zip(sgn_df['id'], sgn_df[col_name]):
				unique_idx.setdefault(unicode(symbol).lower(), []).append(id_map[id])
		for k, v in unique_idx.iteritems():
			unique_idx[k] = np.array(v)
		# Construct the predicted similarity matrix
		pred_simmt = np.zeros((len(olgt_symbols[0]), len(olgt_symbols[1])), dtype='float32')
		for x, y in itertools.product(range(len(olgt_symbols[0])), range(len(olgt_symbols[1]))):
			# print olgt_symbols[x], olgt_symbols[y]
			# print unique_idx[olgt_symbols[x]], unique_idx[olgt_symbols[y]]
			# print sgn_simmt[unique_idx[olgt_symbols[x]],:][:,unique_idx[olgt_symbols[y]]]
			pred_simmt[x, y] = min(1, sgn_simmt[unique_idx[olgt_symbols[0][x]],:][:,unique_idx[olgt_symbols[1][y]]].max())
		print 'Ground truth and prediction size: %s, %s' % (olgt_simmt.shape, pred_simmt.shape)
		io.write_spmt(olgt_simmt, 'truth_mt_%s.npz'%sgnsim_lb.replace(' ', '_').lower(), sparse_fmt=opts.spfmt, compress=True)
		io.write_spmt(pred_simmt, 'pred_mt_%s.npz'%sgnsim_lb.replace(' ', '_').lower(), sparse_fmt=opts.spfmt, compress=True)
		# Calculate the metrics
		fpr, tpr, roc_auc, thrshd = metric.mltl_roc(olgt_simmt.toarray(), pred_simmt, average=opts.avg)
		roc_data.append([fpr, tpr])
		roc_labels.append('%s (AUC=%0.2f)' % (sgnsim_lb, roc_auc))
	# Plot the figures
	plot.plot_roc(roc_data, roc_labels, groups=[(x, x+1) for x in range(0, len(roc_data), 2)], mltl_ls=True, fname='roc_%s' % truesim_lb.lower().replace(' ', '_'), plot_cfg=plot_common_cfg)
	# plot.plot_prc(prc_data, prc_labels, groups=[(x, x+1) for x in range(0, len(roc_data), 2)], mltl_ls=True, fname='prc_%s' % truesim_lb.lower().replace(' ', '_'), plot_cfg=plot_common_cfg)
	
	
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
	plot.plot_roc(roc_data, roc_labels, fname='roc_%s_' % (sim0_lb, sim1_l), plot_cfg=plot_common_cfg)
	
	
def cmp_sim_list():
	kwargs = {} if opts.cfg is None else ast.literal_eval(opts.cfg)
	excel_path, col_name, db_name, sim_lb, rankl_lb = kwargs['excel_path'], kwargs['col_name'], kwargs['db_name'], kwargs['sim_lb'], kwargs['rankl_lb']
	sgn_df, sim_df, rank_list = pd.read_csv(excel_path), io.read_df(os.path.join(opts.loc, kwargs['sim']), with_idx=True, sparse_fmt=opts.spfmt), io.read_obj(os.path.join(opts.loc, kwargs['rankl']))
	overlaps = set([str(x).lower() for x in sim_df.index]) & set([str(x).lower() for x in sgn_df[col_name]])
	y_true, y_pred = [], []
	for ol in overlaps:
		rankls = [rank_list[x] for x in np.where(sgn_df[col_name] == ol)[0] if len(rank_list[x]) > 0]
		if (not rankls): continue
		max_length = max([len(x) for x in rankls])
		rankl = [collections.Counter([x[l] for x in rankls if len(x) > l]).most_common(1)[0][0] for l in range(max_length)]
		y_true.append(rankl)
		y_pred.append(sim_df.columns[sim_df.ix[ol,:].argsort()[::-1]].tolist())
	fpr, tpr, roc_auc, thrshd = metric.list_roc(y_true, y_pred, average=opts.avg)
	roc_labels = ['%s (AUC=%0.2f)' % (sim_lb, roc_auc)]
	roc_data = [[fpr, tpr]]
	plot.plot_roc(roc_data, roc_labels, fname='roc_%s_%s' % (sim_lb, rankl_lb), plot_cfg=plot_common_cfg)
	
	
def simmt2gml():
	kwargs = {} if opts.cfg is None else ast.literal_eval(opts.cfg)
	simmt = io.read_spdf(kwargs['simmt'], with_idx=True, sparse_fmt=opts.spfmt)
	sgn_dfs, col_names = [pd.read_csv(os.path.join(opts.loc, fname)) for fname in kwargs['sgns'].split(SC)], kwargs['col_names'].split(SC)
	id2gse = dict(reduce(operator.add, [list(zip(df['id'], df['geo_id'])) for df in sgn_dfs]))
	id2subj = dict(reduce(operator.add, [list(zip(df['id'], df[col])) for df, col in zip(sgn_dfs, col_names)]))
	gse_Y = io.read_df(os.path.join(gsc.DATA_PATH, 'gse_Y.npz'), with_idx=True)
	G = nx.Graph()
	G.add_nodes_from([(idx,dict(subj='' if id2subj[idx] is np.nan else id2subj[idx], subj_type=np.where(gse_Y.loc[id2gse[idx]]==1)[0][0].item())) for idx in simmt['index']])
	G.add_edges_from([(simmt['index'][k[0]], simmt['columns'][k[1]], dict(similarity=v.item())) for k, v in simmt['values'].todok().iteritems()])
	nx.write_graphml(G, '%s.graphml' % os.path.splitext(os.path.basename(kwargs['simmt']))[0])
	
	
def plot_simmt():
	import graph_tool.all as gt
	import math
	g = gt.load_graph(opts.loc)
	vertex_cmap = {0:(0,0,1,1), 1:(0,1,0,1), 2:(1,0,0,1)}
	edge_cmap = {True:(1,1,0,1), False:(0,1,1,1)}
	vcolor, ecolor = g.new_vertex_property('vector<double>'), g.new_edge_property('vector<double>')
	g.vertex_properties['vcolor'], g.edge_properties['ecolor'] = vcolor, ecolor
	for v in g.vertices():
		vcolor[v] = vertex_cmap[g.vertex_properties['subj_type'][v]]
	for e in g.edges():
		ecolor[e] = edge_cmap[g.edge_properties['similarity'][e] > 0]
	t = gt.Graph()
	for v in g.vertices():
		t.add_vertex()
	dizs, drug, gene, root = t.add_vertex(), t.add_vertex(), t.add_vertex(), t.add_vertex()
	t.add_edge(root,dizs)
	t.add_edge(root,drug)
	t.add_edge(root,gene)

	for tv in t.vertices():
		if t.vertex_index[tv] < g.num_vertices():
			if g.vertex_properties['subj_type'][tv] == 0:
				t.add_edge(dizs,tv)
			elif g.vertex_properties['subj_type'][tv] == 1:
				t.add_edge(drug,tv)
			elif g.vertex_properties['subj_type'][tv] == 2:
				t.add_edge(gene,tv)
	tpos = pos = gt.radial_tree_layout(t, t.vertex(t.num_vertices() - 1), weighted=True)
	cts = gt.get_hierarchy_control_points(g, t, tpos)
	pos = g.own_property(tpos)

	text_rot = g.new_vertex_property('double')
	g.vertex_properties['text_rot'] = text_rot
	for v in g.vertices():
		if pos[v][0] >0:
			text_rot[v] = math.atan(pos[v][1]/pos[v][0])
		else:
			text_rot[v] = math.pi + math.atan(pos[v][1]/pos[v][0])

	gt.graph_draw(g, pos=pos,
				  vertex_size=10,
				  vertex_color=g.vertex_properties['vcolor'],
				  vertex_fill_color=g.vertex_properties['vcolor'],
				  edge_control_points=cts,
				  vertex_text=g.vertex_properties['subj'],
				  vertex_text_rotation=g.vertex_properties['text_rot'],
				  vertex_text_position=1,
				  vertex_font_size=9,
				  edge_color=g.edge_properties['ecolor'],
				  vertex_anchor=0,
				  bg_color=[0,0,0,1],
				  output_size=[4096,4096],
				  output='hreb.pdf')
				  
				  
def plot_simmt_hrc():
	sys.setrecursionlimit(10000)
	kwargs = {} if opts.cfg is None else ast.literal_eval(opts.cfg)
	simmt = io.read_spdf(kwargs['simmt'], with_idx=True, sparse_fmt=opts.spfmt)
	sgn_dfs, col_names = [pd.read_csv(os.path.join(opts.loc, fname)) for fname in kwargs['sgns'].split(SC)], kwargs['col_names'].split(SC)
	valid_idx = [i for i, idx in enumerate(simmt['index']) if idx in func.flatten_list([df['id'] for df in sgn_dfs])]
	simmt['index'] = simmt['index'][valid_idx]
	simmt['columns'] = simmt['columns'][valid_idx]
	simmt['values'] = simmt['values'][valid_idx,:][:,valid_idx]
	simmt['shape'] = simmt['values'].shape
	id2subj = dict(reduce(operator.add, [list(zip(df['id'], df[col])) for df, col in zip(sgn_dfs, col_names)]))
	axis_label = ['' if id2subj[idx] is np.nan else id2subj[idx] for idx in simmt['index']]
	simmt_values = np.nan_to_num(simmt['values'].toarray())
	# border = max(abs(simmt_values.min()), abs(simmt_values.max()))
	border = 0.03
	plot.plot_clt_hrc(simmt_values, xlabel='Signature', ylabel='Signature', dist_metric='precomputed', dist_func=lambda x: 1-np.abs(x), plot_cfg=plot_common_cfg, mat_size=(0.8,0.9), linkage_method='ward', rcntx_linewidth=0.005, dndrgram_truncate_mode='level', dndrgram_p=10, dndrgram_color_threshold=3.0, matshow_vmin=-border, matshow_vmax=border, cbarclim_vmin=-border, cbarclim_vmax=border, cbartick_fontsize=5)


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
			plot.plot_fzyclt(X.as_matrix(), label.as_matrix(), fname='clustering_%i' % i, plot_cfg=plot_common_cfg)
		else:
			plot.plot_clt(X.as_matrix(), label.as_matrix().reshape((label.shape[0],)), fname='clustering_%i' % i, plot_cfg=plot_common_cfg)
			
			
def plot_sampclt(with_cns=False):
	from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
	import networkx as nx
	import matplotlib.pyplot as plt
	cluster_sets = {
		0:[
		# 'GSM155564|GSM155565|GSM155566|GSM155567',
		# 'GSM155568|GSM155569|GSM155570|GSM155571|GSM155572|GSM155573|GSM155574',
		# 'GSM155592|GSM155593|GSM155594|GSM155595',
		'GSM29441|GSM29442',
		'GSM29437|GSM29438',
		'GSM29297|GSM29298',
		'GSM29270|GSM29271',
		'GSM29289|GSM29290|GSM29305|GSM29306',
		'GSM29262|GSM29263|GSM29264|GSM29265|GSM39415|GSM39416',
		# 'GSM684691|GSM684692|GSM684693',
		# 'GSM684688|GSM684689|GSM684690',
		# 'GSM684685|GSM684686|GSM684687',
		# 'GSM684682|GSM684683|GSM684684'
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
	build_mst = kwargs.setdefault('build_mst', True)
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
		if (build_mst):
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
		nx.draw_networkx_edge_labels(G, pos, edge_labels=dict([(e, '%.3f' % G.edge[e[0]][e[1]]['weight']) for e in edges]), bbox=dict(boxstyle='round,pad=0.,rounding_size=0.2', fc='w', ec='w', alpha=1), font_size='6')
		plt.axis('off')
		if (plot.MON):
			plt.show()
		else:
			plt.savefig('sampclt_%i.pdf' % cltid, format='pdf')
		plt.close()
		
		
def plot_circos(**kw_args):
	import matplotlib as mpl
	import matplotlib.pyplot as plt
	import rpy2.robjects as ro
	from rpy2.robjects import numpy2ri as np2r
	from rpy2.robjects import pandas2ri as pd2r
	from rpy2.robjects.packages import importr
	np2r.activate()
	pd2r.activate()
	ro.r('library(circlize)')
	kwargs = kw_args if len(kw_args) > 0 else ({} if opts.cfg is None else ast.literal_eval(opts.cfg))
	baseloc = kwargs.setdefault('loc', '.') if len(kw_args) > 0 else opts.loc
	top_k = kwargs.setdefault('topk', 10)
	top_i = kwargs.setdefault('topi', 1)
	dizs_sgnfile, dizs_dgeloc = kwargs['dizs'].split(SC)
	drug_sgnfile, drug_dgeloc = kwargs['drug'].split(SC)
	gene_sgnfile, gene_dgeloc = kwargs['gene'].split(SC)
	dizs_sgndf, drug_sgndf, gene_sgndf = pd.read_csv(os.path.join(baseloc, dizs_sgnfile), index_col='id'), pd.read_csv(os.path.join(baseloc, drug_sgnfile), index_col='id'), pd.read_csv(os.path.join(baseloc, gene_sgnfile), index_col='id')
	subjtype_sgn_map = {'Disease':dizs_sgndf, 'Drug':drug_sgndf, 'Gene':gene_sgndf}
	subjtype_col_map = {'Disease':'disease_name', 'Drug':'drug_name', 'Gene':'hs_gene_symbol'}
	gsm_Xs, gsm_Ys, _ = gsc.get_mltl_npz(type='gsm', lbs=['0', '1', '2'], mltlx=False, spfmt=opts.spfmt)
	subjtype_y_map = {'Disease':gsm_Ys[0], 'Drug':gsm_Ys[1], 'Gene':gsm_Ys[2]}
	# Prepare the combined data
	io.inst_print('Preparing the data for circos plot...')
	dfname = kwargs.setdefault('data', 'data.npz')
	cache_f = os.path.join(opts.cache, dfname)
	if (os.path.exists(cache_f)):
		io.inst_print('Reading cache for combined data...')
		data = io.read_df(cache_f, with_idx=True)
		io.inst_print('Finish reading cache for combined data...')
	else:
		sgn_dfs = []
		for subjtype, sgndf in subjtype_sgn_map.iteritems():
			df = sgndf[['ctrl_ids', 'pert_ids', 'geo_id', 'prganism', 'cell_type', 'platform', subjtype_col_map[subjtype]]]
			df.rename(columns={'Organism': 'organism', subjtype_col_map[subjtype]: 'subject'}, inplace=True)
			df['subject_type'] = [subjtype] * df.shape[0]
			sgn_dfs.append(df)
		data = pd.concat(sgn_dfs).reset_index()
		io.write_df(data, cache_f, with_idx=True)
	# Prepare the differential gene expression data, p-values are used to select the data
	dgevalf, dgepvalf = os.path.join(opts.cache, 'dge_df.npz'), os.path.join(opts.cache, 'dgepval_df.npz')
	if (os.path.exists(dgevalf) and os.path.exists(dgepvalf)):
		io.inst_print('Reading cache for differential gene expression data...')
		dge_df = io.read_df(dgevalf, with_idx=True)
		dge_pvals = io.read_npz(dgepvalf)['data']
		io.inst_print('Finish reading cache for differential gene expression data...')
	else:
		subjtype_sgnidx_map = {'Disease':pd.DataFrame(np.arange(dizs_sgndf.shape[0]), index=dizs_sgndf.index), 'Drug':pd.DataFrame(np.arange(drug_sgndf.shape[0]), index=drug_sgndf.index), 'Gene':pd.DataFrame(np.arange(gene_sgndf.shape[0]), index=gene_sgndf.index)}
		subjtype_dgeloc_map = {'Disease':dizs_dgeloc, 'Drug':drug_dgeloc, 'Gene':gene_dgeloc}
		dge_vals, dge_pvals = [], []
		for idx, row in data.iterrows():
			# dge_fname = 'dge_%i.npz' % subjtype_sgnidx_map[row['subject_type']].loc[row['id']]
			dge_fname = 'dge_%i.npz' % row['id']
			df = io.read_df(os.path.join(subjtype_dgeloc_map[row['subject_type']], dge_fname), with_idx=True).replace([np.inf, -np.inf], np.nan).dropna()
			dge_abs = np.abs(df['statistic']).astype('float32')
			# df = df[dge_abs > dge_abs.mean()]
			if (df.shape[0] == 0):
				dge_pvals.append(np.nan)
				continue
			df = sampling.samp_df(df, n=min(30, df.shape[0]), reset_idx=True)
			dge_vals.append(pd.DataFrame(data={'dge_val':df['statistic'].tolist(), 'subject':[row['subject']]*df.shape[0]}, index=[idx]*df.shape[0]))
			dge_vals[-1]['dge_val'] = dge_vals[-1]['dge_val'].astype('float32')
			dge_pvals.append(df['pvalue'].max())
		dge_df = pd.concat(dge_vals)
		# io.inst_print([dge_df.dtypes, dge_df.dge_val.min(), dge_df.dge_val.max()])
		dge_df['dge_val'] = dge_df['dge_val'].astype('float16')
		dge_df.dropna(inplace=True)
		dge_pvals = np.array(dge_pvals)
		io.write_df(dge_df, dgevalf, with_idx=True, compress=True)
		io.write_npz(dge_pvals, dgepvalf)
	# Normalize the differential gene expression value
	# if (dge_df['dge_val'].min() < 0):
		# dge_df['dge_val'] = dge_df['dge_val'] - dge_df['dge_val'].min() + 1
	# dge_df['dge_val'][dge_df['dge_val'] > 1] = 1
	# dge_df['dge_val'][dge_df['dge_val'] <= 0] = 2
	# dge_df['dge_val'][dge_df['dge_val'] == 2] = dge_df['dge_val'].min()
	# dge_df['dge_val'] = dstclc.normdist(dge_df['dge_val'], range=(0.1, 0.9))
	# dge_df['dge_val'] = np.log10(dge_df['dge_val'])
	# dge_df.dropna(inplace=True)
	dge_df['dge_val'] = np.log10(dge_df['dge_val'].astype('float16'))
	# dge_df['dge_val'] = dge_df['dge_val'].astype('float16')
	dge_df = dge_df.replace([np.inf, -np.inf], np.nan).dropna()
	# Append the p-value column
	data['dge_pval'] = dge_pvals
	data.dropna(inplace=True)
	# data['dge_pval'] = np.random.random(data.shape[0])
	io.inst_print('Selecting the data and preparing for the first track...')
	# Select the most significant signatures and transport the combined data
	# top_data = []
	# for k, idx in data.groupby('subject_type').groups.iteritems():
		# top_data.append(data.loc[idx].sort_values('dge_pval').head(top_k))
	# Select the signatures in the case studies
	slct_subj = ['Levetiracetam', 'Phenytoin', 'Estradiol', 'Genistein', 'anemia', 'Lysophosphatidic acid', 'hypertension', 'melanoma in situ', 'prostate cancer', 'Celecoxib', 'Mehp', 'Perfluorooctanoic acid', 'Fluoxetine', 'Decitabine', "3,3',4,4'-tetrachlorobiphenyl", 'POR', 'Estradiol', 'Tretinoin', 'Cobalt dichloride hexahydrate', 'D-serine']
	slct_subjtype = [data['subject_type'].iloc[np.where(data['subject']==x)[0][0]] for x in slct_subj] # subject type
	slct_idx = [i for i, x in enumerate(data['subject']) if x in slct_subj] # selected indices
	subjtype_cnt = collections.Counter(slct_subjtype) # counter for each subject type
	top_data = [data.iloc[slct_idx]]
	for k, idx in data.groupby('subject_type').groups.iteritems():
		if (top_k <= subjtype_cnt[k]): continue
		ordered_data = data.loc[idx].sort_values('dge_pval') # reorder the data according to the dge pvalues
		sup_subj = [x for x in func.remove_duplicate(ordered_data['subject']) if x not in slct_subj]
		sup_subj = sup_subj[:min(len(sup_subj), top_k - subjtype_cnt[k])]# find more subjects to show
		if (len(sup_subj) != 0):
			top_data.append(ordered_data.iloc[[i for i, x in enumerate(ordered_data['subject']) if x in sup_subj]])
	data = pd.concat(top_data)
	data.to_csv('circos_data.csv')
	dge_df = dge_df.loc[data.index]
	
	data['cell_type'] = [x.strip().lower() for x in data['cell_type']]
	name_norm = {'cells':'somatic cells', 'tissue':'somatic cells', 't-cell':'t cells', 't-cells':'t cells', 'heart':'heart tissues', 'ko & s100a10 cells':'somatic cells', 'cancer cell':'somatic cells', 'cancer cells':'somatic cells', 'normal cells':'somatic cells', 'mcf7 cells':'mcf-7 cells', 'hl60 cell line':'hl60 cells', 'cd34+ cells':'hematopoietic stem cells'}
	data['cell_type'] = [name_norm[x] if name_norm.has_key(x) else x for x in data['cell_type']]
	
	rdf = pd2r.py2ri(data)
	ro.r.assign('data', rdf)
	io.inst_print('Finish selecting the data and preparing for the first track...')
	# Prepare the data for the second track, control/perturbation sample points
	io.inst_print('Preparing the data for the second track...')
	ctrl_gsmy_dfs, pert_gsmy_dfs = [], []
	for ctrl_ids, pert_ids, subject, subjtype in zip(data['ctrl_ids'], data['pert_ids'], data['subject'], data['subject_type']):
		ctrl_samps, pert_samps = ctrl_ids.split('|'), pert_ids.split('|')
		ctrl_gsmy, pert_gsmy = subjtype_y_map[subjtype].loc[ctrl_samps], subjtype_y_map[subjtype].loc[pert_samps]
		ctrl_gsmy['subject'] = [subject] * len(ctrl_samps)
		pert_gsmy['subject'] = [subject] * len(pert_samps)
		ctrl_gsmy_dfs.append(ctrl_gsmy)
		pert_gsmy_dfs.append(pert_gsmy)
	ctrl_gsmy, pert_gsmy = pd.concat(ctrl_gsmy_dfs).reset_index().drop_duplicates(), pd.concat(pert_gsmy_dfs).reset_index().drop_duplicates()
	ctrl_gsmy['ctrl'][ctrl_gsmy['ctrl'] == 0], ctrl_gsmy['pert'][ctrl_gsmy['pert'] == 0] = -1, -1
	pert_gsmy['ctrl'][pert_gsmy['ctrl'] == 0], pert_gsmy['pert'][pert_gsmy['pert'] == 0] = -1, -1
	ax_margin = 0.2
	ctrl_gsmy['x'], pert_gsmy['x'] = ctrl_gsmy['ctrl'] * np.random.uniform(ax_margin, 1-ax_margin, ctrl_gsmy.shape[0]), pert_gsmy['ctrl'] * np.random.uniform(ax_margin, 1-ax_margin, pert_gsmy.shape[0])
	ctrl_gsmy['y'], pert_gsmy['y'] = ctrl_gsmy['pert'] * np.random.uniform(ax_margin, 1-ax_margin, ctrl_gsmy.shape[0]), pert_gsmy['pert'] * np.random.uniform(ax_margin, 1-ax_margin, pert_gsmy.shape[0])
	ctrl_gsmy, pert_gsmy = pd2r.py2ri(ctrl_gsmy), pd2r.py2ri(pert_gsmy)
	ro.r.assign('ctrl_gsmy', ctrl_gsmy)
	ro.r.assign('pert_gsmy', pert_gsmy)
	io.inst_print('Finish preparing the data for the second track...')
	# Prpare the data for the third track, differential gene expression data
	io.inst_print('Preparing the data for the third track...')
	# dge_df = sampling.samp_df(dge_df, n=100, key='subject', filt_func=lambda x: x[x['dge_val'] != 0], reset_idx=True)
	dge_df = sampling.samp_df(dge_df, n=100, key='subject', reset_idx=True).drop_duplicates()
	dge_dfs = []
	for k, gp in dge_df.groupby('subject'):
		hist, bins = np.histogram(gp['dge_val'])
		idx = np.digitize(gp['dge_val'], bins)
		gp['bins'] = idx
		gp = sampling.samp_df(gp, n=5, key='bins', reset_idx=True)
		del gp['bins']
		dge_dfs.append(gp)
	dge_df = pd.concat(dge_dfs)
	io.write_df(dge_df, 'dge_df.npz', with_idx=True)
	# io.inst_print([dge_df['subject'].shape, dge_df['dge_val'].shape])
	dge_rdf = pd2r.py2ri(dge_df)
	ro.r.assign('dge_df', dge_rdf)
	# ro.r('''write.csv(dge_df, 'dge_df.csv')''')
	# ro.r('''dge_df <- read.csv('dge_df.csv')''')
	io.inst_print('Finish preparing the data for the third track...')
	# Prepare the data for the fourth track, cell type
	io.inst_print('Preparing the data for the fourth track...')
	unq_celltype = list(set(data['cell_type']))
	# unq_celltype = [x.title() for x in set(data['cell_type'])]
	# data['cell_type'] = [x.title() for x in data['cell_type']]
	celltype_map = dict(zip(unq_celltype, range(len(unq_celltype))))
	# celltype_cmap = plt.cm.get_cmap('Set3', len(unq_celltype))
	# celltype_colors = [mpl.colors.rgb2hex(y) for y in np.array([list(celltype_cmap(celltype_map[x]))[:3] for x in data['cell_type']])]
	cmaps = [(plt.cm.get_cmap('Set3', 12), 12), (plt.cm.get_cmap('Set1', 9), 9), (plt.cm.get_cmap('Dark2', 8), 8), (plt.cm.get_cmap('Paired', 12), 12), (plt.cm.get_cmap('Accent', max(1, len(unq_celltype)-41)), max(1, len(unq_celltype)-41))]
	celltype_cmap = func.flatten_list([[cm(x) for x in range(cnum)] for cm, cnum in cmaps])
	celltype_colors = [mpl.colors.rgb2hex(y) for y in np.array([list(celltype_cmap[celltype_map[x]])[:3] for x in data['cell_type']])]
	# Seaborn Palette
	# import seaborn as sns
	# celltype_cmap = sns.color_palette('husl', n_colors=len(unq_celltype))
	# celltype_colors = [mpl.colors.rgb2hex(y) for y in np.array([celltype_cmap[celltype_map[x]] for x in data['cell_type']])]
	ro.r.assign('unq_celltype', unq_celltype)
	ro.r.assign('celltype_colors', celltype_colors)
	ro.r('names(celltype_colors) <- data$cell_type')
	ctcmap = dict(list(set(zip(data['cell_type'], celltype_colors))))
	ro.r.assign('unq_celltype_col', [ctcmap[x] for x in unq_celltype])
	# ro.r('unq_celltype_col <- celltype_colors[unlist(unq_celltype)]')
	# ro.r('unq_celltype_col <- celltype_colors[!is.na(celltype_colors)]')
	# ro.r('print(unq_celltype_col)')
	io.inst_print('Finish preparing the data for the fourth track...')
	# Prepare the data for the fifth track, signature density (deprecated)
	sgn_subidx = pd.Series(np.zeros(data.shape[0]), index=data['subject'])
	sgn_colors = pd.Series(np.zeros(data.shape[0], dtype='|S7'), index=data['subject'])
	subj_cnt = collections.Counter(data.subject)
	sgn_nums = subj_cnt.values()
	sgn_unq_nums = list(set(sgn_nums))
	sgn_unq_idx = np.arange(len(sgn_unq_nums))
	sgn_unq_idx[np.argsort(sgn_unq_nums)] = sgn_unq_idx
	sgn_cm = dict(zip(sgn_unq_nums, sgn_unq_idx))
	sgnnum_cmap = plt.cm.get_cmap('cool', len(sgn_unq_nums))
	cell_margin = 0.05
	for k, v in subj_cnt.iteritems():
		if (0.1 * v > 1 - 2 * cell_margin):
			sgn_subidx.loc[k] = np.arange(1, v+1, dtype='float32') / (v + 1)
		else:
			sgn_subidx.loc[k] = np.arange(1, v+1, dtype='float32') / (v + 1) * (1 + 2 * cell_margin) - cell_margin
		sgn_colors.loc[k] = [mpl.colors.rgb2hex(sgnnum_cmap(sgn_cm[v])[:3])] * v if (v > 1) else mpl.colors.rgb2hex(sgnnum_cmap(sgn_cm[v])[:3])
	ro.r.assign('sgn_subidx', sgn_subidx.tolist())
	ro.r.assign('sgn_colors', sgn_colors.tolist())
	ro.r('names(sgn_colors) <- data$subject')
	# Prepare the data for the fifth track, p-value of the differential gene expression
	io.inst_print('Preparing the data for the fifth track...')
	dge_pval_range = [max(0,1-data['dge_pval'].max()), min(1,1-data['dge_pval'].min())]
	cm_norm = mpl.colors.Normalize(vmin=dge_pval_range[0], vmax=dge_pval_range[1])
	dgepval_cmap = mpl.cm.ScalarMappable(norm=cm_norm, cmap=mpl.cm.YlOrBr)
	dgepval_colors = [mpl.colors.rgb2hex(func.flatten_list(dgepval_cmap.to_rgba([1-x]).tolist())[:3]) for x in data['dge_pval']]
	ro.r.assign('dgepval_colors', dgepval_colors)
	ro.r('names(dgepval_colors) <- 1:nrow(data)')
	dgepval_range = np.linspace(dge_pval_range[0], dge_pval_range[1], num=5)
	# dgepval_range = np.array([dge_pval_range[0], (dge_pval_range[0]+dge_pval_range[1])/2, dge_pval_range[1]])
	ro.r.assign('dgepval_range', dgepval_range)
	# ro.r.assign('dgepval_colrange', [mpl.colors.rgb2hex(func.flatten_list(dgepval_cmap.to_rgba([x]).tolist())[:3]) for x in dgepval_range][::-1])
	ro.r.assign('dgepval_colrange', ['#FFFFD4', '#FED98E', '#FE9929', '#D95F0E', '#993404'][::-1])
	io.inst_print('Finish preparing the data for the fifth track...')
	# Prepare the data for the sixth track, organisms
	io.inst_print('Preparing the data for the sixth track...')
	unq_orgnsm = list(set(data['organism']))
	orgnsm_map = dict(zip(unq_orgnsm, range(len(unq_orgnsm))))
	orgnsm_cmap = plt.cm.get_cmap('Set2', 8)
	orgnsm_colors = [mpl.colors.rgb2hex(plot.mix_white(orgnsm_cmap(2*orgnsm_map[x])[:3], ratio=0.2)) for x in data['organism']]
	ro.r.assign('unq_orgnsm', unq_orgnsm)
	ro.r.assign('orgnsm_colors', orgnsm_colors)
	ro.r('names(orgnsm_colors) <- data$organism')
	ro.r('unq_orgnsm_col <- orgnsm_colors[unlist(unq_orgnsm)]')
	io.inst_print('Finish preparing the data for the sixth track...')
	# Prepare the data for the Chord diagram
	io.inst_print('Preparing the data for the Chord diagram...')
	simmt = io.read_df(kwargs['simmt'], with_idx=True, sparse_fmt=opts.spfmt)
	# simmt.values[np.diag_indices_from(simmt)] = 0
	simmt = simmt.ix[data['id'], data['id']]
	data_with_idx = data.set_index('id')
	# Change the index to subject
	simmt.index = data_with_idx.loc[simmt.index]['subject']
	simmt.columns = data_with_idx.loc[simmt.columns]['subject']
	# Remove the duplicate index
	simmt.values[:,:] = np.abs(simmt.values) # The reason of taking absolute values is that two subject may have both positive and negative relations. They might offset each other when merging.
	simmt = func.unique_rowcol_df(simmt, merge='sum')
	# Normalize the similarity matrix
	simmt.values[:,:] = dstclc.normdist(simmt.values, range=[0.01,0.99])
	np.fill_diagonal(simmt.values, 0)
	# Offset of the similarity values
	sim_offset = np.concatenate([np.zeros((simmt.shape[0],1)), (simmt.values / simmt.values.sum(axis=1).reshape((-1,1)).repeat(simmt.shape[1], axis=1)).cumsum(axis=1)], axis=1)
	rsim_offset = np2r.numpy2ri(sim_offset)
	ro.r.assign('sim_offset', rsim_offset)
	# Select the top i similarity
	min_colnum = simmt.shape[1] - top_i
	if (min_colnum > 0):
		simmt.values[np.arange(simmt.shape[0]).reshape((-1,1)).repeat(min_colnum, axis=1).flatten(), simmt.values.argsort(axis=1)[:,:min_colnum].flatten()] = 0
		simmt.values[simmt.values.argsort(axis=0)[:min_colnum,:].flatten(), np.arange(simmt.shape[1]).reshape((1,-1)).repeat(min_colnum, axis=0).flatten()] = 0
	# simmt[simmt > 0] = 1
	# rsimmt = pd2r.py2ri(simmt) # py2ri cannot transfer duplicate index
	rsimmt = np2r.numpy2ri(simmt.values)
	ro.r.assign('simmt', rsimmt)
	simmt_idx = pd2r.py2ri(pd.Series(simmt.index))
	ro.r.assign('simmt_idx', simmt_idx)
	ro.r('rownames(simmt) <- simmt_idx')
	ro.r('colnames(simmt) <- simmt_idx')
	ro.r('simmt_size <- dim(simmt)')
	io.inst_print('Finish preparing the data for the Chord diagram...')
	io.inst_print('Finish preparing the data for circos plot...')

	# Run the R commands
	ro.r('subj_fa <- factor(data$subject)')
	ro.r('subj_lev <- levels(subj_fa)')
	ro.r('subjtype_fa <- factor(data$subject_type)')
	ro.r('subjtype_lev <- levels(subjtype_fa)')
	ro.r('par(mar=c(2,0,0,0))')
	ro.r('''circos.par('canvas.xlim'=c(-1.4,1.4))''')
	ro.r('''circos.par('canvas.ylim'=c(-1.4,1.0))''')
	ro.r('''circos.par('track.height'=0.05)''')
	ro.r('''circos.par('track.margin'=c(0.005,0.005))''')
	ro.r('''circos.par('gap.degree'=0)''')
	ro.r('''circos.par('cell.padding'=c(0, 0, 0, 0))''')
	ro.r('''circos.par('points.overflow.warning'=FALSE)''')
	ro.r('''circos.initialize(factors=subj_fa, xlim=c(0,1))''')
	# Chord diagram
	# ro.r(r'''chordDiagram(simmt, keep.diagonal=FALSE, reduce=-1, annotationTrack=NULL, preAllocateTracks=list(list(track.height=0.05, track.margin=c(0.005,0)), list(track.height=0.1, track.margin=c(0.005,0.005), cell.padding=c(0.01,0.01)), list(track.height=0.1, track.margin=c(0.005,0.005), cell.padding=c(0.03,0.03)), list(track.height=0.05, track.margin=c(0.001, 0.005)), list(track.height=0.05, track.margin=c(0.001,0.001)), list(track.height=0.05, track.margin=c(0.001,0.001)), list(track.height=0.05, track.margin=c(0,0.001))))''')
	# First track, subject name & subject type
	ro.r(r'''circos.track(factors=subj_fa, ylim=c(0,1), track.index=1, track.height=0.05, track.margin=c(0.005,0), bg.border=NA, panel.fun=function(x, y) {
		xlim <- get.cell.meta.data('xlim')
		ylim <- get.cell.meta.data('ylim')
		sector.index <- get.cell.meta.data('sector.index')
		circos.text(mean(xlim), ylim[2] + uy(1, 'mm'), 
			sector.index, adj=c(0, 0.025), cex=0.3, facing='clockwise', niceFacing=TRUE)
	})''')
	ro.r(r'''highlight.sector(data$subject[data$subject_type==subjtype_lev[1]], track.index=1, col='blue', border=NA, cex=0.8, text.col='white', niceFacing=TRUE)''')
	ro.r(r'''highlight.sector(data$subject[data$subject_type==subjtype_lev[2]], track.index=1, col='green', border=NA, cex=0.8, text.col='white', niceFacing=TRUE)''')
	ro.r(r'''highlight.sector(data$subject[data$subject_type==subjtype_lev[3]], track.index=1, col='red', border=NA, cex=0.8, text.col='white', niceFacing=TRUE)''')
	# Second track
	ro.r(r'''circos.track(factors=subj_fa, ylim=c(-1,1), track.index=2, track.height=0.1, track.margin=c(0.005,0.005), cell.padding=c(0.01,0.01,0.01,0.01))''')
	ro.r(r'''circos.trackPoints(factors=ctrl_gsmy$subject, x=ctrl_gsmy$x, y=ctrl_gsmy$y, track.index=2, pch=20, cex=0.2, col='#FFBC22')''')
	ro.r(r'''circos.trackPoints(factors=pert_gsmy$subject, x=pert_gsmy$x, y=pert_gsmy$y, track.index=2, pch=20, cex=0.2, col='#5519A1')''')
	# Third track
	ro.r('''circos.par('cell.padding'=c(0.03, 0.05, 0.03, 0.05))''')
	ro.r(r'''circos.trackHist(factors=dge_df$subject, x=scale(dge_df$dge_val), bin.size=0.1, track.index=3, track.height=0.1, force.ylim=FALSE, col='#999999', border='#999999')''')
	ro.r('''circos.par('cell.padding'=c(0, 0, 0, 0))''')
	# ro.r(r'''circos.track(factors=subj_fa, ylim=c(0,1), track.index=3, track.height=0.1)''')
	# Fourth track
	# ro.r(r'''circos.track(factors=subj_fa, ylim=c(0,1), track.index=4, track.height=0.05, track.margin=c(0, 0.005), bg.border=NA)''')
	# ro.r(r'''mapply(function(k, v){
		# highlight.sector(data$subject[data$cell_type==k], track.index=4, col=add_transparency(v, 0.8), border=NA, cex=0.8, text.col='white', niceFacing=TRUE)
	# }, names(celltype_colors), celltype_colors)''')
	ro.r(r'''circos.track(factors=subj_fa, x=sgn_subidx, y=1:nrow(data), track.index=4, track.height=0.05, track.margin=c(0,0), bg.border=NA, panel.fun=function(x, y) {
		xlim <- get.cell.meta.data('xlim')
		ylim <- get.cell.meta.data('ylim')
		xrange <- xlim[2] - xlim[1]
		xcenter <- get.cell.meta.data('xcenter')
		sector.index <- get.cell.meta.data('sector.index')
		numx <- length(x)
		for (sub_idx in 1:numx) {
			x_coord <- xlim[1] + x[[sub_idx]] * xrange
			circos.rect(x_coord-ux(0.05, 'mm'), 0, x_coord+ux(0.05, 'mm'), ylim[2], col=toString(celltype_colors[y[sub_idx]]), border=NA)
		}
	})''')
	# Fifth track
	# ro.r(r'''circos.track(factors=subj_fa, x=sgn_subidx, ylim=c(0,1), track.index=5, track.height=0.05, track.margin=c(0.001,0.001), bg.border=NA, panel.fun=function(x, y) {
		# xlim <- get.cell.meta.data('xlim')
		# ylim <- get.cell.meta.data('ylim')
		# xrange <- xlim[2] - xlim[1]
		# xcenter <- get.cell.meta.data('xcenter')
		# sector.index <- get.cell.meta.data('sector.index')
		# numx <- length(x)
		# for (sub_x in x) {
			# x_coord <- xlim[1] + sub_x * xrange
			# circos.rect(x_coord-ux(0.05, 'mm'), 0, x_coord+ux(0.05, 'mm'), ylim[2], col=toString(sgn_colors[sector.index]), border=NA)
		# }
	# })''')
	ro.r(r'''circos.track(factors=subj_fa, x=sgn_subidx, y=1:nrow(data), track.index=5, track.height=0.05, track.margin=c(0,0.001), bg.border=NA, panel.fun=function(x, y) {
		xlim <- get.cell.meta.data('xlim')
		ylim <- get.cell.meta.data('ylim')
		xrange <- xlim[2] - xlim[1]
		xcenter <- get.cell.meta.data('xcenter')
		sector.index <- get.cell.meta.data('sector.index')
		numx <- length(x)
		for (sub_idx in 1:numx) {
			x_coord <- xlim[1] + x[[sub_idx]] * xrange
			circos.rect(x_coord-ux(0.05, 'mm'), 0, x_coord+ux(0.05, 'mm'), ylim[2], col=toString(dgepval_colors[y[sub_idx]]), border=NA)
		}
	})''')
	# Sixth track
	ro.r(r'''circos.track(factors=subj_fa, ylim=c(0,1), track.index=6, track.height=0.05, track.margin=c(0.001,0.005))''')
	ro.r(r'''mapply(function(k, v){
		highlight.sector(data$subject[data$organism==k], track.index=6, col=v, border=NA, cex=0.8, text.col='white', niceFacing=TRUE)
	}, names(orgnsm_colors), orgnsm_colors)''')
	# Links
	ro.r(r'''col_fun = colorRamp2(c(0.001,0.025,0.05), c('beige','coral2','darkmagenta'))''')
	ro.r(r'''for (i in 1:simmt_size[1]) {
		for (j in 1:simmt_size[2]) {
			if (simmt[i,j] > 0) {
				circos.link(simmt_idx[i], sim_offset[i,j:(j+1)], 
					simmt_idx[j], sim_offset[j,i:(i+1)],
					col=add_transparency(col_fun(simmt[i,j]), 0.4),
					h.ratio=0.6)
			}
		}
	}''')
	# Legend
	ro.r('library(stringr)')
	ro.r('library(ComplexHeatmap)')
	ro.r(r'''lgd_subjtype = Legend(at=c('Disease', 'Drug', 'Gene'), type='points', pch=15, size=unit(4,'mm'), legend_gp=gpar(col=c('blue','green','red')), title_position='topleft', title='Collection')''')
	ro.r(r'''lgd_sample = Legend(at=c('Control', 'Perturbation'), type='points', pch=16, size=unit(3,'mm'), legend_gp=gpar(col=c('#FFBC22','#5519A1')), title_position='topleft', title='Sample')''')
	ro.r(r'''lgd_celltype = Legend(at=unlist(lapply(unq_celltype, str_to_title)), nrow=NULL, ncol=5, type='points', pch=15, size=unit(2,'mm'), legend_gp=gpar(col=unlist(unq_celltype_col)), title_position='topleft', title='Tissue/Cell Annotation', labels_gp=gpar(fontsize=5), title_gp=gpar(fontsize=10, fontface='bold'), grid_height=unit(2,'mm'), grid_width=unit(2,'mm'), gap=unit(0.5,'mm'))''')
	ro.r(r'''lgd_orgnsm = Legend(at=unlist(unq_orgnsm), type='points', pch=15, size=unit(4,'mm'), legend_gp=gpar(col=unlist(unq_orgnsm_col)), title_position='topleft', title='Organism')''')
	ro.r(r'''lgd_dgepval = Legend(at=c(0,1), col_fun=colorRamp2(unlist(dgepval_range), unlist(dgepval_colrange)), title_position='topleft', title='DGE P-value')''')
	ro.r(r'''lgd_links = Legend(at=c(0.01,0.03,0.05), col_fun=col_fun, title_position='topleft', title='Association')''')

	ro.r(r'''pushViewport(viewport(x=unit(45,'mm'), y=unit(2,'mm'), 
		width = grobWidth(lgd_celltype), 
		height = grobHeight(lgd_celltype), 
		just = c('left', 'bottom')))''')
	ro.r('''grid.draw(lgd_celltype)''')
	ro.r('''upViewport()''')
	ro.r(r'''lgd_vlist_l = packLegend(lgd_subjtype, lgd_sample, lgd_orgnsm)''')
	ro.r(r'''lgd_vlist_r = packLegend(lgd_dgepval, lgd_links)''')
	ro.r(r'''pushViewport(viewport(x=unit(2,'mm'), y=unit(2,'mm'), 
		width = grobWidth(lgd_vlist_l), 
		height = grobHeight(lgd_vlist_l), 
		just = c('left', 'bottom')))''')
	ro.r('''grid.draw(lgd_vlist_l)''')
	ro.r('''upViewport()''')
	ro.r(r'''pushViewport(viewport(x=unit(1,'npc')-unit(2,'mm'), y=unit(2,'mm'), 
		width = grobWidth(lgd_vlist_r), 
		height = grobHeight(lgd_vlist_r), 
		just = c('right','bottom')))''')
	ro.r('''grid.draw(lgd_vlist_r)''')
	ro.r('''upViewport()''')
	# Cleanup
	ro.r('circos.clear()')


def _gsea(groups, udrg_sgn_df, probe_gene_map, sgndb_path, sample_path, method='signal_to_noise', permt_type='phenotype', permt_num=100, min_size=15, max_size=500, out_dir='gsea_output', keep_unkown_probe=False, fmt='xml', numprocs=1):
	if (fmt == 'soft'):
		from bionlp.spider import geo
	else:
		from bionlp.spider import geoxml as geo
	import gseapy as gp

	for geo_id, sgn_ids in groups:
		# Select the sub signature table
		sub_sgn_df = udrg_sgn_df.loc[sgn_ids]
		ids = sub_sgn_df['id'] if hasattr(sub_sgn_df, 'id') else sub_sgn_df.index
		# Prepair the gene expression profile and the perturbation labels
		pert_ids, ctrl_ids = list(set('|'.join(sub_sgn_df['pert_ids']).split('|'))), list(set('|'.join(sub_sgn_df['ctrl_ids']).split('|')))
		pert_geo_docs, ctrl_geo_docs = geo.parse_geos([os.path.join(sample_path, '.'.join([pert_id, fmt])) for pert_id in pert_ids], view='full', type='gsm', fmt=fmt), geo.parse_geos([os.path.join(sample_path, '.'.join([ctrl_id, fmt])) for ctrl_id in ctrl_ids], view='full', type='gsm', fmt=fmt)
		pert_ge_dfs, ctrl_ge_dfs = [geo_doc['data']['VALUE'] for geo_doc in pert_geo_docs], [geo_doc['data']['VALUE'] for geo_doc in ctrl_geo_docs]
		pert_df, ctrl_df = pd.concat(pert_ge_dfs, axis=1, join='inner').astype('float32'), pd.concat(ctrl_ge_dfs, axis=1, join='inner').astype('float32')
		pert_lb, ctrl_lb, class_vec = 'pert', 'ctrl', ['pert'] * pert_df.shape[1] + ['ctrl'] * ctrl_df.shape[1]
		join_df = pd.concat([pert_df, ctrl_df], axis=1, join='inner')
		join_df.columns = pert_ids + ctrl_ids
		del pert_geo_docs, ctrl_geo_docs, pert_ge_dfs[:], ctrl_ge_dfs[:], pert_df, ctrl_df
		# Map the probes to gene symbols
		plfm = sub_sgn_df['platform'].iloc[0]
		if (probe_gene_map and probe_gene_map.has_key(plfm) and not probe_gene_map[plfm].empty):
			pgmap = probe_gene_map[plfm]
			if (not keep_unkown_probe):
				probes = [idx for idx in join_df.index if idx in pgmap.index and pgmap.loc[idx] and not pgmap.loc[idx].isspace()]
				join_df = join_df.loc[probes]
			join_df.index = [[x.strip() for x in pgmap.loc[probe].split('///')][0] if (probe in pgmap.index) else [probe] for probe in join_df.index]

		join_df.reset_index(inplace=True)
		join_df.rename(columns={'ID_REF': 'NAME'}, inplace=True)
		join_df['NAME'] = join_df['NAME'].apply(str)
		# Call the GSEA API
		try:
			if (not os.path.exists(os.path.join(out_dir,geo_id+'up')) or (os.path.exists(os.path.join(out_dir,geo_id+'up')) and len(fs.read_file(os.path.join(sgndb_path, '%s_up.gmt'%geo_id))) > len(fs.listf(os.path.join(out_dir,geo_id+'up'), pattern='.*\.gsea\.pdf')))):
				print 'doing '+geo_id+'_up'
				gs_res = gp.gsea(data=join_df, gene_sets=os.path.join(sgndb_path, '%s_up.gmt'%geo_id), cls=class_vec, permutation_type=permt_type, permutation_num=permt_num, min_size=min_size, max_size=max_size, outdir=os.path.join(out_dir,geo_id+'up'), method=method, processes=numprocs, format='pdf')
		except Exception as e:
			print 'Error occured when conducting GSEA for up-regulated genes in %s!' % geo_id
			print e
		try:
			if (not os.path.exists(os.path.join(out_dir,geo_id+'down')) or (os.path.exists(os.path.join(out_dir,geo_id+'down')) and len(fs.read_file(os.path.join(sgndb_path, '%s_down.gmt'%geo_id))) > len(fs.listf(os.path.join(out_dir,geo_id+'down'), pattern='.*\.gsea\.pdf')))):
				print 'doing '+geo_id+'_down'
				gs_res = gp.gsea(data=join_df, gene_sets=os.path.join(sgndb_path, '%s_down.gmt'%geo_id), cls=class_vec, permutation_type=permt_type, permutation_num=permt_num, min_size=min_size, max_size=max_size, outdir=os.path.join(out_dir,geo_id+'down'), method=method, processes=numprocs, format='pdf')
		except Exception as e:
			print 'Error occured when conducting GSEA for down-regulated genes in %s!' % geo_id
			print e
		del join_df

	
def run_gsea():
	input_ext = os.path.splitext(opts.loc)[1]
	if (input_ext == '.xlsx' or input_ext == '.xls'):
		sgn_df = pd.read_excel(opts.loc)
	elif (input_ext == '.csv'):
		sgn_df = pd.read_csv(opts.loc)
	elif (input_ext == '.npz'):
		sgn_df = io.read_df(opts.loc)
	else:
		print 'Unsupported input file extension %s, please use csv or npz file!' % input_ext
		sys.exit(1)
	udrg_sgn_df = sgn_df.set_index('id').dropna()
	kwargs = {} if opts.cfg is None else ast.literal_eval(opts.cfg)
	method = kwargs.setdefault('method', 'signal_to_noise')
	sgn_min_size, sgn_max_size = kwargs.setdefault('min_size', 5), kwargs.setdefault('max_size', 100)
	permt_type, permt_num = kwargs.setdefault('permt_type', 'phenotype'), kwargs.setdefault('permt_num', 100)
	keep_unkown_probe = kwargs.setdefault('all_probes', False)
	probe_gene_map = kwargs.setdefault('pgmap', '')
	probe_gene_map = probe_gene_map if (probe_gene_map and os.path.exists(probe_gene_map)) else None
	par_dir, basename = os.path.abspath(os.path.join(opts.loc, os.path.pardir)), os.path.splitext(os.path.basename(opts.loc))[0].split('_udrg')[0]
	sgndb_path = os.path.join(kwargs.setdefault('sgndb', os.path.join(gsc.GEO_PATH, 'sgndb')), basename)
	sample_path = os.path.join(gsc.GEO_PATH, opts.type, basename, 'samples')
	out_dir = os.path.join(par_dir, 'gsea', method.lower(), basename) if opts.output is None else os.path.join(opts.output, method.lower(), basename)
	groups = udrg_sgn_df.groupby('geo_id').groups.items()
	task_bnd = njobs.split_1d(len(groups), split_num=opts.np, ret_idx=True)
	_gsea(groups, udrg_sgn_df=udrg_sgn_df, probe_gene_map=probe_gene_map, sgndb_path=sgndb_path, sample_path=sample_path, method=method, permt_type=permt_type, permt_num=permt_num, min_size=sgn_min_size, max_size=sgn_max_size, out_dir=out_dir, keep_unkown_probe=keep_unkown_probe, numprocs=opts.np)
	# _ = njobs.run_pool(_gsea, n_jobs=opts.np, dist_param=['groups'], groups=[groups[task_bnd[i]:task_bnd[i+1]] for i in range(opts.np)], udrg_sgn_df=udrg_sgn_df, probe_gene_map=probe_gene_map, sgndb_path=sgndb_path, sample_path=sample_path, method=method, permt_type=permt_type, permt_num=permt_num, min_size=sgn_min_size, max_size=sgn_max_size, out_dir=out_dir, keep_unkown_probe=keep_unkown_probe, numprocs=1) # GSEAPY does not allow it to be a children process


def main():
	
	if (opts.method is None):
		return
	elif (opts.method == 'fuseki'):
		fuseki()
	elif (opts.method == 'n2y'):
		npzs2yaml(opts.loc)
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
	elif (opts.method == 'g2m'):
		gpl2map(opts.type)
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
		sgn2dge()
	elif (opts.method == 'pltdgep'):
		plot_dgepval()
	elif (opts.method == 'd2g'):
		dge2udrg()
	elif (opts.method == 'd2s'):
		dge2simmt()
	elif (opts.method == 'simhrc'):
		simhrc()
	elif (opts.method == 'o2s'):
		onto2simmt()
	elif (opts.method == 'o22s'):
		onto22simmt()
	elif (opts.method == 'oc2s'):
		ontoc2simmt()
	elif (opts.method == 'ddis'):
		ddi2simmt()
	elif (opts.method == 'ppis'):
		ppi2simmt()
	elif (opts.method == 'sgno'):
		sgn_overlap()
	elif (opts.method == 'sgne'):
		sgn_eval()
	elif (opts.method == 'csgne'):
		cross_sgn_eval()
	elif (opts.method == 'c2s'):
		cmp2sim()
	elif (opts.method == 'csl'):
		cmp_sim_list()
	elif (opts.method == 's2gml'):
		simmt2gml()
	elif (opts.method == 'pltsim'):
		plot_simmt()
	elif (opts.method == 'pltsimhrc'):
		plot_simmt_hrc()
	elif (opts.method == 'pltclt'):
		plot_clt(opts.fuzzy, threshold=opts.thrshd)
	elif (opts.method == 'smpclt'):
		plot_sampclt(with_cns=opts.cns)
	elif (opts.method == 'circos'):
		plot_circos()
	elif (opts.method == 'gsea'):
		run_gsea()


if __name__ == '__main__':
	# Parse commandline arguments
	op = OptionParser()
	op.add_option('-p', '--pid', default=-1, action='store', type='int', dest='pid', help='indicate the process ID')
	op.add_option('-n', '--np', default=psutil.cpu_count(), action='store', type='int', dest='np', help='indicate the number of processes used for calculation')
	op.add_option('-f', '--fmt', default='npz', help='data stored format: csv or npz [default: %default]')
	op.add_option('-s', '--spfmt', default='csr', help='sparse data stored format: csr or csc [default: %default]')
	op.add_option('-t', '--type', default='xml', help='file type: soft, xml, txt [default: %default]')
	op.add_option('-c', '--cfg', help='config string used in the utility functions, format: {\'param_name1\':param_value1[, \'param_name1\':param_value1]}')
	op.add_option('-a', '--avg', default='micro', help='averaging strategy for performance metrics: micro or macro [default: %default]')
	op.add_option('-i', '--unified', action='store_true', dest='unified', default=False, help='store the data in the same folder')
	op.add_option('-l', '--loc', default='.', help='the files in which location to be process')
	op.add_option('-o', '--output', help='the path to store the data')
	op.add_option('-u', '--fuzzy', action='store_true', dest='fuzzy', default=False, help='use fuzzy clustering')
	op.add_option('-d', '--cns', action='store_true', dest='cns', default=False, help='use constraint clustering')
	op.add_option('-r', '--thrshd', default='mean', dest='thrshd', help='threshold value')
	op.add_option('-w', '--cache', default='.cache', help='the location of cache files')
	op.add_option('-q', '--ipp', default='', help='the ipyparallel cluster profile')
	op.add_option('-z', '--timeout', help='timeout seconds')
	op.add_option('-m', '--method', help='main method to run')
	op.add_option('-v', '--verbose', action='store_true', dest='verbose', default=False, help='display detailed information')
	
	(opts, args) = op.parse_args()
	if len(args) > 0:
		op.print_help()
		op.error('Please input options instead of arguments.')
		sys.exit(1)

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
		nihgene_cfg = cfgr('bionlp.spider.nihgene', 'init')
		if (len(nihgene_cfg) > 0):	
			if (nihgene_cfg['GENE_PATH'] is not None and os.path.exists(nihgene_cfg['GENE_PATH'])):
				nihgene.GENE_PATH = nihgene_cfg['GENE_PATH']
		nihnuccore_cfg = cfgr('bioinfo.spider.nihnuccore', 'init')
		if (len(nihnuccore_cfg) > 0):	
			if (nihnuccore_cfg['GENE_PATH'] is not None and os.path.exists(nihnuccore_cfg['GENE_PATH'])):
				nihgene.GENE_PATH = nihnuccore_cfg['GENE_PATH']
				
		common_cfg = cfgr('gsx_helper', 'common')
		if (len(common_cfg) > 0):
			if (common_cfg.has_key('RAMSIZE') and common_cfg['RAMSIZE'] is not None):
				RAMSIZE = common_cfg['RAMSIZE']
		plot_cfg = cfgr('bionlp.util.plot', 'init')
		plot_common = cfgr('bionlp.util.plot', 'common')
		init_plot(plot_cfg=plot_cfg, plot_common=plot_common)

	annot.init()
			
	main()