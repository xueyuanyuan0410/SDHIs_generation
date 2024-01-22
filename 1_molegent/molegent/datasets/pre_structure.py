#用于数据增强
import argparse
import numpy as np
import pandas as pd
import os
from copy import deepcopy

from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import pearsonr 
import pickle as pkl

def stats(smiles_list):
    # statics of the dataset
    n_atoms = [0] * len(smiles_list) 
    avg_d = [0] * len(smiles_list)
    for i in range(len(smiles_list)):
        s = smiles_list[i]
        mol = Chem.MolFromSmiles(s)
        if mol != None:
            n = len(mol.GetAtoms())
            e = len(mol.GetBonds())
            n_atoms[i]=n
            avg_d[i] = e

    #print(n_atoms)
    print(np.min(n_atoms), np.median(n_atoms), np.mean(n_atoms), np.max(n_atoms))
    print(np.min(avg_d), np.median(avg_d), np.mean(avg_d), np.max(avg_d))
    with open(save_dir+'1_' + 'stats.pkl', 'wb') as f:
        pkl.dump([n_atoms, avg_d], f)


import heapq 
def nearest_neighbor(sim_mat, n_nb):
    n = sim_mat.shape[0] #
    res = np.zeros([n, n])
    for i in range(n):
        r = sim_mat[i, :] 
        idx = heapq.nlargest(n_nb, range(len(r)), r.take)
        res[i, idx] = 1
        if i%100==0:
            print(i)
    res_sum_v = np.sum(res, axis=1)
    print('row sum: ', res_sum_v)
    d = res.diagonal()
    print('diagonal: ', d)
    return res

# other functions
def get_morgan_fingerprint(mol, radius, nBits, FCFP=False):
    m = Chem.MolFromSmiles(mol)
    fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nBits, useFeatures=FCFP)
    fp_bits = fp.ToBitString()
    finger_print = np.fromstring(fp_bits, 'u1')-ord('0')
    return finger_print

def get_drug_fp_batch(drug_smiles, radius=3, length=1024, FCFP=False):
    fp = []
    for mol in drug_smiles:
        fp.append(get_morgan_fingerprint(mol, radius, length, FCFP))
    fp = np.array(fp)
    return fp

import json
def rule_indicate(smiles_list):
    rules = json.load(open('/home/fungicide_generation/molegent_C2/molegent/datasets/isostere_transformations_new.json'))
    print('# rules {:d}'.format(len(rules)))
    rule_indicator = np.zeros([len(smiles_list), len(rules)], dtype=np.int)
    for i in range(len(smiles_list)):
        if i%100==0:
            print(i)
        s = smiles_list[i]
        mol_obj = Chem.MolFromSmiles(s)
        if mol_obj != None:
            for j in range(len(rules)):
                rule = rules[j]
                rxn = AllChem.ReactionFromSmarts(rule['smarts'])
                products = rxn.RunReactants((mol_obj,))
                rule_indicator[i, j] = len(products)
    print(rule_indicator)
    print(np.sum(rule_indicator, axis=1))
    return rule_indicator 

def main():
    parser = argparse.ArgumentParser(description='Get domain rule indicator and neighbor matrix')
    parser.add_argument('--dataset', type=str, default = 'fentn', help='root directory of dataset. For now, only classification.')
    args = parser.parse_args()

    #set up dataset
    input_df = pd.read_csv(args.dataset)
    smiles_list = input_df['smiles'].tolist()#I add
    print('smiles length {:d}'.format(len(smiles_list)))
    # save_dir = 'results/' + args.dataset + '/'
    save_dir = '/home/fungicide_generation/molegent_C2/molegent/datasets/results/'

    #判断文件夹是否存在，若不存在则建立文件夹
    if not os.path.exists(save_dir):
        os.system('mkdir -p %s' % save_dir)

    # rule indicator
    # from utils import rule_indicator, sim_mat
    rule_indicator = rule_indicate(smiles_list)
    with open(save_dir + 'C2_rule_indicator_new.pkl', 'wb') as f:
        pkl.dump([rule_indicator], f)

    # # similarity matrix
    # sim_matrix = sim_mat(smiles_list)
    # with open(save_dir + 'sim_matrix.pkl', 'wb') as f:
    #     pkl.dump([sim_matrix], f)

    # neighbor matrix based on sim_matrix and n_nb (nearest neighbor size)
    # if args.dataset in ['tox21', 'toxcast']:
    #     n_nb_list = [600, 800, 1000]
    # else:
    #     n_nb_list = [10, 50, 100, 150, 300]

    # # with open(save_dir + 'sim_matrix.pkl', 'rb') as f:
    # #     df = pkl.load(f)
    # #     sim_matrix = df[0]
    # print(np.min(sim_matrix), np.max(sim_matrix))
    # print(np.sum(sim_matrix, axis=1))
    # for n_nb in n_nb_list:
    #     print('generate nb: ', n_nb)
    #     sim_matrix_idx = nearest_neighbor(sim_matrix, n_nb)
    #     with open(save_dir + 'sim_matrix_nb_' + str(n_nb) + '.pkl', 'wb') as f:
    #         pkl.dump([sim_matrix_idx], f)


if __name__ == "__main__":
    main()
