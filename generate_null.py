
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from collections import defaultdict
from sklearn import preprocessing
import itertools
from pprint import pprint
from tqdm import tqdm
from glob import glob


def get_consistent_comps(nm_feats, model_comps):
    loocv2bestfeats = {}
    for test_index, feats in enumerate(nm_feats):
        comps = model_comps[test_index]
        sorted_comps_inds = np.argsort(np.abs(comps))[::-1]
        sorted_comps_inds = np.roll(sorted_comps_inds,
                                -np.count_nonzero(np.isnan(comps)))
        first_zeros = np.where(comps[sorted_comps_inds] == 0)[0]
        assert model_name not in ["LASSO", "STACK-LASSO"] or first_zeros[0] == int(feats)
        best_comps = np.array(featgrp_names)[sorted_comps_inds][:int(feats)]
        # print(best_comps)
        loocv2bestfeats[test_index] = best_comps.tolist()

    d = set.intersection(*map(set,list(loocv2bestfeats.values())))
    return d


obs_fnames = glob("out111323/experiment2models_*_1_1_1_1.pkl")
model2obs_const = defaultdict(list)
model2obs_feats = {}
model2obs_counts = defaultdict(list)
model2obs_ylabels = defaultdict(list)
for iii, obs_fname in tqdm(enumerate(obs_fnames)):
    with open(obs_fname, "rb") as f:
        exp2models = pickle.load(f)["prot_meta_lipid_mirna"]

    (model2pred, model2proba, model2tru,
        y, model2comp, stack2feat, featgrp_names,
        model2numfeats) = exp2models

    for model_name, num_feats in model2numfeats.items():
        if model_name == "LR":
            continue
        assert len(model2comp[model_name]) == 7
        const_comps = get_consistent_comps(num_feats, model2comp[model_name])
        if iii == 3:
            model2obs_feats[model_name] = const_comps
            model2obs_ylabels[model_name].append(y)
        # print(num_feats)
        model2obs_const[model_name].append(
            len(const_comps))
        [model2obs_counts[model_name].append(n) for n in num_feats]
        # print(model_name, model2obs_const[model_name])
        # print(d)


print(model2obs_const)
with open("feat_obs_const_111323.pkl", "wb") as f:
    pickle.dump([model2obs_feats, model2obs_ylabels], f)
print("feat_obs_const_111323.pkl")

null_fnames = glob("out111323-null/experiment2models_*_1_1_1_1.pkl")
print(len(null_fnames))
model2null_const = defaultdict(list)
model2null_feats = defaultdict(list)
model2null_counts = defaultdict(list)
model2null_ylabels = defaultdict(list)
for null_fname in tqdm(null_fnames):
    with open(null_fname, "rb") as f:
        exp2models_null = pickle.load(f)["prot_meta_lipid_mirna"]
    (model2pred_null, _,  model2tru_null, df_y_null,
     model2comp_null, _, _, model2numfeats_null) = exp2models_null

    for model_name, num_feats in model2numfeats_null.items():
        assert len(model2comp_null[model_name]) == 7
        const_comps = get_consistent_comps(num_feats, model2comp_null[model_name])
        model2null_ylabels[model_name].append(df_y_null)
        model2null_feats[model_name].append(const_comps)
        model2null_const[model_name].append(len(const_comps))
        [model2null_counts[model_name].append(n) for n in num_feats]

# pprint(model2null_ylabels['STACK-LASSO'])
# exit()
for model_name in model2null_ylabels:
    assert len(model2null_ylabels[model_name]) == len(model2null_feats[model_name])

with open("feat_null_const_111323.pkl", "wb") as f:
    pickle.dump([model2null_feats, model2null_ylabels], f)
print("feat_null_const_111323.pkl")

df_null = pd.DataFrame(model2null_const)
df_null['dist'] = "NULL"

df_obs = pd.DataFrame(model2obs_const)
df_obs['dist'] = "OBSERVED"

pd.concat([df_null, df_obs]).to_csv("feat_hyp_test_111323.csv", index=False)

df_null = pd.DataFrame(model2null_counts)
df_null['dist'] = "NULL"
df_obs = pd.DataFrame(model2obs_counts)
df_obs['dist'] = "OBSERVED"
pd.concat([df_null, df_obs]).to_csv("feat_counts_111323.csv", index=False)

