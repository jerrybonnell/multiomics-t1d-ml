import pandas as pd
from sklearn.decomposition import PCA
import math
from sklearn.model_selection import LeaveOneOut
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler, KMeansSMOTE, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from imblearn.combine import SMOTETomek
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import precision_recall_fscore_support
from sklearn import metrics
from sklearn.decomposition import PCA
# from mlxtend.classifier import StackingCVClassifier
# import umap
from collections import defaultdict
import itertools
from tqdm import tqdm

from stacking import train_master, MyStackingClassifer

from datetime import datetime

def get_formatted_timestamp():
    now = datetime.now()
    month = now.strftime("%m")
    day = now.strftime("%d")
    year = now.strftime("%y")
    timestamp = month + day + year
    return timestamp
formatted_timestamp = get_formatted_timestamp()

df_prot = pd.read_excel(
    "Source multi-omics datasets.xlsx", sheet_name='Proteomics')
df_lipid = pd.read_excel("Source multi-omics datasets.xlsx", sheet_name='Lipidomics')\
    .drop(labels=['Class', 'FA', 'FA Group Key'], axis=1)
df_meta = pd.read_excel("Source multi-omics datasets.xlsx", sheet_name='Metabolomics')\
    .drop(labels=['CAS'], axis=1)
df_mirna = pd.read_excel("Source multi-omics datasets.xlsx", sheet_name='miRNAs')\
    .drop(labels=['Mature miRNA Accession'], axis=1)


def fix_header_col(df, col, replacement_prefix):
    replacement_counter = 1
    # Iterate over the column and replace NaN values with unique strings
    for i in df.index:
        if pd.isna(df.at[i, col]):
            replacement_value = f"{replacement_prefix}{replacement_counter}"
            df.at[i, col] = replacement_value
            replacement_counter += 1
    return df


df_meta0 = fix_header_col(
    df_meta, "Metabolite name (annotation)", "unnamed_meta")


def transpose_df(df):
    df = df.T
    df.columns = df.iloc[0]
    df = df.reset_index(drop=True)
    df = df.rename_axis(None, axis=1)
    df = df.iloc[1:, :]
    df = df.replace(0, np.nan)
    assert [type(e) != str for e in list(df.columns)].count(True) == 0
    df = df.apply(pd.to_numeric)

    nan_cols = df.columns[df.isnull().any()].tolist()
    df = df.drop(columns=nan_cols)
    #df = df[df.columns].apply(pd.to_numeric)
    # df_mirna_t[df_mirna_t.columns].apply(pd.to_numeric).dtypes
    return df.reset_index(drop=True)


# 2330 cols -> 1714 cols (74%)
df_prot_t = transpose_df(df_prot).iloc[:-2].copy()
df_prot_t.columns = [f"prot_{c}" for c in df_prot_t.columns]
# 238 cols -> 122 cols (51%)
df_meta_t = transpose_df(df_meta0).iloc[:-2].copy()
df_meta_t.columns = [f"meta_{c}" for c in df_meta_t.columns]
def log2_to_decimal(x):
    return 2 ** x
for col in list(df_meta_t.columns):
    df_meta_t[col] = df_meta_t[col].apply(log2_to_decimal)

# 66 cols -> 65 cols (98%)
df_lipid_t = transpose_df(df_lipid).iloc[:-2].copy()
df_lipid_t.columns = [f"lipid_{c}" for c in df_lipid_t.columns]
# 329 cols -> 329 cols (100%)
df_mirna_t = transpose_df(df_mirna).iloc[:-2].copy()
df_mirna_t.columns = [f"mirna_{c}" for c in df_mirna_t.columns]
# df_mirna_t = transpose_df(df_mirna).iloc[[0,1,3,4,5,6]].copy()

df_y = pd.DataFrame({'class': ['Healthy', 'Healthy', 'Healthy', 'Healthy',
                               'T1D High-Risk', 'T1D High-Risk', 'T1D High-Risk',
                               ],
                     'subject': list(range(1, 8))})

le = preprocessing.LabelEncoder()
df_y['class'] = le.fit_transform(df_y['class'])


print(le.inverse_transform([0, 1]))

def fold_change_preprocess(df_X, df_y, thresh=1.5):
    # pre-selection for the proteomics
    df_X['class'] = df_y['class']
    healthy_data = df_X[df_X['class'] == le.transform(['Healthy'])[0]]
    case_data = df_X[df_X['class'] == le.transform(['T1D High-Risk'])[0]]
    df_X.drop(columns=['class'], inplace=True)
    # Initialize an empty dictionary to store fold changes
    fold_changes = {}

    # Calculate fold change for each feature
    for feature in list(df_X.columns):
        case_mean = case_data[feature].mean()
        healthy_mean = healthy_data[feature].mean()

        if healthy_mean != 0:
            fold_change = case_mean / healthy_mean
            fold_changes[feature] = fold_change
        else:
            raise ValueError("healthy_mean = 0")
    fld_chng_df = pd.DataFrame(fold_changes.items(),
                                   columns=['Feature', 'Fold Change'])

    fld_chng_df = fld_chng_df[
        (fld_chng_df["Fold Change"] > thresh) |
        (fld_chng_df["Fold Change"] < 1/thresh)]
    return fld_chng_df['Feature'].tolist()



print("prot", (len(df_prot_t), len(df_prot_t.columns)))
print("meta", (len(df_meta_t), len(df_meta_t.columns)))
print("lipid", (len(df_lipid_t), len(df_lipid_t.columns)))
print("mirna", (len(df_mirna_t), len(df_mirna_t.columns)))
print((len(df_y), len(df_y.columns)))

model_names = ["LASSO", "RIDGE", "LR",
               "STACK-LASSO", "STACK-RIDGE", "STACK-LR"]
name2model = {}


all_df = [[df_prot_t, 'prot'],
          [df_meta_t, 'meta'],
          [df_lipid_t, 'lipid'],
          [df_mirna_t, 'mirna']]

comp2len = {'prot': len(df_prot_t.columns),
            'meta': len(df_meta_t.columns),
            'lipid': len(df_lipid_t.columns),
            'mirna': len(df_mirna_t.columns)}



def run_supervised_exp(index_num):
    import time
    import os
    np.random.seed((os.getpid() * int(time.time())) % 123456789)
    experiment2models = {}

    for iiii in [1, 2, 3, 4]:
        for comb in tqdm(list(itertools.combinations(all_df, iiii))):
            dfs = [d[0] for d in comb]
            experiment_name = "_".join([d[1] for d in comb])
            print(experiment_name)
            featgrp_names = []

            # NOTE toggle for generation of null
            # df_y["class"] = np.random.permutation(df_y["class"]) #NULL

            loo = LeaveOneOut()
            model2pred = {}
            model2proba = {}
            model2tru = {}
            model2numfeats = {}
            for name in model_names:
                model2pred[name] = np.zeros(len(dfs[0]))
                model2proba[name] = np.zeros(len(dfs[0]))
                model2tru[name] = np.zeros(len(dfs[0]))
                model2numfeats[name] = np.zeros(len(dfs[0]))

            model2comp = {}
            stack2feat = {}
            for model in ['SVM-LINEAR', 'LASSO', 'RIDGE',"LR",
                          'STACK-SVM','STACK-RIDGE','STACK-LASSO', 'STACK-LR']:
                model2comp[model] = np.empty((7, sum([len(d.columns) for d in dfs])))
                model2comp[model][:] = np.nan
            # for model in ['STACK-MLP','STACK-RF','STACK-SVM',
            #               'STACK-RIDGE','STACK-LASSO','STACK-LDA']:
            #     model2comp[model] = np.empty((7, len(comb)))
            #     model2comp[model][:] = np.nan

            for i, (train_index, test_index) in enumerate(loo.split(dfs[0])):
                print(f"subject {i} test_i {test_index}")

                # thrsh_int_mask = []
                thrsh_sel_cols = []
                f2newlen = {}
                for df_index, f_comp_name in enumerate([d[1] for d in comb]):
                    thrshed_feats = fold_change_preprocess(
                        dfs[df_index].iloc[train_index, :].copy(),
                        df_y, thresh=index_num[f_comp_name])
                    f2newlen[f_comp_name] = len(thrshed_feats)
                    thrsh_sel_cols += thrshed_feats

                # for df_index in range(len(dfs)):
                #     print(dfs[df_index].shape)
                # [df_X.columns.get_loc(c) for c in top_prot_cols]

                df_merged = pd.concat(dfs, axis=1)
                if i == 0:
                    for col_name in list(df_merged.columns):
                        featgrp_names.append(col_name)
                thrsh_int_mask = [df_merged.columns.get_loc(c) for c in thrsh_sel_cols]

                assert len(df_merged.columns[df_merged.isnull().any()].tolist()) == 0
                assert len(df_merged.columns) == len(featgrp_names)
                # assert len(df_merged.columns) in np.multiply(
                #     np.array([1, 2, 3, 4]), obs_thresh)

                X_train, y_train = df_merged.iloc[train_index, thrsh_int_mask].copy(),\
                    df_y.iloc[train_index]['class']
                X_test, y_test = df_merged.iloc[test_index,thrsh_int_mask].copy(),\
                    df_y.iloc[test_index]['class']


                # y_train = y_train.replace(
                #     {le.transform(['T1D New-Onset'])[0]: le.transform(['T1D High-Risk'])[0]})
                # y_test = y_test.replace(
                #     {le.transform(['T1D New-Onset'])[0]: le.transform(['T1D High-Risk'])[0]})
                print(f"X_train dataset size: {X_train.shape}")
                print(f"y_train dataset size: {y_train.shape}")
                print(f"X_test dataset size: {X_test.shape}")
                print(f"y_test dataset size: {y_test.shape}")
                # remove any features with non-zero variance
                # sel = VarianceThreshold(threshold=0.2)
                # sel.fit(X_train)
                # X_train_transf0 = X_train.loc[:, sel.get_support()]
                # # X_test_transf = sel.transform(X_test)
                # X_test_transf0 = X_test.loc[:, sel.get_support()]


                # for col_i in [i for i, x in enumerate(sel.get_support()) if not x]:
                #     print(f"delete {list(X_train_transf.columns)[col_i]}")
                # print(f"before {len(X_train.columns)} after {len(X_train_transf0.columns)}")

                scaler = StandardScaler()
                X_train_transf = pd.DataFrame(scaler.fit_transform(X_train),
                                              columns=X_train.columns)
                X_test_transf = pd.DataFrame(scaler.transform(X_test),
                                             columns=X_test.columns)


                ovrsampler = RandomOverSampler(
                    sampling_strategy={0: 500, 1: 500}, shrinkage=3)

                X_train_aug, y_train_aug = ovrsampler.fit_resample(
                    X_train_transf, y_train)

                for name in model_names:
                    if iiii < 2 and "STACK" in name:
                        continue

                    if name == "LR":
                        model = LogisticRegression(penalty=None)
                        probas_ = model.fit(
                                X_train_aug, y_train_aug).predict_proba(X_test_transf)
                        name2model[name] = model
                        model2numfeats[name][test_index] = X_train_aug.shape[1]
                    elif name == "STACK-LR":
                        model = MyStackingClassifer(name.split("-")[1], C = 0)
                        probas_ = model.fit(
                            X_train_aug, y_train_aug, comp2len,
                            experiment_name).predict_proba(X_test_transf)
                        nonzero_coef_num = np.count_nonzero(np.abs(model.indivd_coefs_[0]))
                        name2model[name] = model
                        model2numfeats[name][test_index] = nonzero_coef_num
                    elif "STACK" in name:
                        for penalty_c in [1, 0.5, 0.1, 0.05, 0.03, 0.02, 0.01,
                                          0.005, 0.004, 0.003, 0.002, 0.001,
                                          0.0005, 0.0004, 0.0003, 0.00025, 0.0002, 0.0001,
                                          0.00009,
                                          0.00005, 0.00004, 0.00003, 0.00002,
                                          1.8e-5, 1.5e-5,
                                          0.00001, 0.000009,
                                          0.000007, 6e-6, 5e-6, 4e-6, 3e-6, 2e-6,
                                          1e-6]:
                            for f_comp in f2newlen:
                                comp2len[f_comp] = f2newlen[f_comp]
                            model = MyStackingClassifer(name.split("-")[1], C = penalty_c)
                            probas_ = model.fit(
                                X_train_aug, y_train_aug, comp2len,
                                experiment_name).predict_proba(X_test_transf)
                            coeff_cutoff = {'STACK-RIDGE': 9.99e-3, 'STACK-SVM':9.99e-4}
                            if name in ["STACK-RIDGE", "STACK-SVM"]:
                                ok = np.abs(model.indivd_coefs_[0])
                                mask = ok < coeff_cutoff[name]
                                ok[mask] = 0
                                nonzero_coef_num = np.count_nonzero(ok)
                            else:
                                nonzero_coef_num = np.count_nonzero(np.abs(model.indivd_coefs_[0]))
                            assert 0 < nonzero_coef_num

                            print(f"{name} {penalty_c} {nonzero_coef_num}")

                            if nonzero_coef_num <= round(0.2 * X_train_aug.shape[1]):
                                print(f"stack {name} {penalty_c} {nonzero_coef_num}")
                                name2model[name] = model
                                model2numfeats[name][test_index] = nonzero_coef_num
                                break
                    elif name in ["LASSO", "RIDGE", "SVM-LINEAR"]:
                        def model_converter(x, c):
                            if x == "LASSO":
                                return LogisticRegression(solver='liblinear',
                                               penalty='l1', C = c)
                            elif x == "RIDGE":
                                return LogisticRegression(solver='liblinear',
                                               penalty='l2', C = c)
                            elif x == "SVM-LINEAR":
                                return SVC(kernel='linear', probability=True,
                                           C = c)
                        for penalty_c in [20, 10, 5, 3, 2, 1.5, 1, 0.5, 0.4,
                                          0.3, 0.2, 0.1, 0.0545, 0.04, 0.03,
                                          0.02,
                                          0.01, 0.009,
                                          0.005, 0.004, 0.003, 0.002, 0.001,
                                          0.0005, 0.0004, 0.0003, 0.0002, 0.0001,
                                          0.00005, 0.00004, 0.00003, 2.8e-5,
                                          2.5e-5, 0.00002,
                                          1.8e-5, 1.5e-5,
                                          0.00001,
                                          0.000009,
                                          0.000007, 6e-6, 5e-6, 4e-6, 3e-6, 2e-6,
                                          1e-6]:
                            model = model_converter(name, penalty_c)
                            probas_ = model.fit(
                                X_train_aug, y_train_aug)\
                                .predict_proba(X_test_transf)
                            coeff_cutoff = {'RIDGE': 9.99e-3, 'SVM-LINEAR':9.99e-4}
                            if name in ["RIDGE", "SVM-LINEAR"]:
                                ok = np.abs(model.coef_[0])
                                mask = ok < coeff_cutoff[name]
                                ok[mask] = 0
                                nonzero_coef_num = np.count_nonzero(ok)
                            else:
                                nonzero_coef_num = np.count_nonzero(np.abs(model.coef_[0]))
                            assert 0 < nonzero_coef_num
                            print(f"{name} {penalty_c} {nonzero_coef_num}")
                            if nonzero_coef_num <= round(0.2 * X_train_aug.shape[1]):
                                print(f"ok {name} {penalty_c} {nonzero_coef_num}")
                                name2model[name] = model
                                model2numfeats[name][test_index] = nonzero_coef_num
                                break
                    else:
                        raise ValueError(f'hi {name}')

                    probas_true = [p[1] for p in probas_]
                    y_pred = name2model[name].predict(X_test_transf)
                    model2pred[name][test_index] = y_pred
                    model2proba[name][test_index] = probas_true
                    model2tru[name][test_index] = y_test

                    if "STACK" in name:

                        model2comp[name][test_index,thrsh_int_mask] = np.abs(
                            name2model[name].indivd_coefs_[0])
                        assert len(test_index) == 1
                        stack2feat[test_index[0]] = name2model[name]\
                            .predict_conv_func(X_test_transf)
                        assert len(model2comp[name][test_index][0]) == len(featgrp_names)
                    elif name in ["LASSO", 'RIDGE', "SVM-LINEAR"]:
                        model2comp[name][test_index,thrsh_int_mask] = np.abs(
                            name2model[name].coef_[0])
                        assert len(model2comp[name][test_index][0]) == len(featgrp_names)
                    elif name == "LR":
                        model2comp[name][test_index,thrsh_int_mask] = np.zeros(
                            (len(df_merged.columns),))
                    else:
                        raise ValueError("hi")

            for model in model2tru:
                assert iiii == 1 or not sum(df_y["class"].values != model2tru[model])
                assert iiii == 1 or sum(model2tru[model]) > 0
            experiment2models[experiment_name] = [model2pred, model2proba,
                                                  model2tru, df_y['class'].values,
                                                  model2comp, stack2feat,
                                                  featgrp_names, model2numfeats]

            for name in model_names:
                print(name, np.mean(model2pred[name] == model2tru[name]))
            for name in model_names:
                print(name, model2numfeats[name])
    import pickle
    import os

    # NOTE toggle these for generation of null
    # os.makedirs(f"out{formatted_timestamp}-null", exist_ok=True)
    os.makedirs(f"out{formatted_timestamp}", exist_ok=True)
    # with open(f"out{formatted_timestamp}-null/experiment2models_{index_num['name']}.pkl", "wb") as f:
    with open(f"out{formatted_timestamp}/experiment2models_{index_num['name']}.pkl", "wb") as f:
        pickle.dump(experiment2models, f)

def thresh_exp():
    prot_thr = [1, 1.1, 1.2, 1.3, 2]
    meta_thr = [1, 1.1, 1.2, 1.3, 2]
    lipid_thr = [1, 1.1, 1.2, 1.3, 2]
    mirna_thr = [1, 1.1, 1.2, 1.3, 2, 3]
    combinations = list(itertools.product(prot_thr, meta_thr, lipid_thr, mirna_thr))
    comb_dic_lis = []
    for cmb in combinations:
        for cmb_i in range(5):
            comb_dic_lis.append({'prot':cmb[0], 'meta':cmb[1], 'lipid':cmb[2], 'mirna':cmb[3],
                                'name': f'{cmb_i}_{"_".join([str(c) for c in cmb])}'})
    return comb_dic_lis

def normal_exp():
    comb_dic_lis = []
    cmb = [1, 1, 1, 1]
    # for cmb_i in range(1000): # for null, only quadromics
    for cmb_i in range(20):
        comb_dic_lis.append({'prot':cmb[0], 'meta':cmb[1], 'lipid':cmb[2], 'mirna':cmb[3],
                            'name': f'{cmb_i}_{"_".join([str(c) for c in cmb])}'})
    return comb_dic_lis



if __name__ == '__main__':
    comb_dic_lis = normal_exp()
    ## toggle next two for one run
    print(comb_dic_lis[-2])
    exit()
    # run_supervised_exp(comb_dic_lis[10])
    # exit()

    arguments = comb_dic_lis
    def run_wrapper(arg):
        return run_supervised_exp(arg)

    import multiprocessing
    def run_in_parallel():
        num_processes = 10#15#multiprocessing.cpu_count()

        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(run_supervised_exp, arguments)

        return results


    results = run_in_parallel()
    print(results)




