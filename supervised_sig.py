import pandas as pd
from sklearn.decomposition import PCA
import math
from sklearn.model_selection import LeaveOneOut
import numpy as np
import matplotlib.pyplot as plt
import os
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
    "data/Source multi-omics datasets.xlsx", sheet_name='Proteomics')
df_lipid = pd.read_excel("data/Source multi-omics datasets.xlsx", sheet_name='Lipidomics')\
    .drop(labels=['Class', 'FA', 'FA Group Key'], axis=1)
df_meta = pd.read_excel("data/Source multi-omics datasets.xlsx", sheet_name='Metabolomics')\
    .drop(labels=['CAS'], axis=1)
df_mirna = pd.read_excel("data/Source multi-omics datasets.xlsx", sheet_name='miRNAs')\
    .drop(labels=['Mature miRNA Accession'], axis=1)


def fix_header_col(df, col, replacement_prefix):
    replacement_counter = 1
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

# print(le.inverse_transform([0, 1]))
# print(df_merged.groupby(by='class').size())

def get_fold_changes(df_X, df_y, test_columns, model_name):
    df_X['class'] = df_y['class'].copy()
    healthy_data = df_X[df_X['class'] == le.transform(['Healthy'])[0]]
    case_data = df_X[df_X['class'] == le.transform(['T1D High-Risk'])[0]]
    df_X.drop(columns=['class'], inplace=True)
    fold_changes = {}

    # Calculate fold change for each feature
    for feature in test_columns:
        case_mean = case_data[feature].mean()
        healthy_mean = healthy_data[feature].mean()

        if healthy_mean != 0:  # Avoid division by zero
            # if feature X is 2 for high-risk and 1 for healthy,
            # then ratio is 2, two-fold change
            fold_change = case_mean / healthy_mean
            fold_changes[feature] = fold_change
        else:
            # fold_changes[feature] = None  # Handle division by zero case
            raise ValueError("healthy_mean = 0")

    fld_chng_df = pd.DataFrame(fold_changes.items(),
                                   columns=['Feature', 'Fold Change'])

    def which_omic(x):
        for o in ["prot", "mirna", "meta", "lipid"]:
            if o in x:
                return o
        raise ValueError("g")

    fld_chng_df['omic'] = fld_chng_df['Feature'].apply(which_omic)
    fld_chng_df['model'] = model_name

    thrsh_list = [1, 1.1, 1.2, 1.3, 2, 3]
    for thresh in thrsh_list:
        fld_chng_df[f"t{thresh}"] = \
            (fld_chng_df["Fold Change"] > thresh) | (fld_chng_df["Fold Change"] < 1/thresh)

    os.makedirs(f"out111323-sig", exist_ok=True)
    return fld_chng_df


print("prot", (len(df_prot_t), len(df_prot_t.columns)))
print("meta", (len(df_meta_t), len(df_meta_t.columns)))
print("lipid", (len(df_lipid_t), len(df_lipid_t.columns)))
print("mirna", (len(df_mirna_t), len(df_mirna_t.columns)))
print((len(df_y), len(df_y.columns)))

all_df = [[df_prot_t, 'prot'],
          [df_meta_t, 'meta'],
          [df_lipid_t, 'lipid'],
          [df_mirna_t, 'mirna']]

comp2len = {'prot': len(df_prot_t.columns),
            'meta': len(df_meta_t.columns),
            'lipid': len(df_lipid_t.columns),
            'mirna': len(df_mirna_t.columns)}


def feat_sets_present(sig_feats):
    present = set()
    for feat in sig_feats:
        if "prot" in feat:
            present.add("prot")
        elif "meta" in feat:
            present.add("meta")
        elif "lipid" in feat:
            present.add("lipid")
        elif "mirna" in feat:
            present.add("mirna")
        else:
            raise ValueError("hey")
    return "_".join(sorted(list(present)))


def sig_exp():
    import pickle
    from pprint import pprint
    cmb = [1, 1, 1, 1]

    tm_stmp = "111323"
    with open("data/feat_obs_const_111323.pkl", "rb") as f:
        [feat_obs_const, model2obs_ylabels] = pickle.load(f)
    with open("data/feat_null_const_111323.pkl", "rb") as f:
        [feat_null_const, model2null_ylabels] = pickle.load(f)

    sig_lis = []
    df_merged = pd.concat([df[0] for df in all_df], axis=1)
    df_fc_lis = []
    for model, const_feats in feat_obs_const.items():
        sig_lis.append(const_feats)
        df_fc_lis.append(get_fold_changes(df_merged, df_y, const_feats, model))

    pd.concat(df_fc_lis).to_csv(f"out111323-sig/cnst_feat_fc_111323.csv", index=False)
    os.makedirs("out111323-sig", exist_ok=True)
    print("out111323-sig/cnst_feat_fc_111323.csv")


sig_exp()
