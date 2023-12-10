import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
# import IPython
from tqdm import tqdm
from collections import Counter
from sklearn.ensemble import BaggingClassifier
import json
import copy
from sklearn.decomposition import PCA
import itertools
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import LeaveOneOut


class MyStackingClassifer():
    def __init__(self, base_clf, hidden_size_m=(45, 45),
                 hidden_size_indiv=(50, 50), C=None):
        self.model = None
        self.conv_func = None
        self.master_coef_ = None
        self.base_clf_use = base_clf
        self.hidden_size_m = hidden_size_m
        self.hidden_size_indiv = hidden_size_indiv
        self.C = C

    def fit(self, df_X, df_y, comp2len, experiment_name):
        prot_len = None
        meta_len = None
        lipid_len = None
        mirna_len = None
        for comp in comp2len:
            if comp == "prot":
                prot_len = comp2len[comp]
            elif comp == "meta":
                meta_len = comp2len[comp]
            elif comp == "lipid":
                lipid_len = comp2len[comp]
            elif comp == "mirna":
                mirna_len = comp2len[comp]
        assert prot_len is not None
        assert meta_len is not None
        assert lipid_len is not None
        assert mirna_len is not None
        assert self.C is not None


        clf, conv_func, indivd_coefs = train_master(pd.DataFrame(
                                        df_X), df_y,
                                      self.base_clf_use,
                                      experiment_name,
                                      prot_len,
                                      meta_len,
                                      lipid_len,
                                      mirna_len,
                                      self.hidden_size_m, self.hidden_size_indiv,
                                      C=self.C)
        self.model = clf
        self.conv_func = conv_func
        self.indivd_coefs_ = indivd_coefs
        self.master_coef_ = clf.coef_
        return self

    def predict_conv_func(self,X_test):
        return self.conv_func(pd.DataFrame(X_test))

    def predict(self, X_test):
        if np.mean(self.conv_func(pd.DataFrame(X_test))) < 0.5:
            return 1
        else:
            return 0
        # return self.model.predict(self.conv_func(pd.DataFrame(X_test)))

    def predict_proba(self, X_test):
        # return np.mean(self.conv_func(pd.DataFrame(X_test)))
        return self.model.predict_proba(self.conv_func(pd.DataFrame(X_test)))



def train_individual(df_X, df_y, ft_grp,
                     hidden_size=None, clf_use="RIDGE", C=None):
    # subset examples variable-wise to a single feature set
    # assert ft_grp in [0, 1, 2, 3]
    assert C is not None
    print(f"feature group: {ft_grp} {clf_use}")
    if type(ft_grp[0]) == tuple:
        raise ValueError("hi")
        sub_df_lis = []
        for tup_grp in ft_grp:
            sub_df_lis.append(df_X.iloc[:, tup_grp[0]:tup_grp[1]])
        sub_X = pd.concat(sub_df_lis, axis=1)
    else:
        sub_X = df_X.iloc[:, ft_grp[0]:ft_grp[1]]
    # assert len(df_X.columns) == ftgrp2lens[ftgrp_name]
    # clf = MLPClassifier(
    #                     max_iter=50, solver='lbfgs')
    # clf = LinearDiscriminantAnalysis()
    if clf_use == "MLP":
        raise ValueError("hi")
        # clf = LogisticRegression(solver='liblinear', penalty='l2')
        # clf = SVC(kernel="linear", probability=True)
        # clf = LinearDiscriminantAnalysis()
        # clf = RandomForestClassifier()
        clf = MLPClassifier(max_iter=50, solver='lbfgs')
    elif clf_use == "SVM":
        clf = SVC(kernel="linear", probability=True, C=C)
        # clf = LogisticRegression(solver='liblinear', penalty='l1')
    elif clf_use == "LASSO":
        clf = LogisticRegression(solver='liblinear', penalty='l1', C=C)
    elif clf_use == "RIDGE":
        clf = LogisticRegression(solver='liblinear', penalty='l2', C=C)
    elif clf_use == "LR":
        clf = LogisticRegression(penalty=None)
    else:
        raise ValueError("hi")
        # clf = KNeighborsClassifier(n_neighbors=15)
    # clf = np.random.choice([LogisticRegression(solver='liblinear', penalty='l2'),
    #                         LogisticRegression(solver='liblinear', penalty='l1'),
    #                         MLPClassifier(max_iter=50, solver='lbfgs')])
    # clf = GaussianNB()
    return clf.fit(sub_X, df_y)


def train_master(df_X, df_y, base_clf_use, experiment_name,
                 prot_len, meta_len, lipid_len, mirna_len,
                 hidden_size_m, hidden_size_indiv, C):
    # y_values = list(set(y))
    # get a list of n choose 2 combinations
    # pairwise = list(itertools.combinations(y_values, 2))
    indivd_clfs = []  # list of individual learners
    # plot_tsne(X, y, exit_when_done=False)
    # ft_lis = [ft2lens[l] for l in lens2ftgrp[len(df_X.columns)].split("_")]

    ftgrp_names = experiment_name  # lens2ftgrp[len(df_X.columns)]

    # prot, meta, lipid, mirna
    col_order = ["prot","meta","lipid","mirna"]
    col_lens = [prot_len, meta_len, lipid_len, mirna_len]  # adjust as needed
    # col_lens = [1714, 122, 65, 329]  # adjust as needed
    col_locs = [0]

    to_del = []
    for i in range(len(col_order)):
        if col_order[i] not in ftgrp_names.split("_"):
            to_del.append(i)
    for index in sorted(to_del, reverse=True):
        del col_order[index]
        del col_lens[index]

    for i in range(len(col_lens)):
        col_locs.append(np.sum(col_lens[:i+1]))
    col_loc_tups = [(col_locs[i], col_locs[i + 1])
                    for i in range(len(col_locs) - 1)]
    # ft2loc = {"prot": 0, "meta": 1, "lipid": 2, "mirna": 3}

    # ft_pos_lis = [col_loc_tups[ft2loc[o]] for o in ftgrp_name.split("_")]
    # ft_pos_lis = [0]
    # for i in range(len(ft_lis)):
    #     ft_pos_lis.append(np.sum(ft_lis[:i+1]))
    # ft_pos_lis = [(ft_pos_lis[i], ft_pos_lis[i + 1])
    #               for i in range(len(ft_pos_lis) - 1)]
    groupings = copy.deepcopy(col_loc_tups)
    # NOTE toggle this for loop if want to switch between a stacking that purely consists
    # of one-omic learners, or also incorporates learners up to (n-1)-omics where n is
    # the total number of -omics features in the input dataset
    # for i in range(2, len(col_loc_tups) - 1):
    #     [groupings.append(fg) for fg in list(itertools.combinations(col_loc_tups, i))]
    # NOTE toggle following if want to incorporate other learners into the stacking scheme
    groupings1 = copy.deepcopy(groupings)
    # groupings2 = copy.deepcopy(groupings)
    for i in range(len(groupings1)):
        groupings1[i] = [groupings1[i], "RIDGE"]
    # for i in range(len(groupings2)):
    #     groupings2[i] = [groupings2[i], "LASSO"]
    # groupings = groupings1 + groupings2
    groupings = groupings1

    for ft_grp, _ in groupings:
        indivd_clfs.append(
            train_individual(df_X, df_y, ft_grp,
                             clf_use=base_clf_use, C=C))
    assert len(indivd_clfs) == len(groupings)
    def converter(df_X, y_val=None, keep_x=False):
        # transform the x to a new x with appropriate dimension
        tmp = []
        # index = 0
        for index, clf in enumerate(indivd_clfs):
            if type(groupings[index][0][0]) == tuple:
                raise ValueError("hi")
                sub_df_lis = []
                for tup_grp in groupings[index][0]:
                    sub_df_lis.append(df_X.iloc[:, tup_grp[0]:tup_grp[1]])
                sub_X = pd.concat(sub_df_lis, axis=1)
            else:
                # if index >= len(col_locs) - 1:
                #     index = 0
                sub_X = df_X.iloc[:, col_locs[index]:col_locs[index+1]]
            #sub_X = df_X.iloc[:, col_locs[index]:col_locs[index+1]]
            tmp.append(clf.predict_proba(sub_X)[:, 0, None])
            # index += 1
        tmp = np.hstack(tmp)

        if keep_x:
            # prefers the hybrid model instead of stacking
            tmp.extend(x)
        x = tmp
        return x
        # if y_val is None:
        #     # when testing an example, don't zero out any components
        #     return x
        # else:
        #     # we know which to zero out
        #     for i in range(len(pairwise)):
        #         pair = pairwise[i]
        #         # assert y_val in [2,4,5,7,9]
        #         if y_val not in pair:
        #             x[2*i] = 0
        #             x[2*i+1] = 0

        #     return x

    X_new = converter(df_X)  # need to create a new dataset
    # uses the negative signal as an additional feature isnt good as it can
    # be derived from the other class
    # for index, x in enumerate(df_X):
    #     X_new.append(converter(x))
        # X_new.append(converter(x))  # won't set components to 0 deliberately

    # try 3 or 4
    # pca = PCA(n_components=3, svd_solver='auto')
    # X_new = pca.fit_transform(X_new)
    # selected_feats = []
    # for clf_i, (ft_grp, _) in enumerate(groupings):
    #     subsub_X = df_X.iloc[:, ft_grp[0]:ft_grp[1]].copy()
    #     top_feat_inds = np.abs(indivd_clfs[clf_i].coef_[0]).argsort()[::-1][:50]
    #     selected_feats.extend(list(subsub_X.iloc[:, top_feat_inds].columns))
    # # selected_cols = ['lipid_LPC(18:1)','lipid_PC(38:6)','meta_alpha-Tocopherol','meta_1-Hexadecanol/Pentadecanoic acid','mirna_hsa-miR-3605-3p','mirna_hsa-miR-3605-5p','prot_sp|P32119|PRDX2_HUMAN','prot_sp|P00915|CAH1_HUMAN']

    # X_new = df_X.loc[:, selected_cols]#pd.concat(subsub_X_lis, axis=1)

    # def new_converter(df_X, y_val=None, keep_x=False):
    #     return df_X.loc[:, selected_cols]


    # plot_tsne(X_new, y, exit_when_done=True)

    clf = LogisticRegression(solver='liblinear', penalty='l2')
    # clf = SVC(kernel='linear', probability=True)
    # clf = MLPClassifier(
    #     max_iter=10, hidden_layer_sizes=hidden_size_m, solver='lbfgs')

    # toggle to use PCA for the master classifier as well
    # return clf.fit(X_new, df_y.values), lambda x: pca.transform([converter(x)])[0]
    # return clf.fit(X_new, df_y.values), lambda x: pca.transform(converter(x))
    if base_clf_use != "MLP":
        indivd_coefs = np.concatenate([indvd.coef_ for indvd in indivd_clfs],axis=1)
    else:
        indivd_coefs = np.zeros((1,len(df_X.columns)))
    return clf.fit(X_new, df_y.values), converter, indivd_coefs


