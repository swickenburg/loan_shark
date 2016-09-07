#!/usr/bin/python
# -*- coding: utf-8 -*-
# can't fit entire script

from sklearn.linear_model import LogisticRegression
import skll
from sklearn import preprocessing
from sklearn.neighbors import KernelDensity
import brewer2mpl
import cPickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from palettable.tableau import GreenOrange_12, Tableau_10, Tableau_20
my_map = brewer2mpl.get_map('Set1', 'qualitative', 8).mpl_colors
my_map[5] = (139 / 255., 139 / 255., 131 / 255.)
almost_black = '#262626'
my_map[1] = (0.21568627450980393, 0.49411764705882355, 0.82)
dark_grey = (150 / 255., 150 / 255., 150 / 255.)


def main():

    # make_data_set()

    df = load_data()
    df_grade = make_data_grade(df)
    mean_return_list_model = make_return_model(df, num_samples=500,
            top_loans=10)
    (x, pdf_list) = combine_pdf_list(df, df_grade,
            mean_return_list_model)
    plot_pdf(x, pdf_list)
    plot_default_rate(df, df_grade)


def make_data_set():
    df_b = pd.read_csv('data/LoanStats3b.csv', header=1, delimiter=r","
                       , dtype=str)
    df_c = pd.read_csv('data/LoanStats3c.csv', header=1, delimiter=r","
                       , dtype=str)
    df_d = pd.read_csv('data/LoanStats3d.csv', header=1, delimiter=r","
                       , dtype=str)
    df_e = pd.read_csv('data/LoanStats3e.csv', header=1, delimiter=r","
                       , dtype=str)

    df_all = pd.concat((df_b, df_c, df_d, df_e))
    idx = df_all['term'] == ' 36 months'
    df = df_all.ix[idx]

    # unique id for loans

    df.index = np.arange(len(df))

    idx = df['loan_amnt'].notnull()
    df = df.ix[idx]
    df = df.ix[(df['loan_status'] != 'Current') & (df['loan_status']
               != 'In Grace Period')]
    idx = df['loan_status'] == 'Fully Paid'
    df.ix[idx, 'loan_status'] = 1.
    df.ix[~idx, 'loan_status'] = 0.

    save_pickle(df, 'data/loan_df')


def make_data_grade(df):
    columns_cat = np.array([
        'grade',
        'sub_grade',
        'home_ownership',
        'verification_status',
        'loan_status',
        'pymnt_plan',
        'purpose',
        'initial_list_status',
        ])

    df_cat = df.copy()
    df_cat_convert = make_grade_feature(df_cat)

    # pop nan rows

    idx_keep = np.sum(df_cat_convert.isnull(), axis=1) == 0
    df_cat_convert = df_cat_convert.ix[idx_keep]
    df_cat_convert['return'] = df_cat_convert['total_pymnt'] \
        / df_cat_convert['loan_amnt']
    return df_cat_convert


def make_grade_feature(df):
    df_0 = df.copy()
    feature_dict = dict(zip(np.unique(df_0['sub_grade']),
                        np.linspace(1, 0, len(np.unique(df_0['sub_grade'
                        ])))))
    int_rate = df_0['int_rate'].astype(str)

    df_0['int_rate'] = np.array([float(item.split('%')[0]) for item in
                                int_rate])
    for key in feature_dict:

#         print key

        idx = df_0['sub_grade'] == key
        df_0.ix[idx, 'sub_grade'] = feature_dict[key]

    return df_0[['sub_grade', 'int_rate', 'loan_status', 'loan_amnt',
                'total_pymnt']].astype(float)


def make_grade_dist(df_grade, num_loans=20):
    print 'make distributions from loan grades'

    # make sub_grade populations

    sub_grade_pop = []

#     sub_grade_pop.append(df_grade)

    sub_grade_range = np.linspace(0, 1, 5)
    for (i, sub_grade) in enumerate(sub_grade_range[:-1]):
        sub_grade_min = sub_grade_range[i]
        sub_grade_max = sub_grade_range[i + 1]

        sub_grade_pop.append(df_grade.ix[(df_grade['sub_grade']
                             < sub_grade_max) & (df_grade['sub_grade']
                             >= sub_grade_min)])

    # select certain number of loans based on subgrade and get expected return

    mean_return_list = []
    for (j, df_sub) in enumerate(sub_grade_pop):
        print j
        num_samples = 1000
        mean_return_list_0 = np.zeros(num_samples)
        for i in range(num_samples):
            idx = np.random.choice(df_sub.index, num_loans)
            df_sel = df_sub.ix[idx]

    #         mean_return = ((df_sel['total_pymnt']/df_sel['loan_amnt']))

            mean_return_list_0[i] = np.mean(df_sel['return'])
        mean_return_list.append(mean_return_list_0)

    return mean_return_list


def make_return_model(df, num_samples=100, top_loans=10):
    '''Select certain number of loans based on subgrade and get expected return.'''

    print 'make distribution from model'
    mean_return_list_model = np.zeros(num_samples)
    for i in range(num_samples):
        print i

        (
            X_train,
            X_val,
            y_train,
            y_val,
            df_train,
            df_val,
            ) = make_data_clf(df)

        clf = LogisticRegression()

        clf.fit(X_train, y_train)

        y_pred_prob = clf.predict_proba(X_val)[:, 1]

        # y_pred = np.zeros_like(y_val)
        # y_pred[y_pred_prob > 0.5] = 1.
        # y_pred[y_pred_prob <= 0.5] = 0.

        int_rate = np.array(df_val['int_rate'])
        idx_sort = np.argsort(y_pred_prob * int_rate)[::-1]

        # highest n probabilities

        idx_high_prob = idx_sort[:top_loans]
        df_sel = df_val.ix[df_val.index[idx_high_prob]]

        mean_return = df_sel['total_pymnt'] / df_sel['loan_amnt']
        mean_return_list_model[i] = np.mean(mean_return)

    #     print 'kappa: ' + str(skll.metrics.kappa(y_val, y_pred))

        print 'mean return : ' + str(mean_return_list_model[i])
    return [mean_return_list_model]


def make_data_clf(df):
    columns_cont = np.array([
        'total_pymnt',
        'loan_amnt',
        'int_rate',
        'installment',
        'annual_inc',
        'loan_status',
        'desc',
        'dti',
        'delinq_2yrs',
        'inq_last_6mths',
        'mths_since_last_delinq',
        'mths_since_last_record',
        'open_acc',
        'pub_rec',
        'revol_bal',
        'revol_util',
        'total_acc',
        'acc_open_past_24mths',
        'total_rev_hi_lim',
        'avg_cur_bal',
        'bc_open_to_buy',
        'bc_util',
        'delinq_amnt',
        'mo_sin_old_il_acct',
        'mo_sin_old_rev_tl_op',
        'mo_sin_rcnt_rev_tl_op',
        'mo_sin_rcnt_tl',
        'mort_acc',
        'mths_since_recent_bc',
        'num_accts_ever_120_pd',
        'num_actv_bc_tl',
        'num_actv_rev_tl',
        'num_bc_sats',
        'num_bc_tl',
        'num_il_tl',
        'num_op_rev_tl',
        'num_rev_accts',
        'num_rev_tl_bal_gt_0',
        'num_sats',
        'num_tl_120dpd_2m',
        'num_tl_30dpd',
        'num_tl_90g_dpd_24m',
        'num_tl_op_past_12m',
        'pct_tl_nvr_dlq',
        'percent_bc_gt_75',
        'pub_rec_bankruptcies',
        'tax_liens',
        'tot_hi_cred_lim',
        'total_bal_ex_mort',
        'total_bc_limit',
        'total_il_high_credit_limit',
        ])

#     select only some grades

    idx_high_grade = (df['grade'] != 'A') & (df['grade'] != 'B') \
        & (df['grade'] != 'C')
    df = df.ix[idx_high_grade]

    df_cont = df.copy()
    df_cont = df_cont[columns_cont]
    df_cont_convert = make_df_cont(df_cont)

    # pop nan rows

    idx_keep = np.sum(df_cont_convert.isnull(), axis=1) == 0
    df_cont_convert = df_cont_convert.ix[idx_keep]

    df_cont_convert['inc_inst'] = df_cont_convert['annual_inc'] \
        / df_cont_convert['installment']

    # randomize

    df_cont_convert = \
        df_cont_convert.ix[np.random.permutation(df_cont_convert.index)]

    (
        X_train_cont,
        X_val_cont,
        y_train_cont,
        y_val_cont,
        df_train,
        df_val,
        ) = make_train_val(df_cont_convert)

    scaler = preprocessing.StandardScaler().fit(X_train_cont)
    X_train_cont = scaler.transform(X_train_cont)
    X_val_cont = scaler.transform(X_val_cont)

    return (
        X_train_cont,
        X_val_cont,
        y_train_cont,
        y_val_cont,
        df_train,
        df_val,
        )


def make_df_cont(df):
    df_convert = df.copy()
    int_rate = df_convert['int_rate'].astype(str)
    df_convert['int_rate'] = np.array([float(item.split('%')[0])
            for item in int_rate])

#     df_convert = df_convert.drop(['int_rate'],1)

    desc = df['desc'].astype(str)
    df_convert['desc'] = np.array([len(item) for item in desc])

    df_convert.ix[df_convert['mths_since_last_delinq'].isnull(),
                  'mths_since_last_delinq'] = 10000.

    df_convert.ix[df_convert['mths_since_last_record'].isnull(),
                  'mths_since_last_record'] = 10000.

    revol_util = df_convert['revol_util'].astype(str)
    df_convert['revol_util'] = np.array([float(item.split('%')[0])
            for item in revol_util])

    return df_convert.astype(float)


def make_train_val(df):
    df_target = df['loan_status'].astype(int)

    # drop the target columns

    df_feat = df.drop(['loan_status', 'total_pymnt'], 1)
    X = np.array(df_feat)
    y = np.array(df_target)

    # train val split, only make class count even for train

    num_train = int(0.7 * len(y))
    X_train = X[:num_train]
    y_train = y[:num_train]

    X_val = X[num_train:]
    y_val = y[num_train:]

    df_train = df.iloc[:num_train]
    df_val = df.iloc[num_train:]

    # make class count even

    target_min = np.argmin([np.sum(y_train == 0.), np.sum(y_train
                           == 1.)])
    target_max = np.argmax([np.sum(y_train == 0.), np.sum(y_train
                           == 1.)])

    i_y_min = np.where(y_train == target_min)[0]
    i_y_max = np.where(y_train == target_max)[0]
    i_y_max = np.random.choice(i_y_max, len(i_y_min), replace=False)
    i_even = np.sort(np.concatenate((i_y_min, i_y_max)))

    X_train = X_train[i_even]
    y_train = y_train[i_even]
    df_train = df_train.iloc[i_even]

    return (
        X_train,
        X_val,
        y_train,
        y_val,
        df_train,
        df_val,
        )


def combine_pdf_list(df, df_grade, mean_return_list_model):

    x = np.arange(0, 1.45, 0.001)
    mean_return_list = [np.array(df_grade['return'])]
    mean_return_list = mean_return_list + make_grade_dist(df_grade,
            num_loans=100)

    pdf_list = make_pdf(mean_return_list, x, bw=0.03)
    pdf_model = make_pdf(mean_return_list_model, x, bw=0.02)

    pdf_list = pdf_list + pdf_model
    return (x, pdf_list)


def plot_default_rate(df, df_grade):

    grade_list = np.unique(df['sub_grade'])[::-1]
    int_rate_list = []
    default_rate_list = []
    for (i, grade) in enumerate(np.unique(df_grade['sub_grade'])):
        idx = df_grade['sub_grade'] == grade
        int_rate_list.append(np.mean(np.array(df_grade.ix[idx,
                             'int_rate'])) / 100.)

#         print np.sum(df_grade.ix[idx,'loan_status']==0.)

        default_rate_list.append(float(np.sum(df_grade.ix[idx,
                                 'loan_status'] == 0.))
                                 / len(df_grade.ix[idx, 'loan_status']))

    (int_rate_list, default_rate_list) = (np.array(int_rate_list),
            np.array(default_rate_list))
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    for j in np.arange(5, 36, 5):
        plt.axvline(j, ls='--', color=dark_grey, lw=2)
    plt.gca().set_color_cycle(Tableau_10.mpl_colors)
    ax.plot(np.arange(1, len(default_rate_list) + 1),
            default_rate_list, lw=5)
    ax.plot(np.arange(1, len(int_rate_list) + 1), int_rate_list, lw=5)

    spines_to_remove = ['top', 'right']
    for spine in spines_to_remove:
        ax.spines[spine].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.yticks(fontsize=20)
    plt.xticks([])
    plt.xlim(1, len(default_rate_list))

    plt.savefig('figures/fig1.pdf', bbox_inches='tight')


def kde_sklearn(
    x,
    x_grid,
    bandwidth=0.2,
    **kwargs
    ):
    """Kernel Density Estimation with Scikit-learn"""

    # kde from https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/

    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x[:, np.newaxis])

    # score_samples() returns the log-likelihood of the samples

    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)


def make_pdf(dist_list, x, bw=0.3):

    pdf_list = []
    for (j, dist) in enumerate(dist_list):
        len_p_cut = len(dist)
        if len_p_cut > 0:
            pdf_list.append(kde_sklearn(dist, x, bandwidth=bw))
        else:
            pdf_list.append(np.nan)

    for i in range(len(pdf_list)):
        pdf_list[i] = pdf_list[i] / np.max(pdf_list[i])
    return pdf_list


def plot_pdf(x, pdf_list):

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    ax.fill_between(
        x,
        pdf_list[0],
        0,
        facecolor=dark_grey,
        edgecolor=dark_grey,
        lw=3,
        alpha=0.8,
        )
    plt.axvline(1, ls='--', color=almost_black, lw=2)

    for (j, pdf) in enumerate(pdf_list[1:]):
        if j == len(pdf_list[1:]) - 1:
            ax.plot(x, pdf, c=Tableau_10.mpl_colors[1], lw=5)
        elif ~np.isnan(pdf).any():
            ax.plot(x, pdf, c=Tableau_10.mpl_colors[0], lw=5, alpha=0.2
                    * j + 0.4)

    plt.yticks([])
    spines_to_remove = ['top', 'right']
    for spine in spines_to_remove:
        ax.spines[spine].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.xticks(np.arange(0, 1.45, 0.1), fontsize=20)

    plt.xlim(0, 1.45)
    plt.ylim(0.008, 1.02)
    plt.savefig('figures/fig2.pdf', bbox_inches='tight')


def save_pickle(df, file_name, ext=False):
    if ext:
        file_path = file_name
    else:
        file_path = file_name + '.cpickle'
    with open(file_path, 'wr') as f:
        cPickle.dump(df, f, protocol=2)


def load_pickle(file_name, ext=False):
    if ext:
        file_path = file_name
    else:
        file_path = file_name + '.cpickle'
    with open(file_path, 'rb') as f:
        df = cPickle.load(f)
    return df


def load_data():
    df = load_pickle('data/loan_df')
    return df


if __name__ == '__main__':
    main()
