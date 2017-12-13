# mobius

import pandas as pd
import numpy as np
from time import time
import sys
# import gensim

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold

# from sklearn.linear_model import SGDClassifier
from gensim.models import Word2Vec
import lightgbm as lgb
import xgboost as xgb
import logging
import gc
import pickle
import json

from functools import wraps

class F1Optimizer():
    def __init__(self):
        pass

    @staticmethod
    def get_expectations(P, pNone=None):
        expectations = []
        P = np.sort(P)[::-1]

        n = np.array(P).shape[0]
        DP_C = np.zeros((n + 2, n + 1))
        if pNone is None:
            pNone = (1.0 - P).prod()

        DP_C[0][0] = 1.0
        for j in range(1, n):
            DP_C[0][j] = (1.0 - P[j - 1]) * DP_C[0, j - 1]

        for i in range(1, n + 1):
            DP_C[i, i] = DP_C[i - 1, i - 1] * P[i - 1]
            for j in range(i + 1, n + 1):
                DP_C[i, j] = P[j - 1] * DP_C[i - 1, j - 1] + (1.0 - P[j - 1]) * DP_C[i, j - 1]

        DP_S = np.zeros((2 * n + 1,))
        DP_SNone = np.zeros((2 * n + 1,))
        for i in range(1, 2 * n + 1):
            DP_S[i] = 1. / (1. * i)
            DP_SNone[i] = 1. / (1. * i + 1)
        for k in range(n + 1)[::-1]:
            f1 = 0
            f1None = 0
            for k1 in range(n + 1):
                f1 += 2 * k1 * DP_C[k1][k] * DP_S[k + k1]
                f1None += 2 * k1 * DP_C[k1][k] * DP_SNone[k + k1]
            for i in range(1, 2 * k - 1):
                DP_S[i] = (1 - P[k - 1]) * DP_S[i] + P[k - 1] * DP_S[i + 1]
                DP_SNone[i] = (1 - P[k - 1]) * DP_SNone[i] + P[k - 1] * DP_SNone[i + 1]
            expectations.append([f1None + 2 * pNone / (2 + k), f1])

        return np.array(expectations[::-1]).T

    @staticmethod
    def maximize_expectation(P, pNone=None):
        expectations = F1Optimizer.get_expectations(P, pNone)

        ix_max = np.unravel_index(expectations.argmax(), expectations.shape)
        max_f1 = expectations[ix_max]

        predNone = True if ix_max[0] == 0 else False
        best_k = ix_max[1]

        return best_k, predNone, max_f1

    @staticmethod
    def _F1(tp, fp, fn):
        return 2 * tp / (2 * tp + fp + fn)

    @staticmethod
    def _Fbeta(tp, fp, fn, beta=1.0):
        beta_squared = beta ** 2
        return (1.0 + beta_squared) * tp / ((1.0 + beta_squared) * tp + fp + beta_squared * fn)

def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time()
        result = function(*args, **kwargs)
        t1 = time()
        print ("Total time running %s: %s seconds" %
               (function.func_name, str(t1-t0))
               )
        return result
    return function_timer

# https://stackoverflow.com/questions/42920537/finding-increasing-trend-in-pandas
def trendline(data, order=1):
    coeffs = np.polyfit(data.index.values, list(data), order)
    slope = coeffs[-2]
    return float(slope)

class Instacart:
    _products = None

    _aisles = None

    _departments = None

    _orders_prior = None

    _orders_train = None

    _tt_orders = None

    _orders = None

    _users = None

    _train = None

    _test = None

    _user_order = None

    _prior_orders = None

    _train_orders = None

    _prior_order_products = None

    _train_order_products = None

    _merged_order_products = None

    _selected_features = None
    # [
    #     'US_time_since_last_order', 'UP_orders_since_last_order', 'US_unique_to_total_products_bought',
    #     'UP_organic_percent', 'US_days_trend', 'US_average_basket', 'UP_favorite_department', 'US_favorite_product',
    #     'UP_favorite_aisle', 'US_avg_order_hour_of_day', 'US_avg_days_between_orders', 'US_avg_order_dow',
    #     'UP_favorite_cluster', 'US_day_span', 'PR_reorder_ratio', 'US_favorite_product_repetition',
    #     'US_total_unique_products_bought', 'UP_order_rate', 'UP_non_fattening_percent',
    #     'UP_order_rate_since_first_order', 'UP_gluten_free_percent', 'US_days_median', 'PR_unique_total_users',
    #     'UP_orders', 'UP_avg_add_to_cart_order', 'PR_mean_add_to_cart_order', 'UP_last_order',
    #     'PR_total_products_bought', 'PR_products_per_aisle', 'PR_reorder_times', 'US_total_products_bought',
    #     'PO_avg_product_importance', 'PR_mean_order_hour_of_day', 'PR_mean_order_dow', 'PR_cluster_id', 'aisle_id',
    #     'UP_first_order', 'PR_second_most_similar', 'order_hour_of_day', 'PR_reorder_probability',
    #     'OR_avg_aisle_per_hod', 'w2vec_pca_x', 'product_id', 'PR_most_similar', 'PR_num_of_characters', 'department_id',
    #     'PR_products_per_department', 'w2vec_pca_y', 'PO_max_product_importance', 'OR_num_of_orders_per_hod',
    #     'US_number_of_orders', 'order_dow', 'OR_avg_department_per_hod', 'OR_avg_cluster_per_hod', 'UP_vegan_percent',
    #     'UP_recycled_percent',
    #     # 'UP_sugar_free_percent',
    #     'OR_avg_aisle_per_dow', 'OR_avg_department_per_dow',
    #     'PR_num_of_words', 'OR_num_of_orders_per_dow', 'OR_avg_cluster_per_dow', 'OR_weekend'
    #     # , 'PR_organic',
    #     # 'UP_product_reorder_percentage', 'US_days_mean'
    # ]

    _xgb_params = {
        # "eta": 0.1,
        # "max_depth": 6,
        # "min_child_weight": 10,
        "eta": 0.03,
        "max_depth": 7,
        "min_child_weight": 9,
        "gamma": 0.70,
        "subsample": 0.76,
        "colsample_bytree": 0.95,
        "alpha": 2e-05,
        "lambda": 10,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'silent': 1,
        # 'updater': 'grow_gpu'
        # 'min_child_weight': 200
    }

    _lgb_params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'binary_logloss', 'auc'},
        'num_leaves': 256,
        'min_sum_hessian_in_leaf': 20,
        'max_depth': 12,
        'learning_rate': 0.05,
        'feature_fraction': 0.6,
        # 'bagging_fraction': 0.9,
        # 'bagging_freq': 3,
        'verbose': 1
    }

    def load(self, use_classifications=True):
        t0 = time()

        self._products = pd.read_csv('data/products.csv', dtype={
            'product_id': np.uint16,
            'order_id': np.int32,
            'aisle_id': np.uint8,
            'department_id': np.uint8
        })

        pc_file = 'data/product_clusters.csv'
        pc_path = Path(pc_file)

        if pc_path.is_file():
            ac = pd.read_csv(pc_file)

            # print(self._products.shape)
            self._products = pd.merge(self._products, ac, how='outer', on='product_id')
            # print(self._products.shape)

            del ac

            self._products.rename(columns={'cluster_id': 'PR_cluster_id'}, inplace=True)

            print("=> Loaded product cluster information from file")

        # self._departments = pd.read_csv('data/departments.csv')
        # self._aisles = pd.read_csv('data/aisles.csv')
        # print(self._products.shape)
        #
        # self._products = pd.merge(self._products, self._departments, on='department_id')
        # self._products = pd.merge(self._products, self._aisles, on='aisle_id')
        # print(self._products.shape)
        # self._products = self._products.sort_values(by='product_id', axis=0)
        # self._products.to_csv('merged-no-classified.csv', index=False)
        # exit()

        # missing_departments = self._products[self._products.department_id == 21]

        department_classification = Path("data/department-classification.csv")
        print("=> Products shape:", self._products.shape)

        if department_classification.is_file() and use_classifications is True:
            dc = pd.read_csv("data/department-classification.csv")

            self._products = pd.merge(self._products, dc, how='outer', on='product_id')
            self._products.department_id_x = np.where(self._products.department_id_x == 21,
                                                      self._products.department_id_y,
                                                      self._products.department_id_x)

            self._products.rename(columns={'department_id_x': 'department_id'}, inplace=True)
            self._products.drop(['department_id_y'], axis=1, inplace=True)

            del dc

        self._departments = pd.read_csv('data/departments.csv')
        self._products = pd.merge(self._products, self._departments, on='department_id')

        print("=> Products shape after products/departments merge:", self._products.shape)

        aisle_classification = Path("data/aisle-classification.csv")

        if aisle_classification.is_file() and use_classifications is True:
            ac = pd.read_csv("data/aisle-classification.csv")

            self._products = pd.merge(self._products, ac, how='outer', on='product_id')
            self._products.aisle_id_x = np.where(self._products.aisle_id_x == 100,
                                                 self._products.aisle_id_y,
                                                 self._products.aisle_id_x)

            self._products.rename(columns={'aisle_id_x': 'aisle_id'}, inplace=True)
            self._products.drop(['aisle_id_y'], axis=1, inplace=True)

            del ac

        self._aisles = pd.read_csv('data/aisles.csv')
        self._products = pd.merge(self._products, self._aisles, on='aisle_id')

        print("=> Products shape after aisles merge:", self._products.shape)

        self._products['product_id'] = self._products['product_id'].astype('int')
        self._products['department_id'] = self._products['department_id'].astype('int')
        self._products['aisle_id'] = self._products['aisle_id'].astype('int')

        if use_classifications is False:
            # Disable these substitutions if I was to use SMOTE
            self._products['aisle_id'] = np.where(self._products.aisle_id == 100,
                                                  np.nan,
                                                  self._products.aisle_id)

            self._products['department_id'] = np.where(self._products.department_id == 29,
                                                       np.nan,
                                                       self._products.department_id)

        # self._products = self._products.sort_values(by='product_id',axis=0)
        # self._products.to_csv('merged-classified.csv', index=False)

        self._orders = pd.read_csv('data/orders.csv', dtype={
            'order_id': np.int32,
            'user_id': np.int32,
            'eval_set': 'category',
            'order_number': np.int16,
            'order_dow': np.int8,
            'order_hour_of_day': np.int8,
            'days_since_prior_order': np.float32
        })

        self._user_order = self._orders[['user_id', 'order_id']].copy()

        self._orders_prior = pd.read_csv('data/order_products__prior.csv', dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8
        })

        print("=> Prior orders shape:", self._orders_prior.shape)
        self._orders_prior = pd.merge(self._orders_prior, self._user_order, on='order_id', how='left')

        print("=> Prior orders shape after adding userid on the order:", self._orders_prior.shape)

        self._orders_train = pd.read_csv('data/order_products__train.csv', dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8
        })

        print("=> Train orders shape:", self._orders_train.shape)
        self._orders_train = pd.merge(self._orders_train, self._user_order, on='order_id', how='left')

        print("=> Train orders shape after adding userid on the order:", self._orders_train.shape)

        # Get the prior order from the orders list
        self._prior_orders = self._orders[self._orders['eval_set'] == 'prior']

        # Get the train order from the orders list
        self._train_orders = self._orders[self._orders['eval_set'] == 'train']

        print("=> Prior orders shape:", self._prior_orders.shape)

        # and join them with the products
        self._prior_order_products = pd.merge(self._prior_orders, self._orders_prior, on='order_id', how='left')
        self._prior_order_products.drop('user_id_y', axis=1, inplace=True)
        self._prior_order_products.rename(columns={'user_id_x': 'user_id'}, inplace=True)

        print("=> Prior orders + products shape:", self._prior_order_products.shape)

        print("=> Train orders shape:", self._train_orders.shape)

        # and join them with the products
        self._train_order_products = pd.merge(self._train_orders, self._orders_train, on='order_id', how='left')
        self._train_order_products.drop('user_id_y', axis=1, inplace=True)
        self._train_order_products.rename(columns={'user_id_x': 'user_id'}, inplace=True)

        print("=> Train orders + products shape:", self._train_order_products.shape)

        self._merged_order_products = pd.concat([self._prior_order_products, self._train_order_products],
                                                ignore_index=True, axis=0)

        print("=> Concatenated Train + products | Prior + products shape:", self._merged_order_products.shape)

        # print(self._prior_order_products.shape)
        # self._prior_order_products = self._prior_order_products[self._prior_order_products['order_number'] <= 20]
        # print(self._prior_order_products.shape)

        # print(self._prior_order_products.head(1))

        # order_streaks = Path('data/order_streaks.csv')
        #
        # if order_streaks.is_file():
        #     order_streaks = pd.read_csv('data/order_streaks.csv',
        #                                 usecols=['order_id', 'user_id', 'product_id', 'streak'])
        #
        #     print(self._prior_order_products.shape)
        #     self._prior_order_products = pd.merge(self._prior_order_products, order_streaks,
        #                                           on=['order_id', 'user_id', 'product_id'], how='left')
        #
        #     print(self._prior_order_products.shape)

        print("=> Loading completed in %fs" % (time() - t0))

    def create_product_features(self):
        t0 = time()

        #
        # Mark organic products (i.e. having "organic" in product name)
        #

        self._products['PR_organic'] = np.where(self._products['product_name'].str.contains('organic', case=False), 1, 0)

        #
        # Non fattening products (i.e. most products containing 'fat'
        # are advertising the fact that they are not fattening
        #

        self._products['PR_non_fattening'] = np.where(self._products['product_name'].str.contains('fat', case=False), 1, 0)

        #
        # Recycled products
        #

        self._products['PR_recycled'] = np.where(self._products['product_name'].str.contains('recycle', case=False), 1, 0)

        #
        # Vegan products
        #

        self._products['PR_vegan'] = np.where(self._products['product_name'].str.contains('vegan', case=False), 1, 0)

        #
        # Sugar free products
        #

        self._products['PR_sugar_free'] = np.where((self._products['product_name'].str.contains('sugar', case=False)) &
                                                   (self._products['product_name'].str.contains('free', case=False)),
                                                   1, 0)

        #
        # Gluten free products (stupid ass hippies)
        #

        self._products['PR_gluten_free'] = np.where(self._products['product_name'].str.contains('Gluten', case=False),
                                                 1, 0)

        #
        # Frozen foods
        #

        self._products['PR_frozen'] = np.where(self._products['product_name'].str.contains('frozen', case=False),
                                               1, 0)

        #
        # Get an estimate on the product importance in the role. If there is only one product its 1 and
        # the higher the add_to_cart_order in relation to the number of products the higher the importance
        #

        self._prior_order_products['PO_product_importance'] = np.where(
            self._prior_order_products.OR_products == 1,
            1,
            np.nan
        )

        # Set the nan to 0 if I want to use SMOTE
        self._prior_order_products['PO_product_importance'] = np.where(
            self._prior_order_products.PO_product_importance.isnull(),
            1 - (self._prior_order_products.add_to_cart_order / self._prior_order_products.OR_products),
            np.nan
        )

        p = self._prior_order_products[['user_id', 'product_id', 'order_number', 'reordered']].copy()
        p.drop_duplicates(inplace=True)

        p['first_order'] = np.where(p['order_number'] == 1, 1, 0)
        p['second_order'] = np.where(p['order_number'] == 2, 1, 0)

        pp = p.groupby('product_id')['reordered', 'first_order', 'second_order'].sum()
        pc = p.groupby('product_id')['user_id'].count()

        self._products = self._products.join(pp, on='product_id', how='left', rsuffix='_pp')

        self._products.rename(columns={'reordered': 'sum_reordered',
                                       'first_order': 'sum_first_orders',
                                       'second_order': 'sum_second_orders'}, inplace=True)

        self._products = self._products.join(pc, on='product_id', how='left', rsuffix='_pc')
        self._products.rename(columns={'user_id': 'PR_total_products_bought'}, inplace=True)

        self._products['PR_reorder_probability'] = self._products['sum_second_orders'] / self._products['sum_first_orders']
        self._products['PR_reorder_times'] = 1 + self._products['sum_reordered'] / self._products['sum_first_orders']
        self._products['PR_reorder_ratio'] = self._products['sum_reordered'] / self._products['PR_total_products_bought']

        self._products['PR_reorder_probability'] = np.where(self._products['sum_first_orders'] == 0, 0,
                                                            self._products.PR_reorder_probability)
        self._products['PR_reorder_times'] = np.where(self._products['sum_first_orders'] == 0, 0,
                                                      self._products.PR_reorder_times)

        self._products.drop(['sum_first_orders', 'sum_second_orders', 'sum_reordered'], axis=1, inplace=True)

        #
        # Get mean, min, max product importance
        #

        pd_mean = self._prior_order_products.groupby('product_id')['PO_product_importance'].mean()
        pd_min = self._prior_order_products.groupby('product_id')['PO_product_importance'].min()
        pd_max = self._prior_order_products.groupby('product_id')['PO_product_importance'].max()

        self._products = self._products.join(pd_mean, on='product_id', how='left', rsuffix='_pdmean')
        self._products.rename(columns={'PO_product_importance': 'PO_avg_product_importance'}, inplace=True)

        self._products = self._products.join(pd_min, on='product_id', how='left', rsuffix='_pdmin')
        self._products.rename(columns={'PO_product_importance': 'PO_min_product_importance'}, inplace=True)

        self._products = self._products.join(pd_max, on='product_id', how='left', rsuffix='_pdmax')
        self._products.rename(columns={'PO_product_importance': 'PO_max_product_importance'}, inplace=True)

        # print(self._products.head(1))

        ps = pd.read_csv('data/product_similarity.csv')

        # print(self._products.shape)
        self._products = pd.merge(self._products, ps, on='product_id', how='left')
        self._products.rename(columns={'most_similar': 'PR_most_similar',
                                       'second_most_similar': 'PR_second_most_similar'}, inplace=True)
        # print(self._products.shape)
        # print(self._products.head(1))

        # print(self._products.describe())
        del ps

        # print(self._products.describe())
        # print(self._products.shape)
        pivot_order_products_by_productid = self._prior_order_products.groupby(['product_id'])[
            'add_to_cart_order', 'order_hour_of_day', 'order_dow'
        ].mean()

        self._products = self._products.join(pivot_order_products_by_productid, on='product_id', how='left',
                                             rsuffix='_pp')
        self._products.rename(columns={'add_to_cart_order': 'PR_mean_add_to_cart_order',
                                       'order_hour_of_day': 'PR_mean_order_hour_of_day',
                                       'order_dow': 'PR_mean_order_dow'}, inplace=True)

        pivot_order_products_by_productid = self._prior_order_products.groupby(['product_id'])[
            'add_to_cart_order', 'order_hour_of_day', 'order_dow'
        ].min()

        self._products = self._products.join(pivot_order_products_by_productid, on='product_id', how='left',
                                             rsuffix='_pp')
        self._products.rename(columns={'add_to_cart_order': 'PR_min_add_to_cart_order',
                                       'order_hour_of_day': 'PR_min_order_hour_of_day',
                                       'order_dow': 'PR_min_order_dow'}, inplace=True)

        pivot_order_products_by_productid = self._prior_order_products.groupby(['product_id'])[
            'add_to_cart_order', 'order_hour_of_day', 'order_dow'
        ].max()

        self._products = self._products.join(pivot_order_products_by_productid, on='product_id', how='left',
                                             rsuffix='_pp')
        self._products.rename(columns={'add_to_cart_order': 'PR_max_add_to_cart_order',
                                       'order_hour_of_day': 'PR_max_order_hour_of_day',
                                       'order_dow': 'PR_max_order_dow'}, inplace=True)

        unique_counts = self._prior_order_products.groupby(['product_id'])[
            'user_id'
        ].unique().map(len)

        self._products = self._products.join(unique_counts, on='product_id', how='left', rsuffix='_up')
        self._products.rename(columns={'user_id': 'PR_unique_total_users'}, inplace=True)

        # print(self._products.describe())
        # print(self._products.head())
        # print(self._products.shape)

        #
        # Number of words per product name
        #

        self._products['PR_num_of_words'] = self._products.product_name.str.split().str.len()

        #
        # Number of characters per product name
        #

        self._products['PR_num_of_characters'] = self._products.product_name.str.len()

        #
        # Number of products per department / aisle
        #

        products_per_department = self._products.groupby('department_id')['product_id'].count()
        products_per_aisle = self._products.groupby('aisle_id')['product_id'].count()

        self._products = self._products.join(products_per_department, on='department_id', how='left', rsuffix='_ppd')
        self._products = self._products.join(products_per_aisle, on='aisle_id', how='left', rsuffix='_ppa')
        self._products.rename(columns={'product_id_ppd': 'PR_products_per_department',
                                       'product_id_ppa': 'PR_products_per_aisle'}, inplace=True)

        #
        # The more products there are in a department / aisle the more important is the user selection
        #

        self._products['PR_product_in_department'] = np.where(
            self._products.PR_products_per_department == 1,
            1,
            np.nan
        )

        # Set the nan to 0 if I want to use SMOTE
        self._products['PR_product_in_department'] = np.where(
            self._products.PR_product_in_department.isnull(),
            1 - (1 / self._products.PR_products_per_department),
            np.nan
        )

        self._products['PR_product_in_aisle'] = np.where(
            self._products.PR_products_per_aisle == 1,
            1,
            np.nan
        )

        # Set the nan to 0 if I want to use SMOTE
        self._products['PR_product_in_aisle'] = np.where(
            self._products.PR_product_in_aisle.isnull(),
            1 - (1 / self._products.PR_products_per_aisle),
            np.nan
        )

        # FEATURE RESULT: F1 .3796 with this feature on the 5fold. Don't add
        # #
        # # Ethnicity
        # #
        #
        # self._products['PR_ethinicity'] = np.nan
        #
        # ethnicities = ['Afrikaans', 'Albanian', 'Arabic', 'Armenian', 'Azeri', 'Basque', 'Belarusian', 'Bulgarian',
        #                'Catalan', 'Chinese', 'Croatian', 'Czech', 'Danish', 'Dhivehi', 'Dutch', 'English', 'Estonian',
        #                'Faroese', 'Farsi', 'Finnish', 'French', 'Galician', 'Georgian', 'German', 'Greek', 'Gujarati',
        #                'Hebrew', 'Hindi', 'Hungarian', 'Icelandic', 'Indonesian', 'Italian', 'Japanese', 'Kannada',
        #                'Kazakh', 'Konkani', 'Korean', 'Kyrgyz', 'Latvian', 'Lithuanian', 'Macedonian', 'Malay',
        #                'Marathi', 'Mongolian', 'Norwegian', 'Polish', 'Portuguese', 'Punjabi', 'Romanian', 'Russian',
        #                'Sanskrit', 'Serbian', 'Slovak', 'Slovenian', 'Spanish', 'Swahili', 'Swedish', 'Syriac',
        #                'Tamil', 'Tatar', 'Telugu', 'Thai', 'Turkish', 'Ukrainian', 'Urdu', 'Uzbek', 'Vietnamese']
        #
        # for i, val in enumerate(ethnicities):
        #     self._products['PR_ethinicity'] = np.where(self._products['product_name'].str.contains(val, case=False),
        #                                                i,
        #                                                self._products['PR_ethinicity'])

        self._products.drop(['product_name', 'department', 'aisle'], axis=1, inplace=True)

        #
        # Get averages per department / aisle
        #

        x = pd.merge(self._prior_order_products, self._products, on='product_id', how='left')

        pivot_order_products_by_department = x.groupby(['department_id', 'product_id'])[
            'add_to_cart_order', 'order_hour_of_day', 'order_dow', 'reordered'
        ].mean()

        # min_pivot_order_products_by_department = x.groupby(['department_id', 'product_id'])[
        #     'add_to_cart_order', 'order_hour_of_day', 'order_dow', 'reordered'
        # ].min()
        #
        # max_pivot_order_products_by_department = x.groupby(['department_id', 'product_id'])[
        #     'add_to_cart_order', 'order_hour_of_day', 'order_dow', 'reordered'
        # ].max()

        # median_pivot_order_products_by_department = x.groupby(['department_id', 'product_id'])[
        #     'add_to_cart_order', 'order_hour_of_day', 'order_dow', 'reordered'
        # ].median()

        pivot_order_products_by_aisle = x.groupby(['aisle_id', 'product_id'])[
            'add_to_cart_order', 'order_hour_of_day', 'order_dow', 'reordered'
        ].mean()

        # min_pivot_order_products_by_aisle = x.groupby(['aisle_id', 'product_id'])[
        #     'add_to_cart_order', 'order_hour_of_day', 'order_dow', 'reordered'
        # ].min()
        #
        # max_pivot_order_products_by_aisle = x.groupby(['aisle_id', 'product_id'])[
        #     'add_to_cart_order', 'order_hour_of_day', 'order_dow', 'reordered'
        # ].max()

        # median_pivot_order_products_by_aisle = x.groupby(['aisle_id', 'product_id'])[
        #     'add_to_cart_order', 'order_hour_of_day', 'order_dow', 'reordered'
        # ].median()

        self._products = self._products.join(pivot_order_products_by_department, on=['department_id', 'product_id'],
                                             how='left', rsuffix='_mpd')
        # self._products = self._products.join(min_pivot_order_products_by_department, on=['department_id', 'product_id'],
        #                                      how='left', rsuffix='_minpd')
        # self._products = self._products.join(max_pivot_order_products_by_department, on=['department_id', 'product_id'],
        #                                      how='left', rsuffix='_maxpd')
        # self._products = self._products.join(median_pivot_order_products_by_department, on=['department_id', 'product_id'],
        #                                      how='left', rsuffix='_medianpd')

        self._products = self._products.join(pivot_order_products_by_aisle, on=['aisle_id', 'product_id'],
                                             how='left', rsuffix='_mpa')
        # self._products = self._products.join(min_pivot_order_products_by_aisle, on=['aisle_id', 'product_id'],
        #                                      how='left', rsuffix='_minpa')
        # self._products = self._products.join(max_pivot_order_products_by_aisle, on=['aisle_id', 'product_id'],
        #                                      how='left', rsuffix='_maxpa')
        # self._products = self._products.join(median_pivot_order_products_by_aisle, on=['aisle_id', 'product_id'],
        #                                      how='left', rsuffix='_medianpa')

        self._products.rename(columns={'add_to_cart_order': 'PR_department_avg_atco',
                                       'order_hour_of_day': 'PR_department_avg_hod',
                                       'order_dow': 'PR_department_avg_dow',
                                       'reordered': 'PR_department_avg_reordered',
                                       'add_to_cart_order_mpa': 'PR_aisle_avg_atco',
                                       'order_hour_of_day_mpa': 'PR_aisle_avg_hod',
                                       'order_dow_mpa': 'PR_aisle_avg_dow',
                                       'reordered_mpa': 'PR_aisle_avg_reordered'}, inplace=True)

        print(self._products.head(5))
        print("=> Created product features in %fs" % (time() - t0))

    def create_order_features(self):
        t0 = time()

        self._tt_orders = self._orders[self._orders['eval_set'] != 'prior'].copy()
        self._tt_orders.rename(columns={'days_since_prior_order': 'US_time_since_last_order'}, inplace=True)

        products_per_order = self._prior_order_products.groupby('order_id')['product_id'].count()

        self._prior_order_products = self._prior_order_products.join(products_per_order, on='order_id', how='left',
                                                                     rsuffix='_pc')
        self._prior_order_products.rename(columns={'product_id_pc': 'OR_products'}, inplace=True)

        orders_per_dow = self._orders.groupby('order_dow')['order_id'].count()
        orders_per_hod = self._orders.groupby('order_hour_of_day')['order_id'].count()

        self._tt_orders = self._tt_orders.join(orders_per_dow, on='order_dow', how='left', rsuffix='_opd')
        self._tt_orders = self._tt_orders.join(orders_per_hod, on='order_hour_of_day', how='left', rsuffix='_oph')

        self._tt_orders.rename(columns={'order_id_opd': 'OR_num_of_orders_per_dow',
                                        'order_id_oph': 'OR_num_of_orders_per_hod'}, inplace=True)

        self._tt_orders['OR_weekend'] = np.where((self._tt_orders.order_dow == 0) | (self._tt_orders.order_dow == 6),
                                                 1, 0)
        self._tt_orders['OR_weekday'] = np.where((self._tt_orders.order_dow != 0) & (self._tt_orders.order_dow != 6),
                                                 1, 0)

        # Get favorite department / aisle per dow / hod

        x = pd.merge(self._prior_order_products, self._products, on='product_id', how='left')
        dac_per_dow = x.groupby('order_dow')['aisle_id', 'department_id', 'PR_cluster_id'].mean()
        dac_per_hod = x.groupby('order_hour_of_day')['aisle_id', 'department_id', 'PR_cluster_id'].mean()

        self._tt_orders = self._tt_orders.join(dac_per_dow, on='order_dow', how='left', rsuffix='_dacpd')
        self._tt_orders = self._tt_orders.join(dac_per_hod, on='order_hour_of_day', how='left', rsuffix='_dacph')

        self._tt_orders.rename(columns={'aisle_id': 'OR_avg_aisle_per_dow',
                                        'department_id': 'OR_avg_department_per_dow',
                                        'PR_cluster_id': 'OR_avg_cluster_per_dow',
                                        'aisle_id_dacph': 'OR_avg_aisle_per_hod',
                                        'department_id_dacph': 'OR_avg_department_per_hod',
                                        'PR_cluster_id_dacph': 'OR_avg_cluster_per_hod'}, inplace=True)

        print("=> Created order features in %fs" % (time() - t0))

    def create_user_features(self):
        t0 = time()

        # Create a new dataframe only for the users
        self._users = pd.DataFrame({'user_id': self._orders['user_id'].unique()})

        print("=> Users dataset shape:", self._users.shape)

        #
        # Get mean per user: days_since_prior_order, order_dow, order_hour_of_day
        #

        num_of_orders_per_user = self._prior_orders.groupby(['user_id'])['order_id'].count()
        day_span_between_orders = self._prior_orders.groupby(['user_id'])['days_since_prior_order'].sum()
        mean_day_span_between_orders = self._prior_orders.groupby(['user_id'])['days_since_prior_order'].mean()
        median_day_span_between_orders = self._prior_orders.groupby(['user_id'])['days_since_prior_order'].median()

        pivot_per_user = pd.pivot_table(self._prior_orders, index='user_id', aggfunc=np.mean)

        self._users = self._users.join(pivot_per_user, on='user_id', how='left', rsuffix='_ppu')
        self._users.drop(['order_id', 'order_number'], axis=1, inplace=True)

        self._users.rename(columns={'days_since_prior_order': 'US_avg_days_between_orders',
                                    'order_dow': 'US_avg_order_dow',
                                    'order_hour_of_day': 'US_avg_order_hour_of_day'}, inplace=True)

        del pivot_per_user

        self._users = self._users.join(num_of_orders_per_user.to_frame(), on='user_id')
        self._users = self._users.join(day_span_between_orders.to_frame(), on='user_id')

        self._users.rename(columns={'days_since_prior_order': 'US_day_span',
                                    'order_id': 'US_number_of_orders'}, inplace=True)

        self._users = self._users.join(mean_day_span_between_orders.to_frame(), on='user_id')
        self._users.rename(columns={'days_since_prior_order': 'US_days_mean'}, inplace=True)

        self._users = self._users.join(median_day_span_between_orders.to_frame(), on='user_id')
        self._users.rename(columns={'days_since_prior_order': 'US_days_median'}, inplace=True)

        unique_counts = self._prior_order_products.groupby(['user_id'])['product_id'].unique().map(len)
        product_counts = self._prior_order_products.groupby(['user_id'])['product_id'].count()

        self._users = self._users.join(product_counts.to_frame(), on='user_id', how='left')
        self._users.rename(columns={'product_id': 'US_total_products_bought'}, inplace=True)

        self._users = self._users.join(unique_counts.to_frame(), on='user_id', how='left')
        self._users.rename(columns={'product_id': 'US_total_unique_products_bought'}, inplace=True)

        #
        # Favorite product
        #

        # Slow, so cache it to a file
        favorite_products_file = Path("data/favorite-products.csv")

        if favorite_products_file.is_file():
            favorite_products = pd.read_csv("data/favorite-products.csv")
        else:
            fp = self._prior_order_products.groupby(['user_id', 'product_id'])['order_id'].count() \
                .reset_index(name='count')

            # I don't know how to fucking make this work in pandas

            d = dict()
            for row in fp.itertuples():
                if row.user_id not in d:
                    d[row.user_id] = {
                        'product_id': None,
                        'count': 0
                    }

                if d[row.user_id]['count'] < row.count:
                    d[row.user_id]['count'] = row.count
                    d[row.user_id]['product_id'] = row.product_id

            favorite_products = pd.DataFrame.from_dict(d, orient='index')
            favorite_products.reset_index(inplace=True)
            favorite_products.columns = ['user_id', 'product_id', 'count']

            favorite_products.to_csv("data/favorite-products.csv", index=False)

        self._users = pd.merge(self._users, favorite_products, on='user_id')

        self._users.rename(columns={'product_id': 'US_favorite_product'}, inplace=True)
        self._users['US_favorite_product_repetition'] = self._users['count'] / self._users['US_number_of_orders']

        self._users.drop(['count'], axis=1, inplace=True)

        #
        # Trend of days between orders
        #

        # Slow, so cache it
        trends_file = Path("data/trends.csv")

        if trends_file.is_file():
            dt = pd.read_csv("data/trends.csv")
        else:
            trends = dict()

            # def strictly_increasing(L):
            #     return all(x < y for x, y in zip(L, L[1:]))

            for row in self._users.itertuples():
                orders = self._prior_orders[self._prior_orders['user_id'] == row.user_id]

                # too slow

                orders = orders[['days_since_prior_order']]
                orders = orders.bfill()
                trend = trendline(orders['days_since_prior_order'])

                # trend = strictly_increasing(orders['days_since_prior_order'].values)

                trends[row.user_id] = trend

            dt = pd.DataFrame.from_dict(trends, orient='index')
            dt.reset_index(inplace=True)
            dt.columns = ['user_id', 'US_days_trend']

            dt.to_csv("data/trends.csv", index=False)

        self._users = pd.merge(self._users, dt, on='user_id')

        #
        # Get a unique to total products bought percent
        #

        self._users['US_unique_to_total_products_bought'] = \
            self._users['US_total_unique_products_bought'] / self._users['US_total_products_bought']

        #
        # Average basket size
        #

        self._users['US_average_basket'] = self._users['US_total_products_bought'] / self._users['US_number_of_orders']

        del unique_counts
        del product_counts
        del num_of_orders_per_user
        del day_span_between_orders

        #
        # Get user favorite product categories
        #

        order_products = pd.merge(self._prior_order_products, self._products, on='product_id')

        pivot_order_products_by_userid = order_products.groupby(['user_id'])[
            'PR_cluster_id', 'PR_organic', 'PR_non_fattening', 'PR_recycled', 'PR_vegan',
            'PR_sugar_free', 'PR_gluten_free', 'aisle_id', 'department_id'
            # , 'PR_num_of_characters', 'PR_num_of_words'
        ].mean()

        self._users = self._users.join(pivot_order_products_by_userid, on='user_id', how='left', rsuffix='_up')

        self._users.rename(columns={'aisle_id': 'UP_favorite_aisle',
                                    'department_id': 'UP_favorite_department',
                                    'PR_cluster_id': 'UP_favorite_cluster',
                                    'PR_gluten_free': 'UP_gluten_free_percent',
                                    'PR_sugar_free': 'UP_sugar_free_percent',
                                    'PR_vegan': 'UP_vegan_percent',
                                    'PR_recycled': 'UP_recycled_percent',
                                    'PR_non_fattening': 'UP_non_fattening_percent',
                                    'PR_organic': 'UP_organic_percent',
                                    'PR_frozen': 'UP_frozen_percent'
                                    # , 'PR_num_of_characters': 'UP_num_of_pcharacters',
                                    # 'PR_num_of_words': 'UP_num_of_pwords'
                                    }, inplace=True)

        print(self._users.describe())

        print("=> Created user features in %fs" % (time() - t0))

    def create_dataset(self, force=False, save=True):
        data_file = Path('data/data.csv')

        t0 = time()

        if data_file.is_file() is False or force is True:
            self.load()

            self.create_order_features()
            self.create_product_features()
            self.create_user_features()

            # print(self._users.shape)
            # print(tt_orders.shape)
            self._users = pd.merge(self._users, self._tt_orders, on='user_id')
            # print(self._users.shape)

            order_stat = self._prior_order_products.groupby('order_id').agg({'order_id': 'size'}) \
                .rename(columns={'order_id': 'order_size'}).reset_index()

            self._prior_order_products = pd.merge(self._prior_order_products, order_stat, on='order_id')
            self._prior_order_products[
                'add_to_cart_order_inverted'] = self._prior_order_products.order_size - \
                                                self._prior_order_products.add_to_cart_order
            self._prior_order_products[
                'add_to_cart_order_relative'] = self._prior_order_products.add_to_cart_order / \
                                                self._prior_order_products.order_size

            pivot = pd.pivot_table(self._prior_order_products, index=['user_id', 'product_id'],
                                   aggfunc=(np.mean, 'count', 'min', 'max', 'median', 'sum'))

            data = self._prior_order_products[['user_id', 'product_id']].copy()
            data.drop_duplicates(inplace=True)
            # print(data.shape)

            d = dict()

            d['num'] = pivot['order_id']['count']
            d['first_order'] = pivot['order_number']['min']
            d['last_order'] = pivot['order_number']['max']
            d['mean_add_to_cart_order'] = pivot['add_to_cart_order']['mean']
            d['median_add_to_cart_order'] = pivot['add_to_cart_order']['median']

            d['mean_days_since_prior_order'] = pivot['days_since_prior_order']['mean']
            d['median_days_since_prior_order'] = pivot['days_since_prior_order']['median']

            d['mean_order_dow'] = pivot['order_dow']['mean']
            d['median_order_dow'] = pivot['order_dow']['median']

            d['mean_order_hour_of_day'] = pivot['order_hour_of_day']['mean']
            d['median_order_hour_of_day'] = pivot['order_hour_of_day']['median']

            d['mean_add_to_cart_order_inverted'] = pivot['add_to_cart_order_inverted']['mean']
            d['median_add_to_cart_order_inverted'] = pivot['add_to_cart_order_inverted']['median']

            d['mean_add_to_cart_order_relative'] = pivot['add_to_cart_order_relative']['mean']
            d['median_add_to_cart_order_relative'] = pivot['add_to_cart_order_relative']['median']

            d['reordered_sum'] = pivot['reordered']['sum']

            for i in d:
                data = data.join(d[i].to_frame(), on=['user_id', 'product_id'], how='left', rsuffix='_' + i)

            del d
            del pivot
            del order_stat

            gc.collect()

            data.rename(columns={'count': 'UP_orders',
                                 'min': 'UP_first_order',
                                 'max': 'UP_last_order',
                                 'mean': 'UP_avg_add_to_cart_order'}, inplace=True)

            data = pd.merge(data, self._products, on='product_id', how='left')
            data = pd.merge(data, self._users, on='user_id', how='left')

            data['UP_order_rate'] = data['UP_orders'] / data['US_number_of_orders']
            data['UP_orders_since_last_order'] = data['US_number_of_orders'] - data['UP_last_order']
            data['UP_order_rate_since_first_order'] = data['UP_orders'] / (data['US_number_of_orders'] -
                                                                           data['UP_first_order'] + 1)

            user_dep_stat = data.groupby(['user_id', 'department_id']).agg(
                {'product_id': lambda v: v.nunique(),
                 'reordered': 'sum'
                 })

            user_dep_stat.rename(columns={'product_id': 'dep_products',
                                          'reordered': 'dep_reordered'}, inplace=True)
            user_dep_stat.reset_index(inplace=True)

            user_aisle_stat = data.groupby(['user_id', 'aisle_id']).agg(
                {'product_id': lambda v: v.nunique(),
                 'reordered': 'sum'
                 })
            user_aisle_stat.rename(columns={'product_id': 'aisle_products',
                                            'reordered': 'aisle_reordered'}, inplace=True)
            user_aisle_stat.reset_index(inplace=True)

            print(data.shape)
            data = pd.merge(data, user_dep_stat, on=['user_id', 'department_id'])
            data = pd.merge(data, user_aisle_stat, on=['user_id', 'aisle_id'])

            print(data.head(1))
            print(data.shape)

            data = pd.merge(data, self._orders_train[['user_id', 'product_id', 'reordered']],
                            on=['user_id', 'product_id'],
                            how='left')

            #
            # Try to see in how many orders of the total this product is in
            #

            data['UP_product_reorder_percentage'] = data['UP_orders'] / data['US_number_of_orders']

            #
            # Build PR_unique_total_users / total_users ratio
            #

            data['UP_utu_to_total_ratio'] = data['PR_unique_total_users'] / self._users.shape[0]

            # print(data.head(100))
            # print(data.shape)

            data['reordered'] = np.where(data.reordered.isnull(), 0, 1)

            # Set the train = 0 and the test = 1 to save some memory
            data['eval_set'] = np.where(data.eval_set == 'train', 0, 1)

            print("Final data shape:", data.shape)
            print("Dtypes:", data.columns.to_series().groupby(data.dtypes).groups)

            if save is True:
                t1 = time()
                print("=> Saving dataset to file")

                data.to_csv('data/data.csv', index=False)
                print("=> Saved the dataset in %fs" % (time() - t1))

            del self._tt_orders

            del self._products
            del self._aisles
            del self._departments
            del self._orders_prior
            del self._orders_train
            del self._orders
            del self._users
            del self._user_order
            del self._prior_orders
            del self._prior_order_products

            print("=> Created the dataset in %fs" % (time() - t0))
        else:
            data = pd.read_csv('data/data.csv', dtype={
                'product_id': np.uint16,
                'order_id': np.int32,
                'aisle_id': np.uint8,
                'department_id': np.uint8,
                'user_id': np.int32,

                'UP_orders': np.uint16,
                'UP_first_order': np.uint16,
                'UP_last_order': np.uint16,
                'UP_avg_add_to_cart_order': np.float32,

                'PR_cluster_id': np.uint16,
                'PR_organic': np.int8,
                'PR_non_fattening': np.int8,
                'PR_recycled': np.int8,
                'PR_vegan': np.int8,
                'PR_sugar_free': np.int8,
                'PR_gluten_free': np.int8,
                'PR_total_products_bought': np.uint32,

                'PR_reorder_probability': np.float32,
                'PR_reorder_times': np.float32,
                'PR_reorder_ratio': np.float32,

                'PO_avg_product_importance': np.float32,
                'PO_min_product_importance': np.float32,
                'PO_max_product_importance': np.float32,

                'PR_most_similar': np.uint16,
                'PR_second_most_similar': np.uint16,

                'PR_mean_add_to_cart_order': np.float32,
                'PR_mean_order_hour_of_day': np.float32,
                'PR_mean_order_dow': np.float32,

                'PR_unique_total_users': np.int32,
                'US_avg_days_between_orders': np.float32,
                'US_avg_order_dow': np.float32,
                'US_avg_order_hour_of_day': np.float32,
                'US_number_of_orders': np.uint32,
                'US_day_span': np.uint16,

                'US_total_products_bought': np.uint32,
                'US_total_unique_products_bought': np.uint32,
                'US_unique_to_total_products_bought': np.float32,

                'US_average_basket': np.float32,
                'UP_favorite_cluster': np.float32,
                'UP_organic_percent': np.float32,
                'UP_non_fattening_percent': np.float32,
                'UP_recycled_percent': np.float32,
                'UP_vegan_percent': np.float32,
                'UP_sugar_free_percent': np.float32,
                'UP_gluten_free_percent': np.float32,
                'UP_favorite_aisle': np.float32,
                'UP_favorite_department': np.float32,

                'eval_set': np.int8,
                'order_number': np.int16,
                'order_dow': np.int8,
                'order_hour_of_day': np.int8,
                'US_time_since_last_order': np.float32,  # perhaps not

                'OR_num_of_orders_per_dow': np.uint32,
                'OR_num_of_orders_per_hod': np.uint32,
                'OR_weekend': np.int8,
                'OR_weekday': np.int8,
                'OR_avg_aisle_per_dow': np.float32,
                'OR_avg_department_per_dow': np.float32,
                'OR_avg_cluster_per_dow': np.float32,
                'OR_avg_aisle_per_hod': np.float32,
                'OR_avg_department_per_hod': np.float32,
                'OR_avg_cluster_per_hod': np.float32,

                'UP_order_rate': np.float32,
                'UP_orders_since_last_order': np.uint32,
                'UP_order_rate_since_first_order': np.float32,

                'reordered': np.int8,

                'UP_product_reorder_percentage': np.float32,
                'UP_utu_to_total_ratio': np.float32
            })

            print("=> Loaded dataset from file in %fs" % (time() - t0))

        self._train = data[data['eval_set'] == 0].copy()
        self._test = data[data['eval_set'] == 1].copy()

        self._train.is_copy = False
        self._test.is_copy = False

        del data

        gc.collect()

    def handle_missing_departments(self, save=True):
        vectorizer = TfidfVectorizer(min_df=0.00009, smooth_idf=True,
                                     encoding='utf8', strip_accents='unicode',
                                     stop_words='english',
                                     use_idf=True,
                                     sublinear_tf=False,
                                     )

        train = self._products[self._products.department_id != 21]
        test = self._products[self._products.department_id == 21]

        train_length = train.shape[0]

        dummy = pd.concat([train, test], ignore_index=True, axis=0)

        x_all = vectorizer.fit_transform(dummy['product_name'])
        x = x_all[:train_length]

        x_test = x_all[train_length:]

        y_all = train['department_id']
        y = y_all[:train_length]

        testProductIds = test.product_id

        # X_train, X_test_dummy, Y_train, Y_test_dummy = train_test_split(x, y, test_size=0.2, random_state=42)

        # from sklearn.model_selection import KFold
        # from sklearn.model_selection import GridSearchCV

        # kf = KFold(n_splits=3)
        # kf.get_n_splits(x)

        # for train_index, test_index in kf.split(x):
        #     X_train, X_test = x[train_index], x[test_index]
        #     y_train, y_test = y[train_index], y[test_index]

        # parameters = {'estimator__kernel': ('linear', 'rbf'), 'estimator__C': [1, 10]}
        #
        # model_tunning = GridSearchCV(clf, parameters, cv=kf, n_jobs=-1)
        # model_tunning.fit(x, y)  # set the best parameters
        #
        # print(model_tunning.best_score_)
        # print(model_tunning.best_params_)

        # Best: {'estimator__C': 10, 'estimator__kernel': 'linear'}

        clf = OneVsOneClassifier(estimator=SVC(random_state=0, verbose=0, C=10, kernel='linear'), n_jobs=-1)

        # clf.fit(X_train, Y_train)
        # print("=> CLF Score:", clf.score(X_test_dummy, Y_test_dummy))

        clf.fit(x, y)
        y_pred = clf.predict(x_test)

        output = pd.DataFrame({"product_id": testProductIds, "department_id": y_pred})

        print(output.shape)

        if save is True:
            output.to_csv('data/department-classification.csv', index=False)

    def handle_missing_aisle(self, save=True):
        vectorizer = TfidfVectorizer(min_df=0.00009, smooth_idf=True,
                                     encoding='utf8', strip_accents='unicode',
                                     stop_words='english',
                                     use_idf=True,
                                     sublinear_tf=False,
                                     )

        train = self._products[self._products.aisle_id != 100]
        test = self._products[self._products.aisle_id == 100]

        train_length = train.shape[0]

        dummy = pd.concat([train, test], ignore_index=True, axis=0)

        dummy['new_product_name'] = dummy['product_name'].map(str) + ' ' + dummy['department'].map(str)

        x_all = vectorizer.fit_transform(dummy['new_product_name'])
        x = x_all[:train_length]
        x_test = x_all[train_length:]

        y_all = train['aisle_id']
        y = y_all[:train_length]
        print(train[train['aisle_id'].isnull()])

        testProductIds = test.product_id

        clf = OneVsOneClassifier(estimator=SVC(random_state=0, verbose=0, C=10, kernel='linear'), n_jobs=-1)

        clf.fit(x, y)
        y_pred = clf.predict(x_test)

        output = pd.DataFrame({"product_id": testProductIds, "aisle_id": y_pred})

        print(output.shape)

        if save is True:
            output.to_csv('data/aisle-classification.csv', index=False)

    def find_cluster_k(self, data, k_range=range(1, 50)):
        t0 = time()

        from sklearn.cluster import KMeans
        from scipy.spatial.distance import cdist, pdist

        import matplotlib.pyplot as plt
        plt.style.use('ggplot')

        KM = [KMeans(n_clusters=k, n_jobs=-1).fit(data) for k in k_range]
        centroids = [k.cluster_centers_ for k in KM]

        D_k = [cdist(data, cent, 'euclidean') for cent in centroids]
        cIdx = [np.argmin(D, axis=1) for D in D_k]
        dist = [np.min(D, axis=1) for D in D_k]
        avgWithinSS = [sum(d) / data.shape[0] for d in dist]

        # Total with-in sum of square
        wcss = [sum(d ** 2) for d in dist]
        tss = sum(pdist(data) ** 2) / data.shape[0]
        bss = tss - wcss

        kIdx = 10 - 1

        # elbow curve
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(k_range, avgWithinSS, 'b*-')
        ax.plot(k_range[kIdx], avgWithinSS[kIdx], marker='o', markersize=12,
                markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
        plt.grid(True)
        plt.xlabel('Number of clusters')
        plt.ylabel('Average within-cluster sum of squares')
        plt.title('Elbow for KMeans clustering')

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(k_range, bss / tss * 100, 'b*-')
        plt.grid(True)
        plt.xlabel('Number of clusters')
        plt.ylabel('Percentage of variance explained')
        plt.title('Elbow for KMeans clustering')

        print("=> Completed in %fs" % (time() - t0))

    def cluster_users(self, k):
        t0 = time()

        from sklearn.cluster import KMeans

        x = self._users

        kmeans_clustering = KMeans(n_clusters=k, n_jobs=-1)
        idx = kmeans_clustering.fit_predict(x)

        unique, counts = np.unique(idx, return_counts=True)
        print(dict(zip(unique, counts)))

        labels = kmeans_clustering.labels_
        name = 'kmeans'
        sample_size = 300

        from sklearn import metrics

        print('% 9s   %.2fs    %i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
              % (name, (time() - t0), kmeans_clustering.inertia_,
                 metrics.homogeneity_score(labels, kmeans_clustering.labels_),
                 metrics.completeness_score(labels, kmeans_clustering.labels_),
                 metrics.v_measure_score(labels, kmeans_clustering.labels_),
                 metrics.adjusted_rand_score(labels, kmeans_clustering.labels_),
                 metrics.adjusted_mutual_info_score(labels, kmeans_clustering.labels_),
                 metrics.silhouette_score(x, kmeans_clustering.labels_,
                                          metric='euclidean',
                                          sample_size=sample_size)))

        self._users['cluster'] = labels
        print(self._users['cluster'])

    def do_cv(self, X=None, Y=None, params=None, model_type='xgb', max_rounds=1500, get_f1=False):
        if X is None:
            X = self._train

        if Y is None:
            Y = self._train['reordered']

        X.drop(['reordered'], axis=1, inplace=True)

        if params is None and model_type == 'xgb':
            params = self._xgb_params

        if params is None and model_type == 'lgb':
            params = self._lgb_params

        # X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

        if get_f1 is True:
            X_train, X_known_test, Y_train, Y_known_test = train_test_split(X_train, Y_train, test_size=0.2,
                                                                            random_state=42)
            X_known_test.is_copy = False

            x_known_ids = X_known_test['order_id']
            X_known_test.drop(['order_id'], axis=1, inplace=True)

        # Turn off the SettingWithCopyWarning warning
        # X_train.is_copy = False
        # X_val.is_copy = False

        # Drop the order_id from all x datasets
        # X_train.drop(['order_id'], axis=1, inplace=True)
        # X_val.drop(['order_id'], axis=1, inplace=True)

        if model_type == 'lgb':
            train_data = lgb.Dataset(X_train, Y_train, free_raw_data=False)
            eval_data = lgb.Dataset(X_val, Y_val, free_raw_data=False)

            model = lgb.train(params,
                              train_data,
                              num_boost_round=max_rounds,
                              valid_sets=eval_data,
                              early_stopping_rounds=50)

            best = model.best_iteration
            print("Best iteration at %d" % best)

            # It looks like if the lgb runs out of rounds it sets the best to 0
            if best == 0:
                best = max_rounds

            # predict
            y_pred = model.predict(X_val, num_iteration=best)

            # from sklearn.metrics import f1_score
            # print('[*] The F1 score of prediction in the validation set is:', f1_score(Y_val, y_pred))

            return best, 0
        else:
            def fpreproc(dtrain, dtest, param):
                label = dtrain.get_label()
                ratio = float(np.sum(label == 0)) / np.sum(label == 1)
                param['scale_pos_weight'] = ratio

                return (dtrain, dtest, param)

            # train_data = xgb.DMatrix(X_train, Y_train, feature_names=X.columns)
            # dval = xgb.DMatrix(X_val, Y_val, feature_names=X_val.columns)

            # cv_xgb = xgb.train(params, train_data, num_boost_round=max_rounds, evals=[(dval, 'val')],
            #                    early_stopping_rounds=50,
            #                    verbose_eval=20)

            train_data = xgb.DMatrix(X.values, Y.values, feature_names=X.columns)

            del X
            del Y
            gc.collect()

            t1 = time()
            print("=> xgboost: Performing cross validation")

            # def decay_eta(boosting_round, num_boost_round):
            # @todo

            cv_xgb = xgb.cv(params, train_data, num_boost_round=max_rounds, nfold=3,
                            metrics=('logloss'), stratified=True,
                            seed=0, fpreproc=fpreproc, verbose_eval=True, show_stdv=True
                            )  # ,learning_rates=decay_eta)

            print(cv_xgb)

            # num_boost_round = cv_xgb.best_iteration
            # best_score = cv_xgb.best_score
            #
            # print("=> Best round:", num_boost_round)
            # print("=> Best score:", best_score)
            # print("=> CV completed in %fs" % (time() - t1))
            #
            # fscores = cv_xgb.get_fscore()
            #
            # features = pd.DataFrame()
            #
            # features['features'] = fscores.keys()
            # features['importance'] = fscores.values()
            # features.sort_values(by=['importance'], ascending=False, inplace=True)
            #
            # print(features.head(100))

            if get_f1 is True:
                # Get a prediction from the validation set
                p = xgb.DMatrix(X_known_test, feature_names=X_known_test.columns)
                predicted = cv_xgb.predict(p)

                X_known_test['order_id'] = x_known_ids
                X_known_test['pred'] = predicted

                y_pred = self.select(X_known_test, save=False)

                y_true = X_known_test.groupby('order_id').apply(
                    lambda x: ' '.join(str(y) for y in x.product_id))
                y_true = y_true.to_frame()

                # print(y_true.shape)
                # print(y_pred.shape)
                # print(y_true.head(1))
                # print(y_pred.head(1))
                merged_y = y_pred.join(y_true, on='order_id', how='left', rsuffix='_true')
                # print(merged_y.shape)
                print(merged_y.head(1))

                res = list()
                for entry in merged_y.itertuples():
                    res.append(self.eval_fun(entry[3], entry[2]))

                res = pd.DataFrame(np.array(res), columns=['precision', 'recall', 'f1'])

                print(res.describe())
                print("F1 score:", (res['f1'].sum() / res.shape[0]))
                exit()

            # return num_boost_round, best_score

    def train(self, best_round, model_type='xgb'):
        print("=> Training the model")
        t0 = time()

        X = self._train
        Y = self._train['reordered']

        # Test if code works
        # X = X[:100]
        # Y = Y[:100]

        X.drop(['eval_set', 'user_id', 'reordered', 'order_id'], axis=1, inplace=True)

        test_order_ids = self._test['order_id']
        self._test.drop(['eval_set', 'user_id', 'reordered', 'order_id'], axis=1, inplace=True)

        if self._selected_features:
            X = X[self._selected_features]
            self._test = self._test[self._selected_features]

        if model_type == 'lgb':
            train_all_data = lgb.Dataset(X, Y, free_raw_data=False)

            model = lgb.train(self._lgb_params, train_all_data, num_boost_round=best_round)
            pickle.dump(model, open("lgb.model.dat", "wb"))

            to_predict = self._test
        else:
            # => Best round: 4251
            # => Best score: 0.238789 => 0.3810519 We have overfitted

            # => Best round: 1499
            # => Best score: 0.240287

            # => Best round: 119
            # => Best score: 0.244491 ====> 0.3803099 instead of 0.3825387 we got on the 1499 rounds

            train_all_data = xgb.DMatrix(X, Y, feature_names=X.columns)

            model = xgb.train(self._xgb_params, train_all_data, num_boost_round=best_round)
            pickle.dump(model, open("xgboost.model.dat", "wb"))

            to_predict = xgb.DMatrix(self._test, feature_names=self._test.columns)

        print("=> Training completed in %fs" % (time() - t0))
        print("=> Predicting...")

        if model_type == 'lgb':
                predicted = model.predict(to_predict, num_iteration=best_round)
        else:
                predicted = model.predict(to_predict)

        self._test['pred'] = predicted
        self._test['order_id'] = test_order_ids

        la = self._test[['order_id', 'product_id', 'pred']]
        la.to_csv('predictions.csv', index=False)

        print("=> Saved raw predictions to disk")

        # print(self._test.head(10))
        # print(predicted)

        # Add the none probability ?

        none_p = self.get_none_probability(la)
        self._test = pd.merge(self._test, none_p, on='order_id', how='left')

        #self.select(self._test)
        self.select_with_max_f1(self._test, none_p, save=True, from_p_n=10)

        print("=> Done! Good luck!")

    def select_with_max_f1(self, data, none, save=True, from_p_n=False):
        # best_per_order = self.get_all_probabilities_in_order(data, 4, debug=True)
        # data = pd.merge(data, best_per_order, how='left', on='order_id')
        # data['best_products'] = np.where(data.best_products.isnull(), 0, data.best_products)
        # print(data['best_products'].describe())

        data.sort_values(by='pred', ascending=False, inplace=True)

        orders = data.groupby('order_id')['product_id', 'pred'].aggregate(lambda x: tuple(x))
        orders = orders.reset_index()

        predictions_from_mf1 = dict()

        for row in orders.itertuples():
            if from_p_n and len(row.pred) < from_p_n:
                continue

            pNone = none.loc[none['order_id'] == row.order_id, 'none_probability']
            pNone = float(pNone)

            best_k, predNone, max_f1 = F1Optimizer.maximize_expectation(row.pred, pNone)

            if predNone or best_k == 0:
                predictions_from_mf1[row.order_id] = 'None'
                continue

            s = row.product_id[:best_k]

            predictions_from_mf1[row.order_id] = ' '.join(str(i) for i in s)

        count_per_order = data.groupby('order_id')['product_id'].count()

        data = data.join(count_per_order, on='order_id', how='left', rsuffix='_c')
        data.rename(columns={'product_id_c': 'count'}, inplace=True)

        d = dict()

        for row in data.itertuples():
            if row.order_id in predictions_from_mf1:
                continue

            if row.none_probability > 0.2:
                th = row.none_probability
            else:
                if row.count >= 0 and row.count < 10:
                    th = 0.29
                elif row.count >= 10 and row.count < 20:
                    th = 0.23
                elif row.count >= 20 and row.count < 40:
                    th = 0.20
                elif row.count >= 40 and row.count < 80:
                    th = 0.19
                elif row.count >= 80 and row.count < 160:
                    th = 0.17
                elif row.count >= 160 and row.count < 320:
                    th = 0.16
                else:
                    th = 0.15

            if row.pred > th:
                try:
                    d[row.order_id] += ' ' + str(row.product_id)
                    # added[row.order_id] += 1
                except:
                    d[row.order_id] = str(row.product_id)
                    # added[row.order_id] = 1

        # The orders that don't have any order assigned should get the None product
        # (i.e. either the classifier sucks (70%) or there was actually no new order

        z = predictions_from_mf1.copy()
        z.update(d)

        uniqueOrders = data['order_id'].unique()

        for order in uniqueOrders:
            if order not in z:
                z[order] = 'None'

        # data.sort_values(by='pred', ascending=False, inplace=True)

        # for row in data.itertuples():
        #     if z[row.order_id] == 'None' and row.none_probability < 0.05:
        #         z[row.order_id] += ' ' + str(row.product_id)

        # z = predictions_from_mf1

        sub = pd.DataFrame.from_dict(z, orient='index')
        sub.reset_index(inplace=True)
        sub.columns = ['order_id', 'products']

        print(sub.shape)

        if save is True:
            sub.to_csv('sub.csv', index=False)

        return sub

    def select(self, data, threshold=0.2, save=True):
        print("=> Selecting products for order")

        count_per_order = data.groupby('order_id')['product_id'].count()

        data = data.join(count_per_order, on='order_id', how='left', rsuffix='_c')
        data.rename(columns={'product_id_c': 'count'}, inplace=True)

        d = dict()
        # added = dict()

        for row in data.itertuples():
            #
            # if row.best_products:
            #     # Skip the orders that have a best_products set. Add them bulk at the end
            #     continue

            # Don't get the threshold. This considers that predictions are ordered descending
            # and we get as many as the average of the user

            # if row.pred > threshold and \
            #         (
            #                     (row.order_id in added and added[row.order_id] < row.num_products)
            #                 or row.order_id not in added
            #         ):

            # Doesn't work => Using the average prediction per order as a threshold
            # if it is bigger than 0.22 dropped the lb from 0.3825 to 0.3814

            # if row.pred_average > threshold:
            #     th = row.pred_average
            # else:
            #     th = threshold

            if 'none_probability' in data.columns:
                if row.none_probability > threshold:
                    th = row.none_probability
                else:
                    if row.count >= 0 and row.count < 10:
                        th = 0.29
                    elif row.count >= 10 and row.count < 20:
                        th = 0.23
                    elif row.count >= 20 and row.count < 40:
                        th = 0.20
                    elif row.count >= 40 and row.count < 80:
                        th = 0.19
                    elif row.count >= 80 and row.count < 160:
                        th = 0.17
                    elif row.count >= 160 and row.count < 320:
                        th = 0.16
                    else:
                        th = 0.15

                    # th = threshold
            else:
                th = threshold

            if row.pred > th:
                try:
                    d[row.order_id] += ' ' + str(row.product_id)
                    # added[row.order_id] += 1
                except:
                    d[row.order_id] = str(row.product_id)
                    # added[row.order_id] = 1

        # The orders that don't have any order assigned should get the None product
        # (i.e. either the classifier sucks (70%) or there was actually no new order

        uniqueOrders = data['order_id'].unique()

        for order in uniqueOrders:
            if order not in d:
                d[order] = 'None'

        # If we have a none_probability (as generated from the method) and the probability is < than
        # 0.05 it means that the mixed probability of not having any products classified is less than 0.5
        # which means that our classifier has fucked up and it makes more sense to add the best ranking product in the
        # list

        # Seem to be getting a +0.0001 on the kfold

        if 'none_probability' in data.columns:
            data.sort_values(by='pred', ascending=False, inplace=True)

            for row in data.itertuples():
                if d[row.order_id] == 'None' and row.none_probability < 0.05:
                    d[row.order_id] += ' ' + str(row.product_id)

        # z = d.copy()
        # z.update(best_per_order)

        z = d

        sub = pd.DataFrame.from_dict(z, orient='index')
        sub.reset_index(inplace=True)
        sub.columns = ['order_id', 'products']

        print(sub.shape)

        if save is True:
            sub.to_csv('sub.csv', index=False)

        return sub

    def get_all_probabilities_in_order(self, data, up_to=10, debug=False):
        orders = data.groupby('order_id')['product_id', 'pred'].aggregate(lambda x: tuple(x))

        x = data[['order_id']]
        x.is_copy = False

        x.drop_duplicates(inplace=True)
        x = x.join(orders, on='order_id', how='left')

        probabilities = dict()

        import itertools

        best_per_order = dict()

        for row in x.itertuples():
            if len(row.product_id) > up_to:
                probabilities[row.order_id] = None

                continue

            probabilities[row.order_id] = list()

            if debug:
                print("Processing products in order %d:" % row.order_id)

            product_prediction_pair = dict(zip(row.product_id, row.pred))

            if debug:
                print(product_prediction_pair)

            combinations = list()
            largest = None

            for L in range(0, len(row.product_id) + 1):
                for subset in itertools.combinations(row.product_id, L):
                    if largest is None or len(subset) > len(largest):
                        largest = subset

                    combinations.append(subset)

            for positive in combinations:
                probability = 1

                if debug:
                    print("P", positive, end='')
                    print(" = 1", end='')

                if len(positive) > 0:
                    for p in positive:
                        probability *= product_prediction_pair[p]

                        if debug:
                            print(" * %f" % product_prediction_pair[p], end='')

                negative = set(positive).symmetric_difference(set(largest))

                if len(negative) > 0:
                    for n in negative:
                        probability *= (1 - product_prediction_pair[n])

                        if debug:
                            print(" * (1-%f)" % product_prediction_pair[n], end='')

                if debug:
                    print(" = %f" % probability)

                probabilities[row.order_id].append({
                    'positive': positive,
                    'negative': negative,
                    'probability': probability
                })

            for order_id in probabilities:
                if probabilities[order_id] is None:
                    continue

                order_probabilities = probabilities[order_id]

                best_expected_f1 = 0
                best_expected_f1_products = None

                for k in order_probabilities:
                    f1_for = k['positive']

                    # if len(f1_for) == 0:
                    #     # Skip P(None)
                    #     continue

                    total_f1 = 0

                    for l in order_probabilities:
                        ground_truth = l['positive']

                        if debug:
                            print("Calculating expected F1 score for", f1_for, end='')
                            print(" against ground truth", ground_truth, end='')

                        tp = len(list(set(f1_for).intersection(ground_truth)))
                        n = len(set(f1_for).symmetric_difference(set(ground_truth)))

                        if tp == 0 and n == 0 and len(f1_for) == 0 and len(ground_truth) == 0:
                            expected_f1 = l['probability']
                        else:
                            if debug:
                                print(" = %f * ( (2 * %d) / ((2 * %d) + %d) )" % (l['probability'], tp, tp, n), end='')

                            if tp == 0:
                                expected_f1 = 0
                            else:
                                expected_f1 = l['probability'] * ((2 * tp) / ((2 * tp) + n))

                        if debug:
                            print(" = %f" % expected_f1)

                        total_f1 += expected_f1

                    if debug:
                        print("Total expected F1 score for", f1_for, end='')
                        print(" = %.10f\n" % total_f1)

                    if best_expected_f1 < total_f1:
                        best_expected_f1 = total_f1
                        best_expected_f1_products = f1_for

                best_per_order[order_id] = ' '.join(str(i) for i in best_expected_f1_products)

                print(best_per_order)
                exit()

        bpo = pd.DataFrame.from_dict(best_per_order, orient='index')
        bpo.reset_index(inplace=True)
        bpo.columns = ['order_id', 'best_products']

        return bpo

    def cluster_product_vectors(self):
        t0 = time()
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        model_filename = 'data/word2vec-productIds'
        model_file = Path(model_filename)

        if model_file.is_file() is False:
            print("No word2vec model file. Run Instacart.ordered_products_to_vector() first")
            exit()

        model = Word2Vec.load(model_filename)

        word_vectors = model.wv.syn0

        # num_clusters = word_vectors.shape[0] // 5       # the // is the int division operator (!)
        # we have 135 departments. Lets get 5x clusters
        num_clusters = 5*135

        from sklearn.cluster import KMeans

        # Initalize a k-means object and use it to extract centroids
        kmeans_clustering = KMeans(n_clusters=num_clusters, n_jobs=-1)
        idx = kmeans_clustering.fit_predict(word_vectors)

        word_centroid_map = dict(zip(model.wv.index2word, idx))
        m = pd.DataFrame.from_dict(word_centroid_map, orient='index')

        m['product_id'] = m.index
        m.columns = ['cluster_id', 'product_id']
        m.to_csv('data/product_clusters.csv', index=False)

        print("=> Completed clustering in %fs" % (time() - t0))

    def ordered_products_to_vector(self):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        model_filename = 'data/word2vec-productIds'
        model_file = Path(model_filename)

        if model_file.is_file():
            model = Word2Vec.load(model_filename)

            # vocab = list(model.vocab.keys())
            # print(len(vocab))

            d = dict()

            for i, row in self._products.iterrows():
                try:
                    print("Getting most similar for", row.product_id)

                    ms = model.wv.most_similar(positive=[str(row.product_id)])

                    self._products.set_value(i, 'most_similar', ms[0][0])
                    self._products.set_value(i, 'second_most_similar', ms[1][0])

                except KeyError:
                    # Set the nan to 0 if I want to use SMOTE
                    self._products.set_value(i, 'most_similar', np.nan)
                    self._products.set_value(i, 'second_most_similar', np.nan)

            self._products[['product_id', 'most_similar', 'second_most_similar']].to_csv('data/product_similarity.csv',
                                                                                         index=False)
            # ms = model.wv.most_similar(positive=['31717'])
            # print(ms[0][0])
            return

        filename = "data/order-product-pairs.csv"

        file = Path(filename)

        if file.is_file() is False:
            t0 = time()
            print("=> Order product pairs csv file not there. Creating...")

            self._merged_orders = pd.concat([self._orders_prior, self._orders_train], ignore_index=True, axis=0)

            product_sentences = self._merged_orders.groupby('order_id').apply(
                lambda x: ' '.join(str(y) for y in x.product_id))
            product_sentences.to_frame().to_csv(filename)

            print("=> Completed in %fs" % (time() - t0))

        opp = pd.read_csv(filename)
        opp.columns = ['order_id', 'products']

        t0 = time()
        print("=> Training word2vec model")

        opp['sentences'] = opp.products.str.split(' ')
        # print(opp.head(2))

        model = Word2Vec(opp.sentences, size=100, window=5, min_count=1, workers=4, sg=1, sample=0)
        model.save('data/word2vec-productIds')

        print("=> Completed in %fs" % (time() - t0))

    # https://www.kaggle.com/hongweizhang/how-to-calculate-f1-score
    def f1_score_single(self, labels, preds):
        labels = labels.split(' ')
        preds = preds.split(' ')

        rr = (np.intersect1d(labels, preds))
        precision = np.float(len(rr)) / len(preds)
        recall = np.float(len(rr)) / len(labels)

        try:
            f1 = 2 * precision * recall / (precision + recall)
        except ZeroDivisionError:
            return precision, recall, 0.0

        return precision, recall, f1

    def f1_score_single_alt(self, y_true, y_pred):
        y_true = y_true.split(' ')
        y_pred = y_pred.split(' ')

        y_true = set(y_true)
        y_pred = set(y_pred)
        # try:
        #     y_true.remove(' ')
        #     y_pred.remove(' ')
        # except KeyError:
        #     pass

        cross_size = len(y_true & y_pred)

        if cross_size == 0:
            return 0, 0, 0

        p = 1. * cross_size / len(y_pred)
        r = 1. * cross_size / len(y_true)

        f1 = 2 * p * r / (p + r)

        return p, r, f1

    def f1_score(self, y):
        res = list()

        for entry in y.itertuples():
            if pd.isnull(entry[3]):
                true = 'None'
            else:
                true = entry[3]

            res.append(self.f1_score_single_alt(true, entry[2]))

        res = pd.DataFrame(np.array(res), columns=['precision', 'recall', 'f1'])
        res['order_id'] = y['order_id']

        # print(res.describe())

        return np.mean(res['f1']), res

    def n_fold(self, folds=False, num_rounds=80, params=None, use_lgb=False):
        if folds is False:
            folds = 10

        kf = GroupKFold(n_splits=folds)

        # self._train = self._train.sample(frac=0.02)
        x = self._train
        y = self._train['reordered']

        if use_lgb:
            p = self._lgb_params
        else:
            p = self._xgb_params

        if params:
            # https://stackoverflow.com/questions/38987/how-to-merge-two-python-dictionaries-in-a-single-expression
            p = p.copy()
            p.update(params)

        scores = list()

        for i, (train_index, test_index) in enumerate(kf.split(x, y, groups=x['user_id'].values)):
            print('=> Training fold %d' % i)

            x_train = x.iloc[train_index]
            y_train = y.iloc[train_index]

            x_train.is_copy = False

            x_test = x.iloc[test_index]
            y_test = y.iloc[test_index]

            x_test.is_copy = False

            x_train.drop(['eval_set', 'user_id', 'order_id', 'reordered'], axis=1, inplace=True)

            x_testIds = x_test['order_id']
            x_reordered = x_test['reordered']

            x_test.drop(['eval_set', 'user_id', 'order_id', 'reordered'], axis=1, inplace=True)

            if self._selected_features:
                x_train = x_train[self._selected_features]
                x_test = x_test[self._selected_features]

            print("=> Train shape:", x_train.shape)
            print("=> Test shape:", x_test.shape)

            print('=> Training')

            columns = x_train.columns
            # x_train, y_train = self.resample(x_train, y_train)

            if use_lgb:
                train_data = lgb.Dataset(x_train, y_train, free_raw_data=False,
                                         categorical_feature=['aisle_id', 'department_id'])

                dval = lgb.Dataset(x_test, y_test)

                model = lgb.train(p, train_data, num_boost_round=num_rounds,
                                  valid_sets=dval, early_stopping_rounds=30)

                best = model.best_iteration

                # It looks like if the lgb runs out of rounds it sets the best to 0
                if best == 0:
                    best = num_rounds

                print(best)
                print('=> Predicting')

                pred = model.predict(x_test)
            else:
                train_data = xgb.DMatrix(x_train, y_train, feature_names=columns)

                dval = xgb.DMatrix(x_test, y_test, feature_names=x_test.columns)

                model = xgb.train(p,
                                  train_data,
                                  num_boost_round=num_rounds,
                                  evals=[(dval, 'val')],
                                  early_stopping_rounds=20,
                                  verbose_eval=20)

                # model = xgb.train(p, train_data, num_boost_round=num_rounds)

                fscores = model.get_fscore()

                features = pd.DataFrame()

                features['features'] = fscores.keys()
                features['importance'] = fscores.values()
                features.sort_values(by=['importance'], ascending=False, inplace=True)

                print("=> Fscores:")

                features.to_csv(sys.stdout)
                features.to_json("folds/fscores-" + str(i) + ".js")

                test_data = xgb.DMatrix(x_test, feature_names=x_test.columns)

                print('=> Predicting')

                pred = model.predict(test_data)

            x_test['reordered'] = x_reordered
            x_test['order_id'] = x_testIds
            x_test['pred'] = pred

            none_p = self.get_none_probability(x_test)
            x_test = pd.merge(x_test, none_p, on='order_id', how='left')

            y_pred = self.select_with_max_f1(x_test, none_p, save=False, from_p_n=10)

            # Get the true result for our fold
            reordered_only = x_test[x_test['reordered'] == 1]
            y_true = reordered_only.groupby('order_id').apply(
                lambda row: ' '.join(str(product) for product in row.product_id))
            y_true = y_true.to_frame()

            merged_y = y_pred.join(y_true, on='order_id', how='left', rsuffix='_true')

            f1, results = self.f1_score(merged_y)

            merged_y = pd.merge(merged_y, results, on='order_id', how='left')
            merged_y.columns = ['order_id', 'products_predicted', 'products_true', 'precision', 'recall', 'f1']

            predictions = x_test[['order_id', 'product_id', 'pred']]

            predictions.to_csv('folds/predictions-' + str(i) + '.csv', index=False)
            merged_y.to_csv('folds/fold-' + str(i) + '.csv', index=False)

            print("=> Average f1 score for kfold %d is %f" % (i, f1))

            scores.append(f1)

        print(scores)

        p['scores'] = scores
        p['mean_f1'] = np.mean(scores)
        p['num_rounds'] = num_rounds
        p['folds'] = folds

        json.dump(p, open("folds/details.md", 'w'))

        print("=> Average f1 score across all folds:", np.mean(scores))

    def examine_fold(self, fold_num=0):
        fold = 'folds/fold-' + str(fold_num) + '.csv'
        merged_y = pd.read_csv(fold)

        # Create a new submission set for the select function to fiddle

        pred = 'folds/predictions-' + str(fold_num) + '.csv'
        predictions = pd.read_csv(pred)

        # Add the none probability ?

        none_p = self.get_none_probability(predictions)
        predictions = pd.merge(predictions, none_p, on='order_id', how='left')

        # Get a new resultset

        # results = self.select_with_max_f1(predictions, none_p, save=False)
        #
        # true = pd.merge(merged_y, results, on='order_id', how='left')
        #
        # true = true[['order_id', 'products', 'products_true']]
        #
        # # Get the f1 score
        # f1, results = self.f1_score(true)
        #
        # print("Fold f1 score (all) is:", f1)

        #for i in [10, 20, 30, 40, 50, 60, 70, 80, 100, 150, 200]:
        i = 20
        results_new = self.select_with_max_f1(predictions, none_p, save=False, from_p_n=i)
        true_new = pd.merge(merged_y, results_new, on='order_id', how='left')
        true_new = true_new[['order_id', 'products', 'products_true']]

        f1, results = self.f1_score(true_new)

        print("Fold f1 score with threshold at %d products is" % i, f1)

        # # Compare it to the original
        # f1, results = self.f1_score(merged_y)
        #
        # print("Fold f1 score original is:", f1)

    def reselect(self):
        pred = 'predictions.csv'
        predictions = pd.read_csv(pred)

        # Add the none probability ?

        none_p = self.get_none_probability(predictions)
        predictions = pd.merge(predictions, none_p, on='order_id', how='left')

        # Get a new resultset

        self.select_with_max_f1(predictions, none_p, save=True, from_p_n=10)

        print("=> Done! Good luck!")

    def get_none_probability(self, data):
        d = dict()

        for row in data.itertuples():
            if row.order_id not in d:
                d[row.order_id] = 1

            d[row.order_id] *= (1-row.pred)

        none_probs = pd.DataFrame.from_dict(d, orient='index')
        none_probs.reset_index(inplace=True)
        none_probs.columns = ['order_id', 'none_probability']

        return none_probs

    def resample(self, x, y):
        # There are 12479129 examples marked as 0 and only 828824 as 1
        # The sample appears to be impalanced. Let's try to remedy this

        print("=> Resampling data")

        from imblearn.over_sampling import SMOTE
        from collections import Counter

        sm = SMOTE(random_state=42, n_jobs=-1)

        x_new, y_new = sm.fit_sample(x, y)

        print("=> Resampled dataset shape {}".format(Counter(y_new)))

        return x_new, y_new

i = Instacart()
#i.examine_fold(0)
#i.reselect()
#exit()
# for x in range(0,5):
#     i.examine_fold(x)
#
# exit()
# i.load()
#i.create_order_features()
i.create_dataset(force=True, save=False)
#i.load()

i.n_fold(5, num_rounds=1500, use_lgb=True)
#, params={
#     'learning_rate': 0.1
# })

#i.do_cv(max_rounds=5000)
#i.find_cluster_k(i._users)

# i.create_product_features()
# i.create_user_features()
# i.cluster_product_vectors()
# exit()
# i.load()
# i.create_dataset(force=True, save=False)
#i.train(best_round=530, model_type='lgb')
#i.reselect()
# pred = pd.read_csv("predictions.csv")
# #
# # # Get average per order
# a = pred.groupby('order_id')['pred'].median()
# # a.to_csv(sys.stdout)
# # print(pred.describe())
# # print(a.describe())
# # print(pred.shape)
# pred = pred.join(a, on='order_id', how='left', rsuffix='_average')
# # print(pred.describe())
# # print(pred.shape)
# ua = pred[['order_id', 'pred_average']]
# ua.drop_duplicates(inplace=True)
# print(ua.head(100))
# #print(pred.head(100))

#i.select(pred)

# i.ordered_products_to_vector()
# i.handle_missing_departments(save=False)
