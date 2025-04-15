import json
import operator
import pickle
import pandas as pd
import numpy as np
from scipy import optimize
import spacy
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
import math
import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import preprocessing
from datasets import DATASET
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.utils import shuffle
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import random
import os
from statistics import mean
class CustomMinMaxScaler:
    def fit(self, X):
        X = X.astype(float)
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        self.range_ = self.max_ - self.min_
        self.range_[self.range_ == 0] = 1e-9

    def transform(self, X):
        X = X.astype(float)
        X_scaled = (X - self.min_) / self.range_
        X_scaled = np.clip(X_scaled, 0, 1)
        return X_scaled

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

def get_sorted_ids_by_time(bug_reports):
    # Sắp xếp src_files theo report_time và lấy các id
    sorted_ids = sorted(bug_reports.items(), key=lambda item: item[1].report_time)
    return [item[0] for item in sorted_ids]


def get_all_text_report(report_id,bug_reports):
    """
    Get all text of a bug report
    :param report: bug report
    :return: all text of a bug report
    """
    report = bug_reports[report_id]
    return " ".join(report.summary['stemmed']) + " " + " ".join(report.description['stemmed'])
                           
def cos_simi(text1, text2):
    # Khởi tạo vectorizer
    tfidf = TfidfVectorizer(sublinear_tf=True, smooth_idf=True)
    
    # Fit và transform cho cả hai văn bản
    tfidf_matrix = tfidf.fit_transform([text1, text2])  # Fit trên cả hai văn bản
    
    # Tính cosine similarity (dot product của các vector TF-IDF)
    cosine_sim = (tfidf_matrix[0] * tfidf_matrix[1].T).toarray()[0, 0]

    
    return cosine_sim

def evaluate(src_files, bug_reports, df,top_n = (1,5,10)):
    
    top_n_rank = [0] * len(top_n)
    mrr = []
    
    for i, (bug_id, report) in enumerate(bug_reports.items()):
        scores_i = [df.at[bug_id,filename] for filename in src_files.keys()]
        # Finding source codes from the simis indices
        src_ranks, _ = zip(*sorted(zip(src_files.keys(), scores_i),
                                   key=operator.itemgetter(1), reverse=True))

        # Getting reported fixed files
        fixed_files = report.fixed_files

        # Iterating over top n
        for k, rank in enumerate(top_n):
            hit = set(src_ranks[:rank]) & set(fixed_files)

            # Computing top n rank
            if hit:
                top_n_rank[k] += 1

            # Computing precision and recall at n
            
        # Getting the ranks of reported fixed files
        relevant_ranks = sorted(src_ranks.index(fixed) + 1
                                for fixed in fixed_files if fixed in src_ranks)

        # If required fixed files are not in the codebase anymore
        if not relevant_ranks:
            mrr.append(0)
            continue

        # MRR
        min_rank = relevant_ranks[0]
        mrr.append(1 / min_rank)
    mrr_score = np.mean(mrr)
    top_n_score = [x / len(bug_reports) for x in top_n_rank]
    return top_n_score,mrr_score

def get_days_between(d1, d2):
    """Calculates the number of days between two date objects

    Arguments:
        d1 {datetime} -- date 1
        d2 {datetime} -- date 2
    """
    diff_in_days = abs((d1 - d2).days)
    return diff_in_days

def previous_reports(filename, bug_rp_ids,report, bug_reports):
    """ Returns a list of previously filed bug id reports that share a file with the current bug report

    Arguments:
        filename {string} -- the name of the shared Java file
        bug_rp_ids  -- list bug report ids train sorted by report_time 
        report -- current bug report
        bug_reports  -- current bug report id
    """
    previous_reports = []
    if report in bug_rp_ids:
        index = bug_rp_ids.index(report)

        for id_bug in bug_rp_ids[:index]:
            if filename in bug_reports[id_bug].fixed_files:
                previous_reports.append(id_bug)


    return previous_reports

def splitKfold(all_bug_ids,k): # train on fold indexStart, test on fold indexStart+1
    fold_size = len(all_bug_ids) // k  # Kích thước fold
    folds = []
    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size if i != k - 1 else len(all_bug_ids)
        folds.append(all_bug_ids[start:end])
    return folds
def top_k_wrong_file(report,bug_reports,src_files,df_rVSM):
    """ Returns the top k wrong files from pool random for a bug report

    Arguments:
        report {object} -- bug report
        bug_rp_ids_train {list} -- list of all bug report ids in the training set
        k {int} -- number of top files to return
    """
    # Get the top k wrong files for the current bug report
    
    all_files = []
    right_files = bug_reports[report].fixed_files
    
    for filename in list(set(src_files.keys()) - set(right_files)):
        all_files.append([filename,df_rVSM.at[report,filename]])
    # Sort the files by their similarity scores in descending order
    all_files.sort(key=lambda x: x[1], reverse=True)
    if (len(src_files)>1000):
        return [filename for filename, _ in all_files[:1000]]
    else :
        return list(set(src_files.keys()) - set(right_files))
def evaluate_test(src_files, bug_test, scores, top_n=(1, 5, 10)):
    top_n_rank = [0] * len(top_n)
    mrr = []
    ap_scores = []

    for i, bug_id in enumerate(bug_test):
        # Sắp xếp source files theo điểm dự đoán
        src_ranks, _ = zip(*sorted(zip(src_files.keys(), scores[i]),
                                   key=operator.itemgetter(1), reverse=True))

        fixed_files = bug_reports[bug_id].fixed_files
        if not fixed_files:
            mrr.append(0)
            ap_scores.append(0)
            continue

        # Tính top-n hit
        for k, rank in enumerate(top_n):
            hit = set(src_ranks[:rank]) & set(fixed_files)
            if hit:
                top_n_rank[k] += 1

        # Tính các vị trí đúng (relevant ranks)
        relevant_ranks = []
        for fixed in fixed_files:
            if fixed in src_ranks:
                relevant_ranks.append(src_ranks.index(fixed) + 1)  # +1 vì xếp hạng bắt đầu từ 1

        if not relevant_ranks:
            mrr.append(0)
            ap_scores.append(0)
            continue

        # MRR
        min_rank = min(relevant_ranks)
        mrr.append(1 / min_rank)

        # AP (Average Precision)
        num_hits = 0
        sum_precisions = 0
        for rank_idx, file in enumerate(src_ranks):
            if file in fixed_files:
                num_hits += 1
                precision_at_k = num_hits / (rank_idx + 1)
                sum_precisions += precision_at_k
        ap = sum_precisions / len(fixed_files)
        ap_scores.append(ap)

    mrr_score = np.mean(mrr)
    map_score = np.mean(ap_scores)
    top_n_score = [x / len(bug_test) for x in top_n_rank]
    return top_n_score, mrr_score, map_score

def dnn_score_keras_drop(bug_trains, bug_tests, src_files, scaler, model,i_):
    scores = []
    bug_ids_test = []
    
    for bug_id in bug_tests:
        # Tạo dữ liệu kiểm tra từ bug_reports, bug_trains, src_files
        X_test = np.array(create_test(bug_id, bug_trains, src_files, bug_reports, df_rVSM, df_token, df_class_simi, bug_simi_df, df_class_token_simi))
        
        # Chọn các cột từ 2 trở đi và chuyển sang kiểu dữ liệu float32
        X_test = X_test[:, 2:].astype(np.float32)
        if(len(i_)>0):
            
                # Xóa cột thứ i trong X_test
            X_test = np.delete(X_test, i_, axis=1)
        # Chuẩn hóa dữ liệu kiểm tra
        X_test = scaler.transform(X_test)
        
        # Dự đoán xác suất với mô hình DNN (sử dụng Keras model)
        y_pred = model.predict(X_test,verbose=0)
        
        # Thêm kết quả dự đoán vào danh sách scores
        scores.append(y_pred.flatten().tolist())
        
        # Thêm bug_id vào danh sách bug_ids_test
        bug_ids_test.append(bug_id)
    
    return scores, bug_ids_test


def create_mini_batches(negative_set, positive_set, Sn, Sp):
    """
    Tạo K mini-batches cho DNN bằng bootstrapping.
    
    Args:
        negative_set (list): Tập âm tính với Nneg mẫu (tệp không chứa lỗi).
        positive_set (list): Tập dương tính với Npos mẫu (tệp chứa lỗi).
        K (int): Số mini-batches.
        Sn (int): Số mẫu âm tính mỗi mini-batch.
        Sp (int): Số mẫu dương tính mỗi mini-batch.
    
    Returns:
        list: Danh sách K mini-batches, mỗi mini-batch là danh sách [âm tính, dương tính].
    """
    
    Nneg = len(negative_set)
    Npos = len(positive_set)
    K = math.ceil(Nneg / Sn)
    if Nneg < K * Sn:
        K = K-1
    if Npos < 1:
        raise ValueError("Tập dương tính rỗng")

    # Bước 1: Chia tập âm tính
    negative_indices = list(range(Nneg))
    random.shuffle(negative_indices)
    X = [negative_indices[i * Sn:(i + 1) * Sn] for i in range(K)]
    X = [[negative_set[idx] for idx in subset] for subset in X]
    
    # Bước 2-3: Thêm Sp mẫu dương tính với sampling with replacement
    for i in range(K):
        posi = []
        for j in range(Sp):
            t = random.choice(positive_set)  # Chọn ngẫu nhiên, không xóa
            posi.append(t)
        X[i].extend(posi)
    
    return X

def prepare_data_for_dnn(mini_batches):
    """
    Chuyển mini-batches thành dữ liệu và nhãn, lấy nhãn từ thuộc tính cuối.
    
    Args:
        mini_batches (list): Danh sách K mini-batches, mỗi mẫu có nhãn ở cuối.
    
    Returns:
        tuple: (X_train, y_train)
    """
    X_train = []
    y_train = []
    
    for batch in mini_batches:
        # Chuyển batch thành mảng NumPy
        batch_array = np.array(batch)
        # Tách đặc trưng (tất cả trừ cột cuối) và nhãn (cột cuối)
        batch_features = batch_array[:, :-1]  # [Sn + Sp, feature_dim]
        batch_labels = batch_array[:, -1]    # [Sn + Sp]
        X_train.append(batch_features)
        y_train.append(batch_labels)
        
    
    # Gộp thành mảng
    X_train = np.vstack(X_train)  # [K * (Sn + Sp), feature_dim]
    y_train = np.concatenate(y_train)  # [K * (Sn + Sp)]
    y_train = y_train.astype(int)
    # Kiểm tra nhãn
    
    
    return X_train, y_train
                                       