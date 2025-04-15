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
from utils import *


dataset = 'tomcat'

def preprocess():
    src_prep,report_prep = preprocessing.main()

    src_files= src_prep.src_files
    bug_reports = report_prep.bug_reports
    for id,bug_report in bug_reports.items():
        if '.' in bug_report.fixed_files:
            bug_report.fixed_files.remove('.') 

    valid_bug_reports = {}

    for id, bug_report in bug_reports.items():
        # Giữ lại các file có dấu cách ở đầu/cuối đã được xóa
        bug_report.fixed_files = [f.strip() for f in bug_report.fixed_files]

        # Giữ lại các file có tồn tại trong src_files (kiểm tra trong keys của src_files)
        bug_report.fixed_files = [f for f in bug_report.fixed_files if f in src_files.keys()]
        
        # Nếu có ít nhất một file hợp lệ thì giữ lại bug report
        if bug_report.fixed_files:  # Không cần matched_files, chỉ cần kiểm tra fixed_files sau khi lọc
            valid_bug_reports[id] = bug_report

    # Cập nhật lại biến bug_reports nếu muốn
    bug_reports = valid_bug_reports
    return src_files, bug_reports

def rVSM(src_files, bug_reports):

    src_strings = [' '.join(src.file_name['stemmed'] + src.class_names['stemmed']
                                     + src.method_names['stemmed']
                                     + src.pos_tagged_comments['stemmed']
                                     + src.attributes['stemmed'])
                            for src in src_files.values()]
    reports_strings = [' '.join(report.summary['stemmed'] + report.description['stemmed']+report.pos_tagged_summary['stemmed']+report.pos_tagged_description['stemmed'])
                           for report in bug_reports.values()]
    
    tfidf = TfidfVectorizer(sublinear_tf=True, smooth_idf=True)
    src_tfidf = tfidf.fit_transform(src_strings)
    reports_tfidf = tfidf.transform(reports_strings)

    src_lengths = np.array([float(len(src_str.split()))
                            for src_str in src_strings]).reshape(-1, 1)
    min_max_scaler = MinMaxScaler()
    normalized_src_len = min_max_scaler.fit_transform(src_lengths)

    # Tính g(#terms)
    src_len_score = 1 / (1 + np.exp(-15 * normalized_src_len))

    simis = []
    for report in reports_tfidf:
        s = cosine_similarity(src_tfidf, report)

        # rVSM = g(#terms) * cos
        rvsm_score = s * src_len_score
        simis.append(np.array(rvsm_score).flatten().tolist())
    return simis

def load_glove_embeddings(glove_path, dim=300):
    embeddings_index = {}
    with open(glove_path, encoding='utf8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype='float32')
            except:
                continue  # bỏ qua dòng lỗi
            if len(coefs) == dim:
                embeddings_index[word] = coefs
    return embeddings_index

def document_vector(doc, tfidf_vectorizer, embeddings_index):
    words = doc.split()
    tfidf_scores = tfidf_vectorizer.transform([doc]).toarray().flatten()
    feature_names = tfidf_vectorizer.get_feature_names_out()

    word_vectors = []
    for word in words:
        if word in embeddings_index and word in feature_names:
            idx = np.where(feature_names == word)[0]
            if len(idx) > 0:
                tfidf_weight = tfidf_scores[idx[0]]
                word_vector = embeddings_index[word]
                word_vectors.append(tfidf_weight * word_vector)

    if len(word_vectors) == 0:
        return np.zeros(300)

    weighted_sum = np.sum(word_vectors, axis=0)
    avg_vector = weighted_sum / len(word_vectors)  # chia theo số từ

    return avg_vector

def semantic_simi_glove(src_files, bug_reports, glove_path='glove.840B.300d.txt'):
    embeddings_index = load_glove_embeddings(glove_path, dim=300)

    # Tạo văn bản từ source files và bug reports
    src_texts = [
        ' '.join(src.file_name['unstemmed'] + src.class_names['unstemmed'] +
                 src.attributes['unstemmed'] + src.comments['unstemmed'] +
                 src.method_names['unstemmed'])
        for src in src_files.values()
    ]

    report_texts = [
        ' '.join(report.summary['unstemmed'] + report.description['unstemmed'])
        for report in bug_reports.values()
    ]

    # Tạo TF-IDF vectorizer
    all_texts = report_texts + src_texts
    tfidf = TfidfVectorizer()
    tfidf.fit(all_texts)

    # Tính vector đại diện cho mỗi văn bản
    report_vecs = np.array([
    document_vector(text, tfidf, embeddings_index)
    for text in report_texts
    ])

    src_vecs = np.array([
        document_vector(text, tfidf, embeddings_index)
        for text in src_texts
    ])

    # Tính cosine similarity
    sim_matrix = cosine_similarity(report_vecs, src_vecs)

    # Trả về DataFrame
    df_sim = pd.DataFrame(sim_matrix,
                          index=[str(k) for k in bug_reports.keys()],
                          columns=src_files.keys())
    df_sim.to_csv("semantic_scores_"+dataset+".csv")
    
def bug_fixing_recency(curr_bug_id, prev_reports, bug_reports):
    """ Calculates the Bug Fixing Recency as defined by Lam et al.

    Arguments:
        curr_bug_id {string} -- current bug report id to get current time
        prev_reports {list of string} -- all previous bug reports id fixed that file
        bug_reports {dictionary} -- all bug reports
    """
    if len(prev_reports) > 0:
        recent_report = max(prev_reports, key=lambda x: bug_reports[x].report_time)
        
    else :
        return 1
    
    if curr_bug_id and recent_report:
        return 1 / float(
            get_days_between(bug_reports[curr_bug_id].report_time, bug_reports[recent_report].report_time) + 1
        )

    return 0

def bug_simi(bug_reports):
    """ Calculates the Bug Similarity as defined by Lam et al.

    Arguments:
        bug_reports {dictionary} -- all bug reports
    """
    bug_simis = []
    for report1 in bug_reports.keys():
        bug_simi = []
        for report2 in bug_reports.keys():
            
                
            bug_simi.append(cos_simi(get_all_text_report(report1, bug_reports), get_all_text_report(report2, bug_reports)))
        bug_simis.append(bug_simi)
    # Chuyển đổi danh sách thành DataFrame
    bug_simi_df = pd.DataFrame(bug_simis, index=bug_reports.keys(), columns=bug_reports.keys())
    print(bug_simi_df.shape)
    bug_simi_df.to_csv("bug_simi_"+dataset+".csv")
def simi_previous_report(bug_id, prev_reports, bug_reports,bug_simi_df):
    """[summary]

    Arguments:
        bug_id {string} -- bug id of curent report
        prev_reports {list} -- list of previous reports id fix that file
    """
    
    sims = []
    for prev_report in prev_reports:
        sims.append((prev_report,bug_simi_df.at[bug_id, prev_report]))
    sims_sorted = sorted(sims, key=lambda x: x[1], reverse=True)
    if(len(prev_reports)<3):
        k=len(prev_reports)
    else:
        k=3
    score = 0.0
    for i, (_, sim) in enumerate(sims_sorted[:k]):
        score += sim / (i + 1)  # i+1 để tránh chia cho 0
    return score

def stack_trace (report, file_name, bug_reports):
    """[summary]

    Arguments:
        report {string} -- bug report id
        bug_reports {dictionary} -- all bug reports
        src_files {dictionary} -- all source files
    """
    stack_trace = bug_reports[report].stack_traces
    #print(stack_trace)
    for file,func in stack_trace:
        parts = file.split('.')
        # Lấy toàn bộ phần trước tên hàm
        class_path = '.'.join(parts[:-1])  # bỏ tên hàm
        class_path = class_path.replace('.', os.sep) + '.java'
        #print(class_path)

        if file_name.endswith(class_path):
            return 1
    return 0

def check_token_matchings(src_files, bug_reports):
    scores = []

    for report in bug_reports.values():
        matched_count = []
        summary_tokens = set(report.summary['stemmed'])
        pos_tokens = set(report.summary['stemmed'] + report.pos_tagged_description['stemmed'])

        for src in src_files.values():
            file_tokens = set(src.file_name['stemmed'])
            common = len(summary_tokens & file_tokens)
            matched_count.append(common)

        # Nếu không có file nào khớp với tên tóm tắt
        if sum(matched_count) == 0:
            matched_count = []
            for src in src_files.values():
                tokens = set(src.file_name['stemmed'] +
                             src.class_names['stemmed'] +
                             src.method_names['stemmed'] +
                             src.comments['stemmed'] +
                             src.attributes['stemmed'])
                common = len(pos_tokens & tokens)
                matched_count.append(common)

        # Chuẩn hóa
        
        scores.append(matched_count)

    return scores

def class_simi_token(src_files, bug_reports):
    """
    Tính ma trận tương đồng giữa bug report và file mã nguồn dựa trên độ dài
    của tên class dài nhất có xuất hiện trong bug report.

    Args:
        src_files (dict): từ điển chứa thông tin source file, mỗi phần tử có field class_names['original'].
        bug_reports (dict): từ điển chứa thông tin bug report, mỗi phần tử có summary và description.

    Returns:
        pd.DataFrame: ma trận tương đồng, index là bug ID, columns là source file.
    """
    sim_matrix = []

    for bug_id, report in bug_reports.items():
        bug_words = set(report.summary['unstemmed'] + report.description['unstemmed'])

        row_scores = []
        for src_id, src in src_files.items():
            class_names = src.class_names['unstemmed']  # danh sách tên class gốc
            matched_names = [cn for cn in class_names if cn in bug_words]
            score = max([len(cn) for cn in matched_names], default=0)
            row_scores.append(score)
        
        sim_matrix.append(row_scores)

    df_sim = pd.DataFrame(sim_matrix, 
                          index=[str(bug_id) for bug_id in bug_reports.keys()],
                          columns=src_files.keys())
    df_sim.to_csv("class_token_simi_"+dataset+".csv")

def class_simi(src_files, bug_reports):
    src_class_strings = [' '.join(src.class_names['stemmed']) for src in src_files.values()]
    report_strings = [' '.join(report.summary['stemmed'] + report.description['stemmed']) 
                      for report in bug_reports.values()]
    
    tfidf = TfidfVectorizer(sublinear_tf=True, smooth_idf=True)
    all_docs = report_strings + src_class_strings
    tfidf_matrix = tfidf.fit_transform(all_docs)

    # Chia ma trận TF-IDF
    report_vecs = tfidf_matrix[:len(report_strings)]
    src_vecs = tfidf_matrix[len(report_strings):]

    # Tính cosine similarity giữa từng report và từng source file
    sim_matrix = cosine_similarity(report_vecs, src_vecs)

    # Tạo DataFrame
    df_sim = pd.DataFrame(sim_matrix, 
                          index=[str(k) for k in bug_reports.keys()],
                          columns=src_files.keys())
    
    return df_sim
def main():
    #Edit this
    


    src_files, bug_reports = preprocess()
    rVSM_score = rVSM(src_files, bug_reports)
    df_rVSM = pd.DataFrame(rVSM_score, index=[str(x) for x in bug_reports.keys()], columns=src_files.keys())
    df_rVSM.to_csv("rVSM_scores_"+dataset+".csv")

    semantic_simi_glove(src_files, bug_reports)

    bug_rp_ids =get_sorted_ids_by_time(bug_reports)
    src_file_name = list(src_files.keys())
    bug_simi(bug_reports)
    token_simi = check_token_matchings(src_files, bug_reports)
    df_token = pd.DataFrame(token_simi, index=[str(x) for x in bug_reports.keys()], columns=src_files.keys())
    df_token.to_csv("token_scores_"+dataset+".csv")

    class_simi_token(src_files, bug_reports)
    df_class_simi = class_simi(src_files,bug_reports )
    #df_class_simi = pd.DataFrame(class_simi_score, index=[str(x) for x in bug_reports.keys()], columns=src_files.keys())
    df_class_simi.to_csv("class_scores_"+dataset+".csv")




