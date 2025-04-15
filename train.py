mport json
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
from extract_features import *


dataset = 'tomcat'

def create_data_train(src_files,bug_reports, bug_report_id_train, df_token, df_rVSM,df_class_simi,bug_simi_df,df_class_token_simi,df_semantic,up_sample = 1):
    # bug_report_ids là danh sách các id của bug reports đã qua sort
    # tạm thời đang có: semantic_score, rVSM_score, prev_rp_simi, prev_rp_recency, num_fixing
    #prev_rp_simi: so sánh độ tương đồng của rp với các rp trước đó đã fix file này
    #prev_rp_recency: thời gian từ lần gần nhất fix
    #num_fixing: số lần đã fix file này trước đó
   
    X_pos=  []
    X_neg =  []
    for report in bug_report_id_train:
        #them case với label true
        for fixed_file_name in bug_reports[report].fixed_files:
            semantic_score = df_semantic.at[report,fixed_file_name]
            rVSM_sc = df_rVSM.at[report,fixed_file_name]
            prev_rps = previous_reports(fixed_file_name, bug_rp_ids, report, bug_reports)
            prev_rp_simi = simi_previous_report(report, prev_rps, bug_reports,bug_simi_df)
            prev_rp_recency = bug_fixing_recency(report, prev_rps, bug_reports)
            stack_trace_score = stack_trace(report, fixed_file_name, bug_reports)
            token_simi_score = df_token.at[report,fixed_file_name]
            class_simi = df_class_simi.at[report,fixed_file_name]
            class_token_simi = df_class_token_simi.at[report,fixed_file_name]
            num_fixing = len(prev_rps)
            for i in range(up_sample):
                #X.append([report,fixed_file_name, rVSM_sc, prev_rp_simi,stack_trace_score,token_simi_score,class_simi,num_fixing,prev_rp_recency])
                X_pos.append([report,fixed_file_name, rVSM_sc, prev_rp_simi,class_simi,stack_trace_score,token_simi_score,prev_rp_recency,class_token_simi,semantic_score,num_fixing,1])
                
            #X.append([report,fixed_file_name,semantic_score, rVSM_sc, prev_rp_simi, prev_rp_recency, num_fixing])
            
            #y.append(1)
        #thêm case với label false
        #wrong_files = list(set(src_files.keys()) - set(bug_reports[report].fixed_files))
        wrong_files  = top_k_wrong_file(report,bug_reports,src_files,df_rVSM)
        for wrong_file_name in wrong_files:
            semantic_score = df_semantic.at[report,wrong_file_name]
            rVSM_sc = df_rVSM.at[report,wrong_file_name]
            prev_rps = previous_reports(wrong_file_name, bug_rp_ids, report, bug_reports)
            prev_rp_simi = simi_previous_report(report, prev_rps, bug_reports,bug_simi_df)
            prev_rp_recency = bug_fixing_recency(report, prev_rps, bug_reports)
            class_token_simi = df_class_token_simi.at[report,wrong_file_name]
            stack_trace_score = stack_trace(report, wrong_file_name, bug_reports)
            token_simi_score = df_token.at[report,wrong_file_name]
            class_simi = df_class_simi.at[report,wrong_file_name]
            num_fixing = len(prev_rps)
            X_neg.append([report,wrong_file_name, rVSM_sc, prev_rp_simi,class_simi,stack_trace_score,token_simi_score,prev_rp_recency,class_token_simi,semantic_score,num_fixing,0])
            #X.append([report,wrong_file_name, rVSM_sc, prev_rp_simi,stack_trace_score,token_simi_score,class_simi,num_fixing,prev_rp_recency])
           
            
    return X_pos,X_neg

def create_test(report,bug_report_id_train,src_files,bug_reports, df_rVSM,df_token,df_class_simi,bug_simi_df,df_class_token_simi,df_semantic):
    '''
    Create test data for a specific bug report
    :param report: bug report id
    :param bug_report_id_train: list of bug report ids in the training set
    :param src_files: source files
    :param bug_reports: bug reports
    :param df_rVSM: rVSM scores
    :param df_semantic: semantic scores
    :return: test data for one bug report
    '''
    X_test = []
    for file_name in src_files.keys():
        semantic_score = df_semantic.at[report,file_name]
        rVSM_sc = df_rVSM.at[report,file_name]
        prev_rps = previous_reports(file_name, bug_rp_ids, report,bug_reports)
        prev_rp_simi = simi_previous_report(report, prev_rps, bug_reports,bug_simi_df)
        prev_rp_recency = bug_fixing_recency(report, prev_rps, bug_reports)
        class_token_simi = df_class_token_simi.at[report,file_name]
        stack_trace_score = stack_trace(report, file_name, bug_reports)
        token_simi_score = df_token.at[report,file_name]
        class_simi = df_class_simi.at[report,file_name]
        num_fixing = len(prev_rps)
        X_test.append([report,file_name, rVSM_sc, prev_rp_simi,class_simi,stack_trace_score,token_simi_score,prev_rp_recency,class_token_simi,semantic_score,num_fixing])
        
    return X_test

import tensorflow as tf

def focal_loss(gamma=2., alpha=0.75):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Tính p_t: xác suất dự đoán cho lớp đúng
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        
        # Cross entropy
        cross_entropy = -tf.math.log(p_t)
        
        # Focal weight
        focal_weight = tf.pow(1 - p_t, gamma)
        
        # Focal loss với alpha
        loss = alpha * focal_weight * cross_entropy
        
        return tf.reduce_mean(loss)
    return focal_loss_fixed

def read_feature_file():
    
    df_rVSM = pd.read_csv("rVSM_scores_"+dataset+".csv", index_col=0)
    df_rVSM.index = df_rVSM.index.astype(str)
    
    df_semantic = pd.read_csv("semantic_scores_"+dataset+".csv", index_col=0)
    df_semantic.index = df_semantic.index.astype(str)

    bug_simi_df = pd.read_csv("bug_simi_"+dataset+".csv", index_col=0)
    bug_simi_df.index = bug_simi_df.index.astype(str)

    df_token = pd.read_csv("token_scores_"+dataset+".csv", index_col=0)
    df_token.index = df_token.index.astype(str)

    df_class_token_simi = pd.read_csv("class_token_simi_"+dataset+".csv", index_col=0)
    df_class_token_simi.index = df_class_token_simi.index.astype(str)

    df_class_simi = pd.read_csv("class_scores_"+dataset+".csv", index_col=0)
    df_class_simi.index = df_class_simi.index.astype(str)
    return df_rVSM,df_semantic,df_token,bug_simi_df,df_class_token_simi,df_class_simi

def main():
    
    df_rVSM,df_semantic,df_token,bug_simi_df,df_class_token_simi,df_class_simi = read_feature_file()
    src_files, bug_reports = preprocess()
    bug_rp_ids =get_sorted_ids_by_time(bug_reports)
    folds = splitKfold(bug_rp_ids, 3)
    top_n_score_dnns = []
    mrrs = []
    map_scores = []

    chooses = [[0, 1, 2, 3, 4, 5, 7, 8]]
    for choose in chooses:
        top_n_score_dnns = []
        mrrs = []
        map_scores = []

        for i in range(len(folds)-1):

            #
            
            bug_train = folds[i]
            bug_test = folds[i+1]
            X_pos, X_neg = create_data_train(src_files,bug_reports, bug_train, df_token, df_rVSM,df_class_simi,bug_simi_df,df_class_token_simi,up_sample =1)
            BATCH_SIZE = 16
            Sn = 14
            Sp = BATCH_SIZE - Sn
            mini_batchs = create_mini_batches(X_neg, X_pos, Sn=Sn, Sp=Sp)
            X_train, y_train = prepare_data_for_dnn(mini_batchs)
            
            info_cols = X_train[:, :2]      # Lấy 2 cột đầu
            X_train = X_train[:, 2:] # Lấy các cột còn lại
            #[ rVSM_sc, prev_rp_simi,class_simi,stack_trace_score,token_simi_score,prev_rp_recency,class_token_simi,semantic_score]
            dele = [i for i in range(9) if i not in choose]
            if len(dele) > 0:
                X_train = np.delete(X_train, dele, axis=1)
            scalerMinMax = CustomMinMaxScaler()
            X_train = scalerMinMax.fit_transform(X_train)
            '''
            clf = MLPClassifier(
            hidden_layer_sizes=(300,150),
            alpha=1e-5,
            solver='sgd',
            random_state=1,
            max_iter=10000,
            n_iter_no_change=30
            )
            
            
            clf.fit(X_train, y_train)
            score_dnn, bug_id_tests = dnn_score(bug_train,bug_test,src_files,scalerMinMax,clf)
            '''
            model = Sequential()

            # First hidden layer (300 nodes, ReLU activation)
            model.add(Dense(300, activation='relu', input_dim=X_train.shape[1]))  # 'input_dimension' should be the number of input features
            
            # Second hidden layer (150 nodes, ReLU activation)
            model.add(Dense(150, activation='relu'))
            #model.add(Dropout(0.2))
            # Output layer (1 node, sigmoid activation)
            model.add(Dense(1, activation='sigmoid'))

            # Define the optimizer (SGD)
            optimizer = SGD(learning_rate=0.01)

            # Compile the model with binary cross-entropy loss
            model.compile(optimizer=optimizer, loss=focal_loss(gamma=2.5, alpha=0.99), metrics=['accuracy'])

            # Train the model
            model.fit(X_train, y_train, epochs=30, batch_size=BATCH_SIZE,verbose=1)
            # Evaluate the model on the training data
            score_dnn, bug_id_tests = dnn_score_keras_drop(bug_train,bug_test,src_files,scalerMinMax,model,dele)
            top_n = (1,2,3,4, 5, 10,15)
            top_n_score,mrr_score,map_score = evaluate_test(src_files,bug_id_tests,score_dnn,top_n)

            top_n_score_dnns.append(top_n_score)
            mrrs.append(mrr_score)
            map_scores.append(map_score)
            print(f"fold {i+1}")
            for i,score in zip(top_n,top_n_score):
                print(f"Top{i}: {score}")
            print(mrr_score)
            print(map_score)
            print("__________________________________")
        top_n_score_mean = [mean(col) for col in zip(*top_n_score_dnns)]
        print("__________________________________")
        print("ALL FOLDS "+dataset)
        print(choose)
        for i,score in zip(top_n,top_n_score_mean):
            print(f"Top{i}: {score}")
        print(mean(mrrs))
        print(mean(map_scores))
        print("__________________________________")
