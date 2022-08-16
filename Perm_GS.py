import pickle 
import scipy
import numpy as np
import statsmodels.api as sm
from sklearn.utils import shuffle
import pandas as pd
import concurrent.futures
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import os

def get_necessary_data(network_type, TR_type, session_type, selected_file, SubsColName):
    
    aa = {'FBN_aseg_2009': 'fmri_all_sub_FC_Z_NoScale', 'FBN_aseg': 'fmri_all_sub_FC_Z_NoScale_big'}
    bb = {'TR_2890': '_2890.pickle', 'TR_1980': '_1980.pickle'}
    
    fmri_file_name = aa[network_type] + bb[TR_type] 
    with open('data/%s'%fmri_file_name, 'rb') as handle:      
        all_sub_avg_hist = pickle.load(handle)
    
    if(os.path.splitext(selected_file)[1] == '.csv'):
        measure_df = pd.read_csv('data/%s'%selected_file)
    else:
        measure_df = pd.read_excel('data/%s'%selected_file)
    
    if(session_type == 'BL'):
        fmri_BL_AL = '_01'
    else:
        fmri_BL_AL = '_02'
    
    if(TR_type == 'TR_2890'):
        with open('data/all_sub_average_motion_2890.pickle', 'rb') as handle:      
            all_sub_average_motion = pickle.load(handle)
    elif(TR_type == 'TR_1980'):
        with open('data/all_sub_average_motion_1980.pickle', 'rb') as handle:      
            all_sub_average_motion = pickle.load(handle)
        
    # print(measure_df.shape)
    count = 0
    to_drop = []        
    for sub_id in measure_df[SubsColName]:
        if not (sub_id + fmri_BL_AL) in all_sub_avg_hist:
            to_drop.append(count) 
        elif(all_sub_average_motion[sub_id + fmri_BL_AL]>0.3):
            to_drop.append(count)
        count = count + 1
    df_to_use = measure_df.drop(to_drop)
    # print(df_to_use.shape)
    
    if(TR_type == 'TR_2890'):
        if(network_type == 'FBN_aseg_2009'):
            imp_roi = np.array([10,11,12,13,17,18,49,50,51,52,53,54,
                                11101, 11102, 11104, 11105, 11106, 11107, 11108, 11109, 11111, 11112, 11113, 11114, 11115, 11116, 11117, 11118, 11119, 11120,
                                11121, 11122, 11123, 11124, 11125, 11126, 11127, 11128, 11129, 11130, 11131, 11133, 11134, 11135, 11136, 11137, 11138, 11141, 11143,
                                11144, 11145, 11146, 11147, 11148, 11149, 11150, 11151, 11153, 11154, 11155, 11157, 11158, 11159, 11160, 11161, 11162, 11164, 11165, 11166,                                   11167,
                                11168, 11169, 11170, 11171, 11172, 11173, 11174, 
                                12101, 12102, 12104, 12105, 12106, 12107, 12108, 12109, 12111, 12112, 12113, 12114, 12115, 12116, 12117, 12118, 12119, 12120,
                                12121, 12122, 12123, 12124, 12125, 12126, 12127, 12128, 12129, 12130, 12131, 12133, 12134, 12135, 12136, 12137, 12138, 12141, 12143, 12144,
                                12145, 12146, 12147, 12148, 12149, 12150, 12151, 12153, 12154, 12155, 12157, 12158, 12159, 12160, 12161, 12162, 12164, 12165, 12166, 12167,
                                12168, 12169, 12170, 12172, 12173, 12174])  
        else:
            imp_roi = np.array([10,11,12,13,17,18,49,50,51,52,53,54,
                                1001, 1002, 1003, 1005, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 
                                1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1033, 1034, 1035,
                                2001, 2002, 2003, 2005, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018,
                                2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2033, 2034, 2035])
    else:
        if(network_type == 'FBN_aseg_2009'):
            imp_roi = np.array([10,11,12,13,17,18,49,50,51,52,53,54,
                                11101, 11102, 11104, 11105, 11106, 11107, 11108, 11109, 11111, 11112, 11113, 11114, 11115, 11116, 11117, 11118, 11119, 11120,
                                11121, 11122, 11123, 11124, 11125, 11126, 11127, 11128, 11129, 11130, 11133, 11134, 11135, 11136, 11137, 11138, 11141, 11143,
                                11144, 11145, 11146, 11147, 11148, 11149, 11150, 11153, 11154, 11155, 11157, 11158, 11159, 11160, 11161, 11162, 11165, 11166,
                                11167, 11168, 11169, 11170, 11171, 11172, 11173, 11174, 
                                12101, 12102, 12104, 12105, 12106, 12107, 12108, 12109, 12111, 12112, 12113, 12114, 12115, 12116, 12117, 12118, 12119, 12120,
                                12121, 12122, 12123, 12124, 12125, 12126, 12127, 12128, 12129, 12130, 12133, 12134, 12135, 12136, 12137, 12138, 12141, 12143,
                                12144, 12145, 12146, 12147, 12148, 12149, 12150, 12153, 12154, 12155, 12157, 12158, 12159, 12160, 12161, 12162, 12165, 12166,
                                12167, 12168, 12169, 12170, 12172, 12173, 12174])
        else:
            imp_roi = np.array([10,11,12,13,17,18,49,50,51,52,53,54,
                                1001, 1002, 1003, 1005, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 
                                1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1033, 1034, 1035,
                                2001, 2002, 2003, 2005, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018,
                                2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2033, 2034, 2035])     
            
            
    cur_sub_ids = df_to_use[SubsColName].to_numpy() + fmri_BL_AL
    
    if((all_sub_avg_hist[cur_sub_ids[0]].shape[0] == all_sub_avg_hist[cur_sub_ids[0]].shape[1]) and all_sub_avg_hist[cur_sub_ids[0]].shape[0] == len(imp_roi)):
        no_nodes = len(imp_roi)
    else:
        raise Exception("Correlation matrix not concistant with selected parameters")
        
    no_subs = len(cur_sub_ids)
    
    ixes = np.where(np.triu(np.ones((no_nodes, no_nodes)), 1))
    no_links = np.size(ixes, axis=1)
    X = np.zeros((no_subs, no_links))
    for i in range(no_subs):
        X[i,:] = all_sub_avg_hist[cur_sub_ids[i]][ixes]
    
    return X, ixes, cur_sub_ids, df_to_use, imp_roi            

def run_test(X,Y_init,test_type, vartable_type,  perm_number = -1, nuisance = None):
    if(perm_number == -1):
        Y = Y_init
    else:
        Y = shuffle(Y_init, random_state = perm_number)
        
    f_stat = np.zeros(X.shape[1])
    if(test_type == 't-test'):
        X_IBS = X[Y == 1,:]
        X_HC = X[Y == 0,:]
        for i in range(X.shape[1]):
            f_stat[i] , junk = scipy.stats.ttest_ind(X_IBS[:,i], X_HC[:,i], equal_var=True)
            if(np.isnan(f_stat[i])):
                f_stat[i] = 0
    elif(test_type == 'OLS'):
        if(nuisance == None):
            Y_mat = np.ones((len(Y), 2))
            Y_mat[:,1] = Y 
            for i in range(X.shape[1]):
                results = sm.OLS(X[:,i], Y_mat).fit()
                f1 =  results.f_test([0,1], cov_p=results.cov_params())
                if(np.isnan(f1.fvalue)):
                    f_stat[i] = 0
                else:
                    f_stat[i] = f1.fvalue
        else:
            Y_mat = np.ones((len(Y), 2 + nuisance.shape[1]))
            Y_mat[:,1] = Y
            Y_mat[:,2:] = nuisance
            var_pos = [0]*Y_mat.shape[1]
            var_pos[1] = 1
            for i in range(X.shape[1]):
                results = sm.OLS(X[:,i], Y_mat).fit()
                f1 =  results.f_test([0,1], cov_p=results.cov_params())
                if(np.isnan(f1.fvalue)):
                    f_stat[i] = 0
                else:
                    f_stat[i] = f1.fvalue
                    
    elif(test_type == 'MI'):
        if(vartable_type == 'Categorical'):
            f_stat = mutual_info_classif(X,Y)
        else:
            f_stat = mutual_info_regression(X,Y)
            
    return f_stat

def run_G_stat(X,Y,test_type, vartable_type, no_perm, run_parallel, nuisance = None):
    original_test_stat =  run_test(X,Y,test_type, vartable_type, -1, nuisance)
    perm_test_stat = []
    
    if(run_parallel < 2):
        for i in range(no_perm):
            perm_test_stat.append(run_test(X,Y,test_type,vartable_type, i,nuisance))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=run_parallel) as executor:
            future_to_perm = {executor.submit(run_test, X, Y, test_type, perm_num, nuisance): perm_num for perm_num in range(no_perm)}
            for future in concurrent.futures.as_completed(future_to_perm):
                perm_test_stat.append(future.result())  
    
    perm_test_stat = np.array(perm_test_stat)
    all_stats = {}
    all_stats['original'] = original_test_stat
    all_stats['permuted'] = perm_test_stat
    return all_stats

def run_all_comb_stats(settings_dict, no_perm, no_parallel_runs):
    if(settings_dict['network_type'] == 'Both'):
        all_network_types = ['FBN_aseg_2009','FBN_aseg']
    else:
        all_network_types = [settings_dict['network_type']]

    if(settings_dict['TR_type'] == 'Both'):    
        all_TR_types = ['TR_2890','TR_1980']
    else:
        all_TR_types = [settings_dict['TR_type']]

    if(settings_dict['session_type'] == 'Both'):    
        all_session_type = ['BL', 'AL']
    else:
        all_session_type = [settings_dict['session_type']]    

    all_types = []
    for NT in all_network_types:
        for TR in all_TR_types:
            for ST in all_session_type:
                all_types.append([NT,TR,ST,settings_dict['measure_file_name']])
    
    for types in all_types:
        X, ixes, cur_sub_ids, df_to_use, imp_roi = get_necessary_data(types[0],types[1],types[2],settings_dict['measure_file_name'],settings_dict['SubsColName'])
    
    for var_name in settings_dict['measure_var_list']:
        cur_Y =  df_to_use[var_name].to_numpy()
        file_name = '%s_%s_%s_%s_%s'%(var_name.split('.')[0],types[0],types[1],types[2],settings_dict['test_type'])
        if not(os.path.isfile('Results/%s.pickle'%file_name)):
            test_statistics = run_G_stat(X,cur_Y,settings_dict['test_type'],settings_dict['measure_var_type'][var_name],settings_dict['No_perm'],                                                                      settings_dict['No_par_jobs'])
            print(file_name)
            data_to_save = {} 
            data_to_save['settings_dict'] = settings_dict
            data_to_save['ixes'] = ixes
            data_to_save['cur_sub_ids'] = cur_sub_ids
            data_to_save['df_to_use'] = df_to_use
            data_to_save['ROIS'] = imp_roi
            data_to_save['test_statistics'] = test_statistics
            with open('Results/%s.pickle'%file_name, 'wb') as handle:
                pickle.dump(data_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)             