import numpy as np
import ipywidgets as widgets
import os
import pandas as pd


def MeasureFileName_SubsColName_callback(FN, SCN):
    global settings_dict
    
    valid = True
    if not ((FN[-4:] == '.csv') or (FN[-5:] == '.xlsx')):
        print('Specified file does not contain Valid extensions: only .csv or .xlsx is valid')
        valid = False
    
    if not (os.path.isfile('data/%s'%FN)):
        print('Specified file does not exists in the data folder')
        valid = False
        
    if(valid):
        if(FN[-4:] == '.csv'):
            measure_df = pd.read_csv('data/%s'%FN)
        else:
            measure_df = pd.read_excel('data/%s'%FN)
        
        if not SCN in measure_df.columns:
            print('The specified column name does not exists in the file')
            valid = False
    
    if(valid):
        print('File and subject column are present')
    settings_dict['measure_file_name'] = FN
    settings_dict['SubsColName'] = SCN
    
def MeasureFileName_SubsColName():
    global settings_dict
    
    yy = os.listdir('data')
    file_name = 'No_file_avaiable'
    for i in yy:
        if((i[-4:] == '.csv') or (i[-5:] == '.xlsx')):
            file_name = i
            break
    if file_name is None:
        print('No Valid Measure file is avaiable in the Data folder Please select one before proceding')
    
    settings_dict = {'measure_file_name': file_name, 'SubsColName': 'Sub_ID', 'test_type': 'MI', 'network_type': 'Both', 'TR_type': 'Both', 'session_type': 'Both',
                     'measure_var_list': [], 'measure_var_type': {}, 'No_par_jobs': 1, 'No_perm': 5000}
    
    FileName_wid = widgets.Text(value=file_name, description='File Name', disabled=False, layout = widgets.Layout(width='500px'))
    SubsColName_wid = widgets.Text(value='Sub_ID', description='Subjects column name', disabled=False,
                                   style = {'description_width': 'initial'}, layout = widgets.Layout(width='500px'))
    
    UI_set1 = widgets.interact_manual(MeasureFileName_SubsColName_callback, FN = FileName_wid, SCN = SubsColName_wid)
    display(UI_set1)    
    return settings_dict

#-----------------------------------------------------

def selected_measure_list_callback(var_list):
    global settings_dict
    settings_dict['measure_var_list'] = var_list
    print('Variable list selected')

def Measure_selection_UI():
    global settings_dict
    if(settings_dict['measure_file_name'][-4:] == '.csv'):
        measure_df = pd.read_csv('data/%s'%settings_dict['measure_file_name'])
    else:
        measure_df = pd.read_excel('data/%s'%settings_dict['measure_file_name'])
    
    measure_list =  np.setdiff1d(np.array(measure_df.columns),np.array([settings_dict['SubsColName']]))
    box_height = 20*len(measure_list)
    UI_MS = widgets.interact_manual(selected_measure_list_callback, var_list = widgets.SelectMultiple(options=measure_list,value= [measure_list[0]],description='Measure List',
                                                                                                layout = widgets.Layout(height='%dpx'%box_height),disabled=False))
    display(UI_MS)
    
#-----------------------------------------------------

def variable_type_callback(**kwargs):
    global settings_dict
    for key, value in kwargs.items():
        settings_dict['measure_var_type'][key] = value
        # print("%s == %s" % (key, value))    
    print('Variable type stored')    
        
def Measure_type_UI():
    global settings_dict
    
    var_list = settings_dict['measure_var_list']
    wid_list = []
    var_UI_dict = {}
    for var_name in var_list:
        wid_list.append(widgets.ToggleButtons(options=['Continues', 'Catagorical'], description = var_name, style = {'description_width': 'initial'}, disabled=False) )
        var_UI_dict[var_name] = wid_list[-1]
        settings_dict['measure_var_type'][var_name] = 'Continues'    
    UI_VT = widgets.VBox(wid_list)
    UI_VT_out = widgets.interactive_output(variable_type_callback, var_UI_dict)
    display(UI_VT,UI_VT_out)    


def Remaining_settings_callback(TT,PT,IT,TRT):
    global settings_dict
    
    if(TT == 'Mutual information'):
        settings_dict['test_type'] = 'MI'
    elif(TT == 'linear model'):
        settings_dict['test_type'] = 'OLS'
    else:
        settings_dict['test_type'] = 't-test'
        
    if(PT == 'Both'):
        settings_dict['network_type'] = 'Both'
    elif(PT == 'FBN_aseg_2009'):
        settings_dict['network_type'] = 'FBN_aseg_2009'
    else:
        settings_dict['network_type'] = 'FBN_aseg'        
        
    if(IT == 'Both'):
        settings_dict['session_type'] = 'Both'
    elif(IT == 'After Lactoluse'):
        settings_dict['session_type'] = 'AL'
    else:
        settings_dict['session_type'] = 'BL'     

    if(TRT == 'Both'):
        settings_dict['TR_type'] = 'Both'
    elif(TRT == 'TR = 2.89s'):
        settings_dict['TR_type'] = 'TR_2890'
    else:
        settings_dict['TR_type'] = 'TR_1980'      
        
    print('Settings stored')
    
def Remaining_settings_UI():
    test_wid = widgets.ToggleButtons(options=['Mutual information', 'linear model', 't-test'], description='Test Type:',style = {'description_width': 'initial'}, disabled=False)
    parc_wid = widgets.ToggleButtons(options=['Both', 'FBN_aseg_2009', 'FBN_aseg'], description='Network and parcellation type:',style = {'description_width': 'initial'}, disabled=False)
    intervention_wid = widgets.ToggleButtons(options=['Both','Before Lactoluse', 'After Lactoluse'], description='Intervention Type:',style = {'description_width': 'initial'}, disabled=False)
    TR_wid = widgets.ToggleButtons(options=['Both','TR = 2.89s', 'TR = 1.98s'], description='Repetation time Type:',style = {'description_width': 'initial'}, disabled=False)
    
    UI_set2 = widgets.interact_manual(Remaining_settings_callback, TT=test_wid, PT=parc_wid, IT=intervention_wid, TRT=TR_wid)
    display(UI_set2)

def parallel_NoPerm_callback(No_P_J,No_perm):
    global settings_dict
    
    settings_dict['No_par_jobs'] = No_P_J
    settings_dict['No_perm'] = No_perm

def parallel_NoPerm_UI():
    global settings_dict
    
    Parallel_wid = widgets.BoundedIntText(value = settings_dict['No_par_jobs'], min = 1, max = os.cpu_count(), step=1, description = 'Parallel Jobs', style = {'description_width': 'initial'}, disabled=False)
    Perm_wid = widgets.BoundedIntText(value = settings_dict['No_perm'], min = 1, max = 10000, step=1, description = 'Number of permutations', style = {'description_width': 'initial'}, disabled=False)
    
    UI_PP = widgets.interact_manual(parallel_NoPerm_callback, No_P_J = Parallel_wid, No_perm = Perm_wid)
    display(UI_PP) 