import pickle 
import numpy as np
from bct import get_components
from nilearn import plotting
import ipywidgets as widgets
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns


# this function is developed using parts of code from nbs.py in bctpy package 
# https://github.com/aestrivex/bctpy/blob/master/bct/nbs.py
def get_comp_size(t_stat, thrs, g_l, nn, ixes, ret_much):
    
    if(g_l == 1):
        ind_t, = np.where(t_stat > thrs)
    else:
        ind_t, = np.where(t_stat < thrs)
        
    adj = np.zeros((nn, nn))
    adj[(ixes[0][ind_t], ixes[1][ind_t])] = 1
    # adj[ixes][ind_t]=1
    adj = adj + adj.T

    a, sz = get_components(adj)
    ind_sz, = np.where(sz > 1)
    ind_sz += 1
    nr_components = np.size(ind_sz)
    sz_links = np.zeros((nr_components,))
    for i in range(nr_components):
        nodes, = np.where(ind_sz[i] == a)
        sz_links[i] = np.sum(adj[np.ix_(nodes, nodes)]) / 2
        adj[np.ix_(nodes, nodes)] *= (i + 2)

    # subtract 1 to delete any edges not comprising a component
    adj[np.where(adj)] -= 1

    if np.size(sz_links):
        max_sz = np.max(sz_links)
    else:
        max_sz=0
    if(ret_much):
        adj1 = np.zeros(adj.shape)
        max_comp_loc = np.where(sz_links == max_sz)[0][0]
        adj1[np.where(adj == (max_comp_loc+1))] = 1        
        return max_sz, sz_links, adj1 
    else:
        return max_sz  


def prepare_data(file_name, sub_network = None):
    global orig_stat, perm_stat, ixes, ROIS, imp_coord, all_parc
    
#    load origianl and permuted stats and other settings
    with open('Results/%s.pickle'%file_name, 'rb') as handle:      
        test_stat_data = pickle.load(handle)

# sect the parcellation image based on ROI's        
    if np.any(test_stat_data['ROIS']>10000):
        imag = 'data/ASEG_2009_MNI_parc.nii'
    else:
        imag = 'data/ASEG_MNI_parc.nii'          

    if sub_network is None: # is sub-network (eg: DMN, saliance network,pain network) is not given perform analysis on whole brain 
        orig_stat = test_stat_data['test_statistics']['original']
        perm_stat = test_stat_data['test_statistics']['permuted']
        ixes = test_stat_data['ixes']
        ROIS = test_stat_data['ROIS']
    else: # Select the stats corresponding to sub-network given by user
        temp_roi = np.intersect1d(test_stat_data['ROIS'],sub_network)
        if not (len(sub_network) == len(temp_roi)):
            raise Exception('Sub-network is not within the large network')
        loc_vec = np.zeros(len(sub_network), dtype = int)
        for i in range(len(sub_network)):
            loc_vec[i] = np.where(test_stat_data['ROIS'] == sub_network[i])[0][0]
        loc_comm_mat = np.zeros((len(test_stat_data['ROIS']),len(test_stat_data['ROIS'])),dtype = int)    
        for i in loc_vec:
            for j in loc_vec:
                loc_comm_mat[i,j] = 1
        up_tri = np.triu_indices(len(test_stat_data['ROIS']), k=1, m=None)

        orig_stat = test_stat_data['test_statistics']['original'][yy == 1]
        perm_stat = test_stat_data['test_statistics']['permuted'][:,yy == 1]        
        ROIS = sub_network
        ixes = np.where(np.triu(np.ones((len(ROIS), len(ROIS))), 1))  
        
    # select the co-ordinates of the nodes based on selected parrcellation in MNI space
    aa, bb  = plotting.find_parcellation_cut_coords(imag,return_label_names = True)
    bb = np.array(bb)
    imp_coord = []
    for i in ROIS:
        imp_loc = np.where(bb == i)[0][0]
        imp_coord.append(aa[imp_loc,:])
    imp_coord = np.array(imp_coord)
        
    # fet ROI names
    all_parc = {}
    with open(r'data/all_parc.txt', 'r') as f:
        x = f.readlines()
    for i in range(0,len(x)):
        temp = x[i].split()
        all_parc[int(temp[0])] = temp[1]
    
#-------------functions related to network based staistics----------------------    
def get_plot_largest_network(cut_off):
    global orig_stat, perm_stat, ixes, ROIS, imp_coord, all_parc  
    
    nn = len(ROIS)
    orig_no_comp, sz_links, adj = get_comp_size(orig_stat, cut_off, 1, nn, ixes, 1)
    no_runs = perm_stat.shape[0]
    perm_no_comp = np.zeros(no_runs)
    pro_bar = widgets.IntProgress(min=0, max=no_runs-1) # instantiate the bar
    display(pro_bar)
    for i in range(no_runs):
        perm_no_comp[i] = get_comp_size(perm_stat[i,:], cut_off, 1, nn, ixes, 0)
        pro_bar.value += 1
    p_value = np.sum(perm_no_comp>orig_no_comp)/(no_runs+1)
    del pro_bar
    
    rows_col_to_delete = []
    for i in range(adj.shape[0]):
        if(np.sum(adj[i,:]) == 0):
            rows_col_to_delete.append(i)
    red_adj_mat = np.delete(adj,rows_col_to_delete, axis = 0)
    red_adj_mat = np.delete(red_adj_mat,rows_col_to_delete, axis = 1)
    aa = np.setdiff1d(np.arange(0,adj.shape[0]),rows_col_to_delete)  
    
    fig, ax = plt.subplots(1,2, figsize = (20,6))
    plotting.plot_connectome(red_adj_mat,imp_coord[aa,:], axes = ax[0])    
    junk = ax[0].text(0.7, 0.9, 'p-value = %f'%(p_value), fontsize = 'x-large')
    ax[0].set_title('Selected Network connectome',fontsize = 'xx-large')
    sns.kdeplot(perm_no_comp, ax = ax[1])
    junk = ax[1].axvline(x = orig_no_comp, color = 'r')
    junk =  ax[1].text(orig_no_comp+0.1, ax[1].get_ylim()[1]*0.75, 'Original NCC = %d'%(orig_no_comp), fontsize = 'x-large')
    ax[1].set_title('Distribution of number of connected components (permuted)', fontsize = 'xx-large')

    G1 = nx.from_numpy_matrix(red_adj_mat)
    labeldict = {}
    cell_text = []
    for i in G1.nodes:
        labeldict[i] = ROIS[aa[i]]
        cell_text.append([str(ROIS[aa[i]]),all_parc[ROIS[aa[i]]]])
    fig, plt_ax = plt.subplots(figsize = (12,9))
    nx.draw_networkx(G1,labels=labeldict, with_labels = True, ax = plt_ax, font_color = 'r')
    # nx.draw_planar(G1,labels=labeldict, with_labels = True, ax = plt_ax, font_color = 'r')
    plt_ax.table(cellText=cell_text, colLabels = ['ROI-number','ROI-label'], loc='right', colWidths = [0.1,0.4],fontsize = 20)    
    plt_ax.set_title('Selected network graph (ROI labels in table)', fontsize = 'xx-large')

def Network_BS(file_name, sub_network = None):
    global orig_stat, perm_stat, ixes, ROIS, imp_coord, all_parc
    
    prepare_data(file_name, sub_network)
    lower_cut_off = np.percentile(orig_stat,85)
    higher_cut_off = np.amax(orig_stat) - 0.001
    cut_off_init = np.percentile(orig_stat,99)
    UI_network = widgets.interact_manual(get_plot_largest_network, cut_off = widgets.BoundedFloatText(value=cut_off_init, min = lower_cut_off, max = higher_cut_off, step = 0.001,
                                                                                  description='Cut-off:', disabled=False))
    display(UI_network)

#-------------functions related to node based staistics----------------------    

def plot_selected_node_conn(sel_node):
    global selected_node_conn, imp_coord, ROIS
    roi = int(sel_node.split('--->')[0])
    
    adj = selected_node_conn[roi]['adj_mat']
    rows_col_to_delete = []
    for i in range(adj.shape[0]):
        if(np.sum(adj[i,:]) == 0):
            rows_col_to_delete.append(i)
    red_adj_mat = np.delete(adj,rows_col_to_delete, axis = 0)
    red_adj_mat = np.delete(red_adj_mat,rows_col_to_delete, axis = 1)
    aa = np.setdiff1d(np.arange(0,adj.shape[0]),rows_col_to_delete)    
   
    fig, ax = plt.subplots(1,2, figsize = (20,6))
    plotting.plot_connectome(red_adj_mat,imp_coord[aa,:], axes = ax[0])    
    ax[0].set_title('Selected Node connectome',fontsize = 'xx-large')
    sns.kdeplot(selected_node_conn[roi]['perm_NC'], ax = ax[1])
    junk = ax[1].axvline(x = selected_node_conn[roi]['orig_NC'], color = 'r')
    junk =  ax[1].text(selected_node_conn[roi]['orig_NC']+0.1, ax[1].get_ylim()[1]*0.75, 'Original NCC = %d'%(selected_node_conn[roi]['orig_NC']), fontsize = 'x-large')
    ax[1].set_title('Distribution of number of connected nodes (permuted)', fontsize = 'xx-large')
    
    G1 = nx.from_numpy_matrix(red_adj_mat)
    labeldict = {}
    cell_text = []
    for i in G1.nodes:
        labeldict[i] = ROIS[aa[i]]
        cell_text.append([str(ROIS[aa[i]]),all_parc[ROIS[aa[i]]]])
    fig, plt_ax = plt.subplots(figsize = (12,9))
    nx.draw_networkx(G1,labels=labeldict, with_labels = True, ax = plt_ax, font_color = 'r')
    # nx.draw_planar(G1,labels=labeldict, with_labels = True, ax = plt_ax, font_color = 'r')
    plt_ax.table(cellText=cell_text, colLabels = ['ROI-number','ROI-label'], loc='right', colWidths = [0.1,0.4],fontsize = 20) 
    plt_ax.set_title('Selected node graph (ROI labels in table)', fontsize = 'xx-large')
    
def get_significant_node(cut_off, p_cut):
    global orig_stat, perm_stat, ROIS, imp_coord, all_parc, node_conn_locs, selected_node_conn
    
    original_cut_off = orig_stat>cut_off
    perm_cut_off = perm_stat>cut_off
    orig_cut_off_node_conn = {}
    node_p_value = {}
    no_runs = perm_stat.shape[0]
    temp_range = np.arange(0, len(ROIS), 1, dtype=int)
    selected_node_conn = {}
    sign_node_list = []
    for i,roi in enumerate(ROIS):
        orig_cut_off_node_conn[roi] = np.sum(original_cut_off[node_conn_locs[roi] == 1])
        perm_temp = np.sum(perm_cut_off[:,node_conn_locs[roi] == 1], axis = 1)
        node_p_value[roi] = np.sum(perm_temp>=orig_cut_off_node_conn[roi])/(no_runs+1)
        if(node_p_value[roi]<p_cut):
            cur_ind = np.setdiff1d(temp_range,np.array([i],dtype=int))
            yy = np.ones(len(ROIS))
            yy[cur_ind] = original_cut_off[node_conn_locs[roi] == 1]
            yy1 = np.zeros((len(ROIS),len(ROIS)))
            yy1[:,i] = yy
            yy1[i,:] = yy
            yy1[i,i] = 0
            selected_node_conn[roi] = {'adj_mat':yy1, 'orig_NC': orig_cut_off_node_conn[roi], 'perm_NC': perm_temp}
            sign_node_list.append('%d--->%s--->%0.4f'%(roi,all_parc[roi],node_p_value[roi]))

    UI1 = widgets.interact_manual(plot_selected_node_conn, sel_node = widgets.Select(options=sign_node_list,value=sign_node_list[0],rows= len(sign_node_list),description = 'Significant_nodes',
                                                                                   disabled=False,layout = widgets.Layout(width='500px')))    
    
    display(UI1)       
    
    
def Node_BS(file_name, sub_network = None):
    global orig_stat, perm_stat, ROIS, imp_coord, all_parc, node_conn_locs, selected_node_conn
    
    prepare_data(file_name, sub_network)
    lower_cut_off = np.percentile(orig_stat,85)
    higher_cut_off = np.amax(orig_stat) - 0.001
    cut_off_init = np.percentile(orig_stat,98) 
    
    up_tri = np.triu_indices(len(ROIS), k=1, m=None)
    node_conn_locs = {} 
    for i,roi in enumerate(ROIS):
        temp_conn_mat = np.zeros((len(ROIS),len(ROIS)))
        temp_conn_mat[i,:] = 1
        temp_conn_mat[:,i] = 1
        node_conn_locs[roi] = temp_conn_mat[up_tri]      
    # print(all_parc[10])
    UI_node = widgets.interact_manual(get_significant_node, cut_off = widgets.BoundedFloatText(value=cut_off_init, min = lower_cut_off, max = higher_cut_off, step = 0.001,
                                                                                  description='Cut-off:', disabled=False),
                                p_cut = widgets.BoundedFloatText(value=0.01, min = 0.001, max = 0.1, step = 0.001,description='p-cut:', disabled=False))
    display(UI_node)        