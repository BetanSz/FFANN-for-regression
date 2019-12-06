"""
This module containes all functions using the torch module.
The most important function of the script, 'train' is here. Also ANN definitions
Other auxiliary functions can be found as well
"""

import collections
import copy as cp

import time

import numpy as np

import packages.common_tools as myctools
from sklearn.model_selection import train_test_split
import torch


def myloss_l1(y_pred, y_true):
    """
    Custom loss function
    """
    y_error = y_pred - y_true
    loss = torch.mean(torch.abs(100 * torch.div(y_error, y_true)))
    return loss


def myloss_max(y_pred, y_true):
    """
    Custom loss function
    """
    y_error = y_pred - y_true
    loss = torch.max(torch.abs(y_error))
    return loss


def train(net, optimizer, loss, train_test_dict, case, targets, verbose=True, lib_net='not given',
          target_idx='not given'):
    """
    This function perfoms the training of the ANNs for the user provided functions.
    One or several function may be approximated by a single ANN.
    When performing to(case['device']) objects are loaded to the GPU.
    Note that time estimation does not considers that the hooking vector is log.

    TODO: if out of memory onload each chuck of training data one used
    TODO: data preparation in another function and usng a loop
    """
    xs_per_net = len(case['tot_targets']) * case['n_nets']
    if verbose:
        print '... preparing batches by hand of ', targets
    # Data stored in disctionary is prepared for be feed to the ANN
    # by transofrming it to a tensor of y.shape=(number of points, number of functions)
    y_tr = []
    y_te = []
    normdatas = []
    normtypes = []
    for f_name in targets:
        y_tr.append(train_test_dict[f_name]['train']['y'])
        y_te.append(train_test_dict[f_name]['test']['y'])
        normdatas.append(train_test_dict[f_name]['normdata'])
        normtypes.append(train_test_dict[f_name]['OS_nm'])
    f_name = targets[0]
    x_tr = train_test_dict[f_name]['train']['x']
    x_te = train_test_dict[f_name]['test']['x']
    y_tr = torch.cat(y_tr, dim=1)
    y_te = torch.cat(y_te, dim=1)
    # Seting loop variables and estimators
    log_dict = collections.OrderedDict()
    log_dict['net_par'] = collections.OrderedDict()
    log_dict['train'] = collections.OrderedDict()
    log_dict['test'] = collections.OrderedDict()
    log_dict['times'] = {}
    log_dict['times']['batch'] = []
    log_dict['times']['train'] = []
    log_dict['times']['save'] = []
    log_dict['times']['print'] = [0]  # avoiding mean of ampty array in first case
    device = case['device']
    net.to(device)  # the net to be trained is loaded in divice. If OOM here== game over
    # Proceding to training
    # embed()
    log_counter = 0
    beg_real_train_time = time.time()
    for epoch in range(case['epochs']):
        # preparing batches
        beg = time.time()
        if case['shuffle'] == 'Syes_':
            tr_permutator = torch.randperm(len(x_tr))
            x_tr_chunk = x_tr[tr_permutator].chunk(case['train_div'])
            y_tr_chunk = y_tr[tr_permutator].chunk(case['train_div'])
        elif case['shuffle'] == 'Sno_':
            # batch_permutator=range(train_div)
            x_tr_chunk = x_tr.chunk(case['train_div'])
            y_tr_chunk = y_tr.chunk(case['train_div'])
        else:
            raise RuntimeError('Check case-shuffle option. Possible syntax error')
            # embed()
        log_dict['times']['batch'].append(time.time() - beg)
        beg = time.time()
        # Working with index has prove performant wrt loading the data directly
        batch_idxs = range(len(x_tr_chunk))
        for i in batch_idxs:
            # input X is loaded to device and used for training
            prediction_train = net(x_tr_chunk[i].to(device))
            current_loss = loss(prediction_train, y_tr_chunk[i].to(device))
            optimizer.zero_grad()  # clear gradients for next train
            current_loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradient
        log_dict['times']['train'].append(time.time() - beg)
        if epoch == case['hooking_vec'][log_counter]:  # if epoch in hooking_vec takes FOREVER
            def namespace_save(epoch):
                r"""
                namespace usefull when profiling
                """
                log_dict['net_par'][epoch] = cp.deepcopy(net.state_dict())
                for kind, x_vec, y_vec in zip(['train', 'test'], [x_tr, x_te], [y_tr, y_te]):
                    # if cuda memory problem do here storing by batch
                    log_dict[kind][epoch] = {}
                    for idx, f_name in enumerate(targets):
                        y_pred_nm = net(x_vec.to(device))
                        loss_data = loss(y_pred_nm, y_vec.to(device))
                        y_pred = myctools.y_antinormalize(y_pred_nm.cpu().data.numpy(
                        )[..., idx:idx + 1], norm_type=normtypes[idx], normdata=normdatas[idx])
                        y_true = myctools.y_antinormalize(y_vec.cpu().data.numpy(
                        )[..., idx:idx + 1], norm_type=normtypes[idx], normdata=normdatas[idx])
                        log_dict[kind][epoch][f_name] = {}
                        log_dict[kind][epoch][f_name]['vectorial'] = {}
                        log_dict[kind][epoch][f_name]['scalar'] = {}
                        log_dict[kind][epoch][f_name]['scalar'], log_dict[kind][epoch][
                            f_name]['vectorial'] = myctools.unify_evaluation(y_pred, y_true)
                        log_dict[kind][epoch][f_name]['vectorial'][
                            'x_batch'] = x_vec.cpu().data.numpy()
                        log_dict[kind][epoch][f_name]['scalar']['L'] = loss_data.data.cpu().numpy()

            def namespace_print(epoch):  # dummy function for generate a namespace usefull when profiling
                """
                Times of curent traning process are given.
                These are separed by batch, training, save and printing times
                Error evolution is provided as well.
                Estimation of remindent time for this ANN and
                for the execution of the script is provided as well.
                """
                pe = lambda x: "{:.2E}".format(x)  # print, exponential
                pfs = lambda x: "{0:0.2f}".format((1 / 60.0) * x)  # print, float, second
                pfp = lambda x, y: "{0:0.2f}".format((100 / y) * x)  # print, float, percentage
                pem = lambda x: "{:.2E}".format(np.mean(x))  # print, exponential,mean
                print 'for ', lib_net, ' doing ', targets, 'epoch/epochs,',\
                    epoch, '/', case['epochs']
                print 'Times [s/cicle]: batch, train, save [s]: ',\
                    pem(log_dict['times']['batch']), pem(log_dict['times']['train']),\
                    pem(log_dict['times']['save'])
                for kind in ['train', 'test']:
                    aux = log_dict[kind][epoch][f_name]['scalar']
                    print kind, ' av,av_r,std,M,L:', pe(aux['av_err']), pe(aux['av_err_r']),\
                        pe(aux['std']), pe(aux['M']), pe(aux['L'])
                cicle_time = np.mean(log_dict['times']['batch'] + log_dict['times']['train'])
                print 'f/tot: ', target_idx, '/', xs_per_net, 'Remains[min] current,total:',\
                    pfs(cicle_time * (case['epochs'] - epoch)),\
                    pfs(cicle_time * (xs_per_net - target_idx) * case['epochs'])
                tot_time = sum([sum(var) for var in log_dict['times'].values()])
                print 'Time distribution [%]', \
                    [[key, pfp(sum(val), tot_time)] for key, val in log_dict['times'].iteritems()]
                print ' '
            beg = time.time()
            namespace_save(epoch)
            log_dict['times']['save'].append(time.time() - beg)
            if verbose:
                beg = time.time()
                namespace_print(epoch)
                log_dict['times']['print'].append(time.time() - beg)
            log_counter += 1
            # break  # un-comment for avoiding any actual training
    log_dict['times']['total_real'] = time.time() - beg_real_train_time
    if verbose:
        print 'actual training % time', \
            "{0:0.2f}".format(100 * sum(log_dict['times']['train']) / log_dict['times']['total_real'])
    net.cpu()  # free GPU RAM
    return net, log_dict


def get_models(train_test_dict, case):
    """
    This function prepares the models for the different ANN studies.
    lib_nets holds the ANN models. There are two levels:
    The first which how the multi-outpu is handle, and the
     second which functions are actualy modeled
    """
    lib_nets = collections.OrderedDict()
    if case['ANN_style'] == 'xs_wise_':
        if case['ANN_subkind'] == 'dummy_':
            name = 'XS_indepe_' + 'Tanh_1_2'
            lib_nets[name] = collections.OrderedDict()
            for key in train_test_dict:
                lib_nets[name][key] = {}
                lib_nets[name][key]['net'] = FFSNN(
                    input_size=case['dim'], hidden_size=2, output_size=1, activation=torch.tanh)
        if case['ANN_subkind'] == 'tanhseries_':
            name = 'XS_indepe_' + 'Tanh_1_2'
            lib_nets[name] = collections.OrderedDict()
            for key in train_test_dict:
                lib_nets[name][key] = {}
                lib_nets[name][key]['net'] = FFSNN(
                    input_size=case['dim'], hidden_size=2, output_size=1, activation=torch.tanh)
            name = 'XS_indepe_' + 'Tanh_1_5'
            lib_nets[name] = collections.OrderedDict()
            for key in train_test_dict:
                lib_nets[name][key] = {}
                lib_nets[name][key]['net'] = FFSNN(
                    input_size=case['dim'], hidden_size=5, output_size=1, activation=torch.tanh)
            name = 'XS_indepe_' + 'Tanh_1_8'
            lib_nets[name] = collections.OrderedDict()
            for key in train_test_dict:
                lib_nets[name][key] = {}
                lib_nets[name][key]['net'] = FFSNN(input_size=case['dim'], hidden_size=8,
                                                   output_size=1, activation=torch.tanh)
            name = 'XS_indepe_' + 'Tanh_1_15'
            lib_nets[name] = collections.OrderedDict()
            for key in train_test_dict:
                lib_nets[name][key] = {}
                lib_nets[name][key]['net'] = FFSNN(
                    input_size=case['dim'], hidden_size=15, output_size=1, activation=torch.tanh)
            name = 'XS_indepe_' + 'Tanh_1_20'
            lib_nets[name] = collections.OrderedDict()
            for key in train_test_dict:
                lib_nets[name][key] = {}
                lib_nets[name][key]['net'] = FFSNN(
                    input_size=case['dim'], hidden_size=20, output_size=1, activation=torch.tanh)
            name = 'XS_indepe_' + 'Tanh_1_30'
            lib_nets[name] = collections.OrderedDict()
            for key in train_test_dict:
                lib_nets[name][key] = {}
                lib_nets[name][key]['net'] = FFSNN(
                    input_size=case['dim'], hidden_size=30, output_size=1, activation=torch.tanh)
        if case['ANN_subkind'] == 'tanhextreme2_':
            name = 'XS_indepe_' + 'Tanh_1_100'
            lib_nets[name] = collections.OrderedDict()
            for key in train_test_dict:
                lib_nets[name][key] = {}
                lib_nets[name][key]['net'] = FFSNN(
                    input_size=case['dim'], hidden_size=100, output_size=1, activation=torch.tanh)
            name = 'XS_indepe_' + 'Tanh_1_500'
            lib_nets[name] = collections.OrderedDict()
            for key in train_test_dict:
                lib_nets[name][key] = {}
                lib_nets[name][key]['net'] = FFSNN(
                    input_size=case['dim'], hidden_size=500, output_size=1, activation=torch.tanh)
        if case['ANN_subkind'] == 'tanh20_':
            name = 'XS_indepe_' + 'Tanh_1_20'
            lib_nets[name] = collections.OrderedDict()
            for key in train_test_dict:
                lib_nets[name][key] = {}
                lib_nets[name][key]['net'] = FFSNN(
                    input_size=case['dim'], hidden_size=20, output_size=1, activation=torch.tanh)
        if case['ANN_subkind'] == 'monsters_':
            name = 'XS_indepe_' + 'Tanh_1_50'
            lib_nets[name] = collections.OrderedDict()
            for key in train_test_dict:
                lib_nets[name][key] = {}
                lib_nets[name][key]['net'] = FFSNN(
                    input_size=case['dim'], hidden_size=50, output_size=1, activation=torch.tanh)
            name = 'XS_indepe_' + 'Pyramid'
            lib_nets[name] = collections.OrderedDict()
            for key in train_test_dict:
                lib_nets[name][key] = {}
                lib_nets[name][key]['net'] = FFDNN_pyramid(
                    input_size=case['dim'], output_size=1, activation=torch.tanh)
            name = 'XS_indepe_' + 'C1'
            lib_nets[name] = collections.OrderedDict()
            for key in train_test_dict:
                lib_nets[name][key] = {}
                lib_nets[name][key]['net'] = C1(input_size=case['dim'])
            name = 'XS_indepe_' + 'C2'
            lib_nets[name] = collections.OrderedDict()
            for key in train_test_dict:
                lib_nets[name][key] = {}
                lib_nets[name][key]['net'] = C2(input_size=case['dim'])
            name = 'XS_indepe_' + 'FFenco1NN'
            lib_nets[name] = collections.OrderedDict()
            for key in train_test_dict:
                lib_nets[name][key] = {}
                lib_nets[name][key]['net'] = FFenco1NN(input_size=case['dim'])
        if case['ANN_subkind'] == 'chosen2_':
            name = 'XS_indepe_' + 'Tanh_2_13'
            lib_nets[name] = collections.OrderedDict()
            for key in train_test_dict:
                lib_nets[name][key] = {}
                lib_nets[name][key]['net'] = FFDNN2(
                    input_size=case['dim'], hidden_size=13, output_size=1, activation=torch.tanh)
        if case['ANN_subkind'] == 'tanh5_':
            name = 'XS_indepe_' + 'Tanh_1_5'
            lib_nets[name] = collections.OrderedDict()
            for key in train_test_dict:
                lib_nets[name][key] = {}
                lib_nets[name][key]['net'] = FFSNN(
                    input_size=case['dim'], hidden_size=5, output_size=1, activation=torch.tanh)
        if case['ANN_subkind'] == 'tanh8_':
            name = 'XS_indepe_' + 'Tanh_1_8'
            lib_nets[name] = collections.OrderedDict()
            for key in train_test_dict:
                lib_nets[name][key] = {}
                lib_nets[name][key]['net'] = FFSNN(
                    input_size=case['dim'], hidden_size=8, output_size=1, activation=torch.tanh)
        if case['ANN_subkind'] == 'tanh10_init':
            name = 'XS_indepe_' + 'Tanh_1_10'
            lib_nets[name] = collections.OrderedDict()
            for key in train_test_dict:
                lib_nets[name][key] = {}
                lib_nets[name][key]['net'] = FFDNN3(
                    input_size=case['dim'], hidden_size=10, output_size=1, activation=torch.tanh)
            name = 'XS_indepe_' + 'Tanh_1_10B'
            lib_nets[name] = collections.OrderedDict()
            for key in train_test_dict:
                lib_nets[name][key] = {}
                lib_nets[name][key]['net'] = FFDNN3_B(
                    input_size=case['dim'], hidden_size=10, output_size=1, activation=torch.tanh)
            name = 'XS_indepe_' + 'Tanh_1_10C'
            lib_nets[name] = collections.OrderedDict()
            for key in train_test_dict:
                lib_nets[name][key] = {}
                lib_nets[name][key]['net'] = FFDNN3_C(
                    input_size=case['dim'], hidden_size=10, output_size=1, activation=torch.tanh)
        if case['ANN_subkind'] == 'Cseries_':
            name = 'XS_indepe_' + 'C1'
            lib_nets[name] = collections.OrderedDict()
            for key in train_test_dict:
                lib_nets[name][key] = {}
                lib_nets[name][key]['net'] = C1(input_size=case['dim'])
            name = 'XS_indepe_' + 'Tanh_1_46'
            lib_nets[name] = collections.OrderedDict()
            for key in train_test_dict:
                lib_nets[name][key] = {}
                lib_nets[name][key]['net'] = FFSNN(
                    input_size=case['dim'], hidden_size=46, output_size=1, activation=torch.tanh)
            name = 'XS_indepe_' + 'Tanh_3_9'
            lib_nets[name] = collections.OrderedDict()
            for key in train_test_dict:
                lib_nets[name][key] = {}
                lib_nets[name][key]['net'] = FFDNN3(
                    input_size=case['dim'], hidden_size=9, output_size=1, activation=torch.tanh)
        if case['ANN_subkind'] == 'WIvsDe2_':
            name = 'XS_indepe_' + 'Tanh_1_20'
            lib_nets[name] = collections.OrderedDict()
            for key in train_test_dict:
                lib_nets[name][key] = {}
                lib_nets[name][key]['net'] = FFSNN(
                    input_size=case['dim'], hidden_size=20, output_size=1, activation=torch.tanh)
            name = 'XS_indepe_' + 'Tanh_2_7'
            lib_nets[name] = collections.OrderedDict()
            for key in train_test_dict:
                lib_nets[name][key] = {}
                lib_nets[name][key]['net'] = FFDNN2(
                    input_size=case['dim'], hidden_size=7, output_size=1, activation=torch.tanh)
            name = 'XS_indepe_' + 'Tanh_3_5'
            lib_nets[name] = collections.OrderedDict()
            for key in train_test_dict:
                lib_nets[name][key] = {}
                lib_nets[name][key]['net'] = FFDNN3(
                    input_size=case['dim'], hidden_size=5, output_size=1, activation=torch.tanh)
            name = 'XS_indepe_' + 'Tanh_4_4'
            lib_nets[name] = collections.OrderedDict()
            for key in train_test_dict:
                lib_nets[name][key] = {}
                lib_nets[name][key]['net'] = FFDNN4(
                    input_size=case['dim'], hidden_size=4, output_size=1, activation=torch.tanh)
            name = 'XS_indepe_' + 'Tanh_5_3'
            lib_nets[name] = collections.OrderedDict()
            for key in train_test_dict:
                lib_nets[name][key] = {}
                lib_nets[name][key]['net'] = FFDNN5(
                    input_size=case['dim'], hidden_size=3, output_size=1, activation=torch.tanh)
        if case['ANN_subkind'] == 'all_':
            name = 'XS_indepe_' + 'tanh_1_20'
            lib_nets[name] = collections.OrderedDict()
            for key in train_test_dict:
                lib_nets[name][key] = {}
                lib_nets[name][key]['net'] = FFSNN(
                    input_size=case['dim'], hidden_size=20, output_size=1, activation=torch.tanh)
            name = 'XS_indepe_' + 'ELU_1_20'
            lib_nets[name] = collections.OrderedDict()
            for key in train_test_dict:
                lib_nets[name][key] = {}
                lib_nets[name][key]['net'] = FFSNN(
                    input_size=case['dim'], hidden_size=20, output_size=1,
                    activation=torch.nn.ELU())
            name = 'XS_indepe_' + 'Lrelu_1_20'
            lib_nets[name] = collections.OrderedDict()
            for key in train_test_dict:
                lib_nets[name][key] = {}
                lib_nets[name][key]['net'] = FFSNN(
                    input_size=case['dim'], hidden_size=20, output_size=1,
                    activation=torch.nn.LeakyReLU())
            name = 'XS_indepe_' + 'HS_1_20'
            lib_nets[name] = collections.OrderedDict()
            for key in train_test_dict:
                lib_nets[name][key] = {}
                lib_nets[name][key]['net'] = FFSNN(
                    input_size=case['dim'], hidden_size=20, output_size=1,
                    activation=torch.nn.Hardshrink())
            name = 'XS_indepe_' + 'Sg_1_20'
            lib_nets[name] = collections.OrderedDict()
            for key in train_test_dict:
                lib_nets[name][key] = {}
                lib_nets[name][key]['net'] = FFSNN(
                    input_size=case['dim'], hidden_size=20, output_size=1,
                    activation=torch.nn.Sigmoid())
    if case['ANN_style'] == 'vec_1':
        if case['ANN_subkind'] == 'exp0_':
            name = 'XS_allmix' + 'tanh_1_10'
            lib_nets[name] = collections.OrderedDict()
            key_mix = 'allmix'
            lib_nets[name][key_mix] = {}
            lib_nets[name][key_mix]['net'] = FFSNN(
                input_size=case['dim'], hidden_size=10, output_size=len(train_test_dict),
                activation=torch.tanh)
        if case['ANN_subkind'] == 'exp1_':
            name = 'XS_tranmix' + 'tanh_1_10'
            lib_nets[name] = collections.OrderedDict()
            trans = []
            for key in train_test_dict:
                if 'tran' not in key:
                    lib_nets[name][key] = {}
                    lib_nets[name][key]['net'] = FFSNN(
                        input_size=case['dim'], hidden_size=10, output_size=1,
                        activation=torch.tanh)
                else:
                    trans.append(key)
            key_mix = 'tranmix'
            lib_nets[name][key_mix] = {}
            lib_nets[name][key_mix]['net'] = FFSNN(
                input_size=case['dim'], hidden_size=10, output_size=len(trans),
                activation=torch.tanh)
            lib_nets[name][key_mix]['targets'] = trans
        if case['ANN_subkind'] == 'all_StanH_':
            name = 'XS_allmix' + 'tanh_1_10'
            lib_nets[name] = collections.OrderedDict()
            key_mix = 'allmix'
            lib_nets[name][key_mix] = {}
            lib_nets[name][key_mix]['net'] = FFSNN(
                input_size=case['dim'], hidden_size=10, output_size=len(train_test_dict),
                activation=torch.tanh)
            lib_nets[name][key_mix]['targets'] = train_test_dict.keys()
            name = 'XS_allmix' + 'tanh_1_20'
            lib_nets[name] = collections.OrderedDict()
            key_mix = 'allmix'
            lib_nets[name][key_mix] = {}
            lib_nets[name][key_mix]['net'] = FFSNN(
                input_size=case['dim'], hidden_size=20, output_size=len(train_test_dict),
                activation=torch.tanh)
            lib_nets[name][key_mix]['targets'] = train_test_dict.keys()
            name = 'XS_allmix' + 'tanh_1_30'
            lib_nets[name] = collections.OrderedDict()
            key_mix = 'allmix'
            lib_nets[name][key_mix] = {}
            lib_nets[name][key_mix]['net'] = FFSNN(
                input_size=case['dim'], hidden_size=30, output_size=len(train_test_dict),
                activation=torch.tanh)
            lib_nets[name][key_mix]['targets'] = train_test_dict.keys()
            name = 'XS_allmix' + 'tanh_1_40'
            lib_nets[name] = collections.OrderedDict()
            key_mix = 'allmix'
            lib_nets[name][key_mix] = {}
            lib_nets[name][key_mix]['net'] = FFSNN(
                input_size=case['dim'], hidden_size=40, output_size=len(train_test_dict),
                activation=torch.tanh)
            lib_nets[name][key_mix]['targets'] = train_test_dict.keys()
        if case['ANN_subkind'] == 'exp2_':
            name = 'XS_allmix' + 'tanh_1_10'
            lib_nets[name] = collections.OrderedDict()
            key_mix = 'allmix'
            lib_nets[name][key_mix] = {}
            lib_nets[name][key_mix]['net'] = FFSNN(
                input_size=case['dim'], hidden_size=10, output_size=len(train_test_dict),
                activation=torch.tanh)
            lib_nets[name][key_mix]['targets'] = train_test_dict.keys()
            name = 'XS_allmix' + 'diam1_5_M50'
            lib_nets[name] = collections.OrderedDict()
            key_mix = 'allmix'
            lib_nets[name][key_mix] = {}
            lib_nets[name][key_mix]['net'] = FFDNN_pyramid(
                input_size=case['dim'], output_size=len(train_test_dict), activation=torch.tanh)
            lib_nets[name][key_mix]['targets'] = train_test_dict.keys()
            name = 'XS_allmix' + 'enco1_5_M50'
            lib_nets[name] = collections.OrderedDict()
            key_mix = 'allmix'
            lib_nets[name][key_mix] = {}
            lib_nets[name][key_mix]['net'] = FFenco1NN(
                input_size=case['dim'], output_size=len(train_test_dict))
            lib_nets[name][key_mix]['targets'] = train_test_dict.keys()
    if case['prediction'] == 'float64_':
        for model in lib_nets:
            for target in lib_nets[model]:
                lib_nets[model][target]['net'] = lib_nets[model][target]['net'].double()
    return lib_nets


# Definition of ANN used in [1]
class FFSNN(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size, activation):
        super(FFSNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.fc_1 = torch.nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.fc_2 = torch.nn.Linear(self.hidden_size, self.output_size, bias=True)

        torch.nn.init.uniform_(self.fc_1.weight)
        torch.nn.init.uniform_(self.fc_2.weight)
        torch.nn.init.zeros_(self.fc_1.bias)
        torch.nn.init.zeros_(self.fc_2.bias)

    def forward(self, x):
        output = self.fc_1(x)
        output = self.activation(output)
        output = self.fc_2(output)
        return output


class FFSNN_antierror(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size, activation):
        super(FFSNN_antierror, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.fc_1 = torch.nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.fc_2 = torch.nn.Linear(self.hidden_size, 1, bias=True)
        self.fc_3 = torch.nn.Linear(1, self.hidden_size, bias=True)
        self.fc_4 = torch.nn.Linear(self.hidden_size, self.output_size, bias=True)

        torch.nn.init.uniform_(self.fc_1.weight)
        torch.nn.init.uniform_(self.fc_2.weight)
        torch.nn.init.uniform_(self.fc_3.weight)
        torch.nn.init.uniform_(self.fc_4.weight)
        torch.nn.init.zeros_(self.fc_1.bias)
        torch.nn.init.zeros_(self.fc_2.bias)
        torch.nn.init.zeros_(self.fc_3.bias)
        torch.nn.init.zeros_(self.fc_4.bias)

    def forward(self, x):
        output = self.fc_1(x)
        output = self.activation(output)
        output = self.fc_2(output)
        output = self.fc_3(output)
        output = self.activation(output)
        output = self.fc_4(output)
        return output


class FFDNN_pyramid(torch.nn.Module):

    def __init__(self, input_size, output_size, activation):
        super(FFDNN_pyramid, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.fc_1 = torch.nn.Linear(self.input_size, 3, bias=True)
        self.fc_2 = torch.nn.Linear(3, 5, bias=True)
        self.fc_3 = torch.nn.Linear(5, 17, bias=True)
        self.fc_4 = torch.nn.Linear(17, 5, bias=True)
        self.fc_5 = torch.nn.Linear(5, 3, bias=True)
        self.fc_6 = torch.nn.Linear(3, self.output_size, bias=True)

        torch.nn.init.uniform_(self.fc_1.weight)
        torch.nn.init.uniform_(self.fc_2.weight)
        torch.nn.init.uniform_(self.fc_3.weight)
        torch.nn.init.uniform_(self.fc_4.weight)
        torch.nn.init.uniform_(self.fc_5.weight)
        torch.nn.init.uniform_(self.fc_6.weight)
        torch.nn.init.zeros_(self.fc_1.bias)
        torch.nn.init.zeros_(self.fc_2.bias)
        torch.nn.init.zeros_(self.fc_3.bias)
        torch.nn.init.zeros_(self.fc_4.bias)
        torch.nn.init.zeros_(self.fc_5.bias)
        torch.nn.init.zeros_(self.fc_6.bias)

    def forward(self, x):
        output = self.fc_1(x)
        output = self.activation(output)
        output = self.fc_2(output)
        output = self.activation(output)
        output = self.fc_3(output)
        output = self.activation(output)
        output = self.fc_4(output)
        output = self.activation(output)
        output = self.fc_5(output)
        output = self.activation(output)
        output = self.fc_6(output)
        return output


class FFDNN2(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size, activation):
        super(FFDNN2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.fc_in = torch.nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.fc_2 = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.fc_out = torch.nn.Linear(self.hidden_size, self.output_size, bias=True)
        torch.nn.init.uniform_(self.fc_in.weight)
        torch.nn.init.uniform_(self.fc_2.weight)
        torch.nn.init.uniform_(self.fc_out.weight)

    def forward(self, x):
        output = self.fc_in(x)
        output = self.activation(output)
        output = self.fc_2(output)
        output = self.activation(output)
        output = self.fc_out(output)
        return output


class FFDNN3(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size, activation):
        super(FFDNN3, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.fc_in = torch.nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.fc_2 = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.fc_3 = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.fc_out = torch.nn.Linear(self.hidden_size, self.output_size, bias=True)

        torch.nn.init.uniform_(self.fc_in.weight)
        torch.nn.init.uniform_(self.fc_2.weight)
        torch.nn.init.uniform_(self.fc_3.weight)
        torch.nn.init.uniform_(self.fc_out.weight)
        torch.nn.init.zeros_(self.fc_in.bias)
        torch.nn.init.zeros_(self.fc_2.bias)
        torch.nn.init.zeros_(self.fc_3.bias)
        torch.nn.init.zeros_(self.fc_out.bias)

    def forward(self, x):
        output = self.fc_in(x)
        output = self.activation(output)
        output = self.fc_2(output)
        output = self.activation(output)
        output = self.fc_3(output)
        output = self.activation(output)
        output = self.fc_out(output)
        return output


class FFDNN3_B(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size, activation):
        super(FFDNN3_B, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.fc_in = torch.nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.fc_2 = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.fc_3 = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.fc_out = torch.nn.Linear(self.hidden_size, self.output_size, bias=True)
        if activation.__name__ in ['sigmoid', 'tanh']:
            torch.nn.init.xavier_uniform(self.fc_in.weight)
            torch.nn.init.xavier_uniform(self.fc_2.weight)
            torch.nn.init.xavier_uniform(self.fc_3.weight)
            torch.nn.init.xavier_uniform(self.fc_out.weight)

    def forward(self, x):
        output = self.fc_in(x)
        output = self.activation(output)
        output = self.fc_2(output)
        output = self.activation(output)
        output = self.fc_3(output)
        output = self.activation(output)
        output = self.fc_out(output)
        return output


class FFDNN3_C(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size, activation):
        super(FFDNN3_C, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.fc_in = torch.nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.fc_2 = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.fc_3 = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.fc_out = torch.nn.Linear(self.hidden_size, self.output_size, bias=True)
        if activation.__name__ in ['sigmoid', 'tanh']:
            torch.nn.init.normal_(self.fc_in.weight)
            torch.nn.init.normal_(self.fc_2.weight)
            torch.nn.init.normal_(self.fc_3.weight)
            torch.nn.init.normal_(self.fc_out.weight)

    def forward(self, x):
        output = self.fc_in(x)
        output = self.activation(output)
        output = self.fc_2(output)
        output = self.activation(output)
        output = self.fc_3(output)
        output = self.activation(output)
        output = self.fc_out(output)
        return output


class FFDNN4(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size, activation):
        super(FFDNN4, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.fc_in = torch.nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.fc_2 = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.fc_3 = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.fc_4 = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.fc_out = torch.nn.Linear(self.hidden_size, self.output_size, bias=True)
        torch.nn.init.uniform_(self.fc_in.weight)
        torch.nn.init.uniform_(self.fc_2.weight)
        torch.nn.init.uniform_(self.fc_3.weight)
        torch.nn.init.uniform_(self.fc_4.weight)
        torch.nn.init.uniform_(self.fc_out.weight)

    def forward(self, x):
        output = self.fc_in(x)
        output = self.activation(output)
        output = self.fc_2(output)
        output = self.activation(output)
        output = self.fc_3(output)
        output = self.activation(output)
        output = self.fc_4(output)
        output = self.activation(output)
        output = self.fc_out(output)
        return output


class FFDNN5(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size, activation):
        super(FFDNN5, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.fc_in = torch.nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.fc_2 = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.fc_3 = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.fc_4 = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.fc_5 = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.fc_out = torch.nn.Linear(self.hidden_size, self.output_size, bias=True)

        torch.nn.init.uniform_(self.fc_in.weight)
        torch.nn.init.uniform_(self.fc_2.weight)
        torch.nn.init.uniform_(self.fc_3.weight)
        torch.nn.init.uniform_(self.fc_4.weight)
        torch.nn.init.uniform_(self.fc_5.weight)
        torch.nn.init.uniform_(self.fc_out.weight)

    def forward(self, x):
        output = self.fc_in(x)
        output = self.activation(output)
        output = self.fc_2(output)
        output = self.activation(output)
        output = self.fc_3(output)
        output = self.activation(output)
        output = self.fc_4(output)
        output = self.activation(output)
        output = self.fc_5(output)
        output = self.activation(output)
        output = self.fc_out(output)
        return output


class C1(torch.nn.Module):

    def __init__(self, input_size, output_size=1):
        super(C1, self).__init__()
        self.input_size = input_size
        self.fc_1 = torch.nn.Linear(self.input_size, 16, bias=True)
        self.fc_2 = torch.nn.Linear(16, 8, bias=True)
        self.fc_3 = torch.nn.Linear(8, 5, bias=True)
        self.fc_4 = torch.nn.Linear(5, output_size, bias=True)
        torch.nn.init.uniform_(self.fc_1.weight)
        torch.nn.init.uniform_(self.fc_2.weight)
        torch.nn.init.xavier_uniform(self.fc_3.weight)
        torch.nn.init.uniform_(self.fc_4.weight)
        self.act1 = torch.nn.LeakyReLU()
        torch.nn.init.zeros_(self.fc_1.bias)
        torch.nn.init.zeros_(self.fc_2.bias)
        torch.nn.init.zeros_(self.fc_3.bias)
        torch.nn.init.zeros_(self.fc_4.bias)

    def forward(self, x):
        # print x.device
        # sys.exit()
        output = self.fc_1(x)
        output = torch.tanh(output)
        output = self.fc_2(output)
        output = torch.tanh(output)
        output = self.fc_3(output)
        output = torch.tanh(output)
        output = self.fc_4(output)
        return output


class C2(torch.nn.Module):

    def __init__(self, input_size, output_size=1):
        super(C2, self).__init__()
        self.input_size = input_size
        self.fc_1 = torch.nn.Linear(self.input_size, 5, bias=True)
        self.fc_2 = torch.nn.Linear(5, 8, bias=True)
        self.fc_3 = torch.nn.Linear(8, 18, bias=True)
        self.fc_4 = torch.nn.Linear(18, output_size, bias=True)
        torch.nn.init.uniform_(self.fc_1.weight)
        torch.nn.init.uniform_(self.fc_2.weight)
        torch.nn.init.xavier_uniform(self.fc_3.weight)
        torch.nn.init.uniform_(self.fc_4.weight)
        self.act1 = torch.nn.LeakyReLU()
        torch.nn.init.zeros_(self.fc_1.bias)
        torch.nn.init.zeros_(self.fc_2.bias)
        torch.nn.init.zeros_(self.fc_3.bias)
        torch.nn.init.zeros_(self.fc_4.bias)

    def forward(self, x):
        # print x.device
        # sys.exit()
        output = self.fc_1(x)
        output = torch.tanh(output)
        output = self.fc_2(output)
        output = torch.tanh(output)
        output = self.fc_3(output)
        output = torch.tanh(output)
        output = self.fc_4(output)
        return output


class FFenco1NN(torch.nn.Module):

    def __init__(self, input_size, output_size=1):
        super(FFenco1NN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc_1 = torch.nn.Linear(self.input_size, 13, bias=True)
        self.fc_2 = torch.nn.Linear(13, 5, bias=True)
        self.fc_3 = torch.nn.Linear(5, 3, bias=True)
        self.fc_4 = torch.nn.Linear(3, 5, bias=True)
        self.fc_5 = torch.nn.Linear(5, 13, bias=True)
        self.fc_6 = torch.nn.Linear(13, output_size, bias=True)
        torch.nn.init.xavier_uniform(self.fc_1.weight)
        torch.nn.init.xavier_uniform(self.fc_2.weight)
        torch.nn.init.xavier_uniform(self.fc_3.weight)
        torch.nn.init.xavier_uniform(self.fc_4.weight)
        torch.nn.init.xavier_uniform(self.fc_5.weight)
        torch.nn.init.xavier_uniform(self.fc_6.weight)
        torch.nn.init.zeros_(self.fc_1.bias)
        torch.nn.init.zeros_(self.fc_2.bias)
        torch.nn.init.zeros_(self.fc_3.bias)
        torch.nn.init.zeros_(self.fc_4.bias)
        torch.nn.init.zeros_(self.fc_5.bias)
        torch.nn.init.zeros_(self.fc_6.bias)

    def forward(self, x):
        output = self.fc_1(x)
        output = torch.tanh(output)
        output = self.fc_2(output)
        output = torch.tanh(output)
        output = self.fc_3(output)
        output = torch.tanh(output)
        output = self.fc_4(output)
        output = torch.tanh(output)
        output = self.fc_5(output)
        output = torch.tanh(output)
        output = self.fc_6(output)
        return output


def train_test_generator(x_data, y_data, labels=[], OS_nm=None, IS_nm=None,
                         test_size=0.2, dtype_flag='float32_'):
    """
    A training/test dictionary is generated from X,Y data.
    Normalization is apply to Y (OS_nm, i.e. output space) and to X (IS_nm, i.e. input space)
    """
    # embed()
    _, x_data_nm = myctools.x_normalization(x_data, IS_nm, labels)
    train_test_dict = collections.OrderedDict()
    for name in y_data:
        normdata, y_data_nm = myctools.y_normalize(y_data[name], norm_type=OS_nm)
        x_vec_train, x_vec_test, y_vec_train, y_vec_test = train_test_split(
            x_data_nm, y_data_nm, test_size=test_size, random_state=2)
        train_test_dict[name] = collections.OrderedDict()
        train_test_dict[name]['normdata'] = normdata
        train_test_dict[name]['OS_nm'] = OS_nm
        train_test_dict[name]['IS_nm'] = IS_nm
        if dtype_flag == 'float64_':
            dtype = torch.float64
        else:
            dtype = torch.float32
        for x_vec, y_vec, key in\
                zip([x_vec_train, x_vec_test], [y_vec_train, y_vec_test], ['train', 'test']):
            train_test_dict[name][key] = {}
            train_test_dict[name][key]['x'] = torch.tensor(x_vec, dtype=dtype)
            train_test_dict[name][key]['y'] = torch.unsqueeze(
                torch.tensor(y_vec, dtype=dtype), dim=1)
    return train_test_dict
