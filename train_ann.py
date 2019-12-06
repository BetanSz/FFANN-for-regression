r"""
Examples used are present in package/pytorch_tools.py but custom ones can be 
generated as well A data set on the form X,Y being
X=[x_1,x_2...,x_N] and x_1=(x_11,x_12,..,x_1d)
Y=[y_1,...,y_t] and y_1=[y_11,y_12,...,y_1N]

let |v|=len(v) then

|X|=N, number of data points |x_i|=d number of dimension for 1\leq\leq N
|Y|=t, number of (target) functions to approximate. Each |y_i|=N outputs for 1\leq\leq t 

For the public user the test/division split is perfomed using scikit_learn

The majority of the script is actually handelling the data and the HP parameters. The
core elements are quite reduced, being basically the training statments responsable for the
back-prop process. This is agnostic code as can be equally run in GPU or CPU as defined by the
'device' handle.
"""

from __future__ import division

import cPickle
import collections
import json
import os
import sys
import time

import numpy as np
from IPython import embed


import packages.common_tools as myctools
import packages.pytorch_tools as myttools


def get_device(use_cuda, verbose=True):
    r"""
    Defines the GPU or CPU divice on which training is performed
    """
    # pylint: disable=E1101
    if myttools.torch.cuda.is_available() and use_cuda:
        device = myttools.torch.device("cuda:0")
        myttools.torch.cuda.empty_cache()
        if verbose:
            print device
            print myttools.torch.cuda.current_device()
            print myttools.torch.cuda.get_device_properties(device)
            print myttools.torch.cuda.memory_allocated() / 1E9
            print myttools.torch.cuda.memory_cached() / 1E9
            print myttools.torch.cuda.get_device_properties(device).total_memory / 1E9
    else:
        device = myttools.torch.device("cpu")
        if verbose:
            print "NOT USING GPU!"
    # pylint: enable=E1101
    return device


def only_allowed(y_vec, allowed=False):
    r"""
    selection of allowed functions
    """
    if not allowed:
        allowed = y_vec.keys()
    return {key: y_vec[key] for key in y_vec if key in allowed}


def user_run():
    r"""
    Here the user may define the hyper-parameters.
    Or the case dict that defines the run can be given by input in a bash file.
    e.g.
    python train_ann.py '{"file_name":"Article_example_","ANN_style":"xs_wise_",
    "ANN_subkind":"dummy_","OS_nm":"N4_","IS_nm":"N3_","WD":0.0,
    "loss":"L1_","train_div":20,"lr":0.001,"wd":0.0,"prediction":"float32_",
    "shuffle":"Syes_","optimizer":"Adam_","train_div":50
    }' '5' '5'
    """
    adress = os.getcwd() + '/data_example/'
    name = 'XY_example.pkle'
    # embed()
    obj = cPickle.load(open(adress + name, 'rb'))
    labels = obj['labels']
    x_vec = obj['x']
    y_vec = obj['y']
    y_vec = only_allowed(y_vec, allowed=['U235nufi2', 'MACRtran0121'])
    # embed()
    if len(sys.argv) > 1:
        case = json.JSONDecoder(object_pairs_hook=collections.OrderedDict).decode(sys.argv[1])
        num = float(sys.argv[2])
        exp = int(sys.argv[3])
        # embed()
    else:
        case = collections.OrderedDict()  # this irder is used when saving the file
        case['file_name'] = 'Article_example_'
        case['ANN_style'] = 'xs_wise_'
        case['ANN_subkind'] = 'chosen2_'
        case['OS_nm'] = 'N4_'
        case['IS_nm'] = 'N3_'
        case['WD'] = 0.0
        case['loss'] = 'L1_'
        case['train_div'] = 5
        case['lr'] = 0.001
        case['wd'] = 0.0
        case['prediction'] = 'float32_'
        case['shuffle'] = 'Syes_'
        case['optimizer'] = 'Adam_'
        case['train_div'] = 50
        num = 5
        exp = 4
    case['dim'] = len(x_vec[0])
    case['labels'] = labels
    case['epochs'] = int(num * np.power(10, exp))  # / case['train_div']
    case['hooking_vec'] = sorted(set([num * int(val) for val in np.logspace(0, exp, num=350)]))
    # embed()
    case['test_div'] = 1
    case['tot_targets'] = y_vec.keys()
    test_train_dict = myttools.train_test_generator(
        x_vec, y_vec, labels=labels, OS_nm=case['OS_nm'],
        IS_nm=case['IS_nm'], test_size=0.2, dtype_flag=case['prediction'])
    return case, test_train_dict, {}


def main():
    r"""
    user_case holds all relevant parameter of the run in form os str, int or float 
    It is used to generate the ouptut names unless stated otherwise.
    OrderDict allows for naming consistancy
    Other key can be useful, as for examples in [notes]='date_of_the_run'
    Script variables such as verbose, use_cuda are not stored in this orderdict
    Statistical data of the models is obtained during training in log_dict.
    """
    # Preparing the run case
    verbose = True  # True, Frue
    use_cuda = True
    myttools.torch.manual_seed(1)
    # case,TT,ML_dict=get_user_case_private() # privite library not availabe in public version
    # embed()
    case, test_train_dict, ml_dict = user_run()
    case['device'] = get_device(use_cuda, verbose) # set divse
    lib_nets = myttools.get_models(test_train_dict, case)  # Preparing models
    case['n_nets'] = len(lib_nets)
    beg = time.time()
    log_dict = collections.OrderedDict()
    # embed()
    for lib_net in lib_nets:  # different network architectures
        log_dict[lib_net] = collections.OrderedDict()
        for target_idx, target_n in enumerate(lib_nets[lib_net]):  # different set of y\in Y
            net = lib_nets[lib_net][target_n]['net']  # handle of network
            try:
                targets = lib_nets[lib_net][target_n]['targets']  # multi-output
            except KeyError:
                targets = [target_n]  # single output, i.e. target=y
            if case['optimizer'] == 'Adam_':
                optimizer = myttools.torch.optim.Adam(
                    net.parameters(), lr=case['lr'], weight_decay=case['wd'])
            if case['loss'] == 'L1_':
                loss = myttools.torch.nn.L1Loss(reduction='mean')
            lib_nets[lib_net][target_n]['net'], log_dict[lib_net][target_n] = myttools.train(
                net, optimizer, loss, test_train_dict, case, targets, lib_net=lib_net,
                target_idx=target_idx, verbose=verbose)
    if verbose:
        print 'total training time [h]', (time.time() - beg) / (60 * 60)
        print '.. building dfs'
    # embed()
    lib_nets_dfs = myctools.get_plotable_data(log_dict, case, ml_dict)
    adress = os.getcwd() + '/Results/'
    myctools.save_results(lib_nets_dfs, lib_nets, log_dict, test_train_dict,
                          case, ml_dict, adress=adress, verbose=verbose)
    print 'END'

if __name__ == "__main__":
    main()
