"""
This module containes some common tools used in the main file and training
"""

import cPickle
import collections
import os
import time

import numpy as np

import pandas as pd
# from IPython import embed


def x_normalization(x_arr, nm_key, labels, normdata=False):
    """
    Normalization of input space.
    The function returns a dictionary of pertinent normalization constants and the normalized data
    If not normadata, the noramlization is performed with hardcoded logic and unormalizaed data
    """
    def checklabel(labels):
        """
        Checks if the first dimension is the burnup
        """
        try:
            if labels[0] != 'BURNUP':
                raise ValueError('wrong preconditioning')
        except IndexError:
            pass
    if not isinstance(x_arr, np.ndarray):
        raise ValueError('not numpy vector for normalization')
    # here I make sure to work with a local copy of X, the ref to the X
    # outside the function is now lost
    x_arr = np.array(x_arr)
    if max([max(x_arr[..., i]) for i in range(len(x_arr[0]))]) > 1:
        raise ValueError('unnormlaized x_arr data needs update the\
         whole normadata thing to have equal test/train normalization')
    elif nm_key == 'N0_':
        if not normdata:
            nm_max = x_arr.max(axis=0)
        else:
            nm_max = normdata['max']
        return {'max': nm_max}, x_arr / nm_max
    elif nm_key == 'N2_':
        checklabel(labels)
        return {}, np.apply_along_axis(np.sqrt, 0, x_arr)
    elif nm_key == 'N3_':
        checklabel(labels)
        x_arr = np.apply_along_axis(np.sqrt, 0, x_arr)
        if not normdata:
            nm_mean = [np.mean(x_arr[..., dim]) for dim in range(len(x_arr[0]))]
        else:
            nm_mean = normdata['mean']
        for dim in range(len(x_arr[0])):
            x_arr[..., dim] = x_arr[..., dim] - nm_mean[dim]
            # print np.mean(X[...,dim])
        return {'mean': nm_mean}, x_arr
    else:
        raise ValueError('Unable to find required normalization')


def x_antinormalization(x_arr, nm_key, normdata):
    """
    Anti-normalization of input space .
    Inverse of normalization
    """
    if not isinstance(x_arr, np.ndarray):
        raise ValueError('not numpy vector for normalization')
    # here I make sure to work with a local copy of X, the ref to the X
    # outside the function is now lost
    x_arr = np.array(x_arr)
    if nm_key == 'N0_':
        nm_max = normdata['max']
        x_arr = x_arr * nm_max
    elif nm_key == 'N2_':
        x_arr = np.apply_along_axis(np.square, 0, x_arr)
    elif nm_key == 'N3_':
        no_mean = normdata['mean']
        for dim in range(len(x_arr[0])):
            x_arr[..., dim] = x_arr[..., dim] + no_mean[dim]
            # print np.mean(X[...,dim])
        x_arr = np.apply_along_axis(np.square, 0, x_arr)
    else:
        raise ValueError('Unable to find required normalization')
    return x_arr


def y_normalize(y_vec, norm_type=None, normdata=None):
    """
    Normalization of input space.
    The function returns a dictionary of pertinent normalization constants a,d the normalized data
    If not normadata, the noramlization is performed with hardcoded logic and unormalizaed data
    """
    if min(y_vec) < 0:
        raise ValueError('normilizing negative y_vec')
    elif norm_type == 'N0_':
        return {}, y_vec
    elif norm_type == 'N1_':
        if normdata is None:
            xsmax = max(y_vec)
        else:
            xsmax = normdata['xsmax']
        return {'xsmax': xsmax}, np.array([g / xsmax for g in y_vec], order="C")
    elif norm_type == 'N2_':
        if normdata is None:
            xsmax = max(y_vec)
            gtau = np.array([g / xsmax for g in y_vec], order="C")
            xsmean = np.mean(gtau)
        else:
            xsmax = normdata['xsmax']
            gtau = np.array([g / xsmax for g in y_vec], order="C")
            xsmean = normdata['xsmean']
        return {'xsmax': xsmax, 'xsmean': xsmean}, np.array([g - xsmean for g in gtau], order="C")
    elif norm_type == 'N3_':
        if normdata is None:
            xmean = np.mean(y_vec)
            xvar = np.std(y_vec)
        else:
            xmean = normdata['xmean']
            xvar = normdata['xvar']
        return {'xmean': xmean, 'xvar': xvar}, np.array(
            [(g - xmean) / xvar for g in y_vec], order="C")
    elif norm_type == 'N4_':
        y_vec = np.log(y_vec)
        if normdata is None:
            xmean = np.mean(y_vec)
            xvar = np.std(y_vec)
        else:
            xmean = normdata['xmean']
            xvar = normdata['xvar']
        return {'xmean': xmean, 'xvar': xvar}, np.array(
            [(g - xmean) / xvar for g in y_vec], order="C")
    else:
        raise ValueError('Unable to find required normalization')


def y_antinormalize(y_vec, norm_type, normdata):
    """
    Anti-normalization of output space .
    Inverse of normalization
    """
    if norm_type == 'N0_':
        return y_vec
    elif norm_type == 'N1_':
        xsmax = normdata['xsmax']
        return np.array([g * xsmax for g in y_vec], order="C")
    elif norm_type == 'N2_':
        xsmax = normdata['xsmax']
        xsmean = normdata['xsmean']
        aux = np.array([g + xsmean for g in y_vec], order="C")
        return np.array([g * xsmax for g in aux], order="C")
    elif norm_type == 'N3_':

        xmean = normdata['xmean']
        xvar = normdata['xvar']
        # return np.array([(g+xmean)*xvar for g in xs],order="C")
        return np.array([g * xvar + xmean for g in y_vec], order="C")
    elif norm_type == 'N4_':
        xmean = normdata['xmean']
        xvar = normdata['xvar']
        # return np.array([(g+xmean)*xvar for g in xs],order="C")
        return np.exp(np.array([g * xvar + xmean for g in y_vec], order="C"))
    else:
        raise ValueError('Unable to find required normalization')


def unify_evaluation(y_pred, y_true):
    """
    Statistical processing is encapsulated here
    Two dictionaries are return, one with vectorial and another with scalar values
    """
    y_error = y_pred - y_true
    vec_dict = {}
    vec_dict['predict'] = y_pred
    vec_dict['true'] = y_true
    vec_dict['err'] = y_error
    y_true_mean = np.mean(y_true)
    ss_res = np.sum(np.square(y_error))
    ss_tot = np.sum(np.square(y_true - y_true_mean))
    sca_dict = {}
    sca_dict['av_err'] = np.mean(np.abs(y_error))
    sca_dict['av_err_r'] = np.mean(np.abs(100 * np.divide(y_error, y_true, dtype=float)))
    sca_dict['std'] = np.std(y_error)
    sca_dict['M'] = np.max(np.abs(y_error))
    sca_dict['R2'] = 1 - ss_res / ss_tot
    return sca_dict, vec_dict


def get_plotable_data(lib_hooks, case, verbose=True):
    """
    Data stored in a dictionary or as tensor are transformed to a DataFrame
    """
    lib_nets_dfs = collections.OrderedDict()
    beg = time.time()
    for net_n in lib_hooks:
        lib_nets_dfs[net_n] = collections.OrderedDict()
        for kind in ['train', 'test']:
            lib_nets_dfs[net_n][kind] = {}
            for subkind in ['vectorial', 'scalar']:
                lib_nets_dfs[net_n][kind][subkind] = hook2df_epoch(
                    lib_hooks[net_n], case['labels'], kind, subkind)
        lib_nets_dfs[net_n]['par'] = hook2df_par(lib_hooks[net_n])
    if verbose:
        print 'done dfs in: ', "{0:0.2f}".format((time.time() - beg) / 60)
    return lib_nets_dfs  # ,xs_list


def hook2df_epoch(hook_h, labels, kind, subkind):
    """
    Transforms (vectorial or scalar) data stored in hooks into a single DataFrame
    """
    df_holder = collections.OrderedDict()
    epochs = hook_h.values()[0][kind].keys()
    for epoch in epochs:
        df_vec = []
        for model in hook_h:
            for func in hook_h[model][kind][epoch]:
                if subkind == 'vectorial':
                    # adding X values to the DataFrame with proper labels. df.shape=(N,d)
                    df = pd.DataFrame({key: val for key, val in zip(
                        labels, zip(*hook_h[model][kind][epoch][func][subkind]['x_batch']))})
                    # adding the rest of vectorial quantities, df.shape=(N,d+e). at time of
                    # writing e=4 (['predict', 'true', 'x_batch', 'err'])
                    for key, val in hook_h[model][kind][epoch][func][subkind].iteritems():
                        if key != 'x_batch':
                            df[key] = val
                    # ading f names. df.shape=(N,d+e+1)
                    df["xs"] = [func for _ in range(len(df['true']))]
                    df_vec.append(df)
                if subkind == 'scalar':
                    df = pd.DataFrame(hook_h[model][kind][epoch][func][subkind], index=[0])
                    df["xs"] = func
                    df_vec.append(df)
        # dfs for every f are concatenated into a single one
        df_holder[epoch] = pd.concat(df_vec).reset_index().drop('index', axis=1)
    return df_holder


def hook2df_par(hook_h):
    """
    ANN parameters are stored in a DataFrame for every epoch
    """
    def repeat(thing, times):
        """
        aux function
        """
        return [thing for _ in range(len(times))]
    # embed()
    df_h_out = {}
    for model in hook_h:
        df_vec = []
        for epoch in hook_h[model]['net_par']:
            df_vec_vec = []
            for name, val in hook_h[model]['net_par'][epoch].iteritems():
                into_df = val.view(1, val.numel()).tolist()[0]
                couche, par_type = name.split('.')
                couche_type, couche_number = couche.split('_')
                df_aux_params = pd.DataFrame(
                    {
                        'par_val': into_df,
                        'par_tot_name': repeat(
                            name,
                            into_df),
                        'par_type': repeat(
                            par_type,
                            into_df),
                        'couche_type': repeat(
                            couche_type,
                            into_df),
                        'couche_number': repeat(
                            couche_number,
                            into_df),
                        'epoch': repeat(
                            epoch,
                            into_df),
                        'par_index': [
                            'ii_' +
                            str(w) +
                            couche_number for w in list(
                                range(
                                    len(into_df)))]})
                df_vec_vec.append(df_aux_params)
            df_vec.append(pd.concat(df_vec_vec))
        df = pd.concat(df_vec)
        df.reset_index(inplace=True)
        df.drop('index', inplace=True, axis=1)
        df_h_out[model] = df
    return df_h_out


def save_results(lib_nets_dfs, lib_nets, data_hooks,
                 test_train_dict, case, ML_dict, verbose=True, adress=False, file_name=False):
    """
    Saving data to be later ploted/analyzed.
    test_train_dict is not saved as its data is present in the DataFrames, only normalization
     variables are conserved
    Two files allow to separate evaluated data and actual network, as to not require pytorch
    module for post-procesing.
    """
    norm_dict = {}
    for name in test_train_dict.keys():
        norm_dict[name] = {}
        norm_dict[name]['normdata'] = test_train_dict[name]['normdata']
        norm_dict[name]['IS_nm'] = test_train_dict[name]['IS_nm']
        norm_dict[name]['OS_nm'] = test_train_dict[name]['OS_nm']
    serilize = collections.OrderedDict()
    for model in lib_nets:
        serilize[model] = {}
        for func in lib_nets[model]:
            serilize[model][func] = lib_nets[model][func]['net'].state_dict()
    for kind, add2dict in zip(['data', 'net'], [{'lib_nets_dfs': lib_nets_dfs},
                                                {'serilize': serilize, 'data_hooks': data_hooks}]):
        if not file_name:
            file_name = ''.join([val.replace('.cpkl', '')
                            for val in case.values() if isinstance(val, str)]) + kind + '.cpkl'
        if verbose:
            print '.. saving models ', adress + file_name
        package = {
            'xs_list': test_train_dict.keys(),
            'case': case,
            'norm_data': norm_dict,
            'ML_dict': ML_dict}
        package.update(add2dict)
        grow(adress)
        with open(adress + file_name, 'wb') as serialization_file:
            cPickle.dump(package, serialization_file, cPickle.HIGHEST_PROTOCOL)


def grow(pth):
    """
    Unit function of generation for the tree of results
    """
    # print pth
    # print pth.split('/')

    aux = []
    for folder in pth.split('/'):
        if folder == '':
            continue
        aux.append('/' + folder)
        create = ''.join(aux)
        # print create,os.path.isdir(create)
        if not os.path.isdir(create):
            print "making dir", create
            os.mkdir(create)
            