__author__ = 'phtra'

import time as timer
import sys
from collections import OrderedDict
import cPickle

import numpy
import theano
import theano.tensor as tensor
from theano import config

import readm
import next_diag
import high_risk
import curr_pm
import joint_tasks
import readm_all
import admissions

SEED = 123
numpy.random.seed(SEED)

tasks = {'readm': (readm.load_options, readm.prepare_data, readm.prepare_train, readm.build_model, readm.init_top_params),
         'next_diag': (next_diag.load_options, next_diag.prepare_data, next_diag.prepare_train, next_diag.build_model, next_diag.init_top_params),
         'high_risk': (high_risk.load_options, high_risk.prepare_data, high_risk.prepare_train, high_risk.build_model, high_risk.init_top_params),
         'curr_pm': (curr_pm.load_options, curr_pm.prepare_data, curr_pm.prepare_train, curr_pm.build_model, curr_pm.init_top_params),
         'readm_all':(readm_all.load_options, readm_all.prepare_data, readm_all.prepare_train, readm_all.build_model, readm_all.init_top_params)}

tast_result = {'readm': (readm.init_best, readm.update_best, readm.evaluation, readm.to_string),
               'next_diag': (next_diag.init_best, next_diag.update_best, next_diag.evaluation, next_diag.to_string),
               'high_risk': (high_risk.init_best, high_risk.update_best, high_risk.evaluation, high_risk.to_string),
               'curr_pm': (curr_pm.init_best, curr_pm.update_best, curr_pm.evaluation, curr_pm.to_string),
               'readm_all': (readm_all.init_best, readm_all.update_best, readm_all.evaluation, readm_all.to_string)}

durations = {'diabetes': 365,
             'mental': 180}

def get_task_funcs(task):
    return tasks[task][0], tasks[task][1], tasks[task][2], tasks[task][3]

def get_task_results(task):
    return tast_result[task][0], tast_result[task][1], tast_result[task][2], tast_result[task][3]

def init_params(options):
    params = OrderedDict()

    # lstm params
    if options['pretrain'] == '':
        params['Wemb'] = 1.0 * numpy.random.rand(options['n_words'], options['dim_emb']).astype(config.floatX)
    else:
        file = 'best_model/' + options['disease'] + '/best.' + options['pretrain'] + '.pkl'
        f = open(file, 'rb')
        opt, model = cPickle.load(f)
        params['Wemb'] = model['Wemb']

    params = init_lstm_params(options, params)

    # top layer
    init_top_params = tasks[options['task']][4]
    params = init_top_params(options, params)
    return params


def init_shared_params(params):
    shared_params = OrderedDict()
    for name, value in params.iteritems():
        shared_params[name] = theano.shared(value)
    return shared_params

def ortho_weight(dim1, dim2):
    W = numpy.random.randn(dim2, dim2)
    u, s, v = numpy.linalg.svd(W)
    return u[:dim1].astype(config.floatX)

def init_lstm_params(options, params):
    dim_emb = options['dim_emb']
    dim_prj = options['dim_prj']
    W = numpy.concatenate([ortho_weight(dim_emb, dim_prj),
                           ortho_weight(dim_emb, dim_prj),
                           ortho_weight(dim_emb, dim_prj),
                           ortho_weight(dim_emb, dim_prj)], axis=1)

    U = numpy.concatenate([ortho_weight(dim_prj, dim_prj),
                           ortho_weight(dim_prj, dim_prj),
                           ortho_weight(dim_prj, dim_prj),
                           ortho_weight(dim_prj, dim_prj)], axis=1)

    Pf = ortho_weight(dim_emb, dim_prj)
    Po = ortho_weight(dim_emb, dim_prj)
    Z = 0.05 * ortho_weight(options['dim_time'], dim_prj)

    b = numpy.zeros((4*options['dim_prj'],))

    params['lstm_W'] = W
    params['lstm_U'] = U
    params['lstm_Pf'] = Pf
    params['lstm_Po'] = Po
    params['lstm_Z'] = Z
    params['lstm_b'] = b

    return params

def train_lstm(
        disease='', # can be mental or diabetes
        task='', # can be 'readm', 'next_diag', 'high_risk', 'curr_pm',... choose by parameters
        embed='',
        pretrain='',
        dataset_file='',
        log_file='',
        dict_file='',
        model_saver='',
        reg=''
):
    duration = durations[disease]

    # functions for each task
    load_options, prepare_data, prepare_train, build_model = get_task_funcs(task)
    init_best, update_best, evaluation, to_string = get_task_results(task)

    # load model options
    model_options = load_options()
    model_options['task'] = task
    model_options['disease'] = disease
    model_options['pretrain'] = pretrain
    model_options['embed'] = embed
    model_options['reg'] = reg
    model_options['model_saver'] = model_saver
    lrate, batch_size, max_epochs = model_options['lrate'], model_options['batch_size'], model_options['max_epochs']

    diag_dict, proc_dict, medi_dict = admissions.load_dict(dict_file)
    model_options['n_diag'] = len(diag_dict)
    model_options['n_pm'] = len(proc_dict) + len(medi_dict)
    model_options['n_words'] = len(diag_dict) + len(proc_dict) + len(medi_dict)

    ##### BUILD MODEL ###################################
    print 'Building model...'
    params = init_params(model_options)
    shared_params = init_shared_params(params)

    (x, x_mask, x_time, method, y, adm_list, adm_mask, f_pred, cost, use_noise) = build_model(shared_params, model_options)
    f_cost = theano.function(inputs=[x, x_mask, x_time, method, adm_list, adm_mask, y],
                               outputs=cost, name='f_cost')

    grads = tensor.grad(cost, wrt=shared_params.values())
    update = [(p, p - lrate * g) for p, g in zip(shared_params.values(), grads)]
    f_grad = theano.function(inputs=[x, x_mask, x_time, method, adm_list, adm_mask, y],
                                 outputs=grads,
                                 updates=update, name='f_grad')

    #####LOAD_DATA#############################
    print 'Loading data...'
    f = open(dataset_file, 'rb')
    train, valid, vlen, test, tlen, adm = cPickle.load(f)
    v_x, v_mask, v_time, v_method, v_y = prepare_data(vlen, valid, adm, duration, model_options)
    t_x, t_mask, t_time, t_method, t_y = prepare_data(tlen, test, adm, duration, model_options)
    adm_list, adm_mask = admissions.prepare_adm(adm['diag'], adm['pm'])

    #####OPTIMIZATION##########################
    print 'Optimization...'
    n_samples = len(train)
    n_batches = n_samples / batch_size
    if n_batches * batch_size < n_samples: n_batches += 1

    f_log = open(log_file, 'w')
    f_log.write('Training log:\n')
    f_log.close()
    save_options(model_options, log_file)

    min_train_cost = 1000000
    best_valid, best_test = init_best()
    #max, min valid...

    patient, num_patient = 5, 5
    esp = 0.00005
    first_start_time = timer.time()

    for epoch in range(max_epochs):
        use_noise.set_value(1.0)
        if lrate < esp: break
        idx_list = numpy.arange(n_samples, dtype='int32')
        numpy.random.shuffle(idx_list)

        start_time = timer.time()
        costs = []
        for batch in range(n_batches):
            train_x = [train[t] for t in idx_list[batch*batch_size : (batch+1) * batch_size]]
            x, x_mask, x_time, x_med, y = prepare_train(train_x, adm, duration, model_options)
            if len(y) == 0: continue


            cost = f_cost(x, x_mask, x_time, x_med, adm_list, adm_mask, y)
            f_grad(x, x_mask, x_time, x_med, adm_list, adm_mask, y)

            costs.append(cost)

        mean_cost = numpy.mean(costs)

        use_noise.set_value(0.0)
        v_cost = f_cost(v_x, v_mask, v_time, v_method, adm_list, adm_mask, v_y)

        v_pred = f_pred(v_x, v_mask, v_time, v_method, adm_list, adm_mask)
        t_pred = f_pred(t_x, t_mask, t_time, t_method, adm_list, adm_mask)

        v_eval = evaluation(v_pred, v_y, v_mask, adm, model_options)
        t_eval = evaluation(t_pred, t_y, t_mask, adm, model_options)
        best_valid, best_test, is_updated = update_best(best_valid, best_test, v_eval, t_eval)
        if is_updated == 1:
            f = open(model_options['model_saver'], 'wb')
            cPickle.dump((model_options, copy(shared_params)), f, -1)

        # print result
        end_time = timer.time()
        res_str = to_string(v_eval, t_eval)
        print 'epoch: %d, train cost: %.4f, valid cost: %.4f' % (epoch, mean_cost, v_cost)
        print res_str

        f_log = open(log_file, 'a')
        f_log.write('epoch: %d, train cost: %.4f, valid cost: %.4f\n' % (epoch, mean_cost, v_cost))
        f_log.write(res_str + '\n')
        f_log.write('\truntime: %.1f\n\n' % (end_time - start_time))
        f_log.close()


        if mean_cost < min_train_cost:
            patient = num_patient
            min_train_cost = mean_cost
        else:
            patient -= 1
            if patient == 0:
                num_patient = min(15, num_patient + 2)
                patient = num_patient
                lrate /= 2
                if lrate > esp:
                    print 'adjust learning rate %.6f' % lrate

    best_res_str = to_string(best_valid, best_test)
    last_end_time = timer.time()
    print('Total runtime: %.1f\n' % (last_end_time - first_start_time))
    print best_res_str, '\n'
    f_log = open(log_file, 'a')
    f_log.write('Total runtime: %.1f\n' % (last_end_time - first_start_time))
    f_log.write(best_res_str)
    f_log.close()

def save_options(options, log_file):
    f = open(log_file, 'a')
    for name, value in options.iteritems():
        f.write(str(name) + '\t\t' + str(value) + '\n')
    f.write('\n')
    f.close()

def copy(shared_params):
    params = OrderedDict()
    for name, param in shared_params.iteritems():
        params[name] = param.get_value()
    return params

if __name__ == '__main__':
    disease = 'diabetes'
    task = 'readm'
    embed = 'mean'
    pretrain = ''
    reg = '' # drin, drhid, drfeat, norm
    add = ''

    vars = {'-d': disease, '-t': task, '-e': embed, '-p': pretrain, '-r': reg, '-a': add}

    i = 1
    while i < len(sys.argv):
        vars[sys.argv[i]] = sys.argv[i+1]
        i += 2

    disease, task, embed, pretrain, reg, add = vars['-d'], vars['-t'], vars['-e'], vars['-p'], vars['-r'], vars['-a']

    dataset_file = '../deepcare_data/' + disease + '/' + task + '.pkl'
    dict_file = '../deepcare_data/' + disease + '/dict.pkl'

    log_file = 'train_log/' + disease + '/' + embed + '_embed/' + task
    model_saver = 'best_model/' + disease + '/' + 'best.' + task + '.' + embed
    if len(pretrain) > 0:
        log_file += '_pre_' + pretrain
        model_saver += '.pre_' + pretrain
    if len(reg) > 0:
        log_file += '_' + reg
        model_saver += '.' + reg

    if len(add) > 0:
        log_file += add
        model_saver += add

    log_file += '.txt'
    model_saver += '.pkl'

    train_lstm(disease=disease,
               task=task,
               dataset_file=dataset_file,
               log_file=log_file,
               dict_file=dict_file,
               pretrain=pretrain,
               embed=embed,
               reg=reg,
               model_saver=model_saver)