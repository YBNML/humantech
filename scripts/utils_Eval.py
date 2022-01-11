'''
matplot을 이용한 UI. 
성능 향상 방향성 판단하기 위함.
'''

#####################################################
##              Import                             ##
#####################################################

import numpy as np

#####################################################
##              Function                           ##
#####################################################

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    log_10 = (np.abs(np.log10(gt)-np.log10(pred))).mean()
    return a1, a2, a3, abs_rel, rmse, log_10
    
def del_zero2(var1, var2):
    var1 = np.ravel(var1, order='C')
    var2 = np.ravel(var2, order='C')

    a = []
    for i, value in enumerate(var1):
        if var1[i] <= 0 or var2[i] <= 0:
            a.extend([i])

    var1 = np.delete(var1, a)
    var2 = np.delete(var2, a)

    return var1, var2

def evaluate(gt, pred):
    gt, pred = del_zero2(gt, pred)

    e = compute_errors(gt, pred)

    # print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('a1', 'a2', 'a3', 'rel', 'rms', 'log_10'))
    # print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0],e[1],e[2],e[3],e[4],e[5]))
    
    return e