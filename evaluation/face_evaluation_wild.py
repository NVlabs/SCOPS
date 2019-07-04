"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from numpy.linalg import norm

import sys
import os.path as osp

def mean_error_IOD(fit_kp, gt_kp):

    err = np.zeros(gt_kp.shape[0])

    for i in range(gt_kp.shape[0]):
        fit_keypoints = fit_kp[i,:,:].squeeze()
        gt_keypoints = gt_kp[i, :, :].squeeze()
        face_error = 0
        for k in range(gt_kp.shape[1]):
            face_error += norm(fit_keypoints[k,:]-gt_keypoints[k,:]);
        face_error = face_error/gt_kp.shape[1];

        # pupil dis
        right_pupil = gt_keypoints[0, :];
        left_pupil = gt_keypoints[1, :];

        IOD = norm(right_pupil-left_pupil);

        if IOD != 0:
            err[i] = face_error/IOD
        else:
            print('IOD = 0!')

    return err.mean()



def face_evaluation(train_pred_kp, train_gt_kp, test_pred_kp, test_gt_kp):

    train_gt_kp_flat = train_gt_kp.reshape(train_gt_kp.shape[0], -1)
    train_pred_kp_flat = train_pred_kp.reshape(train_pred_kp.shape[0], -1)

    scaler_pred = StandardScaler()
    scaler_gt = StandardScaler()

    scaler_pred.fit(train_pred_kp_flat)
    scaler_gt.fit(train_gt_kp_flat)

    train_gt_kp_flat_transform = scaler_gt.transform(train_gt_kp_flat)
    train_pred_kp_flat_transform = scaler_pred.transform(train_pred_kp_flat)

    model = LinearRegression(fit_intercept=False)

    model.fit(train_pred_kp_flat_transform, train_gt_kp_flat_transform)

    # train err
    train_fit_kp = scaler_gt.inverse_transform(model.predict(train_pred_kp_flat_transform)).reshape(train_gt_kp.shape)
    mean_error_train = mean_error_IOD(train_fit_kp, train_gt_kp)


    #test
    test_pred_kp_flat = test_pred_kp.reshape(test_pred_kp.shape[0], -1)
    test_pred_kp_flat_transform = scaler_pred.transform(test_pred_kp_flat)

    test_fit_kp = scaler_gt.inverse_transform(model.predict(test_pred_kp_flat_transform)).reshape(test_gt_kp.shape)
    mean_error_test = mean_error_IOD(test_fit_kp, test_gt_kp)

    return mean_error_train, mean_error_test


if __name__ == "__main__":
    train_pred_kp = np.load(osp.join(sys.argv[1], 'train', 'pred_kp.npy'))
    train_gt_kp = np.load(osp.join(sys.argv[1], 'train', 'gt_kp.npy'))

    test_pred_kp = np.load(osp.join(sys.argv[1], 'test', 'pred_kp.npy'))
    test_gt_kp = np.load(osp.join(sys.argv[1], 'test', 'gt_kp.npy'))

    mean_error_train, mean_error_test = face_evaluation(train_pred_kp, train_gt_kp, test_pred_kp, test_gt_kp)

    print('train err {} test err {}'.format(mean_error_train, mean_error_test))
