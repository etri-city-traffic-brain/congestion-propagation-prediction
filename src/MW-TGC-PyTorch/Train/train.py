import time

import numpy as np
from torch.autograd import Variable
from Utils.math_ import *
from sklearn.metrics import f1_score, accuracy_score

import sys

def trainModel(model, train_dataloader, valid_dataloader, lossFunc1, lossFunc2, optimizer,
               scheduler, num_epochs):
    loss_list = []
    valid_loss_list = []
    average_time = []

    start_time = time.time()

    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        print("\nTraining Start for epoch {}".format(epoch + 1))
        # print('LR: ', scheduler.get_lr())
        loss_epoch = []
        MSE_epoch = []
        MAPE_epoch = []
        MAE_epoch = []
        MAD_epoch = []
        MASE_epoch = []

        epoch_start = time.time()
        iter = 0
        for i, (inputs, labels) in enumerate(train_dataloader):
            torch.cuda.empty_cache()

            if torch.cuda.is_available():
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # Clear gradients w.r.t parameters
            optimizer.zero_grad()

            # Forward pass to get output/logits

            outputs = model(inputs)
            outputs = torch.stack(outputs, 1)
            # Calculate Loss:
            loss = lossFunc1(outputs, labels)

            train_error = outputs - labels
            MSE_train = torch.mean(train_error**2, (0, 2)).data.cpu().numpy()
            MAE_train = torch.mean(torch.abs(train_error), (0, 2)).data.cpu().numpy()
            MAPE_train = torch.mean(torch.abs(torch.div(train_error, labels)), (0, 2)).data.cpu().numpy()
            MAD_train = math_mad_loss(outputs, labels)
            MASE_train = math_mase_loss(outputs, labels)

            # batch_size, seq_len, n_links = inputs.size()

            loss_epoch.append(loss.item())

            if torch.cuda.is_available():
                loss.cuda()

            # Getting gradients w.r.t. parameters

            loss.backward()

            # Updating parameters
            optimizer.step()

            MSE_epoch.append(MSE_train)
            MAE_epoch.append(MAE_train)
            MAPE_epoch.append(MAPE_train)
            MAD_epoch.append(MAD_train)
            MASE_epoch.append(MASE_train)

            iter += 1

            del inputs, labels, outputs, MSE_train, MAE_train, MAPE_train, MAD_train, MASE_train

        mean_loss_epoch = np.mean(loss_epoch)
        loss_list.append(mean_loss_epoch)

        MSE_epoch = np.array(MSE_epoch)
        MAE_epoch = np.array(MAE_epoch)
        MAPE_epoch = np.array(MAPE_epoch)
        MAD_epoch = np.array(MAD_epoch)
        MASE_epoch = np.array(MASE_epoch)

        # loss_list.append(np.mean(MSE_epoch))

        MSE_epoch_valid = []
        MAE_epoch_valid = []
        MAPE_epoch_valid = []
        MAD_epoch_valid = []
        MASE_epoch_valid = []

        val_loss_epoch = []
        # del MSE_epoch, MAE_epoch, MAPE_epoch

        # Iterate through valid dataset

        for inputs_val, labels_val in valid_dataloader:
            torch.cuda.empty_cache()
            model.eval()
            if torch.cuda.is_available():
                inputs_val, labels_val = Variable(inputs_val.cuda()), Variable(labels_val.cuda())
            else:
                inputs_val, labels_val = Variable(inputs_val), Variable(labels_val)

            # Forward pass only to get logits/output
            preds = model(inputs_val)
            preds = torch.stack(preds, 1)

            tmp_val_loss = lossFunc1(preds, labels_val)
            val_loss_epoch.append(tmp_val_loss.item())

            valid_error = preds - labels_val
            MSE_valid = torch.mean(valid_error ** 2, (0, 2)).data.cpu().numpy()
            MAE_valid = torch.mean(torch.abs(valid_error), (0, 2)).data.cpu().numpy()
            MAPE_valid = torch.mean(torch.abs(torch.div(valid_error, labels_val)), (0, 2)).data.cpu().numpy()
            MAD_valid = math_mad_loss(preds, labels_val)
            MASE_valid = math_mase_loss(preds, labels_val)

            MSE_epoch_valid.append(MSE_valid)
            MAE_epoch_valid.append(MAE_valid)
            MAPE_epoch_valid.append(MAPE_valid)
            MAD_epoch_valid.append(MAD_valid)
            MASE_epoch_valid.append(MASE_valid)

            del inputs_val, labels_val, preds, MSE_valid, MAE_valid, MAPE_valid, MAD_valid, MASE_valid

        mean_val_loss_epoch = np.mean(val_loss_epoch)

        epoch_mean_MSE_valid = np.mean(np.array(MSE_epoch_valid), 0)
        epoch_mean_MAE_valid = np.mean(np.array(MAE_epoch_valid), 0)
        epoch_mean_MAPE_valid = np.mean(np.array(MAPE_epoch_valid), 0)
        epoch_mean_MAD_valid = np.mean(np.array(MAD_epoch_valid), 0)
        epoch_mean_MASE_valid = np.mean(np.array(MASE_epoch_valid), 0)

        epoch_mean_RMSE_valid = np.sqrt(epoch_mean_MSE_valid)

        valid_loss_list.append(mean_val_loss_epoch)

        epoch_end = time.time() - epoch_start

        average_time.append(epoch_end)
        epoch_mean_RMSE = np.sqrt(np.mean(MSE_epoch, 0))
        epoch_mean_MAPE = np.mean(MAPE_epoch, 0)
        epoch_mean_MAE = np.mean(MAE_epoch, 0)
        epoch_mean_MAD = np.mean(MAD_epoch, 0)
        epoch_mean_MASE = np.mean(MASE_epoch, 0)

        print('epoch: {}/{}. \n'
              'Time for Epoch: {:0.3f} \n'
              '30 min Train: RMSE: {:0.3f} \t MAPE: {:0.3f} \t MAE: {:0.3f} \t MAD: {:0.3f} \t MASE: {:0.3f} \n'
              '30 min Valid: RMSE: {:0.3f} \t MAPE: {:0.3f} \t MAE: {:0.3f} \t MAD: {:0.3f} \t MASE: {:0.3f} \n \n'
              '45 min Train: RMSE: {:0.3f} \t MAPE: {:0.3f} \t MAE: {:0.3f} \t MAD: {:0.3f} \t MASE: {:0.3f} \n'
              '45 min Valid: RMSE: {:0.3f} \t MAPE: {:0.3f} \t MAE: {:0.3f} \t MAD: {:0.3f} \t MASE: {:0.3f} \n \n'
              '60 min Train: RMSE: {:0.3f} \t MAPE: {:0.3f} \t MAE: {:0.3f} \t MAD: {:0.3f} \t MASE: {:0.3f} \n'
              '60 min Valid: RMSE: {:0.3f} \t MAPE: {:0.3f} \t MAE: {:0.3f} \t MAD: {:0.3f} \t MASE: {:0.3f} \n'.format(
            epoch + 1, num_epochs,
            epoch_end,                   #Time
            epoch_mean_RMSE[5],
            epoch_mean_MAPE[5] * 100,
            epoch_mean_MAE[5],
            epoch_mean_MAD[5],
            epoch_mean_MASE[5], # train 30
            epoch_mean_RMSE_valid[5],
            epoch_mean_MAPE_valid[5] * 100,
            epoch_mean_MAE_valid[5],
            epoch_mean_MAD_valid[5],
            epoch_mean_MASE_valid[5],    # Valid 30
            epoch_mean_RMSE[8],
            epoch_mean_MAPE[8] * 100,
            epoch_mean_MAE[8],
            epoch_mean_MAD[8],
            epoch_mean_MASE[8],                      # train 45
            epoch_mean_RMSE_valid[8],
            epoch_mean_MAPE_valid[8] * 100,
            epoch_mean_MAE_valid[8],
            epoch_mean_MAD_valid[8],
            epoch_mean_MASE_valid[8],     # Valid 45
            epoch_mean_RMSE[11],
            epoch_mean_MAPE[11] * 100,
            epoch_mean_MAE[11],
            epoch_mean_MAD[11],
            epoch_mean_MASE[11],         # train 60
            epoch_mean_RMSE_valid[11],
            epoch_mean_MAPE_valid[11] * 100,
            epoch_mean_MAE_valid[11],
            epoch_mean_MAD_valid[11],
            epoch_mean_MASE_valid[11],    # Valid 60
        ))

        min_delta = 0.01
        patience = 5
        es = 0
        if epoch == 0:
            is_best_model = 1
            best_model = model
            patient_epoch = 0
            min_loss_epoch_valid = 10000.0
            if mean_val_loss_epoch < min_loss_epoch_valid:
                min_loss_epoch_valid = mean_val_loss_epoch
        else:

            if min_loss_epoch_valid - mean_val_loss_epoch > min_delta:
                is_best_model = 1
                best_model = model
                min_loss_epoch_valid = mean_val_loss_epoch
                patient_epoch = 0

            else:
                is_best_model = 0
                patient_epoch += 1
                if patient_epoch >= patience:
                    print('Early Stopped at Epoch: ', epoch)
                    break
        print('Best model: ', is_best_model)

        if scheduler is not None:
            scheduler.step()

    end_time = time.time() - start_time

    print(
        '============================= \n Entire training has finished!! \n time elapsed: {}sec \n '
        'Final train loss: {} \n Final valid loss: {} \n ============================='.format(
            np.around(end_time, decimals=3),
            np.around(np.mean(loss_list), decimals=3),
            np.around(np.mean(valid_loss_list), decimals=3)))
    average_time = np.mean(average_time)
    return best_model, loss_list, valid_loss_list, average_time, i



def testModel(model, test_dataloader, lossFunc1, lossFunc2):

    test_MSE = []
    test_MAPE = []
    test_MAE = []
    test_MAD = []
    test_MASE = []

    test_loss = []
    test_link_RMSE_sum = 0
    test_iter = 0
    torch.cuda.empty_cache()

    model.eval()
    i = 0
    for inputs_test, labels_test in test_dataloader:
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            inputs_test, labels_test = Variable(inputs_test.cuda()), Variable(labels_test.cuda())
        else:
            inputs_test, labels_test = Variable(inputs_test), Variable(labels_test)
        timely_diff_label = torch.abs(labels_test[:-1, :, :] - labels_test[1:, :, :])
        preds_test = model(inputs_test)
        preds_test = torch.stack(preds_test, 1)
        if i == 0:
            p_all = preds_test.data.cpu().numpy()
        else:
            p_all = np.concatenate((p_all, preds_test.data.cpu().numpy()), 0)
        test_error = preds_test - labels_test

        MSE_batch = torch.mean(test_error ** 2, (0, 2)).data.cpu().numpy()
        MAE_batch = torch.mean(torch.abs(test_error), (0, 2)).data.cpu().numpy()
        MAPE_batch = torch.mean(torch.abs(torch.div(test_error, labels_test)), (0, 2)).data.cpu().numpy()
        MAD_batch = math_mad_loss(preds_test, labels_test)
        MASE_batch = math_mase_loss(preds_test, labels_test)

        test_loss_batch = lossFunc1(preds_test, labels_test)
        test_loss.append(test_loss_batch.item())

        test_MSE.append(MSE_batch)
        test_MAE.append(MAE_batch)
        test_MAPE.append(MAPE_batch)
        test_MAD.append(MAD_batch)
        test_MASE.append(MASE_batch)

        link_RMSE = torch.mul(test_error, test_error)
        link_RMSE = torch.mean(link_RMSE, 0)
        link_RMSE = link_RMSE.cpu().detach().numpy()
        link_RMSE = np.sqrt(link_RMSE)

        test_link_RMSE_sum = test_link_RMSE_sum + link_RMSE
        test_iter = test_iter + 1
        i += 1

        del inputs_test, labels_test, preds_test, test_error, link_RMSE
        del MSE_batch, MAE_batch, MAPE_batch, MAD_batch, MASE_batch

    test_link_RMSE = test_link_RMSE_sum / test_iter

    MSE_mean = np.mean(np.array(test_MSE), 0)
    MAE_mean = np.mean(np.array(test_MAE), 0)
    MAPE_mean = np.mean(np.array(test_MAPE), 0)
    MAD_mean = np.mean(np.array(test_MAD), 0)
    MASE_mean = np.mean(np.array(test_MASE), 0)

    RMSE_mean = np.sqrt(MSE_mean)

    print('Test outcome \n'
          '30 min Test: RMSE: {:0.3f} \t MAPE: {:0.3f} \t MAE: {:0.3f} \t MAD: {:0.3f} \t MASE: {:0.3f} \n'
          '45 min Test: RMSE: {:0.3f} \t MAPE: {:0.3f} \t MAE: {:0.3f} \t MAD: {:0.3f} \t MASE: {:0.3f} \n'
          '60 min Test: RMSE: {:0.3f} \t MAPE: {:0.3f} \t MAE: {:0.3f} \t MAD: {:0.3f} \t MASE: {:0.3f} \n'.format(
        RMSE_mean[5],
        MAPE_mean[5] * 100,
        MAE_mean[5],
        MAD_mean[5],
        MASE_mean[5],# Test 30
        RMSE_mean[8],
        MAPE_mean[8] * 100,
        MAE_mean[8],
        MAD_mean[8],
        MASE_mean[8],# Test 45
        RMSE_mean[11],
        MAPE_mean[11] * 100,
        MAE_mean[11],
        MAD_mean[11],
        MASE_mean[11] # Test 60
    ))

    return test_loss, test_link_RMSE, p_all


def evaluate_f1score_accuracy(model, data_iter, speed_limit):
    model.eval()
    with torch.no_grad():
        acc, f1 = [], []
        for x, y in data_iter:
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                x, y = Variable(x.cuda()), Variable(y.cuda())
            else:
                x, y = Variable(x), Variable(y)

            y = y.cpu().numpy()

            # y = y.cuda()
            # y_pred = model(x).view(len(x), -1).cpu().numpy()
            # y_pred = model(x).view(len(x), -1).cuda()
            # print(len(y))
            y_pred = model(x)
            y_pred = torch.stack(y_pred, 1)
            y_pred = y_pred.cpu().numpy()

            for i in range(len(y)):
                y_sub = y[i] - speed_limit
                y_pred_sub = y_pred[i] - speed_limit
                result = y_sub * y_pred_sub
                result[result > 0] = 1
                result[result <= 0] = 0
                # acc.append(np.sum(result)/len(speed_limit[0]))

                real_congest = y_sub.copy()
                real_congest[real_congest >= 0] = 0
                real_congest[real_congest < 0] = 1

                pred_congest = y_pred_sub.copy()
                pred_congest[pred_congest >= 0] = 0
                pred_congest[pred_congest < 0] = 1
                # print("real",real_congest.tolist())
                # print("pred",pred_congest.tolist())
                # print("accuracy : ",accuracy_score(real_congest.tolist(), pred_congest.tolist()))
                for idx in range(len(real_congest)):
                    acc.append(accuracy_score(real_congest.tolist()[idx], pred_congest.tolist()[idx]))
                
                # acc.append(accuracy_score(real_congest.tolist(), pred_congest.tolist()))
                # acc.append()

                f1.append(f1_score(real_congest.tolist(), pred_congest.tolist(), average='micro'))

                # free = np.where(real_congest == 0)
                # congest = np.where(real_congest == 1)
                # TP = np.sum(pred_congest[congest])
                # FN = np.sum(real_congest) - TP
                # FP = np.sum(pred_congest[free])
                # pre.append(TP/(TP+FP))
                # rec.append(TP/(TP+FN))

        ACC = np.array(acc).mean()
        # PRE = np.array(pre).mean()
        # REC = np.array(rec).mean()
        # F1_SCORE = 2 * (PRE*REC) / (PRE+REC)
        F1_SCORE = np.array(f1).mean()
        return ACC, F1_SCORE