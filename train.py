import sys
print('Python %s on %s' % (sys.version, sys.platform))

import logging
import math
import numpy as np
import pytorch_lightning as pl
import os

import torch
import torch.nn as nn
from torch.nn.functional import l1_loss, mse_loss,smooth_l1_loss
from loss import weighted_mse_loss,weighted_l1_loss,weighted_focal_mse_loss,weighted_focal_l1_loss
import torch.optim as optim
from pytorch_lightning.callbacks import RichProgressBar  # 进度条
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from transformers import get_constant_schedule, AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import TensorDataset
from s4_dalition.dataset import ImDataModule
from callbacks import LossLogger

from model import s4_u_net
from model_rr import t_f
#from model_freq import FED_s4
from model_freq_time import FED_s4

from fds import FDS

save_file = './result/'

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True





def NegPearsonCorrelation_Loss(y_true, y_pred):

    # 计算均值
    mean_true = torch.mean(y_true)
    mean_pred = torch.mean(y_pred)

    # 计算协方差
    covar = torch.mean((y_true - mean_true) * (y_pred - mean_pred))

    # 计算标准差
    std_true = torch.std(y_true)
    std_pred = torch.std(y_pred)

    # 计算负皮尔逊相关系数
    pearson_corr = covar / (std_true * std_pred + 1e-8)

    # 返回1减去负皮尔逊相关系数作为损失
    return 1 - pearson_corr

def ConcordanceCorrelationCoefficient_Loss(y_true, y_pred):
    # 计算均值
    mean_pred = torch.mean(y_pred)
    mean_true = torch.mean(y_true)

    # 计算协方差
    covar = torch.mean((y_pred - mean_pred) * (y_true - mean_true))

    # 计算方差
    var_pred = torch.var(y_pred)
    var_true = torch.var(y_true)

    # 计算 Concordance Correlation Coefficient
    ccc = 2 * covar / (var_pred + var_true + (mean_pred - mean_true) ** 2)

    # 将 CCC 转换为 Loss，目标是最大化 CCC，因此取 1 - CCC
    loss = 1 - ccc

    return loss

class PLModel(pl.LightningModule):
    def __init__(
            self,
            fds = False,
            d_model = 256,
            learning_rate=0.001,
            start_update = 0,
            start_smooth = 1,
            weight_decay=0.0,
            betas=(0.9, 0.999),
            scheduler_name="cosine",
            steps_per_epoch=None,
            optimizer_name='adamw',
            mode='TBPTT',
            scaler=None,
            log_prefix="",
            file_path='',
            **scheduler_kwargs,
    ):
        super().__init__()
        self.fds = fds
        self.start_update = start_update
        self.start_smooth = start_smooth
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.d_model = d_model
        self.optimizer_name = optimizer_name
        self.betas = betas
        self.scheduler_name = scheduler_name
        self.steps_per_epoch = steps_per_epoch
        self.scaler = scaler
        self.scheduler_kwargs = scheduler_kwargs
        self.log_prefix = log_prefix
        self.mode = mode
        # self.model = vgg_model
        self.file_path = file_path
        self._build_model()


    def _build_model(self):
        self.model = FED_s4(d_input=2)


    def training_epoch_end(self, outputs):

        #if self.fds and self.current_epoch >= self.start_update:
        #判断是否存在self.current_features，从第
        if hasattr(self, 'current_features') and self.current_features is not None and self.fds:
            with torch.no_grad():
            # This is called at the end of each epoch
            #if self.current_epoch >= self.start_update:
                # Collect features and labels for all training samples
                # Example:
                training_features, training_labels = self.current_features, self.current_labels,
                self.model.FDS.update_last_epoch_stats(self.current_epoch)
                self.model.FDS.update_running_stats(training_features, training_labels, self.current_epoch)
        else:
            pass

    def forward(self, batch):
        x, y= batch  # 输入数据  目标数据
        current_epoch = self.current_epoch
        if self.fds and self.training:
            y_hat, feature = self.model(x,y,current_epoch)
            return y_hat, y, feature
        else:
            y_hat = self.model(x)
            return y_hat, y

    def _shared_step(self, batch, mode, batch_idx=None, prefix="train"):
        # x, y = batch   # 输入数据  目标数据
        if self.training and self.fds:
            y_hat, y, weight,feature = self.forward(batch=batch)
            y_hat = y_hat.squeeze()
            loss =  NegPearsonCorrelation_Loss(y_true=y,y_pred=y_hat)
            metric = l1_loss(y_hat.view(-1), y)
            return loss, metric, y_hat, y, feature

        else:
            y_hat, y = self.forward(batch=batch)
            y_hat = y_hat.squeeze()
            loss1 = NegPearsonCorrelation_Loss(y_true=y,y_pred=y_hat)
            loss2 = mse_loss(input=y_hat,target=y)
            a = 0.4
            loss = a * loss1 + ((1-a) * loss2)
            metric = ConcordanceCorrelationCoefficient_Loss(y_pred=y_hat, y_true=y)
            return loss, metric, y_hat, y
        #loss = mse_loss(y_hat.view(-1), y)
        #loss = weighted_l1_loss(y_hat.view(-1), y,weights=weight.view(-1))
        #metric = l1_loss(y_hat.view(-1), y)
        #metric = weighted_focal_l1_loss(inputs=y_hat.view(-1), targets=y, weights=weight.view(-1))
        # self.log(f'{mode}_loss',loss, f'{mode}_metrics',metric,on_step=False, on_epoch=True, prog_bar=True, logger=True, add_dataloader_idx=False)
        # self.log(f'{mode}_metrics',metric, on_step=False, on_epoch=True, prog_bar=True, logger=True, add_dataloader_idx=False)

        #return loss, metric, y_hat, y

    def training_step(self, batch, batch_idx):
        # 每个training_epoch需要更新FDS
        if self.fds:
            train_loss, train_metric, y_hat, y, train_feature = self._shared_step(batch=batch, mode='train', batch_idx=batch_idx,
                                                                   prefix="train")
            self.log('train_loss', train_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True,
                     add_dataloader_idx=False)
            self.current_output = y_hat
            self.current_labels = y
            self.current_features = train_feature

            return train_loss
        else:
            train_loss, train_metric, y_hat, y = self._shared_step(batch=batch, mode='train', batch_idx=batch_idx, prefix="train")
            self.log('train_loss', train_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True,
                     add_dataloader_idx=False)
            #self.current_features = None
            return train_loss
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger

        # metrics = {"train_loss": train_loss, "train_metric": train_metric}
        # self.log("train_loss", train_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True,add_dataloader_idx=False,)
        # self.log("train_metrics",metrics, on_step=True, on_epoch=False, prog_bar=False, logger=True, add_dataloader_idx=False,)
        #self.log('train_loss', train_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True,
                 #add_dataloader_idx=False)

        #return train_loss, y_hat, y

    def validation_step(self, batch, batch_idx):
        if self.training:
            validation_loss, validation_metric,_,_,_ = self._shared_step(batch=batch, mode='validation', prefix="validation")
            self.log('validation_loss', validation_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                 add_dataloader_idx=False)
            return validation_loss  # , validation_metric
        else:
            validation_loss, validation_metric, _, _ = self._shared_step(batch=batch, mode='validation',
                                                                         prefix="validation")
            self.log('validation_loss', validation_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                     add_dataloader_idx=False)
            return validation_loss

    def test_step(self, batch, batch_idx):
        if self.training:
            test_loss, test_metric, _ , _, _ = self._shared_step(batch=batch, mode='test', prefix="test")
            self.log('test_loss', test_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                 add_dataloader_idx=False)
            return test_loss  # , test_metric
        else:
            test_loss, test_metric, _, _ = self._shared_step(batch=batch, mode='test', prefix="test")
            self.log('test_loss', test_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                     add_dataloader_idx=False)
            return test_loss  # , test_metric


    # 设置优化器
    def configure_optimizers(self):
        all_params = list(self.parameters())
        if self.optimizer_name == 'adamw':
            optimizer = optim.AdamW(params=self.parameters(), lr=self.lr, betas=self.betas,
                                    weight_decay=self.weight_decay)
        else:
            raise NotImplementedError
        # 创建固定学习率调度器
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
        # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=[get_constant_schedule()],last_epoch=0,verbose=False)
        # scheduler = get_constant_schedule(optimizer)
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch',  # 学习率调度的间隔（每个训练周期）
                                 'frequency': 1,
                                 'monitor': 'val/loss'}  # 调度频率（每个训练周期）
                }
        # return {"optimizer": optimizer, "lr_scheduler": scheduler}


def main():

    # 保存结果，Model、Loss、Result
    for i in range(49):

        # 数据集
        # 定义数据集
        print("#########################第{}次训练####################".format(i))

        """

        trainx = np.load('/home/mininet/cha/orig/bidmc_cross_domain/trainx3_{}.npy'.format(i))[:, :, 0:2]
        trainy = np.load('/home/mininet/cha/orig/bidmc_cross_domain/trainx3_{}.npy'.format(i))[:, :, -1]
        validx = np.load('/home/mininet/cha/orig/bidmc_cross_domain/validx3_{}.npy'.format(i))[:, :, 0:2]
        validy = np.load('/home/mininet/cha/orig/bidmc_cross_domain/validx3_{}.npy'.format(i))[:, :, -1]
        testx = np.load('/home/mininet/cha/orig/bidmc_cross_domain/testx3_{}.npy'.format(i))[:, :, 0:2]
        testy = np.load('/home/mininet/cha/orig/bidmc_cross_domain/testx3_{}.npy'.format(i))[:, :, -1]
        """
        """
        trainx = np.load(
            '/home/mininet/PycharmProjects/pythonProject4/s4_dalition/ablation/dataset/trainx_{}.npy'.format(i))
        trainy = np.load(
            '/home/mininet/PycharmProjects/pythonProject4/s4_dalition/ablation/dataset/trainy_{}.npy'.format(i))
        validx = np.load(
            '/home/mininet/PycharmProjects/pythonProject4/s4_dalition/ablation/dataset/validx_{}.npy'.format(i))
        validy = np.load(
            '/home/mininet/PycharmProjects/pythonProject4/s4_dalition/ablation/dataset/validy_{}.npy'.format(i))
        testx = np.load(
            '/home/mininet/PycharmProjects/pythonProject4/s4_dalition/ablation/dataset/testx_{}.npy'.format(i))
        testy = np.load(
            '/home/mininet/PycharmProjects/pythonProject4/s4_dalition/ablation/dataset/testy_{}.npy'.format(i))
        """

        trainx = np.load('/home/mininet/cha/final/capnobase/trainx_{}.npy'.format(i))
        trainy = np.load('/home/mininet/cha/final/capnobase/trainy_{}.npy'.format(i))
        validx = np.load('/home/mininet/cha/final/capnobase/validx_{}.npy'.format(i))
        validy = np.load('/home/mininet/cha/final/capnobase/validy_{}.npy'.format(i))
        testx = np.load('/home/mininet/cha/final/capnobase/testx_{}.npy'.format(i))
        testy = np.load('/home/mininet/cha/final/capnobase/testy_{}.npy'.format(i))



        train_dataset = TensorDataset(torch.FloatTensor(trainx), torch.FloatTensor(trainy))

        valid_dataset = TensorDataset(torch.FloatTensor(validx), torch.FloatTensor(validy))

        test_dataset = TensorDataset(torch.FloatTensor(testx), torch.FloatTensor(testy))

        data_module1 = ImDataModule(train_dataset, valid_dataset, test_dataset, batch_size=40)

        # data_module = MyDataModule(trainx_path='/home/mininet/cha/data/dataset/trainx.npy', trainy_path='/home/mininet/cha/data/dataset/trainy.npy', valx_path='/home/mininet/cha/data/dataset/validx.npy',
        # valy_path='/home/mininet/cha/data/dataset/validy.npy',testx_path='/home/mininet/cha/data/dataset/testx.npy',testy_path='/home/mininet/cha/data/dataset/testy.npy')

        import argparse
        import time

        loca = time.strftime('%Y-%m-%d-%H-%M-%S)')
        new_name = str(loca)

        parser = argparse.ArgumentParser("Begin Train")
        parser.add_argument("-is_train", default='no', type=str, choices=['yes', 'no'])

        # FDS
        parser.add_argument('--fds', action='store_true', default=False, help='whether to enable FDS')
        parser.add_argument('--fds_kernel', type=str, default='gaussian',
                            choices=['gaussian', 'triang', 'laplace'], help='FDS kernel type')
        parser.add_argument('--fds_ks', type=int, default=5, help='FDS kernel size: should be odd number')
        parser.add_argument('--fds_sigma', type=float, default=3, help='FDS gaussian/laplace kernel sigma')
        parser.add_argument('--start_update', type=int, default=0, help='which epoch to start FDS updating')
        parser.add_argument('--start_smooth', type=int, default=1,
                            help='which epoch to start using FDS to smooth features')
        parser.add_argument('--bucket_num', type=int, default=100, help='maximum bucket considered for FDS')
        parser.add_argument('--bucket_start', type=int, default=0, choices=[0, 3],
                            help='minimum(starting) bucket for FDS, 0 for IMDBWIKI, 3 for AgeDB')
        parser.add_argument('--fds_mmt', type=float, default=0.9, help='FDS momentum')

        # args = parser.parse_args()
        args1, unknown = parser.parse_known_args()
        #  pytorch、numpy、python.random 的随机种子固定
        pl.seed_everything(0, workers=True)

        print("#############是否开始训练#############", args1.is_train)
        if args1.is_train == 'yes':  # 训练

            # 定义数据集为训练校验(train+validation)阶段
            # data_module.setup(stage='fit')

            # callbacks
            checkpoint_callback = ModelCheckpoint(
                monitor="validation_loss",
                mode="min",
                dirpath='./checkpoint',
                filename=str(i) + "_bidmc__best_model",
                save_last=True,
                save_top_k=1,
                auto_insert_metric_name=False,
                verbose=True
            )

            early_stopping_callback = EarlyStopping(monitor="validation_loss", mode="min", patience=5)

            lr_monitor = LearningRateMonitor(logging_interval="step")

            loss_callback = LossLogger(save_dir='./result/loss/',time=new_name)

            trainer = pl.Trainer(accelerator="gpu",
                                 callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor, loss_callback],
                                 max_epochs=30, )
            # Trainer(callbacks=[RichProgressBar()])
            print('=====> Building model...')
            model = PLModel(fds=False)
            trainer.fit(model, train_dataloaders=data_module1.train_dataloader(),val_dataloaders=data_module1.val_dataloader())
            print("BIDMC_Training DONE.")


        else:
            import pickle
            # 定义数据集为测试(test)阶段
            # data_module.setup("test")

            model = PLModel.load_from_checkpoint('./checkpoint/{}_capnobase__best_model-v1.ckpt'.format(i))
            model.eval()


            train_data = data_module1.train_dataloader()  #
            train_predictions = []
            train_reals = []
            with torch.no_grad():  # 在推断时，不需要计算梯度
                for batch in train_data:
                    batch_predictions = model(batch)  # 使用模型进行预测
                    train_predictions.append(batch_predictions[0])
                    train_reals.append(batch_predictions[1])

            # 保存模型预测结果
            with open('result/train_predictions_c{}.pkl'.format(i), 'wb') as f:
                pickle.dump(train_predictions, f)
            with open('result/train_reals_c{}.pkl'.format(i), 'wb') as f:
                pickle.dump(train_reals, f)
            print('=====> Train Dateset Predict Done...')

            valid_data = data_module1.val_dataloader()
            valid_predictions = []
            valid_reals = []
            with torch.no_grad():  # 在推断时，不需要计算梯度
                for batch in valid_data:
                    batch_predictions = model(batch)  # 使用模型进行预测
                    valid_predictions.append(batch_predictions[0])
                    valid_reals.append(batch_predictions[1])

            # 保存模型预测结果
            with open('result/valid_predictions_c{}.pkl'.format(i), 'wb') as f:
                pickle.dump(valid_predictions, f)
            with open('result/valid_reals_c{}.pkl'.format(i), 'wb') as f:
                pickle.dump(valid_reals, f)

            print('=====> Validation Dateset Predict Done...')


            test_data = data_module1.test_dataloader()
            test_predictions = []
            test_reals = []
            with torch.no_grad():  # 在推断时，不需要计算梯度
                for batch in test_data:
                    batch_predictions = model(batch)  # 使用模型进行预测
                    test_predictions.append(batch_predictions[0])
                    test_reals.append(batch_predictions[1])

            # 保存模型预测结果
            with open('result/test_predictions_c{}.pkl'.format(i), 'wb') as f:
                pickle.dump(test_predictions, f)
            with open('result/test_reals_c{}.pkl'.format(i), 'wb') as f:
                pickle.dump(test_reals, f)

            print('=====> Test Dateset Predict Done...')

"""


    for i in range(1):

        i = 2

        # 数据集
        # 定义数据集

        print("#########################第{}次训练####################".format(i))
        trainx = np.load('/home/mininet/cha/orig/capnobase_cross_domain/trainx3_{}.npy'.format(i))[:, :, 0:2]
        trainy = np.load('/home/mininet/cha/orig/capnobase_cross_domain/trainx3_{}.npy'.format(i))[:, :, -1]
        validx = np.load('/home/mininet/cha/orig/capnobase_cross_domain/validx3_{}.npy'.format(i))[:, :, 0:2]
        validy = np.load('/home/mininet/cha/orig/capnobase_cross_domain/validx3_{}.npy'.format(i))[:, :, -1]
        testx = np.load('/home/mininet/cha/orig/capnobase_cross_domain/testx3_{}.npy'.format(i))[:, :, 0:2]
        testy = np.load('/home/mininet/cha/orig/capnobase_cross_domain/testx3_{}.npy'.format(i))[:, :, -1]

        train_dataset = TensorDataset(torch.FloatTensor(trainx), torch.FloatTensor(trainy))

        valid_dataset = TensorDataset(torch.FloatTensor(validx), torch.FloatTensor(validy))

        test_dataset = TensorDataset(torch.FloatTensor(testx), torch.FloatTensor(testy))

        data_module1 = ImDataModule(train_dataset, valid_dataset, test_dataset, batch_size=40)

        # data_module = MyDataModule(trainx_path='/home/mininet/cha/data/dataset/trainx.npy', trainy_path='/home/mininet/cha/data/dataset/trainy.npy', valx_path='/home/mininet/cha/data/dataset/validx.npy',
        # valy_path='/home/mininet/cha/data/dataset/validy.npy',testx_path='/home/mininet/cha/data/dataset/testx.npy',testy_path='/home/mininet/cha/data/dataset/testy.npy')

        import argparse
        import time

        loca = time.strftime('%Y-%m-%d-%H-%M-%S)')
        new_name = str(loca)

        parser = argparse.ArgumentParser("Begin Train")
        parser.add_argument("-is_train", default='yes', type=str, choices=['yes', 'no'])

        # FDS
        parser.add_argument('--fds', action='store_true', default=False, help='whether to enable FDS')
        parser.add_argument('--fds_kernel', type=str, default='gaussian',
                                choices=['gaussian', 'triang', 'laplace'], help='FDS kernel type')
        parser.add_argument('--fds_ks', type=int, default=5, help='FDS kernel size: should be odd number')
        parser.add_argument('--fds_sigma', type=float, default=3, help='FDS gaussian/laplace kernel sigma')
        parser.add_argument('--start_update', type=int, default=0, help='which epoch to start FDS updating')
        parser.add_argument('--start_smooth', type=int, default=1,
                            help='which epoch to start using FDS to smooth features')
        parser.add_argument('--bucket_num', type=int, default=100, help='maximum bucket considered for FDS')
        parser.add_argument('--bucket_start', type=int, default=0, choices=[0, 3],
                                help='minimum(starting) bucket for FDS, 0 for IMDBWIKI, 3 for AgeDB')
        parser.add_argument('--fds_mmt', type=float, default=0.9, help='FDS momentum')

        # args = parser.parse_args()
        args1, unknown = parser.parse_known_args()
        #  pytorch、numpy、python.random 的随机种子固定
        pl.seed_everything(0, workers=True)

        print("#############是否开始训练#############", args1.is_train)
        if args1.is_train == 'yes':  # 训练

            # 定义数据集为训练校验(train+validation)阶段
            # data_module.setup(stage='fit')

            # callbacks
            checkpoint_callback = ModelCheckpoint(
                    monitor="validation_loss",
                    mode="min",
                    dirpath='./checkpoint',
                    filename=str(i) + "___best_model",
                    save_last=True,
                    save_top_k=1,
                    auto_insert_metric_name=False,
                    verbose=True
                )

            early_stopping_callback = EarlyStopping(monitor="validation_loss", mode="min", patience=5)

            lr_monitor = LearningRateMonitor(logging_interval="step")

            loss_callback = LossLogger(save_dir='./result/loss/', time=new_name)

            trainer = pl.Trainer(accelerator="gpu",
                                     callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor,
                                                loss_callback],
                                     max_epochs=20, )
            # Trainer(callbacks=[RichProgressBar()])
            print('=====> Building model...')
            model = PLModel(fds=False)
            trainer.fit(model, train_dataloaders=data_module1.train_dataloader(),val_dataloaders=data_module1.val_dataloader())
            print("Capnobase_Training DONE.")


        else:
            import pickle
            # 定义数据集为测试(test)阶段
            # data_module.setup("test")

            model = PLModel.load_from_checkpoint('./checkpoint/{}___best_model-v1.ckpt'.format(i))
            model.eval()

            train_data = data_module1.train_dataloader()  #
            train_predictions = []
            train_reals = []
            with torch.no_grad():  # 在推断时，不需要计算梯度
                for batch in train_data:
                    batch_predictions = model(batch)  # 使用模型进行预测
                    train_predictions.append(batch_predictions[0])
                    train_reals.append(batch_predictions[1])

            # 保存模型预测结果
            with open('result/train_predictions_C0{}.pkl'.format(i), 'wb') as f:
                pickle.dump(train_predictions, f)
            with open('result/train_reals_C0{}.pkl'.format(i), 'wb') as f:
                pickle.dump(train_reals, f)
            print('=====> Train Dateset Predict Done...')

            valid_data = data_module1.val_dataloader()
            valid_predictions = []
            valid_reals = []
            with torch.no_grad():  # 在推断时，不需要计算梯度
                for batch in valid_data:
                    batch_predictions = model(batch)  # 使用模型进行预测
                    valid_predictions.append(batch_predictions[0])
                    valid_reals.append(batch_predictions[1])

            # 保存模型预测结果
            with open('result/valid_predictions_C0{}.pkl'.format(i), 'wb') as f:
                pickle.dump(valid_predictions, f)
            with open('result/valid_reals_C0{}.pkl'.format(i), 'wb') as f:
                pickle.dump(valid_reals, f)

            print('=====> Validation Dateset Predict Done...')


            test_data = data_module1.test_dataloader()
            test_predictions = []
            test_reals = []
            with torch.no_grad():  # 在推断时，不需要计算梯度
                for batch in test_data:
                    batch_predictions = model(batch)  # 使用模型进行预测
                    test_predictions.append(batch_predictions[0])
                    test_reals.append(batch_predictions[1])

            # 保存模型预测结果
            with open('result/test_predictions_C0{}.pkl'.format(i), 'wb') as f:
                pickle.dump(test_predictions, f)
            with open('result/test_reals_C0{}.pkl'.format(i), 'wb') as f:
                pickle.dump(test_reals, f)

            print('=====> Test Dateset Predict Done...')

"""



if __name__ == '__main__':
    main()


