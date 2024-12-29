from os.path import join
from options.errnet.train_options import TrainOptions
from engine import Engine
from data.image_folder import read_fns
import torch.backends.cudnn as cudnn
import data.reflect_dataset as datasets
import util.util as util
import data
import os
from multiprocessing import freeze_support

# freeze_support()

opt = TrainOptions().parse()

cudnn.benchmark = True

opt.display_freq = 10

if opt.debug:
    opt.display_id = 1
    opt.display_freq = 20
    opt.print_freq = 20
    opt.nEpochs = 40
    opt.max_dataset_size = 100
    opt.no_log = False
    opt.nThreads = 0
    opt.decay_iter = 0
    opt.serial_batches = True
    opt.no_flip = True

datadir = '/home/hpc/Project/312510232/Final_Project/DL_TermProject/data/'
datadir_syn = join(datadir, 'training_set', 'VOC224_all')
datadir_real = join(datadir, 'training_set', 'Berkeley real dataset')


train_dataset = datasets.CEILDataset(
    datadir_syn, os.listdir(datadir_syn), size=opt.max_dataset_size, enable_transforms=True, 
    low_sigma=opt.low_sigma, high_sigma=opt.high_sigma,
    low_gamma=opt.low_gamma, high_gamma=opt.high_gamma)

train_dataset_real = datasets.CEILTestDataset(datadir_real, enable_transforms=True)

train_dataset_fusion = datasets.FusionDataset([train_dataset, train_dataset_real], [0.7, 0.3])

train_dataloader_fusion = datasets.DataLoader(
    train_dataset_fusion, batch_size=opt.batchSize, shuffle=not opt.serial_batches, 
    num_workers=opt.nThreads, pin_memory=True)

eval_dataset_ceilnet = datasets.CEILTestDataset(join(datadir, 'testing_set', 'testdata_reflection_synthetic_table2'))
eval_dataset_real = datasets.CEILTestDataset(
    join(datadir, 'testing_set', 'Berkeley real dataset'),
    fns=os.listdir(join(datadir, 'testing_set', 'Berkeley real dataset', 'blended')))

eval_dataloader_ceilnet = datasets.DataLoader(
    eval_dataset_ceilnet, batch_size=1, shuffle=False,
    num_workers=opt.nThreads, pin_memory=True)

eval_dataloader_real = datasets.DataLoader(
    eval_dataset_real, batch_size=1, shuffle=False,
    num_workers=opt.nThreads, pin_memory=True)


"""Main Loop"""
engine = Engine(opt)


def set_learning_rate(lr):
    for optimizer in engine.model.optimizers:
        print('[i] set learning rate to {}'.format(lr))
        util.set_opt_param(optimizer, 'lr', lr)

if opt.resume:
    res = engine.eval(eval_dataloader_ceilnet, dataset_name='testdata_table2')

engine.model.opt.lambda_gan = 0
set_learning_rate(5e-4)

if __name__ == '__main__':
    while engine.epoch < 60:
        if engine.epoch == 10:
            set_learning_rate(1e-4)
        if engine.epoch == 20:
            engine.model.opt.lambda_gan = 0.01 # gan loss is added after epoch 20
        if engine.epoch == 30:
            set_learning_rate(5e-5)
        if engine.epoch == 40:
            set_learning_rate(1e-5)
        if engine.epoch == 45:
            ratio = [0.5, 0.5]
            print('[i] adjust fusion ratio to {}'.format(ratio))
            train_dataset_fusion.fusion_ratios = ratio
            set_learning_rate(5e-5)
        if engine.epoch == 50:
            set_learning_rate(3e-5)
        if engine.epoch == 60:
            set_learning_rate(1e-5)
        if engine.epoch == 70:
            set_learning_rate(5e-6)
        if engine.epoch == 85:
            set_learning_rate(1e-6)

        engine.train(train_dataloader_fusion)
        
        if engine.epoch % 5 == 0:
            engine.eval(eval_dataloader_ceilnet, dataset_name='testdata_table2')        
            # engine.eval(eval_dataloader_real, dataset_name='testdata_real20')
