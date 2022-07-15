import csv
from pathlib import Path

import numpy as np
from numpy import random
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from disentangledsg import DisentangledSG
import datamodules


def svm_test(encoder, loader, max_iter=2000, c=0.01, random_state=random.randint(10000)):
    """
    Train a SVM on the latent variables returned by loading data from the dataloader and projecting it to the latent space with the encoder.
    This method uses an one vs rest (ovr) approach, i.e. the margin for every class against all others is retrieved. The larger the margin, the better.

    :param encoder: the encoder that will be used to encode data presented by the dataloader.
    :param datalaoder: loads test data that will projected onto the latent space.
    :param c: the SVM regularization strength, note that it is inverse. 
    :return: a margin for every class against all others AND the classifier object
    """
    clf = make_pipeline(StandardScaler(), LinearSVC(random_state=random_state, tol=1e-5, multi_class='ovr', max_iter=max_iter, C=c))

    # Check if datalaoder batch size is equal to the dataset size to ensure that all data is used for testing
    # Note that the above implemented SVM classifier does not support partial fit which leads to all eval data must be loaded at once
    # This is memory inefficient !
    # https://stackoverflow.com/questions/52670012/convergencewarning-liblinear-failed-to-converge-increase-the-number-of-iterati
    margin = []
    dataset_size = len(loader.dataset)
    if(dataset_size != loader.batch_size):
        loader_attr = dict(filter(lambda item: not item[0].startswith("_"), loader.__dict__.items()))

        loader_attr['batch_sampler'] = None
        loader_attr['batch_size'] = dataset_size
        loader = DataLoader(**loader_attr)

    for i, (data, target) in enumerate(loader):
        latent_vars = encoder(data).detach().numpy()
        target = target.detach().numpy()
        w_train, w_test, target_train, target_test = train_test_split(latent_vars, target, test_size=0.2, random_state=42)

        clf.fit(w_train, target_train)
        linear_svc = clf.named_steps['linearsvc']
        coef = linear_svc.coef_
        margin = 2.0/(np.linalg.norm(coef, axis=1))
        target_pred = clf.predict(w_test)
        test_acc = accuracy_score(target_test, target_pred)
    return (margin, clf, test_acc)


def load_pl_model(load_path):
    """
    Load PyTorch Lightning Model from path.
    :param load_path: Path of model.
    :return: The PyTorch Lightning Model
    """
    sg = DisentangledSG.load_from_checkpoint(load_path)
    return {'generator':sg.generator,'mapping': sg.mapping, 'discriminator':sg.discriminator,'encoder': sg.encoder, 'classifier':sg.classifier}


def get_test_dataloader(datamodule):
    """
    Get PyTorch test dataloader from PyTorch Lightning Datamodule.
    :param datamodule: PyTorch Lightning Datamodule from which the test loader should be returned
    :return: A dataloader that samples the test set
    """
    stage = 'test'
    datamodule.prepare_data()
    datamodule.setup(stage=stage)
    test_loader = datamodule.test_dataloader()
    return test_loader


def get_datamodule(datamodule_name: str = 'mnist', **kargs):
    """
    Return a desired PyTorch Lightning Datamodule based on input string and attributes
    :param datamodule_name: name of the data module that should be returned.
    :param kargs: the attributes that with which the data module will be initialized.
    :return: a PyTorch Lightning Datamodule
    """
    if(datamodule_name == 'mnist'):
        return datamodules.MNISTDataModule(**kargs)
    elif(datamodule_name == 'pcam'):
        return datamodules.PCAMDataModule(**kargs)
    else:
        return None


def calc_metrics(experiment, results, max_iter):
    """
    Calculate important metrics of results and export those: Save raw data, too.
    :param experiment: Path of the experiment
    :param results: Array of margins of the experiment.
    :param max_iter: max number of iterations used to run SVM
    :return: Dict of metrics
    """
    metrics = {"Path": experiment, "Mean": np.mean(results),
               "Median": np.median(results), "Std:": np.std(results),
               "Min": np.min(results), "Max": np.max(results),
               "SVM_Iterations": max_iter, "Raw_Data": results}
    return metrics


def perform_evaluation(experiment_path, dataset, max_iter=4000, c=0.01, save_to='svm_results', **kargs):
    """
    Perform the disentanglement evaluation based on a linear SVM with a set number of SVM iterations for an experiment.
    Prints and saves results and metrics that are calculated on those to CSV file.
    :param experiment_path: Path of the model to be evaluated.
    :param dataset: dataset that was used to train the model -> its test split will be used for evaluation.
    :param max_iter: maximu number of iterations used for the SVM
    :param c: The regularization parameter used for the SVM, note that it strength is inverse.
    :param save_to: the directory where the evaluation results will be saved.
    :param kargs: key word arguments used to initialize the data module.
    """
    encoder = load_pl_model(experiment_path)['encoder'].eval()
    data_module = get_datamodule(dataset, **kargs)
    test_loader = get_test_dataloader(data_module)
    results, _, test_acc = svm_test(encoder, test_loader, max_iter=max_iter, c=c)
    metrics = calc_metrics(experiment_path, results, max_iter)
    metrics['test_acc'] = test_acc
    metrics['c_val'] = c
    experiment_name = Path(experiment_path).stem
    field_names = list(metrics.keys())

    with open(f'{save_to}/Results_{c}_{experiment_name}.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        writer.writerow(metrics)

    print("Following results were saved:")
    print(metrics)

######################################
# Perform Evaluation
# For a pcam performance comparison see: https://ieeexplore.ieee.org/document/9342346
#####################################
folder = "/workspace/ld2/latent_disentanglement/disentangling-stylegan2/to_test_with_svm_007"
pcam = ["pcam-Linear-pl_disentanglement002-epoch-007.ckpt", "pcam-NonLinear-pd_conservation002-epoch-007.ckpt", "pcam-None-pn_reconstruction002-epoch-007.ckpt", "pcam-Resnet-pr_conservation002-epoch-007.ckpt"]
pcam_experiments = [folder + '/' + f for f in pcam]

mnist = ["mnist-Linear-ml_disentanglement002-last.ckpt","mnist-None-mn_reconstruction002-last.ckpt", "mnist-NonLinear-md_conservation002-last.ckpt", "mnist-Resnet-mr_conservation002-last.ckpt"]

mnist_folder = "/workspace/ld2/latent_disentanglement/disentangling-stylegan2/to_test_with_svm_mnist_last"
mnist_experiments = [mnist_folder + '/' + f for f in mnist]

c_values = [0.001, 0.01, 0.1, 1.0, 10]

# Conduct svm mnist experiment
for experiment in mnist_experiments:
    for c_val in c_values:
        perform_evaluation(experiment_path=experiment,
                           dataset='mnist',
                           max_iter=2000,
                           c=c_val,
                           save_to="svm-encoder-results-mnist")

# Conduct svm pcam experiment
for experiment in pcam_experiments:
    for c_val in c_values:
        perform_evaluation(experiment_path=experiment,
                           dataset='pcam',
                           max_iter=2000,
                           c=c_val,
                           save_to="svm-encoder-results-pcam")
