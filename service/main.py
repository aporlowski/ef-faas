# gcloud functions deploy eigenfaces_download_data_http --set-env-vars USER=benchmark --runtime python38 --trigger-http --allow-unauthenticated --memory=1024MB --timeout=540s
# gcloud functions describe eigenfaces_download_data_http
# gcloud functions delete eigenfaces_download_data_http

#gcloud functions deploy eigenfaces_upload_http --set-env-vars USER=benchmark --runtime python38 --trigger-http --allow-unauthenticated --memory=1024MB --timeout=540s

#curl -F example_image.jpg=@example_image.jpg  https://us-central1-anthony-orlowski.cloudfunctions.net/eigenfaces_upload_http

from time import time
import logging
import io
import sys
import os
from werkzeug.utils import secure_filename
import tempfile
from numpy import ndarray
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.datasets._lfw import _load_imgs
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from cloudmesh.common.Benchmark import Benchmark
from sklearn.datasets._base import get_data_home, _fetch_remote, RemoteFileMetadata
from os import listdir, makedirs, remove
from os.path import dirname, join, exists, isdir
from cloudmesh.common.Shell import Shell
from cloudmesh.common.util import path_expand
from joblib import dump, load
from typing import Tuple
from google.cloud import storage

TARGETS = (
    RemoteFileMetadata(
        filename='pairsDevTrain.txt',
        url='https://ndownloader.figshare.com/files/5976012',
        checksum=('1d454dada7dfeca0e7eab6f65dc4e97a'
                  '6312d44cf142207be28d688be92aabfa')),

    RemoteFileMetadata(
        filename='pairsDevTest.txt',
        url='https://ndownloader.figshare.com/files/5976009',
        checksum=('7cb06600ea8b2814ac26e946201cdb30'
                  '4296262aad67d046a16a7ec85d0ff87c')),

    RemoteFileMetadata(
        filename='pairs.txt',
        url='https://ndownloader.figshare.com/files/5976006',
        checksum=('ea42330c62c92989f9d7c03237ed5d59'
                  '1365e89b3e649747777b70e692dc1592')),
)

def eigenfaces_download_data_http(request):
    '''
    '''
    Benchmark.Start()
    images_filename: str = 'lfw-funneled.tgz'
    images_url: str = 'https://ndownloader.figshare.com/files/5976015'
    images_checksum: str = 'b47c8422c8cded889dc5a13418c4bc2abbda121092b3533a83306f90d900100a'
    data_home: str = None
    data_subdir: str = "lfw_home"
    image_subdir: str = "lfw_funneled"
    target_filenames: list = []
    target_urls: list = []
    target_checksums: list = []

    # this function is based on SKLearn's _check_fetch_lfw function.

    archive = RemoteFileMetadata(
        images_filename,
        images_url,
        checksum=(images_checksum))

    if target_filenames != []:
        target_attributes = zip(target_filenames, target_urls, target_checksums)
        targets = ()
        for target in target_attributes:
            filename, url, checksum = target
            targets = targets + (RemoteFileMetadata(filename, url, checksum))

    data_home = get_data_home(data_home=data_home)
    lfw_home = join(data_home, data_subdir)

    if not exists(lfw_home):
        makedirs(lfw_home)

    for target in TARGETS:
        target_filepath = join(lfw_home, target.filename)
        _fetch_remote(target, dirname=lfw_home)

    data_folder_path = join(lfw_home, image_subdir)
    archive_path = join(lfw_home, archive.filename)
    _fetch_remote(archive, dirname=lfw_home)

    import tarfile
    tarfile.open(archive_path, "r:gz").extractall(path=lfw_home)
    remove(archive_path)
    Benchmark.Stop()

    result = f'Data downloaded to {lfw_home}\n'
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    Benchmark.print()
    result += new_stdout.getvalue()
    sys.stdout = old_stdout

    return result, 200, {'Content-Type': 'text/plain'}

def store_model(name: str, model: GridSearchCV, pca: PCA, target_names: ndarray):
    """
    Use joblib to dump the model into a .joblib file

    Stored model can be found in
    Can be found in ~/.cloudmesh/eigenfaces-svm
    """
    model_name = f"{name}_model.joblib"
    pca_name = f"{name}_pca.joblib"
    target_name = f"{name}_target_names.joblib"
    model_path = os.path.join(tempfile.gettempdir(), model_name)
    pca_path = os.path.join(tempfile.gettempdir(), pca_name)
    target_path = os.path.join(tempfile.gettempdir(), target_name)
    files = [(model_name, model_path),(pca_name, pca_path), (target_name, target_path)]

    dump(model, model_path)
    dump(pca, pca_path)
    dump(target_names, target_path)

    bucket_name = "anthony-orlowski-bucket"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    for name, path in files:
        blob = bucket.blob(name)
        blob.upload_from_filename(path)
        print(
            "File {} uploaded as {}.".format(
                path, name))

def eigenfaces_train_http(request):
    """
        run eigenfaces_svm example
        :return type: str
    """
    # print(__doc__)
    Benchmark.Start()
    # Display progress logs on stdout
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    # #############################################################################
    # Download the data, if not already on disk and load it as numpy arrays

    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

    # introspect the images arrays to find the shapes (for plotting)
    n_samples, h, w = lfw_people.images.shape

    # for machine learning we use the 2 data directly (as relative pixel
    # positions info is ignored by this model)
    X = lfw_people.data
    n_features = X.shape[1]

    # the label to predict is the id of the person
    y = lfw_people.target
    target_names = lfw_people.target_names
    n_classes = target_names.shape[0]

    result = "Total dataset size:\n"
    result += "n_samples: %d\n" % n_samples
    result += "n_features: %d\n" % n_features
    result += "n_classes: %d\n" % n_classes

    # #############################################################################
    # Split into a training set and a test set using a stratified k fold

    # split into a training and testing set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    # #############################################################################
    # Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
    # dataset): unsupervised feature extraction / dimensionality reduction
    n_components = 150

    result += "Extracting the top %d eigenfaces from %d faces\n" \
              % (n_components, X_train.shape[0])
    t0 = time()
    pca = PCA(n_components=n_components, svd_solver='randomized',
              whiten=True).fit(X_train)
    result += "done in %0.3fs\n" % (time() - t0)

    eigenfaces = pca.components_.reshape((n_components, h, w))

    result += "Projecting the input data on the eigenfaces orthonormal basis\n"
    t0 = time()
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    result += "done in %0.3fs\n" % (time() - t0)

    # #############################################################################
    # Train a SVM classification model

    result += "Fitting the classifier to the training set\n"
    t0 = time()
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(
        SVC(kernel='rbf', class_weight='balanced'), param_grid
    )
    clf = clf.fit(X_train_pca, y_train)
    result += "done in %0.3fs\n" % (time() - t0)
    result += "Best estimator found by grid search:\n"
    result += "%s\n" % clf.best_estimator_

    # #############################################################################
    # Quantitative evaluation of the model quality on the test set

    result += "Predicting people's names on the test set\n"
    t0 = time()
    y_pred = clf.predict(X_test_pca)
    result += "done in %0.3fs\n" % (time() - t0)

    result += "%s\n" % str(classification_report(y_test, y_pred, target_names=target_names))
    result += "%s\n" % str(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

    store_model("eigenfaces-svm", clf, pca, target_names)

    Benchmark.Stop()
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    Benchmark.print()
    result += new_stdout.getvalue()
    sys.stdout = old_stdout
    print(result)
    return result, 200, {'Content-Type': 'text/plain'}

def get_file_path(filename):
    # Note: tempfile.gettempdir() points to an in-memory file system
    # on GCF. Thus, any files in it must fit in the instance's memory.
    file_name = secure_filename(filename)
    return os.path.join(tempfile.gettempdir(), file_name)

def eigenfaces_upload_http(request):
    Benchmark.Start()
    bucket_name = "anthony-orlowski-bucket"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    fields = {}
    data = request.form.to_dict()
    files = request.files.to_dict()
    for file_name, file in files.items():
        filepath=get_file_path(file_name)
        file.save(filepath)
        print('local save upload file: %s' % file_name)
        blob = bucket.blob(file_name)
        blob.upload_from_filename(filepath)
        print(
            "File {} uploaded to {}.".format(
                filepath, file_name))
    result = f"File {file_name} uploaded.\n"
    Benchmark.Stop()
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    Benchmark.print()
    result += new_stdout.getvalue()
    sys.stdout = old_stdout
    return result, 200, {'Content-Type': 'text/plain'}

#if __name__ =='__main__':
#    eigenfaces_monolith_http('test')