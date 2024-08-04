"""
========================================================================
This script is for training PHIStruct. It takes a CSV file corresponding
to the training dataset as input and outputs a trained scikit-learn
multilayer perceptron (serialized in joblib format).

@author    Mark Edward M. Gonzales
========================================================================
"""

import argparse

import joblib
import pandas as pd
from imblearn.combine import SMOTETomek

from experiments.MLPDropout import MLPDropout

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the training dataset",
    )

    args = parser.parse_args()

    train = pd.read_csv(
        args.input,
        header=None,
        names=["Protein ID", "Host"] + [f"s{i}" for i in range(1, 1281)],
    )
    X_train = train.loc[:, train.columns.isin([f"s{i}" for i in range(1, 1281)])]
    y_train = train.loc[:, train.columns.isin(["Host"])]

    sm = SMOTETomek(sampling_strategy="all")
    X_train, y_train = sm.fit_resample(X_train, y_train)

    assert X_train.shape[1] == 1280 and y_train.shape[1] == 1

    clf = MLPDropout(
        hidden_layer_sizes=(160, 80),
        dropout=0.20,
        batch_size=128,
    )

    clf.fit(X_train.values, y_train.values.ravel())
    joblib.dump(clf, "phistruct_trained.joblib.gz", compress=True)
