"""
=================================================================================
This script is for running PHIStruct. It takes a directory of PDB files as input
and outputs the predicted host genus for each protein. It also displays the 
prediction score (class probability) for each host genus recognized by PHIStruct.

@author    Mark Edward M. Gonzales
=================================================================================
"""

import argparse
import os

import joblib
import torch
from transformers import EsmTokenizer

from SaProt.model.esm.base import EsmBaseModel
from SaProt.utils.foldseek_util import get_struc_seq

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = {
    "task": "base",
    "config_path": "SaProt/SaProt_650M_AF2",
    "load_pretrained": True,
}

model = EsmBaseModel(**config).to(device)
tokenizer = EsmTokenizer.from_pretrained(config["config_path"])
model.eval()


# Adapted from https://github.com/westlake-repl/SaProt?tab=readme-ov-file#convert-protein-structure-into-structure-aware-sequence
def encode(pdb_path):
    _, _, combined_seq = get_struc_seq("SaProt/bin/foldseek", pdb_path, ["A"])["A"]
    return combined_seq


# Adapted from https://github.com/westlake-repl/SaProt/issues/14
def embed(seq):
    inputs = tokenizer(seq, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        embedding = model.get_hidden_states(inputs, reduction="mean")

    return embedding[0].tolist()


def predict(embedding, clf):
    proba = clf.predict_proba([embedding])

    scores = []
    for idx, class_name in enumerate(clf.classes_):
        scores.append((class_name, proba[0][idx]))

    return sorted(scores, key=lambda x: x[1], reverse=True)


def write_results(id, scores, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if id.endswith(".pdb"):
        id = id[: -len(".pdb")]

    with open(f"{output_dir}/{id}.csv", "w") as f:
        for entry in scores:
            f.write(f"{entry[0]},{entry[1]}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the directory storing the PDB files describing the structures of the receptor-binding proteins",
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to the trained model (recognized format: joblib or compressed joblib, framework: scikit-learn)",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to the directory to which the results of running PHIStruct will be written",
    )

    args = parser.parse_args()

    clf = joblib.load(args.model)
    for protein in os.listdir(args.input):
        write_results(
            protein,
            predict(embed(encode(str(f"{args.input}/{protein}"))), clf),
            args.output,
        )
