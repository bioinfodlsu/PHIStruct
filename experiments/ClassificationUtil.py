"""
======================================================================================
This script contains the utility functions for constructing the training and test sets,
building the phage-host interaction prediction model, and evaluating its performance

@author    Mark Edward M. Gonzales
======================================================================================
"""

import copy
import math
import os
import pickle
import random
import statistics

import joblib
import numpy as np
import pandas as pd
from ete3 import NCBITaxa
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from ConstantsUtil import ConstantsUtil
from MLPDropout import MLPDropout
from RBPPredictionUtil import RBPPredictionUtil


class ClassificationUtil(object):
    def __init__(self, complete_embeddings_dir=None, RANDOM_NUM=42):
        """
        Constructor

        Parameters:
        - complete_embeddings_dir: File path of the directory containing the embeddings of the RBPs
        - RANDOM_NUM: Random number for reproducibility
        """
        self.complete_embeddings_dir = complete_embeddings_dir
        self.RANDOM_NUM = RANDOM_NUM
        self.inphared_gb = None

    # =============
    # Preprocessing
    # =============
    def set_inphared_gb(self, inphared_gb):
        """
        Sets the consolidated GenBank entries of the entries fetched via INPHARED

        Parameters:
        - inphared_gb: File path of the consolidated GenBank entries of the entries fetched via INPHARED
        """
        self.inphared_gb = inphared_gb

    def get_phages(self):
        """
        Retrieves the phage IDs of the phages in the dataset

        Returns:
        - Set of phage IDs of the phages in the dataset
        """
        phages = set()
        for file in os.listdir(f"{self.complete_embeddings_dir}"):
            phages.add(file[: -len("-rbp-embeddings.csv")])

        return phages

    def get_rbps(self):
        """
        Retrieves the protein IDs of the RBPs in the dataset

        Returns:
        - Set of protein IDs of the RBPs in the dataset
        """
        rbps = []
        for file in os.listdir(f"{self.complete_embeddings_dir}"):
            phage_name = file[: -len("-rbp-embeddings.csv")]
            with open(f"{self.complete_embeddings_dir}/{file}", "r") as embeddings:
                for embedding in embeddings:
                    rbp_id = embedding.split(",")[0]
                    if rbp_id != "ID":
                        rbps.append([rbp_id, phage_name])

        return rbps

    def get_host_taxonomy(self, rbp_embeddings, host_column="Host"):
        """
        Retrieves the taxonomical information of the hosts (particularly the superkingdom, phylum, class, order, and family)
        since INPHARED returns only the hosts at genus level

        Parameters:
        - rbp_embeddings: DataFrame containing the RBP embeddings
        - host_column: Name of the host genus

        Returns:
        - List containing the taxonomical information of the hosts; each item in the list corresponds to one host
        """
        host_taxonomy = []
        ncbi = NCBITaxa()
        for genus in rbp_embeddings[host_column].unique():
            taxonomy = [None, None, None, None, None, genus]
            try:
                genus_id = ncbi.get_name_translator([genus])[genus][0]
            except KeyError:
                try:
                    print(genus)
                    genus = "candidatus " + genus
                    genus_id = ncbi.get_name_translator([genus])[genus][0]
                except KeyError:
                    print(genus)
                    continue

            lineage_id = ncbi.get_lineage(genus_id)
            lineage = ncbi.get_rank(lineage_id)

            for taxon_id, rank in lineage.items():
                if rank == "superkingdom":
                    taxonomy[0] = ncbi.get_taxid_translator([taxon_id])[
                        taxon_id
                    ].lower()
                elif rank == "phylum":
                    taxonomy[1] = ncbi.get_taxid_translator([taxon_id])[
                        taxon_id
                    ].lower()
                elif rank == "class":
                    taxonomy[2] = ncbi.get_taxid_translator([taxon_id])[
                        taxon_id
                    ].lower()
                elif rank == "order":
                    taxonomy[3] = ncbi.get_taxid_translator([taxon_id])[
                        taxon_id
                    ].lower()
                elif rank == "family":
                    taxonomy[4] = ncbi.get_taxid_translator([taxon_id])[
                        taxon_id
                    ].lower()

            host_taxonomy.append(taxonomy)

        return host_taxonomy

    def get_sequences(self, rbps_with_accession, protein, *fasta_folders):
        """
        Retrieves the RBP protein (or nucleotide) sequences

        Parameters:
        - rbps_with_accession: DataFrame containing the RBP embeddings
        - protein: Protein ID
        - fasta_folders: File paths of the directories containing the FASTA (or FFN) files with the sequences

        Returns:
        - List containing the RBP protein (or nucleotide) sequences; each item in the list corresponds to one RBP
        """
        util = RBPPredictionUtil()

        sequences = []
        for rbp in rbps_with_accession:
            entry = [rbp[0]]
            sequence = util.get_sequence(rbp[0], rbp[1], protein, *fasta_folders)

            if sequence is None:
                print(rbp, sequence)

            entry.append(sequence)
            sequences.append(entry)

        return sequences

    # ===============
    # Embedding Files
    # ===============
    def get_rbp_embeddings(self, folder):
        """
        Retrieves the protein IDs and embeddings of the RBPs in the given directory

        Parameters:
        - folder: File path of the directory containing the embeddings of the RBPs

        Returns:
        - List containing the protein IDs and embeddings of the RBPs in the directory; each item in the list corresponds to one RBP
        """
        rbps = []

        for file in os.listdir(f"{folder}"):
            with open(f"{folder}/{file}", "r") as embeddings:
                for embedding in embeddings:
                    embedding_list = embedding.rstrip("\n").split(",")
                    rbp_id = embedding_list[0]
                    embedding_vals = embedding_list[1:]
                    if rbp_id != "ID":
                        rbps.append([rbp_id] + embedding_vals)

        return rbps

    def get_rbp_embeddings_df(self, plm, folder):
        """
        Constructs a DataFrame containing the protein IDs and embeddings of the RBPs in the given directory

        Parameters:
        - plm: Protein language model used to generate the embeddings
        - folder: File path of the directory containing the embeddings of the RBPs

        Returns:
        - DataFrame containing the protein IDs and embeddings of the RBPs in the directory
        """
        if plm == "PROTTRANSBERT":
            folder += "/master"

        rbp_df = pd.DataFrame(self.get_rbp_embeddings(folder))
        rbp_df.rename(columns={0: "Protein ID"}, inplace=True)

        rbp_df2 = rbp_df.loc[:, rbp_df.columns != "Protein ID"]
        rbp_df2 = rbp_df2.astype(np.float64)

        rbp_df2["Protein ID"] = rbp_df["Protein ID"].values
        col1 = rbp_df2.pop("Protein ID")
        rbp_df2.insert(0, "Protein ID", col1)

        return rbp_df2

    # ===============================
    # Structure & Sequence Similarity
    # ===============================
    def __get_protein_clusters(self, fasta_file):
        """
        Retrieves the protein clusters from the .clstr file produced by CD-HIT

        Parameters:
        - fasta_file: .clstr file produced by CD-HIT

        Returns:
        - Dictionary containing the protein clusters
          (the key is the protein ID of the representative sequence and the value is the list of the protein IDs of the member sequences)
        """
        cluster_reps = []
        cluster_members = []
        with open(fasta_file) as f:
            cluster = []
            for line in f:
                if line.startswith(">"):
                    cluster_members.append(cluster)
                    cluster = []
                else:
                    line = line.strip().split("\t")[1].strip().split(" ")
                    protein = line[1][len(">") : -len("...")]
                    is_rep = line[2]
                    if is_rep == "*":
                        cluster_reps.append(protein)
                    else:
                        cluster.append(protein)

            cluster_members.append(cluster)
            cluster_members.pop(0)

        clusters = {}
        for rep, member in zip(cluster_reps, cluster_members):
            clusters[rep] = member

        return clusters

    def __get_proteins_in_fasta(self, fasta_file):
        """
        Retrieves the representative sequences from the .clstr file produced by CD-HIT

        Parameters:
        - fasta_file: .clstr file produced by CD-HIT

        Returns:
        - List containing the protein IDs of the representative sequences
        """
        return self.__get_protein_clusters(fasta_file).keys()

    def __get_proteins_in_df(self, csv_file, column="Protein ID"):
        """
        Retrieves the protein IDs from a DataFrame

        Parameters:
        - column: Name of the column
        """
        df = pd.read_csv(csv_file, low_memory=False)
        return df[column].tolist()

    def filter_proteins_based_on_struct_and_seq_sim(
        self, embeddings_file, struct_file, seq_sim_file
    ):
        """
        Retrieves the proteins that will comprise the dataset, as well as the clusters, after clustering the protein sequences
        at a set sequence similarity threshold using CD-HIT

        Parameters:
        - embeddings_file: Path to the file containing the protein embeddings
        - struct_file: Path to ANY file containing the structure-aware embeddings
        - seq_sim_file: .clustr file produced by CD-HIT

        Returns:
        - DataFrame containing the representative sequences, alongside the phage-host information and the embeddings
        - DataFrame containing all the proteins, alongside the phage-host information and the embeddings
        - Dictionary containing the protein clusters
          (the key is the protein ID of the representative sequence and the value is the list of the protein IDs of the member sequences)
        """
        proteins_struct = set(self.__get_proteins_in_df(struct_file))
        proteins_fasta = set(self.__get_proteins_in_fasta(seq_sim_file))

        embeddings_df = pd.read_csv(embeddings_file, low_memory=False)
        embeddings_filtered = (
            embeddings_df[embeddings_df["Protein ID"].isin(proteins_fasta)]
            .sort_values(by="Protein ID")
            .reset_index(drop=True)
        )
        embeddings_all = (
            embeddings_df[embeddings_df["Protein ID"].isin(proteins_struct)]
            .sort_values(by="Protein ID")
            .reset_index(drop=True)
        )

        protein_clusters = self.__get_protein_clusters(seq_sim_file)

        return embeddings_filtered, embeddings_all, protein_clusters

    # ==============
    # Classification
    # ==============
    def random_train_test_split(
        self, rbp_embeddings, taxon, embeddings_size=None, feature_columns=None
    ):
        """
        Constructs the training and test sets for model training and evaluation

        Parameters:
        - rbp_embeddings: DataFrame containing the proteins, alongside the phage-host information and the embeddings
        - taxon: Taxonomical level of classification
        - embeddings_size: Dimension of the embedding space
        - feature_columns: List of the column headers corresponding to the features

        Returns:
        - List containing the number of training and test samples for each class label; each item in the list corresponds to one class label
        - List containing the embeddings of the samples in the training set
        - List containing the class labels of the samples in the training set
        - List containing the embeddings of the samples in the test set
        - List containing the class labels of the samples in the test set
        """
        if not feature_columns:
            feature_columns = [str(i) for i in range(1, embeddings_size + 1)]

        X = rbp_embeddings.loc[:, rbp_embeddings.columns.isin(feature_columns)]
        y = rbp_embeddings.loc[:, rbp_embeddings.columns.isin([taxon])]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=0.3, random_state=self.RANDOM_NUM
        )

        all_counts = rbp_embeddings[taxon].value_counts()
        pre_counts = y_train[taxon].value_counts()
        post_counts = y_test[taxon].value_counts()

        counts = []
        for rank in all_counts.index:
            entry = [rank]
            try:
                entry.append(pre_counts[rank])
            except KeyError:
                entry.append(0)

            try:
                entry.append(post_counts[rank])
            except KeyError:
                entry.append(0)

            try:
                entry.append(all_counts[rank])
            except KeyError:
                entry.append(0)

            counts.append(entry)

        return counts, X_train, X_test, y_train, y_test

    def __get_unknown_hosts(
        self, rbp_embeddings, taxon, embeddings_size=None, feature_columns=None
    ):
        """
        Isolates the samples whose host is outside the training class labels and re-labels them under "others"

        Parameters:
        - rbp_embeddings: DataFrame containing the proteins, alongside the phage-host information and the embeddings;
                          this DataFrame should only contain those whose host is outside the training class labels
        - taxon: Taxonomical level of classification
        - embeddings_size: Dimension of the embedding space
        - feature_columns: List of the column headers corresponding to the features

        Returns:
        - List containing the embeddings of the samples whose host is outside the training class labels
        - List containing the class labels of the samples whose host is outside the training class labels;
          the class label of each of these samples is set to "others".
        """
        if not feature_columns:
            feature_columns = [str(i) for i in range(1, embeddings_size + 1)]

        X = rbp_embeddings.loc[:, rbp_embeddings.columns.isin(feature_columns)]
        y = rbp_embeddings.loc[:, rbp_embeddings.columns.isin([taxon])]

        y[taxon] = y[taxon].apply(lambda x: ConstantsUtil().UNKNOWN)

        return X, y

    def __get_other_proteins_in_cluster(
        self,
        rbp_embeddings,
        taxon,
        representative_proteins,
        protein_clusters,
        embeddings_size=None,
        feature_columns=None,
    ):
        """
        Includes entire protein clusters in either the training set or the test set

        Parameters:
        - rbp_embeddings: DataFrame containing the proteins, alongside the phage-host information and the embeddings
        - taxon: Taxonomical level of classification
        - representative_sequences: List of pertinent representative sequences (can either be those in the training set
                                    or those in the test set, depending on the context on which this function is called)
        - protein_clusters: Dictionary containing the protein clusters
        - embeddings_size: Dimension of the embedding space
        - feature_columns: List of the column headers corresponding to the features

        Returns:
        - List containing the embeddings of the proteins belonging to the pertinent clusters
        - List containing the class labels of the proteins belonging to the pertinent clusters
        """
        if not feature_columns:
            feature_columns = [str(i) for i in range(1, embeddings_size + 1)]

        other_proteins = []
        for protein in representative_proteins:
            other_proteins += protein_clusters[protein]

        rbp_embeddings = rbp_embeddings[
            rbp_embeddings["Protein ID"].isin(other_proteins)
        ]

        X = rbp_embeddings.loc[:, rbp_embeddings.columns.isin(feature_columns)]
        y = rbp_embeddings.loc[:, rbp_embeddings.columns.isin([taxon])]

        return X, y

    def predict_with_threshold(
        self, proba, y_test, y_pred, unknown_threshold=0, display=False
    ):
        """
        Adjusts the predicted class labels in view of the given confidence threshold.

        In particular, a sample is classified under its predicted class label if and only if the difference between the largest
        and second largest class probabilities is greater than or equal to the given confidence threshold. Otherwise,
        the sample is classified as "others" (i.e., falling outside the training class labels).

        Parameters:
        - proba: Class probabilities
        - y_test: True class labels
        - y_pred: Predicted class labels
        - unknown_threshold: Confidence threshold
        - display: True if the per-class evaluation results are to be displayed; False, otherwise

        Returns:
        - List containing the per-class precision, recall, and F1
        - List containing the micro-precision, recall, and F1
        - List containing the macro-precision, recall, and F1
        - List containing the weighted precision, recall, and F1
        - Class probabilities
        - True class labels
        - Original predicted class labels
        - Predicted class labels in view of the confidence threshold
        """
        y_pred_copy = copy.deepcopy(y_pred)

        largest = []
        second_largest = []

        for row in proba:
            row_sorted = sorted(row)
            largest.append(row_sorted[-1])
            second_largest.append(row_sorted[-2])

        for idx, (largest_val, second_largest_val) in enumerate(
            zip(largest, second_largest)
        ):
            if largest_val - second_largest_val < unknown_threshold:
                y_pred_copy[idx] = ConstantsUtil().UNKNOWN

        if display:
            print("Confidence threshold k:", str(unknown_threshold * 100) + "%")
            print(classification_report(y_test, y_pred_copy, digits=4, zero_division=0))
            print("===================")

        return (
            precision_recall_fscore_support(
                y_test, y_pred_copy, average=None, zero_division=0
            ),
            precision_recall_fscore_support(
                y_test, y_pred_copy, average="micro", zero_division=0
            ),
            precision_recall_fscore_support(
                y_test, y_pred_copy, average="macro", zero_division=0
            ),
            precision_recall_fscore_support(
                y_test, y_pred_copy, average="weighted", zero_division=0
            ),
            proba,
            y_test,
            y_pred,
            y_pred_copy,
        )

    def __get_only_top_hosts(self, rbp_embeddings, genus=None, top_x_percent=0.25):
        """
        Returns the list of host genera to be considered part of the training set

        Parameters:
        - rbp_embeddings: DataFrame containing the proteins, alongside the phage-host information and the embeddings
        - genus: List of host genera to be considered part of the traning set
                 (that is, every host genus/class label ouside this list will be re-labeled as "others")
                 If not None, it will override the top_x_percent parameter
        - top_x_percent: Percentage of host genera to be considered part of the training set
                         (that is, only the top_x_percent host genera/class labels by class size
                         will be part of the training set; the rest will be re-labeled as "others")

        Returns:
        - List of host genera to be considered part of the training set
        """
        all_counts = rbp_embeddings["Host"].value_counts()
        top_x = math.floor(all_counts.shape[0] * top_x_percent)

        top_genus = set()
        genus_counts = all_counts.index
        for entry in genus_counts[:top_x]:
            top_genus.add(entry)

        if genus:
            top_genus = set(genus)

        return top_genus

    def __get_unknown_hosts_train_test(
        self, rbp_embeddings, top_genus, feature_columns=None
    ):
        """
        Retrieves the list of proteins with hosts outside the training class labels

        Parameters:
        - rbp_embeddings: DataFrame containing the proteins, alongside the phage-host information and the embeddings
        - top_genus: List of host genera considered part of the traning set
        - feature_columns: List of the column headers corresponding to the features

        Returns:
        - List containing the embeddings of the proteins with hosts outside the training class labels
        - List containing the class labels of the proteins with hosts outside the training class labels
        """
        constants = ConstantsUtil()
        if feature_columns is None:
            unknown_hosts_X, unknown_hosts_y = self.__get_unknown_hosts(
                rbp_embeddings[~rbp_embeddings["Host"].isin(top_genus)],
                "Host",
                embeddings_size=rbp_embeddings.shape[1] - constants.INPHARED_EXTRA_COLS,
            )

        else:
            unknown_hosts_X, unknown_hosts_y = self.__get_unknown_hosts(
                rbp_embeddings[~rbp_embeddings["Host"].isin(top_genus)],
                "Host",
                feature_columns=feature_columns,
            )

        return unknown_hosts_X, unknown_hosts_y

    def classify(
        self,
        rbp_embeddings,
        plm,
        similarity,
        feature_columns=None,
        top_x_percent=0.25,
        genus=None,
        include_proteins_in_cluster=False,
        rbp_embeddings_all=None,
        protein_clusters=None,
        undersample_others=False,
        oversample_technique=None,
        model="MLP",
        save_model=True,
        get_training_test_entries_only=False,
        batch_size=128,
        learning_rate=1e-3,
        dropout=0.2,
    ):
        """
        Trains a multilayer perceptron for phage-host interaction prediction and evaluates the model performance

        Parameters:
        - rbp_embeddings: DataFrame containing the representative sequences, alongside the phage-host information and the embeddings
        - plm: Protein language model used to generate the embeddings
        - feature_columns: List of the column headers corresponding to the features
        - top_x_percent: Percentage of host genera to be considered part of the training set
                         (that is, only the top_x_percent host genera/class labels by class size
                         will be part of the training set; the rest will be re-labeled as "others")
        - genus: List of host genera to be considered part of the traning set
                 (that is, every host genus/class label ouside this list will be re-labeled as "others")
                 If not None, it will override the top_x_percent parameter
        - include_proteins_in_cluster: True if entire clusters will be included in the training and test sets;
                                       False if only the representative sequences will be included
        - rbp_embeddings_all: DataFrame containing all the proteins, alongside the phage-host information and the embeddings
        - protein_clusters: Dictionary containing the protein clusters
                            (the key is the protein ID of the representative sequence
                            and the value is the list of the protein IDs of the member sequences)
        - undersample_others: True if the number of "others" samples in the test will be undersampled; False, otherwise
        - oversample_technique: True if data augmentation will be applied to the training set; False, otherwise
        - model: Model to use for classification

        Returns:
        - If display_feature_importance is set to True, the features with the highest Gini importance are returned
        - Otherwise, the function returns None
        """
        constants = ConstantsUtil()
        top_genus = self.__get_only_top_hosts(rbp_embeddings, genus, top_x_percent)

        # Construct the training and test sets
        print("Constructing training and test sets...")

        rbp_embeddings_top = rbp_embeddings[rbp_embeddings["Host"].isin(top_genus)]

        counts = X_train = X_test = y_train = y_test = None
        if feature_columns is None:
            counts, X_train, X_test, y_train, y_test = self.random_train_test_split(
                rbp_embeddings_top,
                "Host",
                embeddings_size=rbp_embeddings.shape[1] - constants.INPHARED_EXTRA_COLS,
            )
        else:
            counts, X_train, X_test, y_train, y_test = self.random_train_test_split(
                rbp_embeddings_top, "Host", feature_columns=feature_columns
            )

        # Handle proteins with hosts outside the training class labels
        unknown_hosts_X, unknown_hosts_y = self.__get_unknown_hosts_train_test(
            rbp_embeddings, top_genus, feature_columns
        )
        X_test = pd.concat([X_test, unknown_hosts_X])
        y_test = pd.concat([y_test, unknown_hosts_y])

        assert X_train.shape[0] + X_test.shape[0] == rbp_embeddings.shape[0]

        # Add other proteins in the cluster
        if include_proteins_in_cluster:
            if feature_columns is None:
                representative_proteins_train = rbp_embeddings.loc[X_train.index][
                    "Protein ID"
                ].values
                cluster_proteins_X_train, cluster_proteins_y_train = (
                    self.__get_other_proteins_in_cluster(
                        rbp_embeddings_all,
                        "Host",
                        representative_proteins_train,
                        protein_clusters,
                        embeddings_size=rbp_embeddings.shape[1]
                        - constants.INPHARED_EXTRA_COLS,
                    )
                )

                representative_proteins_test = rbp_embeddings.loc[X_test.index][
                    "Protein ID"
                ].values
                cluster_proteins_X_test, cluster_proteins_y_test = (
                    self.__get_other_proteins_in_cluster(
                        rbp_embeddings_all,
                        "Host",
                        representative_proteins_test,
                        protein_clusters,
                        embeddings_size=rbp_embeddings.shape[1]
                        - constants.INPHARED_EXTRA_COLS,
                    )
                )

            else:
                representative_proteins_train = rbp_embeddings.loc[X_train.index][
                    "Protein ID"
                ].values
                cluster_proteins_X_train, cluster_proteins_y_train = (
                    self.__get_other_proteins_in_cluster(
                        rbp_embeddings_all,
                        "Host",
                        representative_proteins_train,
                        protein_clusters,
                        feature_columns=feature_columns,
                    )
                )

                representative_proteins_test = rbp_embeddings.loc[X_test.index][
                    "Protein ID"
                ].values
                cluster_proteins_X_test, cluster_proteins_y_test = (
                    self.__get_other_proteins_in_cluster(
                        rbp_embeddings_all,
                        "Host",
                        representative_proteins_test,
                        protein_clusters,
                        feature_columns=feature_columns,
                    )
                )

            assert (
                len(representative_proteins_train) + len(representative_proteins_test)
                == rbp_embeddings.shape[0]
            )

            X_train = pd.concat([X_train, cluster_proteins_X_train])
            y_train = pd.concat([y_train, cluster_proteins_y_train])

            X_test = pd.concat([X_test, cluster_proteins_X_test])
            y_test = pd.concat([y_test, cluster_proteins_y_test])

            assert X_train.shape[0] + X_test.shape[0] == sum(
                [len(x) for x in protein_clusters.values()]
            ) + len(protein_clusters)

        X_train = X_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        # Drop entries in train set with "others" as host (because of clustering)
        y_train_host_known_indices = y_train[y_train["Host"].isin(top_genus)].index
        X_train = X_train.loc[y_train_host_known_indices]
        y_train = y_train.loc[y_train_host_known_indices]

        y_test.loc[~y_test["Host"].isin(top_genus), "Host"] = constants.UNKNOWN

        assert len(y_train[~y_train["Host"].isin(top_genus)].index) == 0

        # Undersample "others" class
        if undersample_others:
            top_genus_sizes = []
            for g in top_genus:
                top_genus_sizes.append(y_test[y_test["Host"] == g].shape[0])

            others_y_test = y_test.loc[y_test["Host"] == constants.UNKNOWN]
            num_samples = min(others_y_test.shape[0], int(min(top_genus_sizes)))

            random.seed(self.RANDOM_NUM)
            samples_to_remove_indices = random.sample(
                list(others_y_test.index), others_y_test.shape[0] - num_samples
            )

            X_test = X_test.loc[~X_test.index.isin(samples_to_remove_indices)]
            y_test = y_test.loc[~y_test.index.isin(samples_to_remove_indices)]

        # In this context, entries refer to the accession IDs of the sequences in the training
        # and test sets -- prior to the application of oversampling techniques
        if get_training_test_entries_only:
            return X_train, y_train, X_test, y_test

        if oversample_technique:
            if oversample_technique == "SMOTETomek":
                sm = SMOTETomek(random_state=self.RANDOM_NUM, sampling_strategy="all")
                X_train, y_train = sm.fit_resample(X_train, y_train)
            else:
                raise Exception(
                    f"Oversampling technique {oversample_technique} not recognized!"
                )

        # Sanity check
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        assert (
            X_train.shape[0] == y_train.shape[0] and X_test.shape[0] == y_test.shape[0]
        )

        # Construct the model
        print("Training the model...")

        if model == "MLP":
            clf = MLPDropout(
                random_state=self.RANDOM_NUM,
                hidden_layer_sizes=(160, 80),
                dropout=dropout,
                batch_size=batch_size,
                learning_rate_init=learning_rate,
            )

        elif model == "MLP-prot":
            clf = MLPDropout(
                random_state=self.RANDOM_NUM,
                hidden_layer_sizes=(128, 64),
                dropout=dropout,
                batch_size=batch_size,
                learning_rate_init=learning_rate,
            )

        elif model == "random-forest":
            clf = RandomForestClassifier(
                random_state=self.RANDOM_NUM,
                class_weight="balanced",
                max_features="sqrt",
                min_samples_leaf=1,
                min_samples_split=2,
                n_estimators=150,
                n_jobs=-1,
            )

        elif model == "svm":
            clf = SVC(
                random_state=self.RANDOM_NUM, kernel="rbf", C=2.0, probability=True
            )

        else:
            raise Exception(f"Model {model} not recognized!")

        clf.fit(X_train, y_train.values.ravel())
        y_pred = clf.predict(X_test)
        proba = clf.predict_proba(X_test)

        if save_model:
            if not os.path.exists(constants.TRAINED_MODEL):
                os.makedirs(constants.TRAINED_MODEL)

            joblib.dump(clf, f"{constants.TRAINED_MODEL}/{plm}-{similarity}.pickle.gz")

        # Save the results
        print("Saving evaluation results...")

        results = []
        for threshold in range(0, 101, 10):
            results.append(
                self.predict_with_threshold(
                    proba,
                    y_test,
                    y_pred,
                    unknown_threshold=threshold / 100,
                    display=(threshold != 100),
                )
            )

        if not os.path.exists(constants.TEMP_RESULTS):
            os.makedirs(constants.TEMP_RESULTS)

        with open(f"{constants.TEMP_RESULTS}/{plm}-{similarity}.pickle", "wb") as f:
            pickle.dump(results, f)

        # Display progress
        print("Finished")
        print("===================")
