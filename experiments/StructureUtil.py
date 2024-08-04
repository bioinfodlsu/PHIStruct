"""
===========================================================================
This script contains the utility functions for consolidating the embeddings 
generated via structure-aware protein language models

@author    Mark Edward M. Gonzales
===========================================================================
"""

import os

import h5py
import pandas as pd
import torch
from tqdm import tqdm


class StructureUtil(object):
    def __init__(self):
        """
        Constructor
        """
        self.LEN_PROSTT5_EMBEDDINGS = 1024
        self.LEN_SAPROT_EMBEDDINGS = 1280
        self.LEN_PST_EMBEDDINGS = 1280

    def convert_prostt5_h5_to_df(self, prostt5_h5, suffix):
        """
        Consolidates the ProstT5 embeddings to a single DataFrame

        Parameters:
        - prostt5_h5: Path to the HDF5 file containing the ProstT5 embeddings
        - suffix: Suffix appended to the protein ID in each ProstT5 embedding database;
                  this suffix is related to parameter at which ColabFold was run

        Returns:
        - DataFrame consolidating the ProstT5 embeddings
        """
        col_mapping = {}
        for i in range(self.LEN_PROSTT5_EMBEDDINGS):
            col_mapping[i] = f"s{i+1}"

        embeddings_df = pd.DataFrame()
        proteins_df = pd.DataFrame()

        with h5py.File(prostt5_h5) as f:
            proteins = list(f.keys())
            for i in tqdm(range(len(proteins))):
                if not proteins[i].endswith(suffix):
                    continue

                embeddings = f[proteins[i]][()].astype("f8")
                embeddings_df = pd.concat(
                    [embeddings_df, pd.DataFrame.from_records([embeddings])],
                    ignore_index=True,
                )
                proteins_df = pd.concat(
                    [proteins_df, pd.DataFrame.from_records([[proteins[i]]])],
                    ignore_index=True,
                )

        embeddings_df = embeddings_df.rename(columns=col_mapping)
        proteins_df = proteins_df.rename(columns={0: "Protein ID"})

        df = pd.concat([proteins_df, embeddings_df], axis=1)

        return df

    def add_prostt5_id_df(self, df):
        """
        Performs the protein ID sanitization routine in the ProstT5 embedding generation script
        (https://github.com/mheinzinger/ProstT5/blob/main/scripts/embed.py)

        Parameters:
        - df: DataFrame containing the protein IDs to be sanitized

        Returns:
        - DataFrame with the protein IDs sanitized
        """
        df["Protein ID (Clean)"] = (
            df["Protein ID"]
            .str.replace("/", "_", regex=False)
            .str.replace(".", "_", regex=False)
        )
        return df

    def sanitize_prostt5_df(self, df, suffix):
        """
        Adds a column to the DataFrame consolidating the ProstT5 embeddings. This column contains
        the protein IDs with the suffix (which is related to parameter at which ColabFold was run)
        removed

        Parameters:
        - df: DataFrame consolidating the ProstT5 embeddings
        - suffix: Suffix appended to the protein ID in each ProstT5 embedding database;
                  this suffix is related to parameter at which ColabFold was run

        Returns:
        - DataFrame with a column added containing the protein IDs with the suffix removed
        """
        df["Protein ID (Clean)"] = df["Protein ID"].str[: -len(suffix)]
        del df["Protein ID"]
        return df

    def convert_saprot_pt_to_df(self, saprot_pt, suffix):
        """
        Consolidates the SaProt embeddings to a single DataFrame

        Parameters:
        - saprot_pt: Path to the directory containing the SaProt embeddings (saved as .pt files)
        - suffix: Suffix appended to the protein ID in the SaProt embedding filename;
                  this suffix is related to parameter at which ColabFold was run

        Returns:
        - DataFrame consolidating the SaProt embeddings
        """
        col_mapping = {}
        for i in range(self.LEN_SAPROT_EMBEDDINGS):
            col_mapping[i] = f"s{i+1}"

        embeddings_df = pd.DataFrame()
        proteins_df = pd.DataFrame()

        suffix += ".pdb.pt"
        for file in tqdm(os.listdir(saprot_pt)):
            if file.endswith(suffix):
                try:
                    embeddings = torch.load(
                        f"{saprot_pt}/{file}", map_location=torch.device("cpu")
                    )[0].tolist()
                except KeyError:
                    print(file)
                    continue

                embeddings_df = pd.concat(
                    [embeddings_df, pd.DataFrame([embeddings])], ignore_index=True
                )
                proteins_df = pd.concat(
                    [proteins_df, pd.DataFrame([file[: -len(suffix)]])],
                    ignore_index=True,
                )

        embeddings_df = embeddings_df.rename(columns=col_mapping)
        proteins_df = proteins_df.rename(columns={0: "Protein ID"})

        df = pd.concat([proteins_df, embeddings_df], axis=1)

        return df

    def convert_pst_pt_to_df(self, pst_pt, suffix):
        """
        Consolidates the PST embeddings to a single DataFrame

        Parameters:
        - pst_pt: Path to the directory containing the PST embeddings (saved as .pt files)
        - suffix: Suffix appended to the protein ID in the PST embedding filename;
                  this suffix is related to parameter at which ColabFold was run

        Returns:
        - DataFrame consolidating the SaProt embeddings
        """
        col_mapping = {}
        for i in range(self.LEN_PST_EMBEDDINGS):
            col_mapping[i] = f"s{i+1}"

        embeddings_df = pd.DataFrame()
        proteins_df = pd.DataFrame()

        suffix += ".pdb.pt"
        for file in tqdm(os.listdir(pst_pt)):
            if file.endswith(suffix):
                try:
                    embeddings = torch.load(
                        f"{pst_pt}/{file}", map_location=torch.device("cpu")
                    ).tolist()
                except KeyError:
                    print(file)
                    continue

                embeddings_df = pd.concat(
                    [embeddings_df, pd.DataFrame([embeddings])], ignore_index=True
                )
                proteins_df = pd.concat(
                    [proteins_df, pd.DataFrame([file[: -len(suffix)]])],
                    ignore_index=True,
                )

        embeddings_df = embeddings_df.rename(columns=col_mapping)
        proteins_df = proteins_df.rename(columns={0: "Protein ID"})

        df = pd.concat([proteins_df, embeddings_df], axis=1)

        return df
