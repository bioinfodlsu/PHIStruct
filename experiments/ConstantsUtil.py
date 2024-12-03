"""
=====================================================================
This script contains the constants used in the notebooks and scripts.

@author    Mark Edward M. Gonzales
=====================================================================
"""


class ConstantsUtil(object):
    # ===========
    # Directories
    # ===========
    DATA = "data"
    FASTA = "fasta"
    CONSOLIDATED = "consolidated"

    HYPOTHETICAL = "hypothetical"
    RBP = "rbp"
    NUCLEOTIDE = "nucleotide"
    COMPLETE = "complete"

    GENBANK = "genbank"
    PROKKA = "prokka"
    MASTER = "master"

    # =============
    # Preprocessing
    # =============
    PREPROCESSING = "preprocessing"
    GENUS_TYPO = f"{PREPROCESSING}/genus_typo.txt"

    BACTERIA_NOT_GENUS = f"{PREPROCESSING}/bacteria_not_genus.txt"
    EXCLUDED_HOSTS = f"{PREPROCESSING}/excluded_hosts.txt"
    ARCHAEA_HOSTS = f"{PREPROCESSING}/archaea.txt"

    NCBI_STANDARD_NOMENCLATURE = f"{PREPROCESSING}/ncbi_standard_nomenclature.txt"

    # Regex for candidate genera
    CANDIDATE_REGEX = r"candidat(e|us)"
    # Regex for selecting annotated RBPs
    RBP_REGEX = r"tail?(.?|\s*)(?:spike?|fib(?:er|re))|recept(?:o|e)r(.?|\s*)(?:bind|recogn).*(?:protein)?|(?<!\w)RBP(?!a)"
    # Regex for token delimiters in gene product annotations
    TOKEN_DELIMITER = "[-\|,.\/\s]"

    HYPOTHETICAL_KEYWORDS = f"{PREPROCESSING}/hypothetical_keywords.txt"
    RBP_RELATED_NOT_RBP = f"{PREPROCESSING}/rbp_related_not_rbp.txt"
    PUTATIVE_FUNCTIONS = f"{PREPROCESSING}/putative_functions.txt"

    # Number of entries for displayed progress
    DISPLAY_PROGRESS = 1000
    # Minimum edit distance to be considered a possible misspelling
    MISSPELLING_THRESHOLD = 2
    # Minimum length for a token to be considered a keyword of interest
    MIN_LEN_KEYWORD = 6

    # Bounds for the length of RBPs (excluding those with outlying lengths)
    LOWER_BOUND_RBP_LENGTH = -536
    UPPER_BOUND_RBP_LENGTH = 1587

    # ===============
    # Temporary Files
    # ===============
    TEMP = "temp"
    TEMP_PREPROCESSING = f"{TEMP}/{PREPROCESSING}"
    RESULTS = "results"
    TEMP_RESULTS = f"{TEMP}/{RESULTS}"

    INPHARED_WITH_HOSTS = "inphared.csv"

    NO_CDS_ANNOT = "no_cds_annot.pickle"
    ANNOT_PRODUCTS = "annot_products.pickle"
    RBP_PRODUCTS = "rbp_products.pickle"
    HYPOTHETICAL_PRODUCTS = "hypothetical_proteins.pickle"

    ADDED_PHAGES = "added_phages.pickle"
    REMOVED_PHAGES = "removed_phages.pickle"

    RBP_LENGTHS = "rbp_lengths.pickle"

    FOR_EMBED = {
        HYPOTHETICAL: f"{TEMP}/hypothetical_for_embed",
        RBP: f"{TEMP}/rbp_for_embed",
    }

    # ========
    # INPHARED
    # ========
    INPHARED = f"{DATA}/inphared"
    INPHARED_RBP_DATA = f"rbp.csv"
    INPHARED_EXTRA_COLS = 36

    TEMP_INPHARED = f"{TEMP}/{INPHARED}"
    TOP_GENUS = f"{TEMP_INPHARED}/top_genus.pickle"
    TOP_GENUS_ACCESSION = f"{TEMP_INPHARED}/top_genus_accession.pickle"
    NOT_TOP_GENUS_ACCESSION = f"{TEMP_INPHARED}/not_top_genus_accession.pickle"
    INPHARED_ACCESSION = f"{TEMP_INPHARED}/inphared_accession.pickle"

    # =======================
    # Protein Language Models
    # =======================
    EMBEDDINGS = "embeddings"
    PLM = {
        "PROTTRANSBERT": f"{EMBEDDINGS}/prottransbert",
    }

    EMBEDDINGS_CSV = "rbp_embeddings"
    FEATURE_IMPORTANCE = f"{TEMP}/feature_impt"
    COMPLETE_EMBEDDINGS = f"{INPHARED}/{PLM['PROTTRANSBERT']}/{COMPLETE}/{MASTER}"
    TRAINED_MODEL = "models"

    # ==============
    # RBP Prediction
    # ==============
    XGB_RBP_PREDICTION = "rbp_prediction/RBPdetect_xgb_model.json"

    # =========
    # Structure
    # =========
    STRUCTURE = "structure"

    STRUCTURE_PROSTT5 = f"{STRUCTURE}/rbp_prostt5_embeddings.h5"
    STRUCTURE_PROSTT5_3Di = f"{STRUCTURE}/rbp_prostt5_3di_embeddings.h5"
    CSV_PROSTT5 = f"{STRUCTURE}/rbp_prostt5"
    CSV_PROSTT5_3Di = f"{STRUCTURE}/rbp_prostt5_3di"

    STRUCTURE_SAPROT = f"{STRUCTURE}/rbp_saprot_embeddings"
    STRUCTURE_SAPROT_MASK = f"{STRUCTURE}/rbp_saprot_mask_embeddings"
    STRUCTURE_SAPROT_STRUCT_MASK = f"{STRUCTURE}/rbp_saprot_struct_mask_embeddings"
    STRUCTURE_SAPROT_SEQ_MASK = f"{STRUCTURE}/rbp_saprot_seq_mask_embeddings"
    CSV_SAPROT = f"{STRUCTURE}/rbp_saprot"
    CSV_SAPROT_MASK = f"{STRUCTURE}/rbp_saprot_mask"
    CSV_SAPROT_STRUCT_MASK = f"{STRUCTURE}/rbp_saprot_struct_mask"
    CSV_SAPROT_SEQ_MASK = f"{STRUCTURE}/rbp_saprot_seq_mask"

    STRUCTURE_PST = f"{STRUCTURE}/rbp_pst_embeddings"
    CSV_PST = f"{STRUCTURE}/rbp_pst"

    STRUCTURE_PST_SO = f"{STRUCTURE}/rbp_pst_so_embeddings"
    CSV_PST_SO = f"{STRUCTURE}/rbp_pst_so"

    PROSTT5_AA_PLM = {
        "prostt5_relaxed_r3": f"{EMBEDDINGS}/prostt5_relaxed_r3",
    }

    PROSTT5_3Di_PLM = {
        "prostt5_3di_relaxed_r3": f"{EMBEDDINGS}/prostt5_3di_relaxed_r3",
    }

    SAPROT_PLM = {
        "saprot_relaxed_r3": f"{EMBEDDINGS}/saprot_relaxed_r3",
    }

    SAPROT_MASK_PLM = {
        "saprot_mask_relaxed_r3": f"{EMBEDDINGS}/saprot_mask_relaxed_r3",
    }

    SAPROT_STRUCT_MASK_PLM = {
        "saprot_struct_mask_relaxed_r3": f"{EMBEDDINGS}/saprot_struct_mask_relaxed_r3",
    }

    SAPROT_SEQ_MASK_PLM = {
        "saprot_seq_mask_relaxed_r3": f"{EMBEDDINGS}/saprot_seq_mask_relaxed_r3",
    }

    PST_PLM = {
        "pst_relaxed_r3": f"{EMBEDDINGS}/pst_relaxed_r3",
    }

    # =========================
    # Exploratory Data Analysis
    # =========================
    TEMP_EDA = f"{TEMP}/eda"

    # =============
    # Miscellaneous
    # =============
    UNKNOWN = "others"

    def __init__(self, date=""):
        """
        Constructor

        Parameters:
        - date: Download date of the dataset
        """
        self.DATE = date

        self.INPHARED_GENOME = f"{self.DATA}/GenomesDB"
        self.INPHARED_TSV = f"{self.DATA}/{self.DATE}_data_excluding_refseq.tsv"
        self.INPHARED_GB = f"{self.DATA}/{self.DATE}_phages_downloaded_from_genbank.gb"
