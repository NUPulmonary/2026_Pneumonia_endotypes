import dataclasses
import itertools
import pathlib
import typing

import numpy as np
import pandas as pd

import utils


ROOT = pathlib.Path(__file__).parent.parent
DATA = ROOT / 'data'
CACHE = DATA / '_cache'
RAW_FLOW = DATA / '2023-10-13-flow.csv.gz'
FLOW = DATA / '2023-10-13-nn_flow_with_clusters.csv'
SC_CLUSTERS = DATA / '2023-10-13-nn_bal_lymph_restr_with_clusters_November_2024.csv'
CLINICAL = DATA / '04_external.csv.gz'
CLINICAL_LABELS = DATA / '01_ehr_labels_v4.csv'
SC_LABELS = DATA / '01_sc_labels_v4.csv'
ANON_IDS = DATA / '00_metadata/serniczek_anon_ids.csv'
EHRFORMER_BINS = DATA / '01_ehrformer/01_ehr_binned_v2.pickle'


_sc_root = DATA / '02b_integration'
SC_LIBRARIES = _sc_root / '00_list-of-libraries.csv'
SC_RAW = _sc_root / '09_raw/09_raw.h5ad'
SC_NORM = _sc_root / '09_final_full-1/09_final_full-1.h5ad'
SC_META = _sc_root / '09_final_full-1/09_final_full-1-metadata.csv'

PASC_MAPPING = DATA / '00_metadata/subject_sample_table.xlsx'
HEALTHY_META = DATA / '00_metadata/Supplemental Data File 2. Healthy volunteer demographics.xlsx'


GENEFORMER_ROOT = ROOT / 'Geneformer'
GENEFORMER_DATA = DATA / '08_geneformer'


TASKS = {
    'vap_onset_d7_empirical': 'vap_onset_emp_test_split',
    'is_episode_cured': 'vap_outcome_test_split',
    'viral_vs_bacterial': 'viral_vs_bacterial_test_split',
    'perturbation_groups_2': 'test_split_pg2_split'
}
CONTROLS = {
    'Healthy': 'H',
    'NPC': 'N',
}
PATHOGENS = {
    'Early SARS-CoV-2': 'eCOVID',
    'Late SARS-CoV-2': 'lCOVID',
    'Gram+': 'G+',
    'Gram-*': 'G-',
    'Pseudomonas aeruginosa': 'P',
    'Early SARS-CoV-2; Gram+': 'eCOVID_G+',
    'Late SARS-CoV-2; Gram+': 'lCOVID_G+',
    'Pseudomonas aeruginosa; SARS-CoV-2': 'P_COVID',
    'Gram-*; Gram+': 'G-_G+',
}
NPC_CATEGORIES = {
    'Pt1C59_S1': 'NI',
    'Pt1J13_S1': 'Inflammation',
    'Pt1C99_S1': 'Infection',
    'Pt0F43_S1': 'NI',
    'Pt0G03_S1': 'Inflammation',
    'Pt1B40_S1': 'Inflammation',
    'Pt0K63_S1': 'Inflammation',
    'Pt0H77_S1': 'Inflammation',
    'Pt0H25_S1': 'Infection',
    'Pt1A56_S1': 'Infection',
    'Pt0F25_S1': 'Inflammation',
    'Pt1K71_S1': 'Inflammation',
    'Pt0J87_S1': 'NI',
    'Pt0G76_S1': 'Inflammation',
    'Pt0G76_S2': 'Infection',
    'Pt0G76_S7': 'Inflammation',
    'Pt1F57_S1': 'Infection',
    'Pt1F74_S1': 'NI',
    'Pt1G91_S1': 'Inflammation',
    'Pt1E23_S1': 'Inflammation',
    'Pt1J24_S1': 'NI',
    'Pt0C66_S1': 'Inflammation',
    'Pt1K80_S1': 'Inflammation',
    'Pt1B27_S1': 'NI',
    'Pt1B31_S1': 'Inflammation',
    'Pt1G79_S1': 'Infection'
}


@dataclasses.dataclass
class TaskInfo:
    pathname: str
    column: str
    column_values: list
    split_column: str


def get_tasks(
        context: typing.Literal['geneformer', 'ehrformer', 'degs'] = 'geneformer',
        restrict_to: typing.Optional[str] = None
    ) -> typing.List[str]:
    """
    Returns all classification tasks available in the dataset.

    @param context: str, one of 'geneformer', 'ehrformer', 'degs'. Ehrformer has
                         the most tasks, degs fewer (filtering on available pseudobulks),
                         geneformer even fewer (because of train/test splits).
    """
    if context not in ('geneformer', 'ehrformer', 'degs'):
        raise ValueError(f'Invalid context: {context}')
    tasks = [t for t in TASKS.keys() if t != 'perturbation_groups_2']
    # Not enough pseudobulks for these tasks
    if context in ('geneformer', 'degs'):
        tasks.remove('vap_onset_d7_empirical')
    if restrict_to is not None:
        tasks = [t for t in tasks if t == restrict_to]
    if restrict_to is None or restrict_to == 'perturbation_groups_2':
        tasks.extend(
            [f'{CONTROLS[c]}_vs_{PATHOGENS[p]}'
            for c, p
            in itertools.product(CONTROLS.keys(), PATHOGENS.keys())]
        )
    return tasks


def get_task_columns(labels: pd.DataFrame):
    columns = []
    for task, split_prefix in TASKS.items():
        columns.append(task)
        for col in labels.columns:
            if col.startswith(split_prefix):
                columns.append(col)
    return columns


def get_task_info(task_path: str, split_path: typing.Optional[str] = None) -> TaskInfo:
    assert split_path is None or split_path.startswith('split_')
    if task_path in TASKS:
        column = task_path
        column_values = [0, 1]
        split_column = TASKS[task_path]
    else:
        control_abbr, pathogen_abbr = task_path.split('_vs_')
        column = 'perturbation_groups_2'
        control = [k for k, v in CONTROLS.items() if v == control_abbr][0]
        pathogen = [k for k, v in PATHOGENS.items() if v == pathogen_abbr][0]
        column_values = [control, pathogen]
        split_column = TASKS[column]
    if split_path is not None:
        split_column += split_path.replace('split', '')
    else:
        split_column = None
    return TaskInfo(task_path, column, column_values, split_column)


categorical_features = [
    'Binary_outcome', 'CRRT_flag', 'Discharge_disposition',
    'ECMO_flag', 'Episode_category', 'Episode_etiology', 'Episode_is_cured',
    'Ethnicity', 'Sex_genes', 'Global_cause_failure', 'Hemodialysis_flag',
    'Immunocompromised_flag', 'Norepinephrine_flag', 'Pathogen_bacteria_detected',
    'Pathogen_fungi_detected', 'Pathogen_resistance_detected', 'Pathogen_virus_detected',
    'Race_sc', 'Smoking_status', 'Tracheostomy_flag', 'clusters_final', 'clusters_final_sccl',
    'perturbation_groups_2', 'general_perturbation_group',
    'is_episode_cured',
    'vap_onset_d7_empirical',
]
na_values = {
    'is_episode_cured': '-1',
    'vap_onset_d7_empirical': '-1',
    'perturbation_groups': 'discard',
    'perturbation_groups_2': 'discard',
    'general_perturbation_group': 'discard',
    'Race_sc': 'Unknown or Not Reported',
    'Ethnicity': 'Unknown or Not Reported',
    'Smoking_status': 'Unknown Smoking Status',
}
alternative_covariates = {
    'Pathogen_groups': ['Healthy_vs_NPC_vs_Pathogen'],
    'Discharge_disposition': ['Binary_outcome'],
}
renames = {
    'Sex_genes': 'Sex',
    'Race_sc': 'Race',
    'clusters_final': 'Flow_clusters',
    'clusters_final_sccl': 'ScRNAseq_clusters',
    'perturbation_groups_2': 'Pathogen_groups',
    'general_perturbation_group': 'Healthy_vs_NPC_vs_Pathogen',
    'is_episode_cured': 'VAP_is_cured_d7',
    'vap_onset_d7_empirical': 'VAP_onset_within_7d',
}

for k, v in renames.items():
    if k in na_values:
        na_values[v] = na_values[k]


@utils.cache(
    folder=CACHE,
    sources=[SC_LABELS, CLINICAL, SC_META, FLOW, SC_CLUSTERS]
)
def get_sc_categorical_covariates():
    """
    This function collects all relevant categorical covariates for the single-cell data
    and returns them as a DataFrame. Each column is a categorical dtype over strings
    without any missing values. All missing values are replaced by value in `na_values` or 'NA'.
    Category for the missing value is always last.
    """
    sc_labels = pd.read_csv(SC_LABELS, index_col=0)
    clin = pd.read_csv(CLINICAL, index_col=0)
    sc_labels = sc_labels.merge(clin, on='bal_barcode', how='left', suffixes=('', '_clin'))
    sc_meta = pd.read_csv(SC_META, index_col=0)
    sc_meta = sc_meta.groupby('individual').head(1)
    sc_meta = sc_meta[['individual', 'Sex_genes', 'Race']]
    sc_labels = sc_labels.merge(
        sc_meta,
        left_on='bal_barcode',
        right_on='individual',
        how='left',
        suffixes=('', '_sc')
    )

    flow = pd.read_csv(FLOW, index_col=0)
    sc_labels = sc_labels.merge(flow, on='bal_barcode', how='left', suffixes=('', '_flow'))

    sc_clusters = pd.read_csv(SC_CLUSTERS, index_col=0)
    sc_labels = sc_labels.merge(sc_clusters, on='bal_barcode', how='left', suffixes=('', '_sccl'))

    sc_labels = sc_labels.set_index('bal_barcode')[categorical_features].copy()

    for col in sc_labels.columns:
        na_value = na_values.get(col, 'NA')

        if pd.api.types.is_numeric_dtype(sc_labels[col]):
            try:
                na_value_num = int(na_value)
            except ValueError:
                na_value_num = na_value
        else:
            na_value_num = na_value

        na_count = sc_labels[col].isna().sum() + sc_labels[col].eq(na_value_num).sum()
        # stupid pandas doesn't work here without `tolist` on bool data
        not_na_values = pd.Series(sc_labels[col][sc_labels[col].notna()].unique().tolist())
        if pd.api.types.is_bool_dtype(not_na_values):
            idx = sc_labels[col].isna()
            sc_labels[col] = sc_labels[col].fillna(False).astype(str)
            sc_labels.loc[idx, col] = na_value
        # numbers are always integers here
        elif pd.api.types.is_numeric_dtype(not_na_values):
            idx = sc_labels[col].isna()
            sc_labels[col] = sc_labels[col].fillna(0).astype(int).astype(str)
            sc_labels.loc[idx, col] = na_value
        elif pd.api.types.is_string_dtype(not_na_values):
            sc_labels[col] = sc_labels[col].fillna(na_value)
        categories = sc_labels[col].unique().tolist()
        if na_value in categories:
            categories.remove(na_value)
        if na_count > 0:
            categories.append(na_value)
        sc_labels[col] = pd.Categorical(
            sc_labels[col],
            categories=categories
        )
    sc_labels = sc_labels.rename(columns=renames)
    return sc_labels


numerical_features = [
    'Age', 'BMI', 'SOFA_score',
    'days_on_ventilator', 'GSC_score', 'PaO2FIO2_ratio',
    'Mean_arterial_pressure', 'Bicarbonate', 'ABG_pH', 'ABG_PaCO2', 'Urine_output',
    'Heart_rate', 'Respiratory_rate', 'Creatinine', 'Albumin', 'Bilirubin',
    'Hemoglobin', 'Platelets', 'Temperature',
    'Driving_pressure', 'Minute_Ventilation', 'Neutrophils', 'Lymphocytes',
    'Procalcitonin_ff', 'Lactic_acid', 'D_dimer_ff', 'LDH_ff',
    'Cumulative_vent_changes',
    'BAL_pct_neutrophils', 'BAL_pct_lymphocytes', 'BAL_amylase',
    'Episode_duration',
    'days_of_icu_abx_until_today', 'cumulative_icu_steroid_dose_until_today',
    'sequencing_depth', 'sequencing_saturation',
    'frac_reads_in_cells', 'viability',
]
@utils.cache(
    folder=CACHE,
    sources=[SC_LABELS, CLINICAL, SC_META, FLOW]
)
def get_sc_numerical_covariates():
    sc_labels = pd.read_csv(SC_LABELS, index_col=0)
    clin = pd.read_csv(CLINICAL, index_col=0)
    sc_labels = sc_labels.merge(clin, on='bal_barcode', how='left', suffixes=('', '_clin'))
    sc_meta = pd.read_csv(SC_META, index_col=0)
    sc_meta = sc_meta.groupby('individual').head(1)
    sc_meta = sc_meta[[
        'individual', 'sequencing_depth', 'sequencing_saturation',
        'frac_reads_in_cells', 'viability'
    ]].replace({-1: np.nan})
    sc_labels = sc_labels.merge(
        sc_meta,
        left_on='bal_barcode',
        right_on='individual',
        how='left',
        suffixes=('', '_sc')
    )

    flow = pd.read_csv(FLOW, index_col=0)
    sc_labels = sc_labels.merge(flow, on='bal_barcode', how='left', suffixes=('', '_flow'))
    sc_labels = sc_labels.set_index('bal_barcode')[numerical_features].copy()
    return sc_labels
