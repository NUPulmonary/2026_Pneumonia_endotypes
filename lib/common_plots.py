import re
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import common_data


PATHOGEN_PALETTE = {
    'Healthy': "#5cd061", # green
    'NPC': "#256b33", # dark green
    'Early SARS-CoV-2' : "#e51d23", # red
    'Late SARS-CoV-2': '#f4868a', # light red
    'Early SARS-CoV-2; Gram+' : '#936e10', # dark gold
    'Late SARS-CoV-2; Gram+': "#ffc326", # gold
    'discard': "#cfcfcf", # grey
    'Pseudomonas aeruginosa; SARS-CoV-2': "#9858d6", # purple
    'Pseudomonas; SARS-CoV-2': "#9858d6", # purple
    'Gram+': "#013265", # blue
    'Pseudomonas aeruginosa': "#fb4e93", # pink
    'Pseudomonas': "#fb4e93", # pink
    'Gram-*; Gram+': "#1ceaf9", # cyan
    'Gram-*': "#770c2e" # dark red
}

PATHOGEN_ORDER = [
    'Healthy', 'NPC', 'Early SARS-CoV-2', 'Late SARS-CoV-2', 'Gram+', 'Gram-*', 'Pseudomonas',
    'Early SARS-CoV-2; Gram+', 'Late SARS-CoV-2; Gram+', 'Pseudomonas; SARS-CoV-2', 'Gram-*; Gram+',
    'discard'
]

CELL_TYPES_DISPLAY = {
    'NUPR1+ Macs': 'NUPR1+ AM',
    'Proliferating NUPR1+ Macs': 'Proliferating NUPR1+ AM',
    'MRC1+C1QA+': 'MRC1+C1QA+ AM',
    'MRC1+C1QA-': 'MRC1+C1QA– AM',
    'gdT cells': 'γδT cells',
    'Proliferating gdT cells': 'Proliferating γδT cells'
}

_CBAR_POS = (0.02, 0.88, 0.02, 0.08)


def setup_plotting():
    FONT_DIR = common_data.ROOT / 'fonts'
    for f in FONT_DIR.iterdir():
        if f.name.endswith('.ttf'):
            mpl.font_manager.fontManager.addfont(str(f))
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams['font.family'] = 'Arial'


def plot_pc_corrs(pc_corrs, pc_corr_pvals, title):
    pc_annot = pd.DataFrame(index=pc_corr_pvals.index, columns=pc_corr_pvals.columns)
    pc_annot[-np.log10(pc_corr_pvals) > -np.log10(0.05)] = '*'
    pc_annot[pc_annot != '*'] = ''
    cg = sns.clustermap(
        pc_corrs.T,
        method='ward',
        cmap='vlag',
        annot=pc_annot.T,
        fmt='s',
        cbar_kws=dict(
            label='Pearson $r$'
        ),
        dendrogram_ratio=0.1,
        center=0,
    )
    cg.ax_col_dendrogram.set_title(title, size=16)
    cg.fig.subplots_adjust(top=0.95)
    cg.ax_cbar.set_position(_CBAR_POS)
    return cg


def plot_pc_cat_assoc(pc_corrs, title):
    pc_corrs = pc_corrs.copy()
    min_pval = pc_corrs.to_numpy()[pc_corrs != 0].min()
    pc_corrs[pc_corrs == 0] = min_pval * 0.01
    pc_annot = pd.DataFrame(index=pc_corrs.index, columns=pc_corrs.columns)
    pc_annot[-np.log10(pc_corrs) > -np.log10(0.05)] = '*'
    pc_annot[pc_annot != '*'] = ''
    cg = sns.clustermap(
        -np.log10(pc_corrs).T,
        method='ward',
        cmap='Blues',
        annot=pc_annot.T,
        fmt='s',
        cbar_kws=dict(
            label='$-log10($p-value$)$'
        ),
        dendrogram_ratio=0.1
    )
    cg.ax_col_dendrogram.set_title(title, size=16)
    cg.fig.subplots_adjust(top=0.95)
    cg.ax_cbar.set_position(_CBAR_POS)
    return cg


def plot_pc_gsea(df):
    stars = np.empty_like(df, dtype=str)
    stars[df.abs().gt(-np.log10(0.05))] = '*'
    cg = sns.clustermap(
        df,
        method='ward',
        figsize=(12, 14),
        dendrogram_ratio=0.15,
        cbar_kws=dict(
            label='$-log10(padj) * sign(ES)$'
        ),
        xticklabels=df.columns,
        yticklabels=df.index.str.replace('HALLMARK_', ''),
        cmap='vlag',
        annot=stars,
        fmt=''
    )
    cg.ax_cbar.annotate('Negative', (0, -0.1), xycoords='axes fraction', va='top')
    cg.ax_cbar.annotate('Positive', (0, 1.1), xycoords='axes fraction', va='bottom')
    cg.ax_heatmap.collections[0].set_rasterized(True)
    cg.ax_heatmap.set_xlabel('')
    cg.ax_cbar.set_position(_CBAR_POS)
    return cg


def get_color_annotations(df, mapping, as_hex=True):
    result = []
    for column, palette in mapping.items():
        values = df[column].unique()
        if pd.api.types.is_categorical_dtype(df[column]):
            values = df[column].cat.categories
        if pd.api.types.is_numeric_dtype(df[column]):
            values = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
            if not isinstance(palette, mpl.colors.Colormap):
                palette = sns.color_palette(palette, as_cmap=True)
            colors = palette(values)
            if as_hex:
                colors = [mpl.colors.to_hex(color) for color in colors]
            colors = pd.Series(colors, index=values.index, name=column)
        else:
            if as_hex:
                lut = dict(zip(values, sns.color_palette(palette, n_colors=values.size).as_hex()))
            else:
                lut = dict(zip(values, sns.color_palette(palette, n_colors=values.size)))
            colors = df[column].map(lut)
        result.append(colors)
    return pd.concat(result, axis=1)



def process_hallmark_name(name):
    short_words = ('ACID', 'VIA', 'BETA', 'LATE', 'BILE', 'HEME')
    result = []
    words = name.split('_')
    for word in words:
        if re.search(r'\d', word) or len(word) < 5 and word not in short_words:
            result.append(word)
        else:
            result.append(word.lower())
    result = ' '.join(result)
    result = result.replace("TNFA", "TNFα").replace("NFKB", "NFκB").replace("TGF beta", "TGFβ").replace("WNT beta ", "WNT/β-")
    result = result[0].capitalize() + result[1:]
    return result
