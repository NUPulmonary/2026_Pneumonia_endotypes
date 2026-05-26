import typing
import itertools

import numpy as np
import pandas as pd
import scipy
import statsmodels.stats.multitest


def test_association(
    sample_factors: pd.DataFrame,
    sample_covariates: pd.DataFrame,
    na_values: typing.Optional[dict] = None,
    fdr: typing.Literal['full', 'row', 'col', 'none'] = 'full',
    alternative_covariates: typing.Optional[typing.Dict[str, typing.Collection[str]]] = None
):
    """
    Test the association between sample factors and categorical covariates.

    @param na_values: mapping of column name for sample_covariates to the values
                        that should be treated as NA (excluded from the test)
    @param alternative_covariates: mapping of column name for sample_covariates to the
                                    alternative categories that should be tried for the test
                                    but not compete for FDR correction
    """
    assert sample_factors.index.equals(sample_covariates.index)
    tests = pd.DataFrame()
    for factor in sample_factors.columns:
        for cov in sample_covariates.columns:
            assert not pd.api.types.is_numeric_dtype(sample_covariates[cov])

            categories = sample_covariates[cov].unique().tolist()
            if na_values is not None and cov in na_values:
                cov_na_values = na_values[cov]
                if type(cov_na_values) is str:
                    cov_na_values = [cov_na_values]
                for na_value in cov_na_values:
                    if na_value in categories:
                        categories.remove(na_value)

            if len(categories) == 2:
                x = sample_factors[factor][sample_covariates[cov].eq(categories[0])]
                y = sample_factors[factor][sample_covariates[cov].eq(categories[1])]
                tests.loc[factor, cov] = scipy.stats.mannwhitneyu(x, y).pvalue
            elif len(categories) < 2:
                tests.loc[factor, cov] = 1
            else:
                cat_vals = []
                for cat in categories:
                    cat_vals.append(
                        sample_factors[factor][sample_covariates[cov].eq(cat)].values
                    )
                pval = scipy.stats.f_oneway(*cat_vals).pvalue
                if np.isnan(pval) or pval == -np.infty:
                    pval = 1
                tests.loc[factor, cov] = pval

    if fdr == 'full':
        if alternative_covariates is None:
            padj = statsmodels.stats.multitest.fdrcorrection(tests.values.flatten())[1]
            padj = padj.reshape(tests.shape)
            tests = pd.DataFrame(padj, index=tests.index, columns=tests.columns)
        else:
            all_alt_cats = list(itertools.chain(*alternative_covariates.values()))
            final_tests = tests.loc[:, tests.columns.difference(all_alt_cats)].copy()
            padj = statsmodels.stats.multitest.fdrcorrection(final_tests.values.flatten())[1]
            padj = padj.reshape(final_tests.shape)
            final_tests = pd.DataFrame(padj, index=final_tests.index, columns=final_tests.columns)
            for alt_cov, alt_cats in alternative_covariates.items():
                for cat in alt_cats:
                    alt_test_columns = tests.columns.difference([alt_cov] + all_alt_cats).union([cat])
                    alt_tests = tests.loc[:, alt_test_columns].copy()
                    padj = statsmodels.stats.multitest.fdrcorrection(alt_tests.values.flatten())[1]
                    padj = padj.reshape(alt_tests.shape)
                    alt_tests = pd.DataFrame(padj, index=alt_tests.index, columns=alt_tests.columns)
                    final_tests[cat] = alt_tests[cat].copy()
            tests = final_tests
    if fdr == 'row' or fdr == 'col':
        raise NotImplementedError
    return tests


def test_correlation(
    sample_factors: pd.DataFrame,
    sample_covariates: pd.DataFrame,
    na_values: typing.Optional[dict] = None,
    fdr: typing.Literal['full', 'row', 'col', 'none'] = 'full'
):
    """
    Test the correlation between sample factors and continuous covariates.

    @param na_values: mapping of column name for sample_covariates to the value
                      that should be treated as NA (excluded from the test)
    """
    assert sample_factors.index.equals(sample_covariates.index)
    corrs = pd.DataFrame()
    pvals = pd.DataFrame()
    for factor in sample_factors.columns:
        for cov in sample_covariates.columns:
            assert pd.api.types.is_numeric_dtype(sample_covariates[cov])

            mask = sample_covariates[cov].notna()
            if na_values is not None and cov in na_values:
                mask = mask & sample_covariates[cov].ne(na_values[cov])

            corr = scipy.stats.spearmanr(
                sample_factors[factor][mask], sample_covariates[cov][mask]
            )
            pvals.loc[factor, cov] = corr.pvalue
            corrs.loc[factor, cov] = corr.statistic

    if fdr == 'full':
        padj = statsmodels.stats.multitest.fdrcorrection(pvals.values.flatten())[1]
        padj = padj.reshape(pvals.shape)
        pvals = pd.DataFrame(padj, index=pvals.index, columns=pvals.columns)
    if fdr == 'row' or fdr == 'col':
        raise NotImplementedError
    return corrs, pvals
