import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind, fisher_exact
from statsmodels.stats.multitest import multipletests
import os

class HypothesisTesting:
    def __init__(self, data):
        self.data = data.copy()
        os.makedirs('plots', exist_ok=True)
        self.data = self.data[(self.data['totalpremium'] > 0) & (self.data['totalclaims'] >= 0)]
        self.data['has_claim'] = (self.data['totalclaims'] > 0).astype(int)
        self.data['margin'] = self.data['totalpremium'] - self.data['totalclaims']
        print(f"Data shape after filtering: {self.data.shape}")

    def calculate_metrics(self, group_col):
        metrics = self.data.groupby(group_col).agg({
            'has_claim': 'mean',
            'totalclaims': lambda x: x[x > 0].mean() if x[x > 0].count() > 0 else np.nan,
            'margin': 'mean'
        }).rename(columns={
            'has_claim': 'claim_frequency',
            'totalclaims': 'claim_severity',
            'margin': 'margin'
        })
        return metrics

    def chi_squared_test(self, group_col, group_a, group_b):
        df_subset = self.data[self.data[group_col].isin([group_a, group_b])]
        contingency_table = pd.crosstab(df_subset[group_col], df_subset['has_claim'])
        if contingency_table.shape[1] < 2 or contingency_table.min().min() < 5:
            print(f"Warning: Insufficient data for chi-squared test between {group_a} and {group_b}")
            return None
        chi2, p, _, _ = chi2_contingency(contingency_table)
        return {'test': 'Chi-Squared', 'group_a': group_a, 'group_b': group_b, 'metric': 'claim_frequency', 'p_value': p}

    def t_test(self, group_col, group_a, group_b, metric):
        df_subset = self.data[self.data[group_col].isin([group_a, group_b])]
        group_a_data = df_subset[df_subset[group_col] == group_a]
        group_b_data = df_subset[df_subset[group_col] == group_b]
        if metric == 'claim_severity':
            group_a_values = group_a_data[group_a_data['totalclaims'] > 0]['totalclaims']
            group_b_values = group_b_data[group_b_data['totalclaims'] > 0]['totalclaims']
        else:  # margin
            group_a_values = group_a_data['margin']
            group_b_values = group_b_data['margin']
        if len(group_a_values) < 2 or len(group_b_values) < 2:
            print(f"Warning: Insufficient data for t-test between {group_a} and {group_b}")
            return None
        t_stat, p = ttest_ind(group_a_values, group_b_values, equal_var=False, nan_policy='omit')
        return {'test': 'T-Test', 'group_a': group_a, 'group_b': group_b, 'metric': metric, 'p_value': p}

    def check_group_equivalence(self, group_col, group_a, group_b, check_cols):
        results = []
        df_subset = self.data[self.data[group_col].isin([group_a, group_b])]
        for col in check_cols:
            if df_subset[col].dtype in ['object', 'category']:
                contingency_table = pd.crosstab(df_subset[group_col], df_subset[col])
                if contingency_table.min().min() >= 5:
                    _, p, _, _ = chi2_contingency(contingency_table)
                    results.append({'column': col, 'test': 'Chi-Squared', 'p_value': p})
            else:
                group_a_values = df_subset[df_subset[group_col] == group_a][col].dropna()
                group_b_values = df_subset[df_subset[group_col] == group_b][col].dropna()
                if len(group_a_values) >= 2 and len(group_b_values) >= 2:
                    _, p = ttest_ind(group_a_values, group_b_values, equal_var=False, nan_policy='omit')
                    results.append({'column': col, 'test': 'T-Test', 'p_value': p})
        return pd.DataFrame(results)

    def run_hypothesis_tests(self):
        results = []
        # H₀: No risk differences across provinces (Gauteng vs. KwaZulu-Natal)
        results.append(self.chi_squared_test('province', 'Gauteng', 'KwaZulu-Natal'))
        results.append(self.t_test('province', 'Gauteng', 'KwaZulu-Natal', 'claim_severity'))
        # H₀: No risk/margin differences between zip codes (by MainCrestaZone)
        if 'maincrestazone' in self.data.columns:
            zone_metrics = self.calculate_metrics('maincrestazone')
            claim_counts = self.data[self.data['has_claim'] == 1]['maincrestazone'].value_counts()
            valid_zones = claim_counts[claim_counts > 10].index
            zone_metrics = zone_metrics.loc[zone_metrics.index.isin(valid_zones)]
            if len(zone_metrics) < 2:
                print("Error: Insufficient valid MainCrestaZones with >10 claims")
                return pd.DataFrame(results)
            n_zones = len(zone_metrics)
            n_select = max(2, int(0.2 * n_zones))  # Top/bottom 20% or at least 2 zones
            high_risk = zone_metrics['claim_frequency'].nlargest(n_select).index
            low_risk = zone_metrics['claim_frequency'].nsmallest(n_select).index
            self.data['zone_group'] = self.data['maincrestazone'].apply(
                lambda x: 'High Risk' if x in high_risk else 'Low Risk' if x in low_risk else np.nan
            )
            df_zone = self.data[self.data['zone_group'].notna()]
            chi_result = self.chi_squared_test('zone_group', 'High Risk', 'Low Risk')
            if chi_result is None:
                contingency_table = pd.crosstab(df_zone['zone_group'], df_zone['has_claim'])
                if contingency_table.shape[1] == 2:
                    _, p = fisher_exact(contingency_table)
                    results.append({
                        'test': 'Fisher Exact', 'group_a': 'High Risk', 'group_b': 'Low Risk',
                        'metric': 'claim_frequency', 'p_value': p
                    })
            else:
                results.append(chi_result)
            results.append(self.t_test('zone_group', 'High Risk', 'Low Risk', 'claim_severity'))
            results.append(self.t_test('zone_group', 'High Risk', 'Low Risk', 'margin'))
        # H₀: No risk differences between Women and Men
        gender_counts = self.data['gender'].value_counts()
        print(f"Gender distribution: {gender_counts.to_dict()}")
        if 'Female' in gender_counts and 'Male' in gender_counts:
            results.append(self.chi_squared_test('gender', 'Female', 'Male'))
            results.append(self.t_test('gender', 'Female', 'Male', 'claim_severity'))

        # Adjust p-values
        results = [r for r in results if r is not None]
        if results:
            p_values = [r['p_value'] for r in results]
            _, p_adjusted, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
            for r, p_adj in zip(results, p_adjusted):
                r['p_value_adjusted'] = p_adj
        return pd.DataFrame(results)

    def save_results(self, filename='hypothesis_test_results.csv'):
        results = self.run_hypothesis_tests()
        os.makedirs('reports', exist_ok=True)
        results.to_csv(f'../reports/{filename}', index=False)
        print(f"Results saved to reports/{filename}")
