## A/B Hypothesis Testing for Risk Drivers

### Objective
Test four null hypotheses to identify risk drivers, supporting ACIS’s goal of optimizing premiums and targeting low-risk segments.

### Methodology
- **Metrics**:
  - Claim Frequency: Proportion of policies with `totalclaims` > 0.
  - Claim Severity: Mean `totalclaims` for claims > 0.
  - Margin: `totalpremium` - `totalclaims`.
- **Null Hypotheses**:
  - H₀: No risk differences across provinces (Gauteng vs. KwaZulu-Natal).
  - H₀: No risk differences between zip codes.
  - H₀: No significant margin difference between zip codes.
  - H₀: No significant risk difference between Women and Men.
- **Segmentation**:
  - Province: Gauteng vs. KwaZulu-Natal.
  - PostalCode: High-risk (top 20%, 170 postcodes) vs. low-risk (bottom 20%, 170 postcodes), filtered for >10 claims.
  - Gender: Female (6.5%) vs. Male (93.5%).
- **Tests**:
  - Chi-squared or Fisher’s exact for Claim Frequency.
  - Welch’s t-test for Claim Severity and Margin.
  - Equivalence checked for `maritalstatus`, `vehicletype`, `covertype`, `cubiccapacity`.
  - P-values adjusted via FDR.
- **Data Cleaning**:
  - Filtered negative `totalpremium`/`totalclaims`.
  - Dropped 5 zero `totalpremium` rows.
  - Gender: ~93.5% Male post-imputation.

### Findings
- **Province (Gauteng vs. KwaZulu-Natal)**:
  - Claim Frequency: p = 3.04e-03 (adjusted 4.26e-03), reject H₀. Gauteng has higher frequency.
  - Claim Severity: p = 3.20e-04 (adjusted 5.59e-04), reject H₀. Gauteng has higher severity.
  - **Limitation**: Vehicle differences (p = 3.74e-79 for `vehicletype`, 7.19e-72 for `cubiccapacity`) may confound results.
  - **Recommendation**: Increase Gauteng premiums by ~10-15%; target KwaZulu-Natal for low-risk marketing.
- **PostalCode (Risk)**:
  - Claim Frequency: p = 1.62e-14 (adjusted 1.13e-13), reject H₀. High-risk postcodes have higher frequency.
  - Claim Severity: p = 8.33e-01 (adjusted 8.38e-01), fail to reject H₀.
  - **Recommendation**: Implement postcode-based pricing for high-risk areas.
- **PostalCode (Margin)**:
  - Margin: p = 3.97e-07 (adjusted 1.39e-06), reject H₀. Low-risk postcodes more profitable.
  - **Recommendation**: Market to low-risk postcodes.
- **Gender**:
  - Claim Frequency: p = 4.21e-05 (adjusted 9.82e-05), reject H₀. Female higher frequency.
  - Claim Severity: p = 8.38e-01 (adjusted 8.38e-01), fail to reject H₀.
  - **Limitation**: ~93.5% Male imputation biases results.
  - **Recommendation**: Exclude gender; improve data collection.

### Business Impact
Targeted pricing in Gauteng and high-risk postcodes, and marketing to low-risk postcodes and KwaZulu-Natal, will enhance profitability. Gender data quality improvement is critical.

