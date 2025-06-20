import pandas as pd
import numpy as np
import re
import os
from datetime import datetime

class DataLoader:
    """Class to load and preprocess messy insurance dataset from text file."""
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.delimiter = None
        self.expected_columns = [
            'UnderwrittenCoverID', 'PolicyID', 'TransactionMonth', 'IsVATRegistered', 'Citizenship',
            'LegalType', 'Title', 'Language', 'Bank', 'AccountType', 'MaritalStatus', 'Gender',
            'Country', 'Province', 'PostalCode', 'MainCrestaZone', 'SubCrestaZone', 'ItemType',
            'Mmcode', 'VehicleType', 'RegistrationYear', 'Make', 'Model', 'Cylinders',
            'Cubiccapacity', 'Kilowatts', 'Bodytype', 'NumberOfDoors', 'VehicleIntroDate',
            'CustomValueEstimate', 'AlarmImmobiliser', 'TrackingDevice', 'CapitalOutstanding',
            'NewVehicle', 'WrittenOff', 'Rebuilt', 'Converted', 'CrossBorder',
            'NumberOfVehiclesInFleet', 'SumInsured', 'TermFrequency', 'CalculatedPremiumPerTerm',
            'ExcessSelected', 'CoverCategory', 'CoverType', 'CoverGroup', 'Section', 'Product',
            'StatutoryClass', 'StatutoryRiskType', 'TotalPremium', 'TotalClaims'
        ]

    def detect_delimiter(self, sample_lines=10):
        """Detect the delimiter used in the text file."""
        delimiters = [',', '\t', '|', ';', ' ']
        delimiter_scores = {d: 0 for d in delimiters}

        with open(self.file_path, 'r') as file:
            lines = [file.readline().strip() for _ in range(sample_lines)]

        for line in lines:
            for d in delimiters:
                delimiter_scores[d] += len(line.split(d))

        self.delimiter = max(delimiter_scores, key=delimiter_scores.get)
        print(f"Detected delimiter: '{self.delimiter}'")
        return self.delimiter

    def load_data(self):
        """Load and parse the dataset from a text file."""
        try:
            # Use default ',' if file is .csv
            if self.file_path.endswith('.csv'):
                sep = ','
            else:
                if not self.delimiter:
                    self.delimiter = self.detect_delimiter()
                sep = self.delimiter

            na_values = ['N/A', 'NA', 'NULL', '', 'missing', 'Unknown']
            self.data = pd.read_csv(
                self.file_path,
                sep=self.delimiter,
                na_values=na_values,
                low_memory=False
            )

            # Standardize column names
            self.data.columns = [col.strip().lower().replace(' ', '_') for col in self.data.columns]

            # Validate columns
            missing_cols = [col.lower().replace(' ', '_') for col in self.expected_columns if col.lower().replace(' ', '_') not in self.data.columns]
            if missing_cols:
                print(f"Warning: Missing columns: {missing_cols}")

            print(f"Data loaded successfully from {self.file_path}")
            print(f"Initial shape: {self.data.shape}")
            return self.data
        except FileNotFoundError:
            raise FileNotFoundError(f"File {self.file_path} not found.")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")

    def clean_data(self):
        """Clean the dataset by handling common issues step-by-step."""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        # Remove duplicate rows
        initial_rows = self.data.shape[0]
        self.data = self.data.drop_duplicates()
        print(f"Removed {initial_rows - self.data.shape[0]} duplicate rows")

        # Map termfrequency
        if 'termfrequency' in self.data.columns:
            freq_map = {'Monthly': 12, 'Annual': 1}
            self.data['termfrequency_mapped'] = self.data['termfrequency'].map(freq_map)
            print("Mapped 'termfrequency' to numeric values.")

        # Step 1: Impute missing gender from title
        self.impute_gender_from_title()

        # Step 2: Drop rows where 'newvehicle' is missing
        if 'newvehicle' in self.data.columns:
            before_drop = self.data.shape[0]
            self.data = self.data.dropna(subset=['newvehicle'])
            print(f"Dropped {before_drop - self.data.shape[0]} rows with missing 'NewVehicle'")

        # Step 3: Drop rows with missing vehicle info
        self.drop_rows_with_missing_vehicle_info()

        # Step 4: Drop sparse columns
        self.drop_sparse_columns()

        # Step 5: Handle remaining missing values
        self.handle_remaining_missing_values()

        # Step 6: Impute totalpremium
        self.impute_totalpremium()

        # Save cleaned data
        save_path = "data/processed/clean_data.csv"
        self.save_cleaned_data(filename="clean_data.csv", directory="data/processed")

        # Reload and return data from saved file
        print(f"Reloading cleaned data from {save_path}")
        cleaned_data = pd.read_csv(save_path)
        print(f"Data shape after reloading: {cleaned_data.shape}")
        return cleaned_data


    def impute_totalpremium(self):
        """
        Imputes missing or zero totalpremium values in self.data based on:
        - termfrequency (Monthly or Annual)
        - VAT registration status (14% adjustment if registered)
        - transaction month
        """
        required_cols = ['totalpremium', 'calculatedpremiumperterm', 'termfrequency', 'transactionmonth', 'isvatregistered']
        if not all(col in self.data.columns for col in required_cols):
            print("Skipping totalpremium imputation: missing required columns.")
            return

        # Convert to datetime
        self.data['transactionmonth'] = pd.to_datetime(self.data['transactionmonth'], errors='coerce')

        # Identify zero premium rows
        zero_premium = self.data['totalpremium'] == 0
        print(f"Rows with totalpremium = 0 before imputation: {zero_premium.sum()}")

        # Adjust premium based on VAT registration
        self.data['adjusted_premium_per_term'] = self.data.apply(
            lambda row: row['calculatedpremiumperterm'] / 1.14 if row['isvatregistered'] else row['calculatedpremiumperterm'],
            axis=1
        )

        # Apply masks
        monthly_mask = zero_premium & (self.data['termfrequency'] == 'Monthly') & self.data['transactionmonth'].notna()
        annual_mask = zero_premium & (self.data['termfrequency'] == 'Annual') & self.data['transactionmonth'].notna()

        # Impute values
        self.data.loc[monthly_mask, 'totalpremium'] = self.data.loc[monthly_mask, 'adjusted_premium_per_term']
        self.data.loc[annual_mask, 'totalpremium'] = self.data.loc[annual_mask, 'adjusted_premium_per_term']

        # Logging
        monthly_filled = monthly_mask.sum()
        annual_filled = annual_mask.sum()
        print(f"Rows filled with totalpremium (Monthly): {monthly_filled}")
        print(f"Rows filled with totalpremium (Annual): {annual_filled}")
        print(f"Total rows filled: {monthly_filled + annual_filled}")

        remaining = (self.data['totalpremium'] == 0).sum()
        print(f"Remaining totalpremium = 0 rows: {remaining}")

        if remaining > 0:
            print("\nDiagnosing remaining rows (first 5):")
            print(
                self.data[zero_premium & (self.data['totalpremium'] == 0)][
                    ['policyid', 'transactionmonth', 'termfrequency', 'calculatedpremiumperterm',
                     'adjusted_premium_per_term', 'isvatregistered']
                ].head()
            )

        print("\nSample of successfully imputed rows:")
        print(
            self.data[zero_premium & (self.data['totalpremium'] != 0)][
                ['policyid', 'transactionmonth', 'termfrequency', 'calculatedpremiumperterm',
                 'adjusted_premium_per_term', 'totalpremium', 'isvatregistered']
            ].head()
        )

        # Drop temporary column
        self.data.drop(columns=['adjusted_premium_per_term'], inplace=True)

    def save_cleaned_data(self, filename="clean_data.csv", directory="data/processed"):
        """Save the cleaned data to a CSV file in the specified directory."""
        if self.data is None:
            raise ValueError("No cleaned data to save. Run clean_data() first.")

        # Ensure the directory exists
        os.makedirs(directory, exist_ok=True)

        # Compose full file path
        save_path = os.path.join(directory, filename)

        # Save the DataFrame
        self.data.to_csv(save_path, index=False)
        print(f"Cleaned data saved to: {save_path}")

    def handle_remaining_missing_values(self):
        """Handle manageable missing values with appropriate strategies."""
        # Fill 'bank' with 'Unknown'
        if 'bank' in self.data.columns:
            self.data['bank'] = self.data['bank'].fillna('Unknown')
#            self.data.fillna({'bank': 'Unknown'}, inplace=True)
            print("Filled missing 'bank' with 'Unknown'.")

        # Fill 'accounttype' with most common category
        if 'accounttype' in self.data.columns:
            mode_accounttype = self.data['accounttype'].mode()[0]
            self.data['accounttype'] = self.data['accounttype'].fillna(mode_accounttype)
#            self.data.fillna({'accounttype':  mode_accounttype}, inplace=True)
#            self.data['accounttype'].fillna(mode_accounttype, inplace=True)
            print(f"Filled missing 'accounttype' with mode: {mode_accounttype}")

        # Fill 'capitaloutstanding' with 0
        if 'capitaloutstanding' in self.data.columns:
            self.data['capitaloutstanding'] = self.data['capitaloutstanding'].fillna(0)
#            self.data.fillna({'capitaloutstanding': 0}, inplace=True)
#            self.data['capitaloutstanding'].fillna(0, inplace=True)
            print("Filled missing 'capitaloutstanding' with 0.")

    def drop_sparse_columns(self):
        """
        Drops columns that are too sparse (i.e., with excessive missing values)
        and are unlikely to contribute significantly to modeling or insights.
        """
        sparse_columns = [
            'customvalueestimate',
            'writtenoff',
            'rebuilt',
            'converted',
            'crossborder',
            'numberofvehiclesinfleet'
        ]

        existing_sparse = [col for col in sparse_columns if col in self.data.columns]
        self.data.drop(columns=existing_sparse, inplace=True)

        print(f"Dropped {len(existing_sparse)} sparse columns: {existing_sparse}")

    def drop_rows_with_missing_vehicle_info(self):
        """
        Drops rows with missing values in high-impact vehicle-related columns
        necessary for A/B testing and predictive modeling.
        """
        required_vehicle_cols = [
            'maritalstatus',
            'mmcode', 'vehicletype', 'make', 'model',
            'cylinders', 'cubiccapacity', 'kilowatts',
            'bodytype', 'numberofdoors', 'vehicleintrodate'
        ]

        # Check if all required columns exist
        missing_cols = [col for col in required_vehicle_cols if col not in self.data.columns]
        if missing_cols:
            print(f"Error: Missing columns in dataset: {missing_cols}")
            return

        # Drop rows with any NaN in required vehicle columns
        rows_before = self.data.shape[0]
        self.data.dropna(subset=required_vehicle_cols, inplace=True)
        rows_after = self.data.shape[0]

        print(f"Dropped {rows_before - rows_after} rows with missing values in critical vehicle-related columns.")

    def impute_gender_from_title(self):
        """Impute missing gender values using the title column."""
        if 'gender' not in self.data.columns or 'title' not in self.data.columns:
            print("Required columns not found.")
            return

        # Drop ambiguous 'Dr' titles
        before_drop = self.data.shape[0]
        self.data['gender'] = self.data['gender'].replace('Not specified', pd.NA)
        self.data = self.data[~((self.data['title'] == 'Dr') & (self.data['gender'].isna()))]
        print(f"Dropped {before_drop - self.data.shape[0]} rows with title 'Dr' and missing gender.")

        # Define mappings
        title_gender_map = {
            'Mr': 'Male',
            'Mrs': 'Female',
            'Ms': 'Female',
            'Miss': 'Female'
        }

        # Fill missing gender based on title
        missing_before = self.data['gender'].isna().sum()
        for title, gender in title_gender_map.items():
            mask = (self.data['gender'].isna()) & (self.data['title'] == title)
            self.data.loc[mask, 'gender'] = gender
        missing_after = self.data['gender'].isna().sum()

        print(f"Imputed gender for {missing_before - missing_after} rows based on title.")

    def check_data_types(self):
        """Display data types of each column."""
        return self.data.dtypes

    def get_data(self):
        """Return the cleaned dataset."""
        return self.data
