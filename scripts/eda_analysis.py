import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


class EDAAnalysis:
    """Class to perform EDA and visualization on insurance dataset."""
    def __init__(self, data):
        self.data = data
        os.makedirs('plots', exist_ok=True)

    def summarize_data(self):
        """Compute descriptive statistics for numerical columns."""
        numerical_cols = [
            'totalpremium', 'totalclaims', 'cylinders', 'cubiccapacity',
            'kilowatts', 'numberofdoors', 'customvalueestimate',
            'capitaloutstanding', 'suminsured', 'calculatedpremiumperterm',
            'numberofvehiclesinfleet'
        ]
        available_cols = [col for col in numerical_cols if col in self.data.columns]
        return self.data[available_cols].describe()

    def calculate_loss_ratio(self, group_by=None):
        """Calculate Loss Ratio (totalclaims / totalpremium)."""
        if 'totalpremium' in self.data.columns and 'totalclaims' in self.data.columns:
            self.data['lossratio'] = self.data['totalclaims'] / \
                self.data['totalpremium'].replace(0, np.nan)
            if group_by and group_by in self.data.columns:
                return self.data.groupby(group_by)['lossratio'].mean()
            return self.data['lossratio'].mean()
        else:
            print("Error: 'totalpremium' or 'totalclaims' not found")
            return None

    def univariate_analysis(self, column, plot_type='histogram'):
        """Perform univariate analysis for a given column."""
        if column not in self.data.columns:
            print(f"Error: Column {column} not found")
            return
        plt.figure(figsize=(10, 6))
        if plot_type == 'histogram' and \
                self.data[column].dtype in ['float64', 'int64']:
            sns.histplot(self.data[column], kde=True)
            plt.title(f'Distribution of {column}')
        elif plot_type == 'bar':
            self.data[column].value_counts().plot(kind='bar')
            plt.title(f'Count of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.savefig(f'plots/univariate_{column}.png')
        plt.close()

    def bivariate_analysis(self, x_col, y_col, plot_type='scatter'):
        """Perform bivariate analysis between two columns."""
        if x_col not in self.data.columns or y_col not in self.data.columns:
            print(f"Error: One or both columns ({x_col}, {y_col}) not found")
            return
        plt.figure(figsize=(10, 6))
        if plot_type == 'scatter':
            sns.scatterplot(data=self.data, x=x_col, y=y_col)
            plt.title(f'{x_col} vs {y_col}')
        elif plot_type == 'box':
            sns.boxplot(data=self.data, x=x_col, y=y_col)
            plt.title(f'{y_col} by {x_col}')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.savefig(f'plots/bivariate_{x_col}_{y_col}.png')
        plt.close()

    def detect_outliers(self, column):
        """Detect outliers using box plots."""
        if column not in self.data.columns:
            print(f"Error: Column {column} not found")
            return
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=self.data[column])
        plt.title(f'Box Plot of {column}')
        plt.savefig(f'plots/outliers_{column}.png')
        plt.close()

    def correlation_matrix(self, columns=None):
        """Compute and plot correlation matrix for specified or numerical columns."""
        if columns is None:
            columns = [
                col for col in self.data.columns
                if self.data[col].dtype in ['float64', 'int64']
            ]
        if not all(col in self.data.columns for col in columns):
            print("Error: Some specified columns not found")
            return None
        corr = self.data[columns].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Matrix')
        plt.savefig('plots/correlation_matrix.png')
        plt.close()
        return corr

    def plot_loss_ratio_by_province(self):
        """Plot Loss Ratio by province."""
        if 'province' not in self.data.columns or 'lossratio' not in self.data.columns:
            print("Error: Required columns for loss ratio by province not found")
            return
        plt.figure(figsize=(12, 6))
        sns.barplot(x='province', y='lossratio', data=self.data)
        plt.title('Loss Ratio by Province')
        plt.xticks(rotation=45)
        plt.savefig('plots/loss_ratio_province.png')
        plt.close()

    def plot_claims_by_vehicle_type(self):
        """Plot Total Claims by Vehicle Type."""
        if 'vehicletype' not in self.data.columns or 'totalclaims' not in self.data.columns:
            print("Error: Required columns for claims by vehicle type not found")
            return
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='vehicletype', y='totalclaims', data=self.data)
        plt.title('Total Claims by Vehicle Type')
        plt.xticks(rotation=45)
        plt.savefig('plots/claims_vehicletype.png')
        plt.close()

    def plot_temporal_trends(self):
        """Plot temporal trends in claims and premiums."""
        if 'transactionmonth' not in self.data.columns or \
                'totalclaims' not in self.data.columns or \
                'totalpremium' not in self.data.columns:
            print("Error: Required columns for temporal trends not found")
            return
        self.data['transactionmonth'] = pd.to_datetime(
            self.data['transactionmonth'], errors='coerce'
        )
        monthly_data = self.data.groupby(
            self.data['transactionmonth'].dt.to_period('M')
        ).agg({
            'policyid': 'count',
            'totalclaims': 'mean',
            'totalpremium': 'mean'
        }).reset_index()
        monthly_data['transactionmonth'] = monthly_data['transactionmonth'].dt.to_timestamp()

        plt.figure(figsize=(12, 6))
        plt.plot(
            monthly_data['transactionmonth'],
            monthly_data['totalclaims'],
            label='Average Claims'
        )
        plt.plot(
            monthly_data['transactionmonth'],
            monthly_data['totalpremium'],
            label='Average Premium'
        )
        plt.title('Temporal Trends in Claims and Premiums')
        plt.xlabel('Transaction Month')
        plt.ylabel('Amount')
        plt.legend()
        plt.savefig('plots/temporal_trends.png')
        plt.close()
