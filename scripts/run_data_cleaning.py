import sys
sys.path.append('scripts')
from data_loader import DataLoader

def main():
    input_path = 'data/raw/MachineLearningRating_v3.txt'
    output_path = 'data/processed/clean_data.csv'
    loader = DataLoader(input_path)
    loader.load_data()
    cleaned_data = loader.clean_data()
    print(f"Data cleaning complete. Output saved to {output_path}")

if __name__ == "__main__":
    main()
