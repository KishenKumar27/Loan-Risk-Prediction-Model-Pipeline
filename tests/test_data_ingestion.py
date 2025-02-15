import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingest_data import DataIngestorFactory
    
def test_data_ingestion():
    # Specify the file path
    file_path = "/Users/kishenkumarsivalingam/Documents/MLE/data.zip"

    # Determine the file extension
    file_extension = os.path.splitext(file_path)[1]

    # Get the appropriate DataIngestor
    data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)

    # Ingest the data and load it into a DataFrame
    df = data_ingestor.ingest(file_path)

    assert df is not None, "Data ingestion returned None"
    assert df.shape[0] > 0, "Dataframe is empty"
    assert 'loanAmount' in df.columns, "Expected column 'loanAmount' not found"
    assert df.isnull().sum().sum() == 0, "Data contains missing values"
    
if __name__ == '__main__':
    test_data_ingestion()