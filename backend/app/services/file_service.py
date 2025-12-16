import io
import pandas as pd
from werkzeug.datastructures import FileStorage
from ..config import ALLOWED_EXTENSIONS


class FileService:
    """Handle file upload, validation, and parsing."""
    @staticmethod
    def is_allowed_file(filename:str) -> bool:
        """Check if file extension is allowed"""
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    @staticmethod
    def read_csv(file: FileStorage) -> pd.DataFrame:
        """Read and parse CSV file into DataFrame"""
        binary = file.stream.read()
        try:
            csv_string = binary.decode('utf-8').replace('\r','')
        except UnicodeDecodeError as e:
            raise ValueError("Could not decode file content as UTF-8. It might be a binary file.") from e
        
        csv_io = io.StringIO(csv_string)
        df = pd.read_csv(csv_io)
        csv_io.close()

        return df
    
    @staticmethod
    def validate_eeg_data(df: pd.DataFrame) -> bool:
        """Validate that DataFrame contains valid EEG data."""
        # Check for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found in file.")
        return True