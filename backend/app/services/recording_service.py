from werkzeug.datastructures import FileStorage
from .file_service import FileService
from .eeg_service import EEGService
from .band_analysis_service import BandAnalysisService
from ..db.repositories.recordings import RecordingsRepository

class RecordingService:
    """Manage EEG recording upload, analysis, and storage"""

    def __init__(self,
                 recordings_repo: RecordingsRepository,
                 file_service: FileService,
                 eeg_service: EEGService,
                 band_analysis_service: BandAnalysisService
                ):
        self.recordings_repo = recordings_repo
        self.file_service = file_service
        self.eeg_service = eeg_service
        self.band_analysis_service = band_analysis_service

    def process_and_store(self, subject_id:int, file: FileStorage, clinician_id: int) -> dict:
        """Process uploaded EEG file and store results."""
        df = self.file_service.read_csv(file)
        self.file_service.validate_eeg_data(df)

        # Create recording entry
        recording_id = self.recordings_repo.create_recording(
            subject_id=subject_id,
            file_name=file.filename,
        )

        # Classify and save results
        classification_result = self.eeg_service.classify_and_save(recording_id,df)

        # Compute and save band powers
        band_powers = self.band_analysis_service.compute_and_save(classification_result['result_id'], df)

        return {
            'recording_id': recording_id,
            'classification': classification_result,
            'band_powers' : band_powers
        }