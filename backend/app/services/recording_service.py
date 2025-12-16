from werkzeug.datastructures import FileStorage
from .file_service import FileService
from .eeg_service import EEGService
from .band_analysis_service import BandAnalysisService
from ..db.repositories.recordings import RecordingsRepository
from ..core.logging_config import get_app_logger

logger = get_app_logger(__name__)

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

    def process_and_store(self, subject_id:int, file: FileStorage, clinician_id: int,
                         sleep_hours=None, food_intake=None, caffeinated=None,
                         medicated=None, medication_intake=None,
                         artifacts_noted=None, notes=None) -> dict:
        """Process uploaded EEG file and store results."""
        logger.info(f"Processing and storing recording for subject {subject_id}: {file.filename}")
        
        df = self.file_service.read_csv(file)
        self.file_service.validate_eeg_data(df)

        # Create recording entry
        logger.debug("Creating recording entry in database")
        recording_id = self.recordings_repo.create_recording(
            subject_id=subject_id,
            file_name=file.filename,
            sleep_hours=sleep_hours,
            food_intake=food_intake,
            caffeinated=caffeinated,
            medicated=medicated,
            medication_intake=medication_intake,
            artifacts_noted=artifacts_noted,
            notes=notes
        )

        # Classify and save results
        logger.info(f"Starting classification for recording {recording_id}")
        classification_result = self.eeg_service.classify_and_save(recording_id,df)

        # Compute and save band powers
        logger.info(f"Computing band powers for result {classification_result['result_id']}")
        band_powers = self.band_analysis_service.compute_and_save(classification_result['result_id'], df)

        logger.info(f"Recording processing complete for recording {recording_id}")
        return {
            'recording_id': recording_id,
            'classification': classification_result,
            'band_powers' : band_powers
        }