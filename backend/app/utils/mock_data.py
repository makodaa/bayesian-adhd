"""
Mock data generator for temporary solution.
Creates placeholder subjects and recordings when actual data is not provided.
"""
import random
from datetime import datetime

class MockDataGenerator:
    """Generate mock data for subjects and recordings."""
    
    @staticmethod
    def generate_subject_code():
        """Generate a mock subject code with timestamp to ensure uniqueness."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_suffix = random.randint(1000, 9999)
        return f"MOCK_{timestamp}_{random_suffix}"
    
    @staticmethod
    def generate_subject_data():
        """Generate mock subject data."""
        return {
            'subject_code': MockDataGenerator.generate_subject_code(),
            'age': random.randint(18, 65),
            'gender': random.choice(['Male', 'Female', 'Other'])
        }
    
    @staticmethod
    def generate_recording_data(file_name: str):
        """Generate mock recording data."""
        return {
            'file_name': file_name or f"mock_file_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv",
            'sleep_hours': round(random.uniform(4.0, 10.0), 2),
            'food_intake': random.choice(['Light breakfast', 'Normal meal', 'Heavy meal', 'Fasting']),
            'caffeinated': random.choice([True, False]),
            'medicated': False,  # Default to False for safety
            'medication_intake': None,
            'artifacts_noted': 'Auto-generated recording - no artifacts manually reviewed',
            'notes': 'This is a mock recording created automatically for testing purposes.'
        }
    
    @staticmethod
    def create_mock_subject_and_recording(subject_service, recording_service, file_name: str = None):
        """
        Create a mock subject and recording in the database.
        
        Args:
            subject_service: SubjectService instance
            recording_service: RecordingService instance
            file_name: Optional file name for the recording
            
        Returns:
            tuple: (subject_id, recording_id)
        """
        # Create mock subject
        subject_data = MockDataGenerator.generate_subject_data()
        subject_id = subject_service.create_subject(
            subject_code=subject_data['subject_code'],
            age=subject_data['age'],
            sex=subject_data['gender']
        )
        
        # Create mock recording
        recording_data = MockDataGenerator.generate_recording_data(file_name)
        recording_id = recording_service.create_recording(
            subject_id=subject_id,
            file_name=recording_data['file_name'],
            sleep_hours=recording_data['sleep_hours'],
            food_intake=recording_data['food_intake'],
            caffeinated=recording_data['caffeinated'],
            medicated=recording_data['medicated'],
            medication_intake=recording_data['medication_intake'],
            artifacts_noted=recording_data['artifacts_noted'],
            notes=recording_data['notes']
        )
        
        print(f"Created mock subject (ID: {subject_id}) and recording (ID: {recording_id})")
        
        return subject_id, recording_id
