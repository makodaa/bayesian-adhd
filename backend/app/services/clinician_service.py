from typing import Optional
from ..db.repositories.clinicians import CliniciansRepository

class ClinicianService:
    """Manage clinician accounts and authentication."""
    
    def __init__(self, clinicians_repo: CliniciansRepository):
        self.clinicians_repo = clinicians_repo

    def create_clinician(self, data:dict) -> int:
        """Register new clinician."""
        return self.clinicians_repo.create_clinician(
            first_name=data['first_name'],
            last_name=data['last_name'],
            middle_name=data['middle_name'],
            occupation=data['occupation']
        )
