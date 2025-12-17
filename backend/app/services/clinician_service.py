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
    
    def get_all_clinicians(self) -> list:
        """Get all clinicians."""
        return self.clinicians_repo.get_all()
    
    def get_clinician_by_id(self, clinician_id: int) -> Optional[dict]:
        """Get clinician by ID."""
        return self.clinicians_repo.get_by_id(clinician_id)
    
    def format_clinicians_for_frontend(self, clinicians: list) -> list:
        """Format clinicians data for frontend consumption."""
        formatted = []
        for c in clinicians:
            name = f"{c.get('first_name', '')} {c.get('last_name', '')}".strip()
            if c.get('middle_name'):
                name = f"{c.get('first_name', '')} {c.get('middle_name', '')} {c.get('last_name', '')}".strip()
            formatted.append({
                'id': c['id'],
                'name': name,
                'occupation': c.get('occupation', '')
            })
        return formatted
