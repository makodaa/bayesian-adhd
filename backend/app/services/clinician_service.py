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
    
    def get_or_create_clinician(self, full_name: str, occupation: str = None) -> int:
        """Get existing clinician by name or create a new one."""
        name_parts = self.parse_clinician_name(full_name)
        
        # Try to find existing clinician
        existing = self.clinicians_repo.get_by_name(
            name_parts['first_name'],
            name_parts['last_name']
        )
        
        if existing:
            return existing['id']
        else:
            # Create new clinician
            return self.clinicians_repo.create_clinician(
                first_name=name_parts['first_name'],
                last_name=name_parts['last_name'],
                middle_name=name_parts['middle_name'],
                occupation=occupation
            )
        
    def parse_clinician_name(self, full_name: str) -> dict:
        """Parse full name into first, middle, and last name."""
        parts = full_name.strip().split()
        if len(parts) == 0:
            return {'first_name': '', 'last_name': '', 'middle_name': None}
        elif len(parts) == 1:
            return {'first_name': parts[0], 'last_name': '', 'middle_name': None}
        elif len(parts) == 2:
            return {'first_name': parts[0], 'last_name': parts[1], 'middle_name': None}
        else:
            # Assume format: First Middle Last (or First Middle1 Middle2... Last)
            return {
                'first_name': parts[0],
                'last_name': parts[-1],
                'middle_name': ' '.join(parts[1:-1])
            }
    
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
                'occupation': c.get('occupation', ''),
                'assessments_count': c.get('assessments_count', 0),
                'last_activity': c.get('last_activity')
            })
        return formatted