from typing import Optional
from ..db.repositories.subjects import SubjectsRepository

class SubjectService:
    """Manage subject records"""

    def __init__(self, subjects_repo: SubjectsRepository):
        self.subjects_repo = subjects_repo

    def create_subject(self, subject_code:int, age: int, sex:str) -> int:
        """Create a new subject"""
        return self.subjects_repo.create_subject(subject_code, age, sex)
    
    def get_subject(self, subject_id:int) -> Optional[dict]:
        raise NotImplementedError

    def get_subject_recordings(self, subject_id:int) -> list[dict]:
        raise NotImplementedError

