from ..db.connection import get_db_connection
from ..db.repositories.subjects import SubjectsRepository
from ..db.repositories.clinicians import CliniciansRepository
from ..db.repositories.recordings import RecordingsRepository
from ..db.repositories.results import ResultsRepository
from ..db.repositories.band_powers import BandPowersRepository
from ..db.repositories.ratios import RatiosRepository
from ..db.repositories.reports import ReportsRepository

def persist_prediction(data):
    with get_db_connection() as conn:
        subjects = SubjectsRepository(conn)
        recordings = RecordingsRepository(conn)
        clinicians = CliniciansRepository(conn)
        results = ResultsRepository(conn)
        band_powers = BandPowersRepository(conn)
        ratios = RatiosRepository(conn) 
        reports = ReportsRepository(conn)

