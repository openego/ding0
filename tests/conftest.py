import pytest
from egoio.tools import db
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def oedb_session():
    """
    Returns an ego.io oedb session and closes it on finishing the test
    """
    engine = db.connection(readonly=True)
    session = sessionmaker(bind=engine)()
    yield session
    print("closing session")
    session.close()