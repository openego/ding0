from ding0.tools import database

class TestDatabase:
    def test_oedb(self):
        with database.session_scope(overwrite_database="oedb") as session:
            print()
            print(session)

    def test_local_database(self):
        with database.session_scope("local") as session:
            print()
            print(session)


