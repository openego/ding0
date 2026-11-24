import pytest

from ding0.tools import database


def write_config(tmp_path, ssh_tunnel=""):
    key = tmp_path / "key"
    key.write_text("")
    config = tmp_path / "egon-data.configuration.yaml"
    config.write_text(
        "egon-data:\n"
        "  --database-name: egon-data\n"
        "  --database-password: secret\n"
        "  --database-port: '59734'\n"
        "  --database-user: egon\n"
        + ssh_tunnel.format(key=key)
    )
    return config, key


class TestEgonDataCredentials:
    def test_without_ssh_tunnel(self, tmp_path):
        config, _ = write_config(tmp_path)

        credentials = database.egon_data_credentials(str(config))

        assert credentials == {
            "name": "egon-data",
            "user": "egon",
            "password": "secret",
            "host": "localhost",
            "port": 59734,
        }

    def test_ssh_tunnel_with_jump_host(self, tmp_path):
        config, key = write_config(
            tmp_path,
            "ssh-tunnel:\n"
            "  ssh-host: '10.0.0.2'\n"
            "  ssh-user: egon\n"
            "  ssh-pkey: '{key}'\n"
            "  ssh-jump-host: '10.0.0.1'\n"
            "  ssh-jump-user: student\n"
            "  pgres-host: localhost\n",
        )

        ssh = database.egon_data_credentials(str(config))["ssh"]

        assert ssh == {
            "host": "10.0.0.2",
            "user": "egon",
            "pkey": str(key),
            "pgres_host": "localhost",
            "jump_host": "10.0.0.1",
            "jump_user": "student",
            "jump_pkey": str(key),
        }

    def test_missing_ssh_key(self, tmp_path):
        config, _ = write_config(
            tmp_path,
            "ssh-tunnel:\n"
            "  ssh-host: '10.0.0.2'\n"
            "  ssh-user: egon\n"
            "  ssh-pkey: '{key}_missing'\n",
        )

        with pytest.raises(ValueError, match="SSH key"):
            database.egon_data_credentials(str(config))

    def test_engine_without_ssh_tunnel(self, tmp_path):
        config, _ = write_config(tmp_path)

        engine = database.get_engine(
            overwrite_database="local", egon_data_config=str(config))

        assert engine.url.host == "localhost"
        assert engine.url.port == 59734
        assert engine.url.database == "egon-data"
        assert engine is database.get_egon_data_engine(str(config))
