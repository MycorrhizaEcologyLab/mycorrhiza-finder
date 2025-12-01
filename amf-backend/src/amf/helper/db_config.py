import os
import sys
from configparser import ConfigParser

import psycopg2
from psycopg2 import sql

import amf.helper.log as AmfLog

if getattr(sys, "frozen", False):  # i.e. a pyinstaller build
    db_ini_path = os.path.join(sys._MEIPASS, "database.ini")  # type: ignore[attr-defined]
else:  # i.e. dev
    db_ini_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "database", "database.ini"
    )


# This uses the local database.ini file, which has dummy credentials in.
# Please update to your local postgres credentials
def config(filename: str = db_ini_path, section: str = "database") -> dict[str, str]:
    parser = ConfigParser()
    parser.read(filename)
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception(
            "Section {0} is not found in the {1} file.".format(section, filename)
        )
    return db


def connect(
    db_name: str, password: str | None = None
) -> psycopg2.extensions.connection:
    connection = None
    try:
        params = config()
        params["database"] = db_name
        if password is not None:
            params["password"] = password
        connection = psycopg2.connect(**params)
        connection.autocommit = True
        return connection
    except (Exception, psycopg2.DatabaseError) as error:
        AmfLog.error(
            "Cannot connect to database, consider restarting your postgres service: "
            f"{error}",
            exit_code=AmfLog.ERR_NO_DATABASE_CONNECTION,
        )


def create_database_if_not_exists(db_name: str, user: str, password: str) -> None:
    connection = connect(
        "postgres", password
    )  # Get a new connection for this operation
    if connection is None:
        print("Connection to database failed.")
        return

    with connection.cursor() as crsr:
        crsr.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
        exists = crsr.fetchone()

        if not exists:
            crsr.execute(
                sql.SQL(
                    "CREATE DATABASE {} WITH OWNER = {} ENCODING = 'UTF8' CONNECTION "
                    "LIMIT = -1;"
                ).format(sql.Identifier(db_name), sql.Identifier(user))
            )
            print(f"Database '{db_name}' created.")
        else:
            print(f"Database '{db_name}' already exists.")

    connection.close()
