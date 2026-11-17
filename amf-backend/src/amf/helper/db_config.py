import os
import sys
from configparser import ConfigParser

import psycopg2
from loguru import logger
from psycopg2 import sql

if getattr(sys, "frozen", False):  # i.e. a pyinstaller build
    db_ini_path = os.path.join(sys._MEIPASS, "database.ini")  # type: ignore[attr-defined]
else:  # i.e. dev
    db_ini_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "database", "database.ini"
    )

# What a blank field in the login prompt stands for (see amf.helper.db_login).
DEFAULT_USER = "postgres"
DEFAULT_PASSWORD = "postgres"
DEFAULT_PORT = 5432

# The login chosen at startup. Every connect() call overlays it on database.ini,
# so the connections opened deep inside load/save/convert reach the same
# account as the startup one instead of falling back to the bundled password.
_active_credentials: dict[str, str] = {}


# This uses the local database.ini file, which has dummy credentials in.
# The user, password and port are overridden by set_credentials at startup.
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


def set_credentials(user: str, password: str, port: int) -> None:
    _active_credentials.clear()
    _active_credentials.update({"user": user, "password": password, "port": str(port)})


def connect(
    db_name: str, password: str | None = None
) -> psycopg2.extensions.connection:
    connection = None
    try:
        params = config()
        params.update(_active_credentials)
        params["database"] = db_name
        if password is not None:
            params["password"] = password
        connection = psycopg2.connect(**params)
        connection.autocommit = True
        return connection
    except (Exception, psycopg2.DatabaseError) as error:
        logger.error(
            f"Cannot connect to database, consider restarting your postgres service: "
            f"{error}"
        )
        # ERR_NO_DATABASE_CONNECTION = 42
        sys.exit(42)


def check_login(
    user: str, password: str, port: int, db_name: str = "amf"
) -> str | None:
    """
    Tries a login without exiting, for the startup prompt to report on.

    Besides the credentials themselves, checks the account can use 'db_name'.
    Create it if it doesn't exist yet, or act as its owner if it does (the
    schema script alters its tables, which only the owner may do).

    :return: None if the login can be used, otherwise a reason to show the user.
    """
    params = config()
    params.update(
        {
            "user": user,
            "password": password,
            "port": str(port),
            "database": "postgres",
            "connect_timeout": "5",
        }
    )
    try:
        connection = psycopg2.connect(**params)
    except psycopg2.Error as error:
        return _describe_login_error(error, params.get("host", "localhost"), port)

    try:
        with connection.cursor() as crsr:
            crsr.execute(
                "SELECT rolsuper OR rolcreatedb FROM pg_roles "
                "WHERE rolname = current_user"
            )
            can_create = crsr.fetchone()[0]
            crsr.execute(
                "SELECT pg_get_userbyid(datdba), "
                "pg_has_role(current_user, datdba, 'USAGE') "
                "FROM pg_database WHERE datname = %s",
                (db_name,),
            )
            existing = crsr.fetchone()
    except psycopg2.Error as error:
        return f"Logged in, but couldn't check the account's permissions: {error}"
    finally:
        connection.close()

    if existing is None and not can_create:
        return (
            f"This account can't create the '{db_name}' database. Log in as an "
            f"account that can, e.g. postgres."
        )
    if existing is not None and not existing[1]:
        return (
            f"The '{db_name}' database belongs to '{existing[0]}'. Log in as that "
            f"account."
        )
    return None


def _describe_login_error(error: psycopg2.Error, host: str, port: int) -> str:
    message = str(error).strip()
    lowered = message.lower()
    if "password authentication failed" in lowered or (
        "role" in lowered and "does not exist" in lowered
    ):
        return "Incorrect username or password."
    if "no pg_hba.conf entry" in lowered:
        return (
            "PostgreSQL doesn't allow this account to log in from this computer "
            "(no pg_hba.conf entry)."
        )
    if any(
        phrase in lowered
        for phrase in ("connection refused", "timeout expired", "could not connect")
    ):
        return (
            f"Couldn't reach PostgreSQL on {host}:{port}. Check it's installed and "
            f"the PostgreSQL service is running."
        )
    return f"Couldn't log in: {message.splitlines()[-1] if message else error!r}"


def create_database_if_not_exists(
    db_name: str, user: str, password: str | None = None
) -> None:
    connection = connect(
        "postgres", password
    )  # Get a new connection for this operation
    if connection is None:
        logger.error("Connection to database failed.")
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
            logger.info(f"Database '{db_name}' created.")
        else:
            logger.debug(f"Database '{db_name}' already exists.")

    connection.close()
