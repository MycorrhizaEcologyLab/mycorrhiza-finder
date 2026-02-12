"""Functionality for connecting to and configuring the database."""

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


def config(filename: str = db_ini_path, section: str = "database") -> dict[str, str]:
    """Read database configuration.

    This uses the local database.ini file, which has dummy credentials in.
    Please update to your local postgres credentials.


    Args:
        filename: Path to database configuration file.
        section: Section of database configuration file to read.

    Returns: Dictionary of database config parameters.
    """
    parser = ConfigParser()
    parser.read(filename)
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception(f"Section {section} is not found in the {filename} file.")
    return db


def connect(
    db_name: str, password: str | None = None
) -> psycopg2.extensions.connection:
    """Connect to PostgreSQL database.

    Args:
        db_name: Database name.
        password: Optional password to override the one in config file.

    Returns: Database connection object.
    """
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
        logger.error(
            f"Cannot connect to database, consider restarting your postgres service: "
            f"{error}"
        )
        # ERR_NO_DATABASE_CONNECTION = 42
        sys.exit(42)


def create_database_if_not_exists(db_name: str, user: str, password: str) -> None:
    """Create the database if it does not already exist.

    Args:
        db_name: Database name.
        user: Database user name.
        password: Database user password.
    """
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
