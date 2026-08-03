import os
from pathlib import Path
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH, override=True)

DATABASE_URL = os.getenv("DATABASE_URL")


def get_connection() -> psycopg.Connection:
    if not DATABASE_URL:
        raise RuntimeError(f"Not found DATABASE_URL in {ENV_PATH}")
    connection = psycopg.connect(DATABASE_URL, row_factory=dict_row)
    register_vector(connection)
    return connection


def check_database_connection() -> bool:
    try:
        with get_connection() as connection:
            result = connection.execute(
                """
                SELECT
                    current_user AS database_user,
                    current_database() AS database_name,
                    EXISTS (
                        SELECT 1
                        FROM pg_extension
                        WHERE extname = 'vector'
                    ) AS vector_enabled;
                """
            ).fetchone()

        print("Connect database successfully!")
        print("User:", result["database_user"])
        print("Database:", result["database_name"])
        print("pgvector:", result["vector_enabled"])
        return True

    except psycopg.Error as error:
        print("Connect database failed:")
        print(error)
        return False


if __name__ == "__main__":
    check_database_connection()