import psycopg
from .connection import get_connection

class FaceRepository:
    def __init__(self):
        self.conn = get_connection()

    def insert_embedding(self, embedding, name):
        insert_query = """INSERT INTO face_embedding (embedding, name)
                                VALUES(%s, %s)"""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(insert_query, (embedding, name))
            self.conn.commit()
        except psycopg.errors:
            self.conn.rollback()
            raise

    def search(self, embedding):
        query = """SELECT name, 1 - (embedding <=> %s) AS similarity 
        FROM face_embedding 
        ORDER BY embedding <=> %s LIMIT 1;"""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(query, (embedding, embedding))
                return cursor.fetchone()
        except psycopg.errors:
            self.conn.rollback()
            raise

    def close(self):
        self.conn.close()