import hashlib
import json
import os
from pathlib import Path
from typing import Union, Optional, Any, Tuple, Generator

import sqlite3


class BinaryDB:

    def __init__(
            self,
            filename: Union[str, Path],
            id_length: int = 128,
    ):
        self.filename = Path(filename)
        self._connection: Optional[sqlite3.Connection] = None
        self._id_length = id_length
        self.__db_created = False

    def __enter__(self):
        self.connection()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def connection(self):
        if self._connection is None:
            os.makedirs(self.filename.parent, exist_ok=True)
            self._connection = sqlite3.connect(self.filename)
        return self._connection

    def commit(self):
        self.connection().commit()
        return self

    def disconnect(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def to_id(self, id: Any) -> str:
        if not isinstance(id, str):
            return hashlib.sha3_256(repr(id).encode()).hexdigest()[:self._id_length]
        return id

    def store(
            self,
            id: Any,
            data: Optional[bytes] = None,
            meta: Optional[dict] = None,
            commit: bool = True,
    ):
        if data is not None:
            data = sqlite3.Binary(data)
        if meta is not None:
            meta = json.dumps(meta)

        self.cursor().execute(
            """
                INSERT INTO binary_table (id, data, meta)
                VALUES (?, ?, ?)
            """,
            (self.to_id(id), data, meta),
        )
        if commit:
            self.commit()

    def get(self, id: Any) -> Optional[Tuple[Optional[bytes], Optional[dict]]]:
        c = self.cursor()
        c.execute(
            """
                SELECT data, meta FROM binary_table
                WHERE id = ?
            """,
            (self.to_id(id), ),
        )
        r = c.fetchone()
        if r is not None and r[1] is not None:
            return (r[0], json.loads(r[1]))
        return r

    def iter(self) -> Generator[Tuple[str, Optional[bytes], Optional[dict]], None, None]:
        c = self.cursor()
        c.execute(
            """
                SELECT id, data, meta FROM binary_table
                ORDER BY id
            """
        )
        for id, data, meta in c.fetchall():
            if meta is not None:
                meta = json.loads(meta)
            yield id, data, meta

    def has(self, id: Any) -> bool:
        c = self.cursor()
        c.execute(
            """
                SELECT id FROM binary_table
                WHERE id = ?
            """,
            (self.to_id(id), ),
        )
        return bool(c.fetchone())

    def cursor(self):
        self.connection()
        cursor = self._connection.cursor()

        if not self.__db_created:
            self.__create_db(cursor)
            self.__db_created = True

        return cursor

    def __create_db(self, cursor: sqlite3.Cursor):
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS binary_table (
                id TEXT PRIMARY KEY,
                data BLOB,
                meta TEXT
            )
            """
        )
        self.commit()
