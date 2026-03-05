import re
from pathlib import Path
from typing import Any

from sqlalchemy import Column, Integer, MetaData, Table, Text, bindparam, create_engine
from sqlalchemy import insert, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.sql import func, literal_column, table, column


DEFAULT_MEMORY_FIELDS = [
    "issue_description",
    "keywords",
    "root_cause",
    "decisions",
    "insights",
    "related_paths",
    "commands",
    "notes",
]
MEMORY_FIELDS = list(DEFAULT_MEMORY_FIELDS)


def _tokenize(text: str) -> list[str]:
    seen: set[str] = set()
    tokens: list[str] = []
    for token in re.findall(r"[a-z0-9_]+", text.lower()):
        if len(token) < 2:
            continue
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens


class MemoryStore:
    def __init__(
        self,
        db_path: str | Path,
        *,
        fields: list[str] | None = None,
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        selected_fields = fields or list(DEFAULT_MEMORY_FIELDS)
        self.fields = tuple(str(field) for field in selected_fields)
        self.retrieval_fields = self.fields
        self.engine: Engine = create_engine(
            f"sqlite+pysqlite:///{self.db_path}", future=True
        )
        self.metadata = MetaData()
        self.memories = Table(
            "memories",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("created_at", Text, nullable=False),
            Column("exit_status", Text, nullable=False),
            Column("task_hash", Text, nullable=False),
            Column("summary_version", Text, nullable=False),
            *(Column(field, Text, nullable=False, server_default="") for field in self.fields),
        )
        self.memories_fts = table(
            "memories_fts",
            column("rowid"),
            *(column(field) for field in self.retrieval_fields),
        )
        self.init_db()

    def init_db(self) -> None:
        field_lines = ",\n            ".join(self.retrieval_fields)
        fts_sql = f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            {field_lines},
            content='memories',
            content_rowid='id'
        )
        """
        trigger_fields = ",\n                    ".join(self.retrieval_fields)
        trigger_new_values = ",\n                    ".join(
            f"new.{field}" for field in self.retrieval_fields
        )
        trigger_old_values = ",\n                    ".join(
            f"old.{field}" for field in self.retrieval_fields
        )
        trigger_sql = [
            f"""
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(
                    rowid,
                    {trigger_fields}
                ) VALUES (
                    new.id,
                    {trigger_new_values}
                );
            END
            """,
            f"""
            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(
                    memories_fts,
                    rowid,
                    {trigger_fields}
                ) VALUES (
                    'delete',
                    old.id,
                    {trigger_old_values}
                );
            END
            """,
            f"""
            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(
                    memories_fts,
                    rowid,
                    {trigger_fields}
                ) VALUES (
                    'delete',
                    old.id,
                    {trigger_old_values}
                );

                INSERT INTO memories_fts(
                    rowid,
                    {trigger_fields}
                ) VALUES (
                    new.id,
                    {trigger_new_values}
                );
            END
            """,
        ]

        with self.engine.begin() as conn:
            self.metadata.create_all(conn)
            try:
                conn.execute(text(fts_sql))
            except OperationalError as exc:
                if "fts5" in str(exc).lower():
                    raise RuntimeError(
                        "SQLite FTS5 is unavailable. Please use a Python build with FTS5 enabled."
                    ) from exc
                raise
            self._validate_schema(conn)
            conn.execute(text("DROP TRIGGER IF EXISTS memories_ai"))
            conn.execute(text("DROP TRIGGER IF EXISTS memories_ad"))
            conn.execute(text("DROP TRIGGER IF EXISTS memories_au"))
            for ddl in trigger_sql:
                conn.execute(text(ddl))

    def _validate_schema(self, conn) -> None:
        memory_columns = {
            row[1] for row in conn.execute(text("PRAGMA table_info(memories)")).fetchall()
        }
        required_memory_columns = {
            "id",
            "created_at",
            "exit_status",
            "task_hash",
            "summary_version",
            *self.fields,
        }
        missing_memory_columns = sorted(required_memory_columns - memory_columns)
        if missing_memory_columns:
            raise RuntimeError(
                "memory.db schema mismatch for table 'memories'. Missing columns: "
                + ", ".join(missing_memory_columns)
                + ". Use a fresh --memory-path when changing memory fields."
            )

        fts_columns = {
            row[1] for row in conn.execute(text("PRAGMA table_info(memories_fts)")).fetchall()
        }
        missing_fts_columns = sorted(set(self.retrieval_fields) - fts_columns)
        if missing_fts_columns:
            raise RuntimeError(
                "memory.db schema mismatch for table 'memories_fts'. Missing columns: "
                + ", ".join(missing_fts_columns)
                + ". Use a fresh --memory-path when changing retrieval_fields."
            )

    def insert_memory(self, record: dict[str, Any]) -> int:
        columns = [
            "created_at",
            "exit_status",
            "task_hash",
            "summary_version",
            *self.fields,
        ]
        values = {col: str(record.get(col, "")) for col in columns}
        stmt = insert(self.memories).values(values)

        with self.engine.begin() as conn:
            result = conn.execute(stmt)
            return int(result.inserted_primary_key[0])

    def search(
        self,
        query_text: str,
        *,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> list[dict[str, Any]]:
        tokens = _tokenize(query_text)
        if not tokens:
            return []
        limited_tokens = tokens[:16]
        match_expr = " OR ".join(f'"{token}"' for token in limited_tokens)
        query_token_set = set(limited_tokens)

        m = self.memories.alias("m")
        stmt = (
            select(
                *[m.c[col.name].label(col.name) for col in self.memories.columns],
                func.bm25(literal_column("memories_fts")).label("bm25"),
            )
            .select_from(self.memories_fts.join(m, m.c.id == self.memories_fts.c.rowid))
            .where(text("memories_fts MATCH :match_expr"))
            .order_by(text("bm25 ASC"))
            .limit(max(candidate_k, top_k))
        )

        with self.engine.begin() as conn:
            rows = conn.execute(stmt, {"match_expr": match_expr}).mappings().all()

        enriched: list[dict[str, Any]] = []
        for row in rows:
            row_dict = dict(row)
            corpus = "\n".join(str(row_dict.get(f, "")) for f in self.retrieval_fields)
            overlap = len(query_token_set.intersection(_tokenize(corpus)))
            row_dict["overlap"] = overlap
            enriched.append(row_dict)

        enriched.sort(
            key=lambda r: (
                -int(r.get("overlap", 0)),
                float(r.get("bm25", 9999.0)),
                -int(r.get("id", 0)),
            )
        )
        return enriched[:top_k]

    def get_by_ids(self, ids: list[int]) -> list[dict[str, Any]]:
        if not ids:
            return []
        stmt = (
            select(self.memories)
            .where(self.memories.c.id.in_(bindparam("ids", expanding=True)))
            .order_by(self.memories.c.id.desc())
        )
        with self.engine.begin() as conn:
            rows = conn.execute(stmt, {"ids": ids}).mappings().all()
        return [dict(r) for r in rows]
