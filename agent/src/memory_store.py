import re
from pathlib import Path
from typing import Any

from sqlalchemy import Column, Integer, MetaData, Table, Text, bindparam, create_engine
from sqlalchemy import insert, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.sql import func, literal_column, table, column


MEMORY_FIELDS = [
    "issue_description",
    "keywords",
    "root_cause",
    "decisions",
    "insights",
    "related_paths",
    "commands",
    "notes",
]


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
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
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
            *(Column(field, Text, nullable=False, server_default="") for field in MEMORY_FIELDS),
        )
        self.memories_fts = table(
            "memories_fts",
            column("rowid"),
            *(column(field) for field in MEMORY_FIELDS),
        )
        self.init_db()

    def init_db(self) -> None:
        fts_sql = """
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            issue_description,
            keywords,
            root_cause,
            decisions,
            insights,
            related_paths,
            commands,
            notes,
            content='memories',
            content_rowid='id'
        )
        """
        trigger_sql = [
            """
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(
                    rowid,
                    issue_description,
                    keywords,
                    root_cause,
                    decisions,
                    insights,
                    related_paths,
                    commands,
                    notes
                ) VALUES (
                    new.id,
                    new.issue_description,
                    new.keywords,
                    new.root_cause,
                    new.decisions,
                    new.insights,
                    new.related_paths,
                    new.commands,
                    new.notes
                );
            END
            """,
            """
            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(
                    memories_fts,
                    rowid,
                    issue_description,
                    keywords,
                    root_cause,
                    decisions,
                    insights,
                    related_paths,
                    commands,
                    notes
                ) VALUES (
                    'delete',
                    old.id,
                    old.issue_description,
                    old.keywords,
                    old.root_cause,
                    old.decisions,
                    old.insights,
                    old.related_paths,
                    old.commands,
                    old.notes
                );
            END
            """,
            """
            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(
                    memories_fts,
                    rowid,
                    issue_description,
                    keywords,
                    root_cause,
                    decisions,
                    insights,
                    related_paths,
                    commands,
                    notes
                ) VALUES (
                    'delete',
                    old.id,
                    old.issue_description,
                    old.keywords,
                    old.root_cause,
                    old.decisions,
                    old.insights,
                    old.related_paths,
                    old.commands,
                    old.notes
                );

                INSERT INTO memories_fts(
                    rowid,
                    issue_description,
                    keywords,
                    root_cause,
                    decisions,
                    insights,
                    related_paths,
                    commands,
                    notes
                ) VALUES (
                    new.id,
                    new.issue_description,
                    new.keywords,
                    new.root_cause,
                    new.decisions,
                    new.insights,
                    new.related_paths,
                    new.commands,
                    new.notes
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
            for ddl in trigger_sql:
                conn.execute(text(ddl))

    def insert_memory(self, record: dict[str, Any]) -> int:
        columns = [
            "created_at",
            "exit_status",
            "task_hash",
            "summary_version",
            *MEMORY_FIELDS,
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
            corpus = "\n".join(str(row_dict.get(f, "")) for f in MEMORY_FIELDS)
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
