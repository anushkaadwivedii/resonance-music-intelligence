"""Add passage-level lyrics vectors without retaining raw lyric text."""

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


revision = "20260829_06"
down_revision = "20260829_05"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "lyrics_chunks",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("song_id", sa.BigInteger(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("embedding", Vector(1536), nullable=False),
        sa.Column("embedding_model", sa.String(length=100), nullable=False),
        sa.Column("embedding_version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["song_id"], ["songs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "song_id", "chunk_index", "embedding_model", "embedding_version",
            name="uq_lyrics_chunks_recipe_position",
        ),
    )
    op.create_index("ix_lyrics_chunks_song_id", "lyrics_chunks", ["song_id"])
    op.execute(
        """
        CREATE INDEX idx_lyrics_chunks_embedding_hnsw
        ON lyrics_chunks USING hnsw (embedding vector_cosine_ops)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_lyrics_chunks_embedding_hnsw")
    op.drop_index("ix_lyrics_chunks_song_id", table_name="lyrics_chunks")
    op.drop_table("lyrics_chunks")
