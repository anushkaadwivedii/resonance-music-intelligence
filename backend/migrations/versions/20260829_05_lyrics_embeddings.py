"""Add a shadow lyrics vector without retaining copyrighted lyric text."""

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


revision = "20260829_05"
down_revision = "20260828_04"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("songs", sa.Column("lyrics_embedding", Vector(1536), nullable=True))
    op.add_column("songs", sa.Column("lyrics_embedding_model", sa.String(length=100), nullable=True))
    op.add_column("songs", sa.Column("lyrics_embedding_version", sa.Integer(), nullable=True))
    op.add_column("songs", sa.Column("lyrics_embedded_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("songs", sa.Column("lyrics_provider", sa.String(length=80), nullable=True))
    op.add_column("songs", sa.Column("lyrics_source_id", sa.String(length=255), nullable=True))
    op.add_column("songs", sa.Column("lyrics_lookup_status", sa.String(length=40), nullable=True))
    op.add_column("songs", sa.Column("lyrics_match_confidence", sa.Float(), nullable=True))
    op.add_column("songs", sa.Column("lyrics_checked_at", sa.DateTime(timezone=True), nullable=True))
    op.execute(
        """
        CREATE INDEX idx_songs_lyrics_embedding_hnsw
        ON songs USING hnsw (lyrics_embedding vector_cosine_ops)
        WHERE lyrics_embedding IS NOT NULL
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_songs_lyrics_embedding_hnsw")
    op.drop_column("songs", "lyrics_checked_at")
    op.drop_column("songs", "lyrics_match_confidence")
    op.drop_column("songs", "lyrics_lookup_status")
    op.drop_column("songs", "lyrics_source_id")
    op.drop_column("songs", "lyrics_provider")
    op.drop_column("songs", "lyrics_embedded_at")
    op.drop_column("songs", "lyrics_embedding_version")
    op.drop_column("songs", "lyrics_embedding_model")
    op.drop_column("songs", "lyrics_embedding")
