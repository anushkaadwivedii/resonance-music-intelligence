"""Add a shadow vector containing sound evidence but no song identity."""

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


revision = "20260828_04"
down_revision = "20260828_03"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("songs", sa.Column("sound_embedding", Vector(1536), nullable=True))
    op.add_column("songs", sa.Column("sound_embedding_model", sa.String(length=100), nullable=True))
    op.add_column("songs", sa.Column("sound_embedding_version", sa.Integer(), nullable=True))
    op.add_column("songs", sa.Column("sound_embedded_at", sa.DateTime(timezone=True), nullable=True))
    op.execute(
        """
        CREATE INDEX idx_songs_sound_embedding_hnsw
        ON songs USING hnsw (sound_embedding vector_cosine_ops)
        WHERE sound_embedding IS NOT NULL
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_songs_sound_embedding_hnsw")
    op.drop_column("songs", "sound_embedded_at")
    op.drop_column("songs", "sound_embedding_version")
    op.drop_column("songs", "sound_embedding_model")
    op.drop_column("songs", "sound_embedding")
