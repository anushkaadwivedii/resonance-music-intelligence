"""Track embedding compatibility and add HNSW cosine search."""

from alembic import op
import sqlalchemy as sa


revision = "20260828_02"
down_revision = "20260828_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("songs", sa.Column("embedding_model", sa.String(length=100), nullable=True))
    op.add_column("songs", sa.Column("embedding_version", sa.Integer(), nullable=True))
    op.add_column("songs", sa.Column("embedded_at", sa.DateTime(timezone=True), nullable=True))

    # Existing vectors were generated with this project's version-2 recipe.
    op.execute(
        """
        UPDATE songs
        SET embedding_model = 'text-embedding-3-small',
            embedding_version = 2,
            embedded_at = CURRENT_TIMESTAMP
        WHERE embedding IS NOT NULL
        """
    )
    op.execute(
        """
        CREATE INDEX idx_songs_embedding_hnsw
        ON songs USING hnsw (embedding vector_cosine_ops)
        WHERE embedding IS NOT NULL
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_songs_embedding_hnsw")
    op.drop_column("songs", "embedded_at")
    op.drop_column("songs", "embedding_version")
    op.drop_column("songs", "embedding_model")
