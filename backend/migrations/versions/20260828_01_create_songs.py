"""Create the PostgreSQL song catalog."""

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision = "20260828_01"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "songs",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("source", sa.String(length=80), nullable=False),
        sa.Column("source_id", sa.String(length=255), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("artist", sa.Text(), nullable=False),
        sa.Column("album", sa.Text(), nullable=True),
        sa.Column("genres", sa.ARRAY(sa.Text()), nullable=False),
        sa.Column("bpm", sa.Float(), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("explicit", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("danceability", sa.Float(), nullable=True),
        sa.Column("energy", sa.Float(), nullable=True),
        sa.Column("acousticness", sa.Float(), nullable=True),
        sa.Column("instrumentalness", sa.Float(), nullable=True),
        sa.Column("speechiness", sa.Float(), nullable=True),
        sa.Column("valence", sa.Float(), nullable=True),
        sa.Column("popularity", sa.Integer(), nullable=True),
        sa.Column("moods", sa.ARRAY(sa.Text()), nullable=False),
        sa.Column("contexts", sa.ARRAY(sa.Text()), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(1536), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("source", "source_id", name="uq_songs_source_identity"),
    )
    op.create_index("idx_songs_bpm", "songs", ["bpm"])
    op.create_index("idx_songs_energy", "songs", ["energy"])
    op.create_index("idx_songs_valence", "songs", ["valence"])


def downgrade() -> None:
    op.drop_index("idx_songs_valence", table_name="songs")
    op.drop_index("idx_songs_energy", table_name="songs")
    op.drop_index("idx_songs_bpm", table_name="songs")
    op.drop_table("songs")
