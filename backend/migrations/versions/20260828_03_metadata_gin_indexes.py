"""Index array metadata used for hybrid candidate recall."""

from alembic import op


revision = "20260828_03"
down_revision = "20260828_02"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE INDEX idx_songs_moods_gin ON songs USING gin (moods)")
    op.execute("CREATE INDEX idx_songs_contexts_gin ON songs USING gin (contexts)")
    op.execute("CREATE INDEX idx_songs_genres_gin ON songs USING gin (genres)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_songs_genres_gin")
    op.execute("DROP INDEX IF EXISTS idx_songs_contexts_gin")
    op.execute("DROP INDEX IF EXISTS idx_songs_moods_gin")
