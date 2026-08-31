"""Store derived lyrical meaning without retaining raw lyrics."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260830_07"
down_revision = "20260829_06"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("songs", sa.Column("lyrics_meaning", postgresql.JSONB(), nullable=True))
    op.add_column("songs", sa.Column("lyrics_meaning_model", sa.String(length=100), nullable=True))
    op.add_column("songs", sa.Column("lyrics_meaning_version", sa.Integer(), nullable=True))
    op.add_column(
        "songs",
        sa.Column("lyrics_meaning_analyzed_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("songs", "lyrics_meaning_analyzed_at")
    op.drop_column("songs", "lyrics_meaning_version")
    op.drop_column("songs", "lyrics_meaning_model")
    op.drop_column("songs", "lyrics_meaning")
