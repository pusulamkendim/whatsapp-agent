"""add image localization jobs

Revision ID: 20260707_0004
Revises: 20260702_0003
Create Date: 2026-07-07
"""
from alembic import op
import sqlalchemy as sa

revision = "20260707_0004"
down_revision = "20260702_0003"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    existing_tables = set(sa.inspect(bind).get_table_names())
    if "image_localization_jobs" in existing_tables and "image_localization_assets" in existing_tables:
        return

    if "image_localization_jobs" not in existing_tables:
        op.create_table(
            "image_localization_jobs",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("target_language", sa.String(), server_default="tr"),
            sa.Column("status", sa.String(), server_default="created"),
            sa.Column("notes", sa.Text(), server_default=""),
            sa.Column("created_at", sa.DateTime()),
            sa.Column("updated_at", sa.DateTime()),
        )
        op.create_index(
            "ix_image_localization_jobs_status",
            "image_localization_jobs",
            ["status"],
        )
    if "image_localization_assets" not in existing_tables:
        op.create_table(
            "image_localization_assets",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("job_id", sa.Integer(), sa.ForeignKey("image_localization_jobs.id"), nullable=False),
            sa.Column("filename", sa.String(), nullable=False),
            sa.Column("content_type", sa.String(), server_default="image/png"),
            sa.Column("original_path", sa.Text(), nullable=False),
            sa.Column("cropped_path", sa.Text(), server_default=""),
            sa.Column("output_path", sa.Text(), server_default=""),
            sa.Column("crop_json", sa.Text(), server_default="{}"),
            sa.Column("ocr_json", sa.Text(), server_default="[]"),
            sa.Column("translations_json", sa.Text(), server_default="[]"),
            sa.Column("approved_texts_json", sa.Text(), server_default="[]"),
            sa.Column("status", sa.String(), server_default="uploaded"),
            sa.Column("error_message", sa.Text(), server_default=""),
            sa.Column("created_at", sa.DateTime()),
            sa.Column("updated_at", sa.DateTime()),
        )
        op.create_index(
            "ix_image_localization_assets_job_id",
            "image_localization_assets",
            ["job_id"],
        )
        op.create_index(
            "ix_image_localization_assets_status",
            "image_localization_assets",
            ["status"],
        )


def downgrade():
    op.drop_index("ix_image_localization_assets_status", table_name="image_localization_assets")
    op.drop_index("ix_image_localization_assets_job_id", table_name="image_localization_assets")
    op.drop_table("image_localization_assets")
    op.drop_index("ix_image_localization_jobs_status", table_name="image_localization_jobs")
    op.drop_table("image_localization_jobs")
