"""Unit tests for scrape module (pure functions, no network)."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tarachat.scrape import (
    download_one,
    has_changed,
    load_metadata,
    meta_path_for,
    save_metadata,
)


class TestMetaPathFor:
    def test_appends_meta_json(self):
        result = meta_path_for(Path("/data/file.pdf"))
        assert result == Path("/data/file.pdf.meta.json")

    def test_txt_file(self):
        result = meta_path_for(Path("docs/readme.txt"))
        assert result == Path("docs/readme.txt.meta.json")


class TestLoadMetadata:
    def test_missing_file_returns_empty(self, tmp_path):
        result = load_metadata(tmp_path / "nonexistent.pdf")
        assert result == {}

    def test_reads_existing_metadata(self, tmp_path):
        file_path = tmp_path / "doc.pdf"
        meta_file = tmp_path / "doc.pdf.meta.json"
        meta_file.write_text(json.dumps({"etag": "abc"}), encoding="utf-8")
        result = load_metadata(file_path)
        assert result == {"etag": "abc"}

    def test_corrupt_json_returns_empty(self, tmp_path):
        file_path = tmp_path / "doc.pdf"
        meta_file = tmp_path / "doc.pdf.meta.json"
        meta_file.write_text("{bad json", encoding="utf-8")
        result = load_metadata(file_path)
        assert result == {}


class TestSaveMetadata:
    def test_writes_metadata(self, tmp_path):
        file_path = tmp_path / "doc.pdf"
        metadata = {"etag": '"xyz"', "last_modified": "Mon, 01 Jan 2024"}
        save_metadata(file_path, metadata)
        meta_file = tmp_path / "doc.pdf.meta.json"
        assert meta_file.exists()
        assert json.loads(meta_file.read_text(encoding="utf-8")) == metadata


class TestHasChanged:
    def test_empty_remote_meta_means_changed(self):
        assert has_changed({"etag": "a"}, {}) is True

    def test_different_etag(self):
        local = {"etag": "a", "last_modified": "x", "content_length": "100"}
        remote = {"etag": "b", "last_modified": "x", "content_length": "100"}
        assert has_changed(local, remote) is True

    def test_different_last_modified(self):
        local = {"etag": "a", "last_modified": "x", "content_length": "100"}
        remote = {"etag": "a", "last_modified": "y", "content_length": "100"}
        assert has_changed(local, remote) is True

    def test_different_content_length(self):
        local = {"etag": "a", "last_modified": "x", "content_length": "100"}
        remote = {"etag": "a", "last_modified": "x", "content_length": "200"}
        assert has_changed(local, remote) is True

    def test_same_metadata_no_change(self):
        meta = {"etag": "a", "last_modified": "x", "content_length": "100"}
        assert has_changed(meta, meta) is False


class TestDownloadOne:
    @pytest.mark.asyncio
    async def test_skips_unchanged(self, tmp_path):
        """When local file exists and metadata matches, skip download."""
        from yarl import URL

        url = URL("http://example.com/file.txt")
        file_path = tmp_path / "file.txt"
        file_path.write_text("existing", encoding="utf-8")

        meta = {"etag": "abc", "last_modified": "Mon", "content_length": "8", "url": str(url)}
        save_metadata(file_path, meta)

        session = MagicMock()
        with patch("tarachat.scrape.fetch_remote_metadata", return_value=meta) as mock_fetch:
            result = await download_one(session, url, tmp_path)

        assert result[2] == "skipped"
        mock_fetch.assert_awaited_once_with(session, url)

    @pytest.mark.asyncio
    async def test_downloads_new_file(self, tmp_path):
        """When file doesn't exist, download it."""
        from yarl import URL

        url = URL("http://example.com/newfile.txt")
        session = MagicMock()

        # GET context manager
        get_resp = AsyncMock()
        get_resp.raise_for_status = MagicMock()
        session.get = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=get_resp),
                __aexit__=AsyncMock(return_value=False),
            )
        )

        with (
            patch("tarachat.scrape.fetch_remote_metadata", return_value={}),
            patch("tarachat.scrape._write_response") as mock_write,
        ):
            result = await download_one(session, url, tmp_path)

        assert result[2] == "downloaded"
        mock_write.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_error_on_failed_download(self, tmp_path):
        """When GET raises, return error status."""
        from yarl import URL

        url = URL("http://example.com/fail.txt")
        session = MagicMock()
        session.get = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(side_effect=Exception("timeout")),
                __aexit__=AsyncMock(return_value=False),
            )
        )

        with patch("tarachat.scrape.fetch_remote_metadata", return_value={}):
            result = await download_one(session, url, tmp_path)

        assert result[2] == "error"
        assert result[1] is None
