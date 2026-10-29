"""Tests for the one-off full re-read that picks up the classical tags of unchanged files."""

from __future__ import annotations

import asyncio
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from music_assistant_models.enums import MediaType

from music_assistant.providers.filesystem_local import LocalFileSystemProvider
from music_assistant.providers.filesystem_local.constants import (
    CONF_CLASSICAL_REPARSE_DONE,
    CONF_ENTRY_LIBRARY_SYNC_PLAYLISTS,
    CONF_ENTRY_LIBRARY_SYNC_TRACKS,
)
from music_assistant.providers.filesystem_local.cue import cue_metadata_checksum
from music_assistant.providers.filesystem_local.helpers import FileSystemItem

TRACK_FILE = "Artist/Album/track.flac"
INSTANCE_ID = "filesystem_local--test"


def _create_provider(reparse_done: bool, sync_tracks: bool = True) -> LocalFileSystemProvider:
    """
    Create a music provider whose library holds one unchanged track file.

    :param reparse_done: Whether the one-off full re-read already completed.
    :param sync_tracks: Whether the tracks sync checkbox is enabled.
    """
    config_values = {
        CONF_ENTRY_LIBRARY_SYNC_TRACKS.key: sync_tracks,
        CONF_ENTRY_LIBRARY_SYNC_PLAYLISTS.key: True,
    }
    with patch.object(LocalFileSystemProvider, "__init__", lambda *_a, **_kw: None):
        provider = LocalFileSystemProvider.__new__(LocalFileSystemProvider)
    provider.config = MagicMock(instance_id=INSTANCE_ID)
    provider.config.get_value = MagicMock(side_effect=config_values.get)
    provider.media_content_type = "music"
    provider.sync_running = False
    provider.logger = MagicMock()
    provider.mass = MagicMock()
    provider.mass.create_task = lambda coro, **_kwargs: asyncio.ensure_future(coro)
    provider.mass.config.get_raw_provider_config_value = MagicMock(
        side_effect=lambda _instance, key, default=None: (
            reparse_done if key == CONF_CLASSICAL_REPARSE_DONE else default
        )
    )
    provider.mass.music.database.get_rows_from_query = AsyncMock(
        return_value=[{"provider_item_id": TRACK_FILE, "details": "1"}]
    )
    provider._process_item_async = AsyncMock(return_value=True)  # type: ignore[method-assign]
    provider._process_deletions = AsyncMock()  # type: ignore[method-assign]
    provider._process_orphaned_albums_and_artists = AsyncMock()  # type: ignore[method-assign]
    provider._prune_all_missing_local_refs = AsyncMock(return_value=False)  # type: ignore[method-assign]
    provider._set_available = MagicMock()  # type: ignore[method-assign]
    provider._enumerate_files_for_sync = _enumerate_unchanged_track(provider)  # type: ignore[method-assign]
    return provider


def _enumerate_unchanged_track(provider: LocalFileSystemProvider, failed_dirs: int = 0) -> Any:
    """Build a scan stub that finds the indexed track file with its checksum unchanged."""

    async def _enumerate(**kwargs: Any) -> None:
        kwargs["scan_errors"].failed_dirs = failed_dirs
        provider._classify_scan_item(
            FileSystemItem(
                filename="track.flac",
                relative_path=TRACK_FILE,
                absolute_path=f"/media/{TRACK_FILE}",
                is_dir=False,
                checksum="1",
            ),
            file_checksums=kwargs["file_checksums"],
            cue_file_checksums=kwargs["cue_file_checksums"],
            cur_filenames=kwargs["cur_filenames"],
            items_to_process=kwargs["items_to_process"],
            unchanged_cue_items=kwargs["unchanged_cue_items"],
            cue_stems=kwargs["cue_stems"],
            ignore_album_playlists=False,
            metadata_files=kwargs["metadata_files"],
        )

    return _enumerate


def _reparse_done_writes(provider: LocalFileSystemProvider) -> list[Any]:
    """Return the values written to the classical re-read flag."""
    set_value = cast("MagicMock", provider.mass.config.set_raw_provider_config_value)
    return [
        call.args[2]
        for call in set_value.call_args_list
        if call.args[1] == CONF_CLASSICAL_REPARSE_DONE
    ]


@pytest.mark.parametrize(
    ("reparse_done", "sync_tracks", "reads", "flag_writes"),
    [
        (False, True, 1, [True]),
        (True, True, 0, []),
        (False, False, 0, []),
    ],
    ids=["first sync re-reads", "skipped after the re-read", "waits for track sync"],
)
async def test_unchanged_files_are_read_again_once(
    reparse_done: bool, sync_tracks: bool, reads: int, flag_writes: list[bool]
) -> None:
    """
    The first sync with track sync on re-reads unchanged files and then marks the re-read as done.

    Once the re-read is done an unchanged file is not read again. Without track sync no re-read
    happens, so it stays pending.
    """
    provider = _create_provider(reparse_done=reparse_done, sync_tracks=sync_tracks)

    await provider.sync_library(MediaType.TRACK)

    assert provider._process_item_async.await_count == reads  # type: ignore[attr-defined]
    assert _reparse_done_writes(provider) == flag_writes
    assert not provider._force_full_reparse


async def test_re_read_is_repeated_after_an_incomplete_scan() -> None:
    """A scan that skipped folders leaves the re-read pending for the next sync."""
    provider = _create_provider(reparse_done=False)
    provider._enumerate_files_for_sync = _enumerate_unchanged_track(  # type: ignore[method-assign]
        provider, failed_dirs=1
    )

    await provider.sync_library(MediaType.TRACK)

    provider._process_item_async.assert_awaited_once()  # type: ignore[attr-defined]
    assert _reparse_done_writes(provider) == []


@pytest.mark.parametrize("filename", ["track.flac", "album.cue", "playlist.m3u"])
def test_the_re_read_covers_only_unchanged_track_files(filename: str) -> None:
    """An unchanged CUE sheet or playlist is not read again, it carries no classical details."""
    provider = _create_provider(reparse_done=False)
    provider._force_full_reparse = True
    provider._sync_tracks = provider._sync_playlists = True
    path = f"Artist/Album/{filename}"
    is_cue = filename.endswith(".cue")
    items_to_process: list[Any] = []
    provider._classify_scan_item(
        FileSystemItem(
            filename=filename,
            relative_path=path,
            absolute_path=f"/media/{path}",
            is_dir=False,
            checksum="1",
        ),
        file_checksums={} if is_cue else {path: "1"},
        cue_file_checksums={path: {cue_metadata_checksum("1")}} if is_cue else {},
        cur_filenames=set(),
        items_to_process=items_to_process,
        unchanged_cue_items=[],
        cue_stems=set(),
        ignore_album_playlists=False,
        metadata_files=[],
    )

    assert len(items_to_process) == int(filename == "track.flac")
