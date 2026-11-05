"""Tests for refreshing a library item from its source and MusicBrainz."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

from music_assistant_models.media_items import Work

from music_assistant.controllers.music import MusicController


async def test_refresh_of_a_work_looks_up_its_whole_tree_again() -> None:
    """Refreshing a work reads its work tree from MusicBrainz again, and nothing else."""
    work = Work(item_id="3", provider="library", name="Work", provider_mappings=set())
    mass = Mock()
    mass.metadata.refresh_work_tree = AsyncMock(return_value=work)
    music = MusicController.__new__(MusicController)
    music.mass = mass

    with patch.object(music, "get_controller", return_value=AsyncMock()) as controller:
        assert await music.refresh_item(work) is work

    mass.metadata.refresh_work_tree.assert_awaited_once_with(work)
    controller.return_value.get_provider_item.assert_not_awaited()
