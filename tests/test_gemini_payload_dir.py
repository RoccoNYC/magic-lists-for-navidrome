import os
import json
import asyncio
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from backend.services.ai_providers import AIProvider, get_payload_dir

def test_get_payload_dir_default(monkeypatch):
    """Test default fallback for payload directory when no env var is set."""
    for var in ["GEMINI_PAYLOAD_DIR", "GEMINI_RESPONSE_DIR", "AI_PAYLOAD_DIR", "AI_RESPONSE_DIR", "PAYLOAD_DIR"]:
        monkeypatch.delenv(var, raising=False)
    
    payload_dir = get_payload_dir()
    if os.path.exists("/app") and os.path.isdir("/app"):
        assert payload_dir == Path("/app/payloads")
    else:
        assert payload_dir == Path("payloads")

def test_get_payload_dir_env_vars(monkeypatch, tmp_path):
    """Test environment variable overrides for payload directory."""
    custom_path = str(tmp_path / "custom_payloads")
    
    # Test GEMINI_PAYLOAD_DIR
    monkeypatch.setenv("GEMINI_PAYLOAD_DIR", custom_path)
    assert get_payload_dir() == Path(custom_path)
    monkeypatch.delenv("GEMINI_PAYLOAD_DIR")
    
    # Test GEMINI_RESPONSE_DIR (legacy / alias)
    monkeypatch.setenv("GEMINI_RESPONSE_DIR", custom_path)
    assert get_payload_dir() == Path(custom_path)
    monkeypatch.delenv("GEMINI_RESPONSE_DIR")
    
    # Test AI_PAYLOAD_DIR
    monkeypatch.setenv("AI_PAYLOAD_DIR", custom_path)
    assert get_payload_dir() == Path(custom_path)
    monkeypatch.delenv("AI_PAYLOAD_DIR")
    
    # Test AI_RESPONSE_DIR
    monkeypatch.setenv("AI_RESPONSE_DIR", custom_path)
    assert get_payload_dir() == Path(custom_path)
    monkeypatch.delenv("AI_RESPONSE_DIR")
    
    # Test PAYLOAD_DIR
    monkeypatch.setenv("PAYLOAD_DIR", custom_path)
    assert get_payload_dir() == Path(custom_path)

def test_missing_directory_auto_created(monkeypatch, tmp_path):
    """Test that missing payload directory is automatically created."""
    target_dir = tmp_path / "nested" / "missing" / "payloads"
    assert not target_dir.exists()
    
    monkeypatch.setenv("GEMINI_PAYLOAD_DIR", str(target_dir))
    
    provider = AIProvider(
        provider_type="google",
        api_key="fake-key",
        model="gemini-2.5-flash",
        base_url="https://generativelanguage.googleapis.com/v1beta"
    )
    
    dummy_payload = {"contents": [{"parts": [{"text": "hello"}]}]}
    saved_file = provider._save_payload(dummy_payload, timestamp=123456)
    
    # a) missing output directory is auto-created
    assert target_dir.exists()
    assert target_dir.is_dir()
    
    # b) write succeeds after auto-create
    assert saved_file.exists()
    with open(saved_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == dummy_payload

def test_write_succeeds_when_directory_already_exists(monkeypatch, tmp_path):
    """Test writing when directory already exists."""
    target_dir = tmp_path / "existing_payloads"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    monkeypatch.setenv("GEMINI_PAYLOAD_DIR", str(target_dir))
    
    provider = AIProvider(
        provider_type="google",
        api_key="fake-key",
        model="gemini-2.5-flash",
        base_url="https://generativelanguage.googleapis.com/v1beta"
    )
    
    dummy_payload = {"test": "data", "count": 42}
    saved_file = provider._save_payload(dummy_payload, timestamp=999999)
    
    assert saved_file.exists()
    with open(saved_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == dummy_payload

def test_permission_or_path_failure_produces_actionable_error(monkeypatch, tmp_path):
    """Test that permission/path failure raises an actionable error with provider and path info."""
    blocker_file = tmp_path / "not_a_dir"
    blocker_file.write_text("blocking file")
    
    invalid_payload_dir = blocker_file / "sub_payloads"
    monkeypatch.setenv("GEMINI_PAYLOAD_DIR", str(invalid_payload_dir))
    
    provider = AIProvider(
        provider_type="google",
        api_key="fake-key",
        model="gemini-2.5-flash",
        base_url="https://generativelanguage.googleapis.com/v1beta"
    )
    
    dummy_payload = {"contents": []}
    with pytest.raises(IOError) as exc_info:
        provider._save_payload(dummy_payload, timestamp=111111)
    
    error_message = str(exc_info.value)
    assert "Google AI" in error_message
    assert str(invalid_payload_dir) in error_message

def test_permission_error_simulation(monkeypatch, tmp_path):
    """Test simulated PermissionError on directory creation or writing."""
    target_dir = tmp_path / "perm_payloads"
    monkeypatch.setenv("GEMINI_PAYLOAD_DIR", str(target_dir))
    
    provider = AIProvider(
        provider_type="google",
        api_key="fake-key",
        model="gemini-2.5-flash",
        base_url="https://generativelanguage.googleapis.com/v1beta"
    )
    
    dummy_payload = {"contents": []}
    with patch("pathlib.Path.mkdir", side_effect=PermissionError("Permission denied")):
        with pytest.raises(IOError) as exc_info:
            provider._save_payload(dummy_payload, timestamp=222222)
        
        error_message = str(exc_info.value)
        assert "Google AI" in error_message
        assert "Permission denied" in error_message
        assert str(target_dir) in error_message

def test_generate_google_payload_save_resilience(monkeypatch, tmp_path):
    """Test that _generate_google still functions even if payload saving fails."""
    target_dir = tmp_path / "resilience_payloads"
    monkeypatch.setenv("GEMINI_PAYLOAD_DIR", str(target_dir))
    
    provider = AIProvider(
        provider_type="google",
        api_key="fake-key",
        model="gemini-2.5-flash",
        base_url="https://generativelanguage.googleapis.com/v1beta"
    )
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "candidates": [{
            "content": {
                "parts": [{"text": '{"track_ids": [1, 2, 3]}'}]
            }
        }]
    }
    mock_response.raise_for_status = MagicMock()
    
    async def run_test():
        with patch.object(provider, "_save_payload", side_effect=IOError("Simulated write error")), \
             patch.object(provider.client, "post", return_value=mock_response):
            result = await provider._generate_google("system prompt", "user prompt")
            assert result == '{"track_ids": [1, 2, 3]}'

    asyncio.run(run_test())
