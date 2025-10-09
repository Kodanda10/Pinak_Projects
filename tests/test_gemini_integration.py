from types import SimpleNamespace

import pytest

from pinak.integrations import gemini


class StubMemoryManager:
    def __init__(self, results=None, error: Exception | None = None):
        self._results = results or []
        self._error = error
        self.search_v2_called_with = None
        self.search_memory_called_with = None

    def search_v2(self, query, layers=None, k=5):
        if self._error:
            raise self._error
        self.search_v2_called_with = {
            "query": query,
            "layers": layers,
            "k": k,
        }
        return list(self._results)

    def search_memory(self, query, k=5):
        if self._error:
            raise self._error
        self.search_memory_called_with = {
            "query": query,
            "k": k,
        }
        return list(self._results)


class StubGenerativeModel:
    def __init__(self, *, should_fail: bool = False, stream_chunks=None):
        self.should_fail = should_fail
        self.stream_chunks = stream_chunks or []
        self.last_call = None

    def generate_content(self, prompt, generation_config=None, safety_settings=None, stream=False):
        self.last_call = {
            "prompt": prompt,
            "generation_config": generation_config,
            "safety_settings": safety_settings,
            "stream": stream,
        }
        if self.should_fail:
            raise RuntimeError("Gemini down")
        if stream:
            return self.stream_chunks
        return SimpleNamespace(text="Stub response")


def test_generate_response_builds_prompt_and_uses_layers():
    manager = StubMemoryManager(
        results=[{"content": "Important fact", "layer": "semantic", "score": 0.9}],
    )
    client = StubGenerativeModel()

    response = gemini.generate_response(
        "What is the fact?",
        layers=["semantic", "episodic"],
        manager=manager,
        temperature=0.5,
        top_p=0.6,
        max_output_tokens=256,
        client=client,
    )

    assert "Important fact" in response.prompt
    assert "Layer: semantic" in response.prompt
    assert "What is the fact?" in response.prompt
    assert response.text == "Stub response"

    assert client.last_call["stream"] is False
    assert client.last_call["generation_config"] == {
        "temperature": 0.5,
        "top_p": 0.6,
        "max_output_tokens": 256,
    }
    assert manager.search_v2_called_with == {
        "query": "What is the fact?",
        "layers": ["semantic", "episodic"],
        "k": 5,
    }


def test_generate_response_raises_integration_error_when_context_fails():
    manager = StubMemoryManager(error=RuntimeError("memory offline"))
    client = StubGenerativeModel()

    with pytest.raises(gemini.GeminiIntegrationError) as exc:
        gemini.generate_response("hi", layers=None, manager=manager, client=client)

    assert "Failed to retrieve context" in str(exc.value)


def test_stream_response_yields_chunks():
    manager = StubMemoryManager(results=[{"content": "context", "layer": "episodic"}])
    stream_chunks = [SimpleNamespace(text="Hello "), SimpleNamespace(text="world!")]
    client = StubGenerativeModel(stream_chunks=stream_chunks)

    stream_response = gemini.stream_response(
        "Say hello",
        layers=None,
        manager=manager,
        client=client,
    )

    assert client.last_call["stream"] is True
    collected = "".join(gemini.iter_text_from_stream(stream_response.stream))
    assert collected == "Hello world!"
