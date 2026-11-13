"""Offline visual notes regressions: real PDF rendering, mock model transport."""
import io
import json
from types import SimpleNamespace

import pytest
from PIL import Image
from pypdf import PdfWriter

from echolingo.assistant.vision import VisualMaterials, encoded, valid_box, render_batch, ensure_vision
from echolingo.assistant.llm import AssistantError, ChatResult
from echolingo.assistant.prompts import MaterialText


class Client:
    provider = 'deepseek'
    model = 'deepseek-flash'

    def __init__(self, answers):
        self.answers = iter(answers)
        self.calls = []

    async def complete(self, messages, **kwargs):
        self.calls.append(messages)
        return ChatResult(next(self.answers), 'stop', {'prompt_tokens': 10}, 0)


class Events:
    def __init__(self):
        self.stages = []

    def progress(self, stage, detail):
        self.stages.append((stage, detail))


def test_pdf_batches_include_blank_scanned_pages_and_rotation(tmp_path):
    writer = PdfWriter()
    for _ in range(5):
        writer.add_blank_page(width=200, height=100)
    writer.pages[0].rotate(90)
    path = tmp_path / 'slides.pdf'
    writer.write(path)
    images, total = render_batch(path, 0)
    assert total == 5 and [n for n, _ in images] == [1, 2, 3, 4]
    assert images[0][1].height > images[0][1].width
    images, _ = render_batch(path, 4)
    assert [n for n, _ in images] == [5]


def test_boxes_reject_nonfinite_and_invalid_coordinates():
    for box in ([0, 0, float('nan'), 1], [-1, 0, 1, 1], [1, 1, 0, 0], [False, 0, 1, 1], [0, 0, .001, 1]):
        assert not valid_box(box)
    assert valid_box([0, 0, 1, 1])


@pytest.mark.asyncio
async def test_visual_image_read_crop_fallback_and_reference(tmp_path):
    path = tmp_path / 'handwriting.png'
    Image.new('RGB', (400, 200), 'white').save(path)
    answer = json.dumps({'pages': [{'page': 1, 'description': 'A model with arrows',
                                   'figures': [{'box': [.2, .2, .7, .7], 'caption': 'Model'}]}]})
    client = Client([answer, 'NO'])
    visual = VisualMaterials()
    docs = await visual.read(client, [{'id': 'a', 'name': path.name, 'path': str(path)}], Events(), {})
    assert docs[0].pages == 1 and 'arrows' in docs[0].text
    figure = visual.figures[0]
    assert figure['box'] == [0, 0, 1, 1]  # Incomplete crop uses whole page.
    messages = visual.augment([{'role': 'user', 'content': 'Write notes'}], [MaterialText(path.name, docs[0].text)])
    assert messages[-1]['content'][-1]['type'] == 'image_url'
    assert visual.used(f'![model](echolingo-asset:{figure["id"]})') == [figure]
    assert visual.used('![fake](echolingo-asset:00000000-0000-0000-0000-000000000000)') == []
    assert all(block.get('type') != 'file' for block in client.calls[0][0]['content'])


@pytest.mark.asyncio
async def test_missing_page_report_retries_then_fails(tmp_path):
    path = tmp_path / 'scan.png'
    Image.new('RGB', (40, 40)).save(path)
    client = Client(['{"pages": []}', '{"pages": []}'])
    with pytest.raises(AssistantError, match='page 1'):
        await VisualMaterials().read(client, [{'id': 'a', 'name': path.name, 'path': str(path)}], Events(), {})
    assert len(client.calls) == 2


@pytest.mark.asyncio
async def test_unknown_model_uses_synthetic_probe_and_known_text_model_refuses():
    client = Client(['red'])
    client.provider = 'custom_openai'
    await ensure_vision(client, {})
    assert client.calls[0][0]['content'][1]['image_url']['url'].startswith('data:image/jpeg;base64,')
    client.provider, client.model = 'deepseek', 'deepseek-v4-pro'
    with pytest.raises(AssistantError, match='does not support images'):
        await ensure_vision(client, {})
