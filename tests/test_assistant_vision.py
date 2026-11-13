"""Offline visual notes regressions: real PDF rendering, mock model transport."""
import io
import json
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image
from pypdf import PdfWriter

from echolingo.assistant.vision import VisualMaterials, encoded, valid_box, render_batch, ensure_vision
from echolingo.assistant.llm import AssistantError, ChatClient, ChatResult, LlmEndpoint
from echolingo.assistant.prompts import MaterialText


class Client:
    provider = 'deepseek'
    model = 'deepseek-flash'

    def __init__(self, answers):
        self.answers = iter(answers)
        self.calls = []

    async def complete(self, messages, *, model=None, max_tokens=1024, temperature=0.2):
        self.calls.append(messages)
        return ChatResult(next(self.answers), 'stop', {'prompt_tokens': 10}, 0)


class Events:
    def __init__(self):
        self.stages = []

    def progress(self, stage, detail):
        self.stages.append((stage, detail))


@pytest.mark.parametrize('model', ['deepseek-flash', 'custom-vision-model'])
async def test_real_chat_client_reads_pdf_batches_and_checks_crops(tmp_path, model):
    """Keep the production client at the visual caller seam; no permissive fake.

    The unknown model also exercises the synthetic capability probe. Only the
    HTTP transport is mocked, so request argument mismatches cannot be hidden.
    """
    path = tmp_path / 'slides.pdf'
    writer = PdfWriter()
    for _ in range(5):
        writer.add_blank_page(width=200, height=100)
    writer.write(path)
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        assert body['thinking'] == {'type': 'disabled'}
        assert body['stream'] is False
        content = body['messages'][0]['content']
        if content[0]['text'].startswith('Name the color'):
            text = 'red'
        elif content[0]['text'].startswith('Reference data: first image'):
            assert sum(block['type'] == 'image_url' for block in content) == 2
            text = 'YES'
        else:
            numbers = [int(block['text'].splitlines()[0].split()[1])
                       for block in content[1:] if block['type'] == 'text']
            assert sum(block['type'] == 'image_url' for block in content) == len(numbers)
            text = json.dumps({'pages': [
                {'page': n, 'description': f'Model diagram on page {n}',
                 'figures': [{'box': [.2, .2, .8, .8], 'caption': 'Model'}] if n == 1 else []}
                for n in numbers]})
        return httpx.Response(200, json={
            'choices': [{'message': {'content': text}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 10},
        })

    endpoint = LlmEndpoint('deepseek', 'DeepSeek', 'https://api.deepseek.com', model, 'test-key')
    usage = {}
    visual = VisualMaterials()
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        client = ChatClient(endpoint, client=http)
        docs = await visual.read(
            client, [{'id': 'a', 'name': path.name, 'path': str(path)}], Events(), usage)
    assert len(requests) == (3 if model == 'deepseek-flash' else 4)
    assert usage['prompt_tokens'] == len(requests) * 10
    assert docs[0].visual_pages == 5
    assert [segment.number for segment in docs[0].segments] == [1, 2, 3, 4, 5]
    assert len(visual.figures) == 1
    assert visual.figures[0]['box'] == pytest.approx([.18, .18, .82, .82])


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
