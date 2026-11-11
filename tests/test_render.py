import os

from prefixanvil.ingest import Request
from prefixanvil.render import FIXTURE_TOKENIZER_DIR, Renderer, load_tokenizer, template_version


def req(messages, tools=None):
    return Request(0, 1, "m", messages, tools, None, None, None)


def test_fixture_tokenizer_offline_and_template_recorded(renderer):
    info = renderer.tok.template_info()
    assert info["tokenizer_is_fixture"] is True
    assert info["template_source"].startswith("tokenizer_bundled:")
    assert info["template_version"].startswith("sha256:")
    assert info["template_uses_strftime_now"] is False


def test_render_and_tokenize(renderer):
    r = renderer.render(req([{"role": "system", "content": "S"}, {"role": "user", "content": "hello"}],
                            [{"type": "function", "function": {"name": "f", "parameters": {}}}]))
    assert r.text.startswith("<|im_start|>system\nS")
    assert '"name": "f"' in r.text and r.text.endswith("<|im_start|>assistant\n")
    assert r.token_ids and all(isinstance(t, int) for t in r.token_ids)
    assert r.template_version == renderer.template_version


def test_append_only_prefix_property(renderer):
    m1 = [{"role": "system", "content": "S"}, {"role": "user", "content": "u1"}]
    m2 = m1 + [{"role": "assistant", "content": "a1"}, {"role": "user", "content": "u2"}]
    t1 = renderer.text(m1, None, True)
    t2 = renderer.text(m2, None, True)
    assert t2.startswith(t1)


def test_no_system_with_tools_and_content_parts(renderer):
    t = renderer.text([{"role": "user", "content": [{"type": "text", "text": "a"},
                                                    {"type": "image_url", "image_url": {}}]}],
                      [{"type": "function", "function": {"name": "g"}}], True)
    assert t.startswith("<|im_start|>system\n# Tools") and "a<image_url>" in t


def test_explicit_template_source(tmp_path):
    p = tmp_path / "tpl.jinja"
    p.write_text("{% for m in messages %}[{{ m.role }}]{{ m.content }}{% endfor %}")
    tok = load_tokenizer(FIXTURE_TOKENIZER_DIR, str(p))
    r = Renderer(tok)
    assert r.template_source == f"explicit:{os.path.abspath(p)}"
    assert r.template_version == template_version(p.read_text())
    assert r.text([{"role": "user", "content": "x"}], None, False) == "[user]x"
