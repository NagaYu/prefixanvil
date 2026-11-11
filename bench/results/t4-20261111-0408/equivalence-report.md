# PrefixAnvil equivalence report

- result: **PASSED**
- build: `sha256:c5b87f48c3860fde00018139` (prefixanvil 0.2.0)
- suite: `/kaggle/working/pf/prefixanvil/fixtures/equiv` (15 cases, `sha256:36401b83d14b1a6bbc05c765`)
- upstream: mock (`prefixanvil.mock-upstream/v1` — deterministic mock, NOT a real model; real-model equivalence is a Phase 3 measurement)

| mode | cases | requests | exact match | rate |
|---|---:|---:|---:|---:|
| L1 | 15 | 19 | 19 | 100.0% |
| L2 | 6 | 6 | 5 | 83.3% |

| case | mode | expect | outcome | ok |
|---|---|---|---|---|
| `content_parts_system` | L1 | match | match | ✓ |
| `content_parts_system` | L2 | match | match | ✓ |
| `dict_arguments` | L1 | match | match | ✓ |
| `duplicate_tool_names` | L1 | match | match | ✓ |
| `json_tool_content_pretty` | L1 | match | match | ✓ |
| `l2_volatile_in_code_block` | L1 | match | match | ✓ |
| `l2_volatile_in_code_block` | L2 | match | match | ✓ |
| `l2_volatile_lines` | L1 | match | match | ✓ |
| `l2_volatile_lines` | L2 | allowed-diff | diff | ✓ |
| `l2_volatile_lookalikes` | L1 | match | match | ✓ |
| `l2_volatile_lookalikes` | L2 | match | match | ✓ |
| `nested_json_string_argument` | L1 | match | match | ✓ |
| `non_json_tool_content` | L1 | match | match | ✓ |
| `stateless_no_session` | L1 | match | match | ✓ |
| `system_whitespace_drift` | L1 | match | match | ✓ |
| `system_whitespace_drift` | L2 | match | match | ✓ |
| `tool_set_change_mid_session` | L1 | match | match | ✓ |
| `tools_key_order` | L1 | match | match | ✓ |
| `unicode_escapes_in_arguments` | L1 | match | match | ✓ |
| `ws_code_block` | L1 | match | match | ✓ |
| `ws_code_block` | L2 | match | match | ✓ |

## Allowed diffs

- `l2_volatile_lines` (L2): L2 relocation changes line order inside the system message, which the mock's equivalence model treats as meaningful by design; whether a real model's output is unaffected must be measured on real models (Phase 3).
