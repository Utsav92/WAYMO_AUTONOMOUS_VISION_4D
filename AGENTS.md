# Notes for coding agents

- Never invent sensor values, labels or metrics. The synthetic fixture (`tests/synth.py`) is for tests / renderer
  self-test only and is flagged `synthetic: true`.
- Real Waymo column names must be confirmed with `datasets/inspect_components.py`; `waymo_adapter.find_col` matches
  by suffix and fails loudly with the available columns.
- TD gotchas (2025.33230): cross-COMP wires silently fail (use Select TOP); Text TOP position origin is top-left
  (y negative); Button COMP `click(val)`; channel-expression uniforms lag one bridge call; Script TOPs must be
  force-cooked; bridge result dicts containing a `name` key are mangled - return strings/lists.
- Run: `python touchdesigner/tools/tdrun.py <file.py>`; own TD = `TouchDesigner.exe touchdesigner\base.tox` (port 9995).
- Waymo licence: non-commercial only.
