#!/usr/bin/env python3
"""vbsm_verify.py: the structural half of rebase-vbsm.yml's "verify before publishing" step (lives on master; the step
runs it with `git show "$GITHUB_SHA:.github/scripts/vbsm_verify.py"`). It reads only trees and the blobs it needs, so a
blobless clone is fine.

  PARAM   a Params key named as a literal in a managed .py must be in C's openpilot/common/params_keys.h. The device
          ships a prebuilt libparams: any other key raises UnknownKeyName (a crash). Keys that upstream's own copy of
          the same file already names are not blamed on the fork.
  IMPORT  every absolute `from openpilot.X import name` in a managed .py, module level or inside a function, must
          resolve in C: the module exists and defines `name` (or has it as a submodule). pyflakes cannot see this; a
          lazy import inside a try/except fails silently at runtime (hold-to-restart after the device.py move), and a
          module-level one is a crash loop (a hud_renderer constant that augmented_road_view still imports).
  ASSET   every gui_app.texture("...png") / IconWidget("...png") path in a managed UI file must exist under
          openpilot/selfdrive/assets (upstream deleted icons_mici/experimental_mode.png; loading a missing texture
          divides by zero in the UI's image loader).
  INDEX   MODS.md's "Managed files (N)" table: N == len(manifest files), the table's file set == the manifest's, and
          every marker a row lists is present in that file.
  DEVPIN  modeld_v2/modeld.py: the DEV pin and the AM_POWER_LIMIT export are module-level statements that come
          before every import except os and math (tinygrad reads DEV at import; compile_modeld evaluates
          Device.DEFAULT at import; tinygrad reads AM_POWER_LIMIT once, at the first AMD boot).
  SEAMS   upstream lines outside the managed set that the fork's GPU logic depends on (the session gate, the retry
          scopes, the wedge watchdog's process match, the power cap). A change there stops the port for a review.
  RETRY   OnroadCycleRequested is set (any put form, any managed .py) only by upstream's line in modeld and by
          ui_watchdog's wedge path: upstream's retry stays the only onroad cycle (MODS.md s4).
  SUBS    hardwared and ui_watchdog read only SubMaster services they subscribe (the fork added selfdriveStateSP and
          modelV2 there): a lost subscription is a KeyError at the first cycle request or watchdog tick.

usage: python3 -I vbsm_verify.py <C> [repo]      exit 1 on any failure, one ::error:: line per finding
"""
import ast
import json
import re
import subprocess
import sys

C = sys.argv[1]
REPO = sys.argv[2] if len(sys.argv) > 2 else "."
PARAM_METHODS = {"get", "get_bool", "put", "put_bool", "put_nonblocking", "put_bool_nonblocking", "remove",
                 "check_key", "get_default_value", "get_type"}
MODELD = "openpilot/sunnypilot/modeld_v2/modeld.py"
WATCHDOG = "openpilot/sunnypilot/ui_watchdog.py"
HARDWARED = "openpilot/system/hardware/hardwared.py"
fail = False


def err(msg):
  global fail
  fail = True
  print(f"::error::{msg}")


def blob(path, commit=C):
  r = subprocess.run(["git", "-C", REPO, "cat-file", "blob", f"{commit}:{path}"], capture_output=True, text=True)
  return r.stdout if r.returncode == 0 else None


manifest = json.loads(blob(".github/vbsm-base.json"))
FILES, NEWBASE = manifest["files"], manifest["base"]
TREE = set(subprocess.run(["git", "-C", REPO, "ls-tree", "-r", "--name-only", C],
                          capture_output=True, text=True, check=True).stdout.splitlines())
KEYS = set(re.findall(r'\{"([A-Za-z0-9_]+)",', blob("openpilot/common/params_keys.h") or ""))
if not KEYS:
  err("no params keys parsed from openpilot/common/params_keys.h")


def params_receiver(n):
  if isinstance(n, ast.Name):
    return "param" in n.id.lower()
  if isinstance(n, ast.Attribute):
    return "param" in n.attr.lower()
  if isinstance(n, ast.Call):
    f = n.func
    return (isinstance(f, ast.Name) and f.id == "Params") or (isinstance(f, ast.Attribute) and f.attr == "Params")
  return False


def keys_used(tree):
  for n in ast.walk(tree):
    if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in PARAM_METHODS
        and params_receiver(n.func.value) and n.args and isinstance(n.args[0], ast.Constant)
        and isinstance(n.args[0].value, str) and re.fullmatch(r"[A-Za-z0-9_]+", n.args[0].value)):
      yield n.lineno, n.args[0].value


def top_names(code):
  """Names a module defines at top level (incl. inside top-level if/try/with); None = cannot tell (star import)."""
  try:
    body = ast.parse(code).body
  except SyntaxError:
    return None
  out, star = set(), False

  def walk(stmts):
    nonlocal star
    for n in stmts:
      if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        out.add(n.name)
      elif isinstance(n, ast.Assign):
        out.update(x.id for t in n.targets for x in ast.walk(t) if isinstance(x, ast.Name))
      elif isinstance(n, (ast.AnnAssign, ast.AugAssign)) and isinstance(n.target, ast.Name):
        out.add(n.target.id)
      elif isinstance(n, (ast.Import, ast.ImportFrom)):
        for a in n.names:
          star |= a.name == "*"
          out.add((a.asname or a.name).split(".")[0])
      elif isinstance(n, (ast.If, ast.Try, ast.With)):
        walk(n.body)
        for h in getattr(n, "handlers", []):
          walk(h.body)
        walk(getattr(n, "orelse", []))
        walk(getattr(n, "finalbody", []))
  walk(body)
  return None if star else out


def module_file(mod):
  base = mod.replace(".", "/")
  for cand in (base + ".py", base + "/__init__.py"):
    if cand in TREE:
      return cand
  if any(p.startswith(base + "/") for p in TREE):
    return ""          # namespace package: nothing to check names against
  leaf = base.rsplit("/", 1)[-1]
  if any(p.startswith(base) and p.rsplit("/", 1)[-1].split(".")[0] == leaf and p.endswith((".so", ".pyx", ".pyi"))
         for p in TREE):
    return ""          # compiled module: opaque
  return None


# ---- PARAM / IMPORT / ASSET ----
TEXTURE = re.compile(r"""(?:texture|IconWidget)\(\s*['"]([^'"]+\.png)['"]""")
for f in FILES:
  if not f.endswith(".py"):
    continue
  code = blob(f)
  try:
    tree = ast.parse(code or "")
  except SyntaxError as e:
    err(f"{f}: does not parse (line {e.lineno})")
    continue
  up = blob(f, NEWBASE)
  try:
    up_keys = {k for _, k in keys_used(ast.parse(up))} if up else set()
  except SyntaxError:
    up_keys = set()
  for line, k in keys_used(tree):
    if k not in KEYS and k not in up_keys:
      err(f"{f}:{line}: Params key '{k}' is not in params_keys.h (UnknownKeyName on the device)")
  for n in ast.walk(tree):
    if isinstance(n, ast.Import):   # `import openpilot.x.y as z`: the module itself must exist
      for a in n.names:
        if a.name.startswith("openpilot.") and module_file(a.name) is None:
          err(f"{f}:{n.lineno}: imports {a.name}, which is not in the tree")
      continue
    if not (isinstance(n, ast.ImportFrom) and n.level == 0 and (n.module or "").startswith("openpilot.")):
      continue
    mf = module_file(n.module)
    if mf is None:
      err(f"{f}:{n.lineno}: imports {n.module}, which is not in the tree")
      continue
    names = top_names(blob(mf) or "") if mf else None
    if names is None:
      continue
    for a in n.names:
      sub = n.module.replace(".", "/") + "/" + a.name
      if a.name not in names and sub + ".py" not in TREE and sub + "/__init__.py" not in TREE:
        err(f"{f}:{n.lineno}: {n.module} defines no '{a.name}'")
  if f.startswith("openpilot/selfdrive/ui/"):
    for p in sorted(set(TEXTURE.findall(code or ""))):
      if f"openpilot/selfdrive/assets/{p}" not in TREE:
        err(f"{f}: uses the missing asset {p}")

# ---- INDEX ----
mods = blob("MODS.md") or ""
sec = mods.split("## Managed files", 1)[1].split("\n## ", 1)[0] if "## Managed files" in mods else ""
m = re.match(r" \((\d+)\)", sec)
if not m or int(m.group(1)) != len(FILES):
  err(f"MODS.md 'Managed files ({m.group(1) if m else '?'})' != {len(FILES)} files in the manifest")
listed = set()
for row in sec.splitlines():
  if not row.startswith("| `"):
    continue
  cells = [c.strip() for c in row.strip().strip("|").split("|")]
  if len(cells) < 3:
    err(f"MODS.md managed-files row has {len(cells)} cells, expected 3: {row[:80]}")
    continue
  for f in re.findall(r"`([^`]+)`", cells[0]):
    listed.add(f)
    body = blob(f)
    if body is None:
      err(f"MODS.md lists {f}, which is not in the tree")
      continue
    for mk in re.findall(r"`([^`]+)`", cells[2]):
      if mk not in body:
        err(f"MODS.md lists marker {mk} for {f}, which has none")
for f in sorted(set(FILES) ^ listed):
  err(f"{f}: in the {'manifest but not MODS.md' if f in FILES else 'MODS.md table but not the manifest'}")

# ---- DEVPIN ----
md = blob(MODELD) or ""
try:
  body = ast.parse(md).body
except SyntaxError:
  body = []
def first(pred):
  return next((i for i, n in enumerate(body) if pred(ast.unparse(n))), None)
pin = first(lambda s: s == "os.environ.setdefault('DEV', 'QCOM')")
export = first(lambda s: "os.environ['AM_POWER_LIMIT'] =" in s and s.startswith("if "))
other = next((i for i, n in enumerate(body) if isinstance(n, (ast.Import, ast.ImportFrom)) and
              not (isinstance(n, ast.Import) and {a.name for a in n.names} <= {"os", "math"})), None)
if pin is None or export is None or other is None or not (pin < other and export < other):
  err(f"{MODELD}: the DEV pin (stmt {pin}) and the AM_POWER_LIMIT export (stmt {export}) must come before every "
      f"import except os/math (first other import: stmt {other})")

# ---- SEAMS (upstream files outside the managed set) ----
def seam(path, text, why, count=None):
  b = blob(path)
  n = (b or "").count(text)
  if b is None or (n == 0 if count is None else n != count):
    err(f"upstream seam changed: {path} has {n} x {text!r} ({why}); review the fork's GPU layer, then update this check")
PK = "openpilot/common/params_keys.h"
seam(PK, '{"ChestnutLoading", {CLEAR_ON_MANAGER_START | CLEAR_ON_OFFROAD_TRANSITION | CLEAR_ON_IGNITION_ON, BOOL}},',
     "modeld's session gate: ChestnutLoading set = an earlier modeld of this onroad session", 1)
seam(PK, '{"ChestnutActive", {CLEAR_ON_MANAGER_START | CLEAR_ON_OFFROAD_TRANSITION | CLEAR_ON_IGNITION_ON, BOOL}},',
     "per-session big-model verdict", 1)
seam(PK, '{"ChestnutRetry", {CLEAR_ON_MANAGER_START | CLEAR_ON_IGNITION_ON, BOOL}},', "one retry per ignition", 1)
seam(PK, '{"OnroadCycleRequested", {CLEAR_ON_MANAGER_START, BOOL}},', "the retry's onroad cycle", 1)
mgr = blob("openpilot/system/manager/manager.py") or ""
try:
  mgr_ifs = [n for n in ast.walk(ast.parse(mgr)) if isinstance(n, ast.If)]
except SyntaxError:
  mgr_ifs = []
if not any(ast.unparse(n.test) == "not started and started_prev" and
           "params.clear_all(ParamKeyFlag.CLEAR_ON_OFFROAD_TRANSITION)" in [ast.unparse(b) for b in n.body] for n in mgr_ifs):
  err("upstream seam changed: manager.py no longer clears CLEAR_ON_OFFROAD_TRANSITION on the falling edge of started "
      "(modeld's session gate depends on it)")
stock = blob("openpilot/selfdrive/modeld/modeld.py") or ""
if re.search(r"environ\[\s*['\"]AM_POWER_LIMIT['\"]\s*\]\s*=(?!=)", stock):
  err("upstream seam changed: stock modeld now ASSIGNS AM_POWER_LIMIT; it would override the fork's cap exported by "
      f"{MODELD} (it used to setdefault)")
pn = re.search(r'^PROCESS_NAME = "([^"]+)"', md, re.M)
wd = re.search(r'^MODELD_PROCTITLE = "([^"]+)"', blob(WATCHDOG) or "", re.M)
if not (pn and wd and pn.group(1) == wd.group(1)):
  err(f"{WATCHDOG} MODELD_PROCTITLE ({wd and wd.group(1)}) must equal {MODELD} PROCESS_NAME ({pn and pn.group(1)}): "
      "the wedge backstop matches the exact proctitle")

# ---- RETRY: upstream's onroad cycle keeps exactly its two sanctioned requesters (any put form, any managed file) ----
CYCLE_WRITERS = {MODELD: 1, WATCHDOG: 1}
writes = {}
for f in FILES:
  if not f.endswith(".py"):
    continue
  try:
    t = ast.parse(blob(f) or "")
  except SyntaxError:
    continue
  for n in ast.walk(t):
    if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr.startswith("put") and n.args
        and isinstance(n.args[0], ast.Constant) and n.args[0].value == "OnroadCycleRequested"
        and not (len(n.args) > 1 and isinstance(n.args[1], ast.Constant) and not n.args[1].value)):
      writes[f] = writes.get(f, 0) + 1
if writes != CYCLE_WRITERS:
  err(f"OnroadCycleRequested is set by {writes}, expected {CYCLE_WRITERS}: upstream's retry must stay the only cycle "
      "(MODS.md s4)")

# ---- SUBS: every SubMaster service the fork's hardwared / ui_watchdog code reads is subscribed ----
SM_ATTRS = {"alive", "updated", "seen", "valid", "freq_ok", "recv_frame", "recv_time", "logMonoTime"}
for f in (HARDWARED, WATCHDOG):
  try:
    t = ast.parse(blob(f) or "")
  except SyntaxError:
    continue
  subs = [n for n in ast.walk(t) if isinstance(n, ast.Call) and ast.unparse(n.func).split(".")[-1] == "SubMaster"]
  if len(subs) != 1 or not subs[0].args or not isinstance(subs[0].args[0], ast.List):
    err(f"{f}: expected one SubMaster([...]) with a literal service list, found {len(subs)}; update vbsm_verify's SUBS")
    continue
  have = {e.value for e in subs[0].args[0].elts if isinstance(e, ast.Constant)}
  for n in ast.walk(t):
    if not (isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str)):
      continue
    v = n.value
    if ((isinstance(v, ast.Name) and v.id == "sm") or (isinstance(v, ast.Attribute) and v.attr in SM_ATTRS and
                                                       isinstance(v.value, ast.Name) and v.value.id == "sm")) \
       and n.slice.value not in have:
      err(f"{f}:{n.lineno}: reads sm['{n.slice.value}'] but its SubMaster does not subscribe it (KeyError at runtime)")

sys.exit(1 if fail else 0)
