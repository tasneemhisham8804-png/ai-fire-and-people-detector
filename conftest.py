"""
Pytest fixture module, auto-loaded by pytest for every test in this project.

The test suite only exercises pure, model-free logic (geometry helpers like
_box_gap, classical computer-vision code like _localize_fire_regions) plus
HTTP-layer behavior in test_main.py with every model call monkeypatched out.
But the modules that logic lives in (FIRE_DETECTOR.py, PEOPLE_DETECTOR.py,
AI_DETECTOR.py, and MAIN.py which imports all three) import
tensorflow/ultralytics/torch/transformers at the top of the file. Without
those packages installed, the import itself would fail before a single test
could run — and installing multi-gigabyte ML frameworks just to satisfy an
import statement, when the tests never actually call into them for real, is
wasteful.

The fix: register fake, empty modules for the heavy dependencies in
sys.modules *before* the real modules get imported. Python's import system
checks sys.modules first and will happily reuse these stand-ins instead of
trying to import the real package, letting every detector module (and
MAIN.py, which pulls all of them in) import cleanly with no functional ML
backend behind them. This only works because the functions under test never
call the stubbed-out APIs (YOLO(), keras.models.load_model(), pipeline())
with real arguments expecting real results — in test_main.py, MAIN's own
model-loading globals and prediction functions are monkeypatched with fakes
instead.
"""
import sys
import types

for name in ["tensorflow", "ultralytics", "torch", "torch.nn", "torchvision",
             "torchvision.transforms", "torchvision.models", "transformers"]:
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)

# A few stubs need specific attributes defined (not just an empty module)
# because the real code touches them at import/module-load time:
# PEOPLE_DETECTOR.py calls YOLO(...) when load_people_model() runs,
# FIRE_DETECTOR.py references tf.keras.models.load_model as a fallback path,
# and AI_DETECTOR.py does `from transformers import pipeline` directly.
sys.modules["ultralytics"].YOLO = lambda *a, **k: None
sys.modules["tensorflow"].keras = types.SimpleNamespace(models=types.SimpleNamespace(load_model=lambda *a, **k: None))
sys.modules["transformers"].pipeline = lambda *a, **k: None
