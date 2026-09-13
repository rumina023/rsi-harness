from staged import OllamaClient, ROOT
from pathlib import Path
import tempfile

with tempfile.TemporaryDirectory() as d:
    result=OllamaClient(Path(d)).call('qwen-smoke', {'task':'Return a valid baseline policy. Use policy keys explore, flip, temperature, restart, guidance.'})
    print(result)
