import pytest
import os
from src.train import main

def test_pipeline_runs():
    try:
        main()
        assert os.path.exists("mlruns"), "El directorio mlruns no fue creado"
    except SystemExit:
        pytest.fail("El pipeline falló durante la ejecución")