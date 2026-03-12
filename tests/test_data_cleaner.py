import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from julia.data.data_cleaner import clean_number, clean_crops_data, OUTPUT_FILE, run_pipeline

def test_clean_number():
    assert clean_number("100.50") == 100.5
    assert clean_number(" 20,000.00 ") == 20000.0
    assert clean_number("$1,000") == 1000.0
    assert clean_number("") == 0.0
    assert clean_number(pd.NA) == 0.0
    assert clean_number(150) == 150.0

@pytest.mark.skipif(not (Path(__file__).parent.parent / "data" / "pr_crops_with_stations.csv").exists(), 
                    reason="Data files not present on build server")
def test_clean_crops_data_shape():
    df = clean_crops_data()
    # We expect some rows dropped (Mataderos etc.)
    # Original is ~502, should be < 500
    assert len(df) > 100 
    assert len(df) <= 502
    
    # Check new columns were created
    assert "planted_ha" in df.columns
    assert "harvested_ha" in df.columns
    assert "harvest_ratio" in df.columns
    assert "crop_base" in df.columns
    assert "growing_method" in df.columns
    
    # Invalid rows dropped
    assert "Mataderos" not in df["Cultivo"].values
    
    # Ratio bounded
    assert df["harvest_ratio"].max() <= 1.0
    assert df["harvest_ratio"].min() >= 0.0

@pytest.mark.skipif(not (Path(__file__).parent.parent / "data" / "pr_crops_with_stations.csv").exists(), 
                    reason="Data files not present on build server")
def test_full_pipeline_run():
    # Attempt to run full pipeline
    df = run_pipeline()
    
    assert Path(OUTPUT_FILE).exists()
    assert not df.empty
    
    # Ensure standard names
    assert "year" in df.columns
    assert "municipality" in df.columns
    assert "barrio" in df.columns
    
    # At least one joined column should be present or the left join succeeded cleanly
    if "is_strong_event" in df.columns:
        assert df["is_strong_event"].isin([0, 1]).all() or df["is_strong_event"].isna().all()
