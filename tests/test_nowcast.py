from src.nowcaster import run_gdp_nowcast_model

def test_nowcast():
    results = run_gdp_nowcast_model(5.2, 4.3)
    assert len(results) == 6
    res27 = results[-1]
    assert 4.2 <= res27.nowcast_point_pct <= 5.2
    assert res27.within_official_range is True
