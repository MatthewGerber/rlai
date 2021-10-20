from rlai.models.feature_extraction import OneHotCategory


def test_one_hot_category():

    booleans = [True, False]
    ohc_1 = OneHotCategory(*booleans)
    assert str(ohc_1) == '_'.join(str(arg) for arg in booleans)

    ohc_2 = OneHotCategory(*booleans)
    assert ohc_1 == ohc_2
