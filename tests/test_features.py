from src.pipeline.preprocessor import get_features, get_preprocessor


def test_column_definitions():
    num_cols, ord_cols, nom_cols = get_features()

    assert len(num_cols) == 11
    assert len(ord_cols) == 4
    assert len(nom_cols) == 6

    # Verify critical columns exist
    assert "Preparation_Time_Min" in num_cols
    assert "Traffic_Level" in ord_cols
    assert "Cuisine_Type" in nom_cols


def test_preprocessor_structure():
    num_cols, ord_cols, nom_cols = get_features()
    transformer = get_preprocessor(num_cols, ord_cols, nom_cols)

    transformer_names = [name for name, _, _ in transformer.transformers]
    assert "Numerical Pipeline" in transformer_names
    assert "Ordinal Pipeline" in transformer_names
    assert "Nominal Pipeline" in transformer_names