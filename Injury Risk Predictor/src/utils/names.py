def extract_feature_names(preprocessor):
    num_cols = preprocessor.transformers_[0][2]
    cat_cols = preprocessor.transformers_[1][2]

    ohe = preprocessor.named_transformers_["cat"]

    # case: no categorical features
    try:
        ohe_names = ohe.get_feature_names_out(cat_cols).tolist()
    except:
        ohe_names = []

    return num_cols + ohe_names