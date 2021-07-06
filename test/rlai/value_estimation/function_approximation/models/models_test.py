from rlai.q_S_A.function_approximation.models.sklearn import SKLearnSGD


def test_model_pickle_no_coefs_dict():

    model = SKLearnSGD(False)

    assert hasattr(model, 'feature_action_coefficients') and model.__getstate__()['feature_action_coefficients'] is None
