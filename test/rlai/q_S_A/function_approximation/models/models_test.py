from rlai.gpi.state_action_value.function_approximation.models.sklearn import SKLearnSGD


def test_model_pickle_no_coefs_dict():
    """
    Test.
    """

    model = SKLearnSGD(False)

    assert hasattr(model, 'feature_action_coefficients') and model.__getstate__()['feature_action_coefficients'] is None
