import pytest

from sklearn.utils.estimator_checks import check_estimator

from sklearn.tdde51module import tdde51Estimator
from sklearn.tdde51module import tdde51Classifier
from sklearn.tdde51module import tdde51Transformer


@pytest.mark.parametrize(
    "estimator",
    [tdde51Estimator(), tdde51Transformer(), tdde51Classifier()]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
