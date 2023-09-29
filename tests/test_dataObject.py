import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from src.outputDataObject import outputDataUpdateObject


def test_exists():
    assert not outputDataUpdateObject.dummy() is None


def test_hasUnknowns():
    assert len(outputDataUpdateObject.dummy().unknowns) > 0
