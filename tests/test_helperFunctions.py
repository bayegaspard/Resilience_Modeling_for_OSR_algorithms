import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/src")
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/src/ML_Model")
import helperFunctions
import Config


def test_seenModels():
    files = helperFunctions.get_saved_models()
    for file in files:
        assert os.path.exists(os.path.join("Saves", "models", file))


def test_loopOverUnknowns():
    unknowns_before_change = Config.parameters["unknowns_clss"][0]
    knowns_before_change = Config.parameters["knowns_clss"][0]
    assert len([x for x in unknowns_before_change if x in knowns_before_change]) == 0
    assert len([x for x in knowns_before_change if x in unknowns_before_change]) == 0
    new_unknowns = [1, 2, 3]
    Config.loopOverUnknowns(new_unknowns)
    unknowns_after_change = Config.parameters["unknowns_clss"][0]
    knowns_after_change = Config.parameters["knowns_clss"][0]
    assert len(unknowns_after_change) == len(new_unknowns)
    assert all([x == y for x, y in zip(new_unknowns, unknowns_after_change)])
    assert len([x for x in unknowns_after_change if x in knowns_after_change]) == 0
    assert len([x for x in knowns_after_change if x in unknowns_after_change]) == 0
