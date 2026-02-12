import pytest
import numpy as np
from phase.main.timeseries import Timeseries
from tests.data import TIMELAPSE


def test_loading():
    ts = Timeseries.from_directory("test", TIMELAPSE)
    # tests if every image was loaded
    assert len(ts.frames) == len(list(TIMELAPSE.iterdir())), "No / not all images loaded"

@pytest.mark.slow
def test_populating(n=3):
    ts = Timeseries.from_directory("test", TIMELAPSE)
    ts.populate_timeseries()

    # checking dishes/stencils init correctly
    stencils = ts.frames[0].dishes
    assert len(stencils) > 0, "No dishes detected in first frame"

    # checking if stencil attributes init correctly
    for stencil in stencils:
        assert stencil.centroid is not None, "Stencil attribute error"
        assert stencil.radius > 0, "Stencil attribute error"
        assert stencil.crop is not None, "Stencil crop error"

    # checking correct stencil propagation
    for frame in ts.frames[1:n]:  # testing first n frames
        assert len(frame.dishes) == len(stencils), "Stencil dish count mismatch"

        # checking if the stencil and subsequent dishes share same geometry but different image crop
        for dish, stencil in zip(frame.dishes, stencils):
            assert dish.centroid == stencil.centroid, "Stencil dish geometry mismatch"
            assert dish.radius == stencil.radius, "Stencil dish geometry mismatch"
            assert not np.array_equal(dish.crop, stencil.crop), "Stencil and dish image identical"


@pytest.mark.slow
def test_preprocessing(n=3):
    ts = Timeseries.from_directory("test", TIMELAPSE)
    ts.populate_timeseries()

    # checks if preprocessing works correctly
    ts.preprocess_timeseries()
    
    # checks if masks have been initialised and if they match the dishes
    assert ts.fg_masks is not None, "fg_mask empty"
    assert ts.bg_masks is not None, "bg_mask empty"
    assert len(ts.fg_masks) == len(ts.frames[0].dishes), "mask length mismatch"
    assert len(ts.bg_masks) == len(ts.frames[0].dishes), "mask length mismatch"

    for frame in ts.frames[1:n]:
        for dish in frame.dishes:
            assert dish.preprocessed is not None, "Preprocessed image empty"
