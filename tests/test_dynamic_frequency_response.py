"""
Tests for the dynamic_frequency_response_refactor branch (see
/home/wvanstra/.claude/plans/lovely-forging-mountain.md): makes H(nu,T) --
CyclicSolver.h_time_freq, the per-channel frequency response with no
Fourier transform applied to the subintegration axis -- the persistent
state FISTA steps and stores between iterations, transforming to
h(tau;omega)/h(tau;t) only where a regularizer specifically needs that
domain. Grouped here (rather than spread across the existing per-module
test files) so every test this refactor adds is easy to find in one place.
"""

import numpy as np

import pycyc

from .helpers import random_complex


def _minimal_cs(rng, nspec=4, nchan=8):
    """Bare CyclicSolver with just enough state for the h_time_delay <->
    h_time_freq / h_doppler_delay transform-wiring methods -- no cyclic
    spectra, profile, or gradient machinery needed for these."""
    CS = object.__new__(pycyc.CyclicSolver)
    CS.nspec = nspec
    CS.nchan = nchan
    CS.h_time_delay = random_complex(rng, (nspec, nchan))
    return CS


def test_sync_time_freq_from_time_delay_matches_direct_transform():
    rng = np.random.default_rng(101)
    CS = _minimal_cs(rng)
    expected = pycyc.time2freq(CS.h_time_delay, axis=1)

    CS._sync_time_freq_from_time_delay()

    np.testing.assert_array_equal(CS.h_time_freq, expected)


def test_sync_time_freq_round_trips_to_time_delay():
    rng = np.random.default_rng(102)
    CS = _minimal_cs(rng)
    original = CS.h_time_delay.copy()

    CS._sync_time_freq_from_time_delay()
    recovered = pycyc.freq2time(CS.h_time_freq, axis=1)

    np.testing.assert_allclose(recovered, original, rtol=1e-9, atol=1e-9)


def test_doppler_delay_view_matches_direct_transform():
    rng = np.random.default_rng(103)
    CS = _minimal_cs(rng)
    expected = pycyc.time2freq(CS.h_time_delay, axis=0)

    view = CS._doppler_delay_view()

    np.testing.assert_array_equal(view, expected)


def test_doppler_delay_view_round_trips_to_time_delay():
    rng = np.random.default_rng(104)
    CS = _minimal_cs(rng)
    original = CS.h_time_delay.copy()

    view = CS._doppler_delay_view()
    recovered = pycyc.freq2time(view, axis=0)

    np.testing.assert_allclose(recovered, original, rtol=1e-9, atol=1e-9)


def test_doppler_delay_view_is_not_persisted():
    """_doppler_delay_view() is documented as on-demand, not cached --
    mutating its return value must not silently change h_time_delay/
    h_time_freq unless the caller explicitly folds it back."""
    rng = np.random.default_rng(105)
    CS = _minimal_cs(rng)
    original = CS.h_time_delay.copy()

    view = CS._doppler_delay_view()
    view[0, 0] = 12345.0 + 6789.0j

    np.testing.assert_array_equal(CS.h_time_delay, original)
