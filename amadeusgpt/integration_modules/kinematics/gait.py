from typing import List

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from scipy import signal

from amadeusgpt.programs.api_registry import register_integration_api


def find_peaks(x, min_dist=50, detrend=False):
    if detrend:
        x = signal.detrend(x[~np.isnan(x)])
    return signal.find_peaks(x, distance=min_dist)[0]


def next_pow_two(val):
    return 1 << (val - 1).bit_length()


def autocorr(x, axis=-1):
    """Compute the autocorrelation of a 1D time series."""
    x = np.atleast_1d(x)
    length = len(x)
    n = next_pow_two(length)
    f = np.fft.fft(x - np.mean(x, axis=axis), n=2 * n, axis=axis)
    acf = np.fft.ifft(f * np.conjugate(f), axis=axis)[:length].real
    acf /= acf[0]
    return acf


def get_events(analysis, event, kpt_names, min_dist=None, detrend=False):

    if event not in ("contact", "lift"):
        raise ValueError('`event` must be either "contact" or "lift".')

    coords = analysis.get_keypoints().copy()
    # center the keypoints
    coords -= np.nanmedian(coords, axis=2)[:, None]
    events = []
    for kpt in kpt_names:
        x = coords[:, 0, analysis.get_keypoint_names().index(kpt), 0]
        if event == "lift":
            x = -x
        if min_dist is None:  # Determine a signal's period from its autocorrelation
            corr = autocorr(x[~np.isnan(x)])
            start = np.argmin(corr)
            period = np.argmax(corr[start:]) + start
            min_dist = int(0.75 * period)  # To be safe
        peaks = find_peaks(x, min_dist, detrend)
        events.append(peaks)
    return events


def calc_stride_durations(contacts):
    return [np.diff(c) for c in contacts]


def calc_stride_lengths(analysis, keypoints, hoof_kpt_names, contacts):
    stride_lengths = []
    for hoof, contacts_ in zip(hoof_kpt_names, contacts):
        x = keypoints[
            contacts_, :, analysis.get_keypoint_names().index(hoof), 0
        ].flatten()
        stride_lengths.append(np.diff(x))
    return stride_lengths


def calc_duty_factors(contacts, lifts):
    duty = []
    for contacts_, lifts_ in zip(contacts, lifts):
        temp = []
        for c1, c2 in zip(contacts_[:-1], contacts_[1:]):  # Individual strides
            lift = lifts_[(lifts_ > c1) & (lifts_ < c2)]
            if not lift.size:
                continue
            d = (lift[0] - c1) / (c2 - c1)
            temp.append(d)
        duty.append(temp)
    return duty


def get_stances(contacts, lifts):
    stances = []
    for contacts_, lifts_ in zip(contacts, lifts):
        temp = set()
        for c1, c2 in zip(contacts_[:-1], contacts_[1:]):  # Individual strides
            lift = lifts_[(lifts_ > c1) & (lifts_ < c2)]
            if not lift.size:
                continue
            temp.add((c1, lift[0]))
        for l1, l2 in zip(lifts_[:-1], lifts_[1:]):
            contact = contacts_[(contacts_ > l1) & (contacts_ < l2)]
            if not contact.size:
                continue
            temp.add((contact[0], l2))
        stances.append(list(temp))
    return stances


@register_integration_api
def run_gait_analysis(self, limb_keypoint_names: List[str]) -> dict:
    """
    This function computse an animal's gait parameters given a list of distal keypoints.
    Parameters
    ----------
    limb_keypoint_names: List[str], list of the names of the distal keypoints. Need to be at least 2 keypoints.
    """
    min_dist = None
    contacts = get_events(self, "contact", limb_keypoint_names, min_dist)
    lifts = get_events(self, "lift", limb_keypoint_names, min_dist)
    stride_durations = calc_stride_durations(contacts)
    keypoints = self.get_keypoints()
    stride_lengths = calc_stride_lengths(self, keypoints, limb_keypoint_names, contacts)
    duty_factors = calc_duty_factors(contacts, lifts)
    stances = get_stances(contacts, lifts)
    return {
        "contacts": contacts,
        "lifts": lifts,
        "stride_durations": stride_durations,
        "stride_lengths": stride_lengths,
        "duty_factors": duty_factors,
        "stances": stances,
    }


def _make_line_collection(
    coords, links, start=0, end=-1, inds=None, color_stance="plum", alpha=0.5
):
    color = mcolors.to_rgb("gray")
    colors = np.array([color] * len(coords))
    if inds is not None:
        mask = np.zeros(coords.shape[0], dtype=bool)
        for ind1, ind2 in inds:
            mask[ind1 : ind2 + 1] = True
        colors[mask] = mcolors.to_rgb(color_stance)
    sl = slice(start, end)
    colors = colors[sl]

    segs = coords[sl, links].reshape((-1, 2, 2))
    colors = np.repeat(colors, len(links), axis=0)
    coll = LineCollection(segs, colors=colors, alpha=alpha)
    return coll, segs


@register_integration_api
def plot_gait_analysis_results(
    self, gait_analysis_results, limb_keypoints, color_stance="plum"
):
    """
    This function plots the gait analysis results returned from the `run_gait_analysis` function.
    Parameters
    ----------
    gait_analysis_results: dict, the results from the `run_gait_analysis` function
    limb_keypoints: List[str], list of the names of the distal keypoints. Need to be at least 2 keypoints.
    color_stance: str, optional, default to be "plum"
    """
    fig, ax = plt.subplots(sharex=True, sharey=True)
    coords = self.get_keypoints()[:, 0]

    if gait_analysis_results["stances"] == [[]]:
        return fig, ax
    stance_inds = gait_analysis_results["stances"][0]

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    skeleton = []
    for kpt1, kpt2 in zip(limb_keypoints, limb_keypoints[1:]):
        skeleton.append(
            (
                self.get_keypoint_names().index(kpt1),
                self.get_keypoint_names().index(kpt2),
            ),
        )
    coll, segs = _make_line_collection(
        coords, skeleton, inds=stance_inds, color_stance=color_stance
    )
    xmin, ymin = np.nanmin(segs, axis=(0, 1))
    xmax, ymax = np.nanmax(segs, axis=(0, 1))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.invert_yaxis()
    ax.add_collection(coll)
    ax.set_ylabel("Limb")
    return fig, ax
