import numpy as np
from scipy import signal
from amadeusgpt.implementation import AnimalBehaviorAnalysis


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


def get_events(event, kpt_names, min_dist=None, detrend=False):
    if event not in ("contact", "lift"):
        raise ValueError('`event` must be either "contact" or "lift".')

    coords = AnimalBehaviorAnalysis.get_keypoints().copy()
    coords -= AnimalBehaviorAnalysis.get_animal_centers()[:, None]
    events = []
    for kpt in kpt_names:
        x = coords[:, 0, AnimalBehaviorAnalysis.get_bodypart_index(kpt), 0]
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


def calc_stride_lengths(df, hoof_kpt_names, contacts):
    stride_lengths = []
    for hoof, contacts_ in zip(hoof_kpt_names, contacts):
        x = df.iloc[contacts_].loc(axis=1)[:, hoof, "x"].to_numpy().flatten()
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


def run_gait_analysis(limb_keypoint_names, min_dist=None) -> dict:
    contacts = get_events("contact", limb_keypoint_names, min_dist)
    lifts = get_events("lift", limb_keypoint_names, min_dist)
    stride_durations = calc_stride_durations(contacts)
    df = AnimalBehaviorAnalysis.get_dataframe()
    stride_lengths = calc_stride_lengths(df, limb_keypoint_names, contacts)
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


AnimalBehaviorAnalysis.run_gait_analysis = staticmethod(run_gait_analysis)
