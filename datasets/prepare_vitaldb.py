"""Preparation of the open VitalDB dataset (https://vitaldb.net) for the ClusterBP v2 pipeline.

VitalDB is open intraoperative monitoring of ~6,400 patients, available without registration. It is used here to test
the protocol and the leakage audit on data any reviewer can download; the clinical conclusions of the paper are not tested on it.

The script writes to cache_v2_vitaldb/ the same files the main notebook builds from the operator's export:
- measurements.parquet — pid, td, sbp, dbp, hr (td is time in MINUTES from the start of the recording; per-minute medians);
- patients.parquet — pid, birth_days, bmi, group (birth_days is set so that age in the features is correct);
- events.parquet — pid, td, code: 90001 — minute with mean arterial pressure < 65 mm Hg (hypotension),
  90002 — minute with SBP ≥ 160 mm Hg (hypertension).
Then clusterbp_v2.ipynb and clusterbp_v2_leakage.ipynb run unchanged with
CLUSTERBP_PRESET=vitaldb CLUSTERBP_TAG=vitaldb: all durations are read in minutes (14-min history window, 3-min stride),
targets are an episode in a 5-min horizon after a 5-min gap (A — hypotension, B — hypertension; OLD — hypotension without gap).

As in the operator's data, events are defined by a threshold on the same signal as the input, so the gap is required:
without it the target is partly computed from the last minutes of the history window.

Usage (requires internet access and the vitaldb package: pip install vitaldb):
    python datasets/prepare_vitaldb.py --cases 600            # 600 eligible operations
    CLUSTERBP_PRESET=vitaldb CLUSTERBP_TAG=vitaldb jupyter nbconvert --execute --to notebook clusterbp_v2.ipynb
    CLUSTERBP_PRESET=vitaldb CLUSTERBP_TAG=vitaldb jupyter nbconvert --execute --to notebook clusterbp_v2_leakage.ipynb
The other notebooks read the operator's reference files (regions, therapy) and are not run on VitalDB.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

# Numeric trends of the Solar 8000 monitor: invasive arterial BP and heart rate, 2-s interval
TRACKS = ['Solar8000/ART_SBP', 'Solar8000/ART_DBP', 'Solar8000/ART_MBP', 'Solar8000/HR']
CASES_URL = 'https://api.vitaldb.net/cases'
HYPO_CODE, HYPER_CODE = 90001, 90002
INTERVAL_S = 2


def clean_minutes(raw, interval_s=INTERVAL_S):
    """Array [n × 4] (SBP, DBP, MBP, HR) sampled every interval_s → per-minute medians with plausibility checks.

    Values outside physiological bounds (line flushing artefacts, disconnection) are removed before averaging.
    Returns a DataFrame: minute, sbp, dbp, mbp, hr (only minutes with SBP, DBP and HR present).
    """
    df = pd.DataFrame(raw, columns=['sbp', 'dbp', 'mbp', 'hr'], dtype=float)
    ok = df.sbp.between(40, 250) & df.dbp.between(20, 150) & df.mbp.between(30, 200) & (df.sbp > df.dbp)
    df.loc[~ok, ['sbp', 'dbp', 'mbp']] = np.nan
    df.loc[~df.hr.between(30, 220), 'hr'] = np.nan
    df['minute'] = (np.arange(len(df)) * interval_s) // 60
    m = df.groupby('minute').median().dropna(subset=['sbp', 'dbp', 'hr']).reset_index()
    return m


def convert(cases, clinical):
    """cases: dict caseid -> DataFrame from clean_minutes; clinical: the api.vitaldb.net/cases table.

    Returns (measurements, patients, events) in the pipeline's format.
    """
    clin = clinical.set_index('caseid')
    meas, pats, evs = [], [], []
    for cid, m in cases.items():
        if len(m) < 60:                      # less than an hour of recording: no windows can be built
            continue
        td = m['minute'].values.astype(float)
        meas.append(pd.DataFrame({'pid': cid, 'td': td, 'sbp': m.sbp.values, 'dbp': m.dbp.values, 'hr': m.hr.values}))
        mbp = m['mbp'].values
        for code, mask in [(HYPO_CODE, mbp < 65), (HYPER_CODE, m.sbp.values >= 160)]:
            mask = np.nan_to_num(mask, nan=0).astype(bool)
            evs.append(pd.DataFrame({'pid': cid, 'td': td[mask], 'code': code}))
        age = pd.to_numeric(clin.at[cid, 'age'], errors='coerce') if cid in clin.index else np.nan
        bmi = pd.to_numeric(clin.at[cid, 'bmi'], errors='coerce') if cid in clin.index else np.nan
        # age in the features = (cut − birth_days) / 365.25; time is in minutes, so the drift during an operation is negligible
        pats.append({'pid': cid, 'birth_days': td[0] - age * 365.25 if np.isfinite(age) else np.nan,
                     'bmi': bmi if 12 <= bmi <= 60 else np.nan, 'group': 'undefined'})
    meas = pd.concat(meas, ignore_index=True).sort_values(['pid', 'td']).reset_index(drop=True)
    meas['pid'] = meas['pid'].astype('int64')
    for c in ['sbp', 'dbp', 'hr']:
        meas[c] = meas[c].astype('float32')
    events = pd.concat(evs, ignore_index=True)
    events['pid'] = events['pid'].astype('int64')
    return meas, pd.DataFrame(pats), events


def download(n_cases, seed=42):
    import vitaldb   # pip install vitaldb (this file is named prepare_vitaldb.py so that it does not shadow the package)
    if not hasattr(vitaldb, 'find_cases'):
        raise ImportError(f'Imported {vitaldb.__file__} instead of the vitaldb package: install it (pip install vitaldb) '
                          'and make sure there is no vitaldb.py file next to this script')
    clinical = pd.read_csv(CASES_URL)
    ids = sorted(set(vitaldb.find_cases(TRACKS)))
    # only operations under general anaesthesia lasting ≥ 1 h
    c = clinical.set_index('caseid').reindex(ids)
    dur = (pd.to_numeric(c['opend'], errors='coerce') - pd.to_numeric(c['opstart'], errors='coerce')) / 60
    ok = (dur >= 60) & (c.get('ane_type', 'General') == 'General')
    ids = [i for i in ids if ok.get(i, False)]
    rng = np.random.default_rng(seed)
    ids = sorted(rng.choice(ids, min(n_cases, len(ids)), replace=False))
    cases = {}
    for j, cid in enumerate(ids):
        raw = vitaldb.load_case(int(cid), TRACKS, INTERVAL_S)
        if raw is None or len(raw) == 0:
            continue
        cases[int(cid)] = clean_minutes(raw)
        if (j + 1) % 50 == 0:
            print(f'{j + 1}/{len(ids)}', flush=True)
    return cases, clinical


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--cases', type=int, default=600)
    ap.add_argument('--out', default='cache_v2_vitaldb')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cases, clinical = download(args.cases)
    meas, pats, events = convert(cases, clinical)
    meas.to_parquet(os.path.join(args.out, 'measurements.parquet'))
    pats.to_parquet(os.path.join(args.out, 'patients.parquet'))
    events.to_parquet(os.path.join(args.out, 'events.parquet'))
    print(f'Operations: {meas.pid.nunique()}, minutes: {len(meas):,}; '
          f'hypotension: {(events.code == HYPO_CODE).mean():.1%} of events, '
          f'share of minutes with hypotension: {(events.code == HYPO_CODE).sum() / len(meas):.1%}')


if __name__ == '__main__':
    sys.exit(main())
