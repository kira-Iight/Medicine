"""Shared code for the additional ClusterBP v2 analyses (clusterbp_v2_leakage / first_events / features / shift_subgroups).

Uses the windows built by clusterbp_v2.ipynb (cache_v2*/windows.parquet and phenotype series).
Folds are taken from oof_predictions.parquet if present, otherwise built with the same StratifiedGroupKFold
as in clusterbp_v2.ipynb, so the folds match the main notebook.
Predictions and models of the main notebook are used only if they were built with the current targets and features
(see cache_is_current in clusterbp_lib.py). Fold phenotype templates are taken from models/templates_fold{k}.npy if present,
otherwise rebuilt (from the fold's training patients only) and saved.
"""
import os
import re
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from clusterbp_lib import (RAW_DIR, DATA_DIR, CACHE_DIR, MODELS_DIR, SEED, N_FOLDS, N_CLUSTERS, TARGETS,
                           fit_phenotypes, assign_phenotypes, set_phenotype_columns, fit_xgb, best_threshold,
                           check_windows, config_hash, cache_is_current, full_history_profiles,
                           event_counts, long_history_features, LONG_SPANS)   # re-exported for the notebooks

N_BOOT = int(os.environ.get('CLUSTERBP_N_BOOT', 1000))


# ---------------------------------------------------------------- data

def parse_int(s):
    return pd.to_numeric(s.astype(str).str.replace(r'\s', '', regex=True), errors='coerce')


def to_days(s):
    """Days since 1970-01-01 (float); empty or unparsable dates are NaN."""
    t = pd.to_datetime(s, errors='coerce')
    out = t.values.astype('datetime64[s]').astype('int64') / 86400.0
    return np.where(t.isna().values, np.nan, out)


def load_windows():
    """Windows and phenotype series from clusterbp_v2.ipynb; the fold column matches the main notebook."""
    check_windows()
    win = pd.read_parquet(os.path.join(DATA_DIR, 'windows.parquet'))
    pheno_win = np.load(os.path.join(DATA_DIR, 'pheno_win.npy'))
    pheno_first = np.load(os.path.join(DATA_DIR, 'pheno_first.npy'))
    pheno_first_pid = np.load(os.path.join(DATA_DIR, 'pheno_first_pid.npy'))
    oof_path = os.path.join(CACHE_DIR, 'oof_predictions.parquet')
    if os.path.exists(oof_path) and cache_is_current('cv'):
        oof = pd.read_parquet(oof_path, columns=['pid', 'cut', 'fold'])
        assert (oof['pid'].values == win['pid'].values).all() and (oof['cut'].values == win['cut'].values).all()
        win['fold'] = oof['fold'].values
    else:
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        folds = np.full(len(win), -1)
        for k, (_, te) in enumerate(sgkf.split(win, win['y_A'], groups=win['pid'].values)):
            folds[te] = k
        win['fold'] = folds
    return win, pheno_win, pheno_first, pheno_first_pid


def load_oof_scores(win):
    """Predictions of clusterbp_v2.ipynb (if present), to avoid retraining XGB_full where it is needed as is."""
    path = os.path.join(CACHE_DIR, 'oof_predictions.parquet')
    if not os.path.exists(path) or not cache_is_current('cv'):
        return None
    oof = pd.read_parquet(path)
    assert (oof['pid'].values == win['pid'].values).all()
    return oof


def phenotypes(win, pheno_win, pheno_first, pheno_first_pid, k, mode='train', meas=None):
    """Sets the phenotype columns in win for fold k and returns window clusters.

    mode:
    - 'train' — correct: templates from the first 30 days of the fold's training patients, windows assigned by their past 30 days;
    - 'all' — error: templates from all patients, including test patients (window assignment as in 'train');
    - 'full_history' — error of the first paper version: templates from all patients and their WHOLE series; the cluster
      is assigned once per patient from the whole series (including the future) and put into all its windows. Needs meas.
    Assignments are cached (GDTW over all windows is the slowest part); the cache is checked by the number of windows.
    """
    key = {'train': f'fold{k}', 'all': 'all', 'full_history': 'full_history'}[mode]
    cache = os.path.join(CACHE_DIR, f'pheno_assign_{key}.npz')
    if os.path.exists(cache):
        z = np.load(cache)
        if len(z['cl']) == len(win):
            set_phenotype_columns(win, z['cl'], z['dist'])
            return z['cl']
    pids = win['pid'].values
    if mode == 'full_history':
        if meas is None:
            raise ValueError("mode='full_history' needs meas")
        P, P_pid = full_history_profiles(meas)
        tpath = os.path.join(MODELS_DIR, 'templates_full_history.npy')
        if os.path.exists(tpath):
            templates = np.load(tpath)
        else:
            templates, _, _ = fit_phenotypes(P, P_pid, P_pid, SEED)
            np.save(tpath, templates)
        cl_p, dist_p = assign_phenotypes(P, templates)
        pos = pd.Series(np.arange(len(P_pid)), index=P_pid).reindex(pids).values
        ok = ~np.isnan(pos)
        cl = np.full(len(win), -1)
        dist = np.full((len(win), dist_p.shape[1]), np.nan)
        cl[ok] = cl_p[pos[ok].astype(int)]
        dist[ok] = dist_p[pos[ok].astype(int)]
    else:
        tpath = os.path.join(MODELS_DIR, f'templates_{key}.npy')
        if os.path.exists(tpath):
            templates = np.load(tpath)
        else:
            train_pids = np.unique(pids) if mode == 'all' else np.unique(pids[win['fold'].values != k])
            templates, _, _ = fit_phenotypes(pheno_first, pheno_first_pid, train_pids, SEED + (0 if mode == 'all' else k))
            np.save(tpath, templates)
        cl, dist = assign_phenotypes(pheno_win, templates)
    np.savez(cache, cl=cl, dist=dist)
    set_phenotype_columns(win, cl, dist)
    return cl


def event_target(win, events, codes, gap, horizon):
    """1 if the window's patient has an event with codes in [cut + gap, cut + gap + horizon)."""
    e = events[events.code.isin(codes)].sort_values(['pid', 'td'])
    by_pid = {p: g.values for p, g in e.groupby('pid')['td']}
    y = np.zeros(len(win), dtype=np.int8)
    cut = win['cut'].values
    for p, idx in win.groupby('pid').indices.items():
        t = by_pid.get(p)
        if t is None:
            continue
        a0 = cut[idx] + gap
        y[idx] = (np.searchsorted(t, a0 + horizon) - np.searchsorted(t, a0)) > 0
    return y


# ---------------------------------------------------------------- models

def fit_predict(win, cols, y, tr, va, seed, X=None):
    """XGBoost with early stopping on va; scores for all windows and the Macro F1 threshold tuned on va."""
    X = win[cols].values.astype(float) if X is None else X
    m = fit_xgb(X[tr], y[tr], X[va], y[va], seed)
    s = m.predict_proba(X)[:, 1]
    thr, _ = best_threshold(y[va], s[va])
    return s, thr, m


# ---------------------------------------------------------------- metrics

def _prep(y, s):
    o = np.argsort(-s, kind='stable')
    ss = s[o]
    g = np.concatenate([[0], np.cumsum(ss[1:] != ss[:-1])]) if len(ss) else np.zeros(0, int)
    return o, g


def _wmetrics(y, pred, o, g, w):
    wo, yo = w[o], y[o]
    pos = np.bincount(g, wo * yo)
    neg = np.bincount(g, wo * (1 - yo))
    P, N = pos.sum(), neg.sum()
    if P == 0 or N == 0:
        return np.nan, np.nan, np.nan
    cpos, cneg = np.cumsum(pos), np.cumsum(neg)
    auroc = (pos * (N - cneg) + 0.5 * pos * neg).sum() / (P * N)
    ap = (pos * cpos / np.maximum(cpos + cneg, 1e-12)).sum() / P
    if pred is None:
        return auroc, ap, np.nan
    tp = (w * y * pred).sum(); fp = (w * (1 - y) * pred).sum()
    fn = (w * y * (1 - pred)).sum(); tn = (w * (1 - y) * (1 - pred)).sum()
    f1 = (2 * tp / max(2 * tp + fp + fn, 1e-12) + 2 * tn / max(2 * tn + fn + fp, 1e-12)) / 2
    return auroc, ap, f1


def paired_boot(entries, pid, n_boot=N_BOOT, seed=SEED):
    """Patient-level bootstrap for several evaluations at once.

    entries — list of (mask, y, score, pred); each evaluation uses its own window mask, but patient weights are shared
    within an iteration, so differences between evaluations are paired (including different masks).
    Returns an array [n_boot, len(entries), 3]: AUROC, AUPRC, Macro F1.
    """
    _, pidx = np.unique(pid, return_inverse=True)
    n_pat = pidx.max() + 1
    prep = []
    for mask, y, s, p in entries:
        idx = np.flatnonzero(mask)
        yy = y[idx].astype(float)
        o, g = _prep(yy, s[idx])
        prep.append((pidx[idx], yy, None if p is None else p[idx].astype(float), o, g))
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, len(entries), 3))
    for b in range(n_boot):
        wp = np.bincount(rng.integers(0, n_pat, n_pat), minlength=n_pat).astype(float)
        for j, (pi, yy, pp, o, g) in enumerate(prep):
            out[b, j] = _wmetrics(yy, pp, o, g, wp[pi])
    return out


def ci(a):
    a = a[np.isfinite(a)]
    return f'[{np.percentile(a, 2.5):.3f}; {np.percentile(a, 97.5):.3f}]' if len(a) else ''


def pval(d):
    d = d[np.isfinite(d)]
    return min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())) if len(d) else np.nan


def point_metrics(y, s, p):
    y = y.astype(int)
    out = {'windows': len(y), 'event rate': y.mean() if len(y) else np.nan}
    if len(y) == 0 or y.min() == y.max():
        return {**out, 'AUROC': np.nan, 'AUPRC': np.nan, 'AUPRC / event rate': np.nan, 'macro_F1': np.nan,
                'sensitivity': np.nan, 'specificity': np.nan, 'PPV': np.nan}
    ap = average_precision_score(y, s)
    out.update({'AUROC': roc_auc_score(y, s), 'AUPRC': ap, 'AUPRC / event rate': ap / y.mean()})
    if p is None:
        out.update({'macro_F1': np.nan, 'sensitivity': np.nan, 'specificity': np.nan, 'PPV': np.nan})
    else:
        p = p.astype(int)
        tp = ((p == 1) & (y == 1)).sum(); fp = ((p == 1) & (y == 0)).sum()
        fn = ((p == 0) & (y == 1)).sum(); tn = ((p == 0) & (y == 0)).sum()
        out.update({'macro_F1': f1_score(y, p, average='macro'), 'sensitivity': tp / max(tp + fn, 1),
                    'specificity': tn / max(tn + fp, 1), 'PPV': tp / max(tp + fp, 1)})
    return out


def compare_table(entries, pid, ref=None, n_boot=N_BOOT):
    """entries: dict name -> (mask, y, score, pred). Metrics, 95% CI (patient-level bootstrap) and differences from ref."""
    names = list(entries)
    B = paired_boot([entries[n] for n in names], pid, n_boot=n_boot)
    rows = []
    for j, n in enumerate(names):
        mask, y, s, p = entries[n]
        r = {'variant': n, **point_metrics(y[mask], s[mask], None if p is None else p[mask])}
        r['AUPRC 95% CI'] = ci(B[:, j, 1])
        r['macro_F1 95% CI'] = ci(B[:, j, 2])
        if ref is not None and n != ref:
            jr = names.index(ref)
            rr = point_metrics(entries[ref][1][entries[ref][0]], entries[ref][2][entries[ref][0]],
                               None if entries[ref][3] is None else entries[ref][3][entries[ref][0]])
            for mi, m in [(1, 'AUPRC'), (2, 'macro_F1'), (0, 'AUROC')]:
                d = B[:, j, mi] - B[:, jr, mi]
                r[f'Δ{m}'] = r[m] - rr[m]
                r[f'Δ{m} 95% CI'] = ci(d)
                r[f'p Δ{m}'] = pval(d)
        rows.append(r)
    return pd.DataFrame(rows).set_index('variant')


def calib(y, p):
    """Calibration-in-the-large (mean prediction / observed rate) and ECE over deciles."""
    if len(y) < 20:
        return np.nan, np.nan
    q = pd.qcut(pd.Series(p).rank(method='first'), 10, labels=False).values
    obs = pd.Series(y).groupby(q).mean()
    pr = pd.Series(p).groupby(q).mean()
    wts = pd.Series(q).value_counts(normalize=True).sort_index()
    return p.mean() / max(y.mean(), 1e-12), float((np.abs(obs - pr) * wts).sum())


# ---------------------------------------------------------------- therapy

# Drug classes by INN (substring or regular expression on the Russian INN in the export). Combination drugs get several classes.
DRUG_CLASSES = [
    ('ACE inhibitor', r'прил'),           # perindopril, perindopril arginine, enalapril…
    ('ARB', r'артан'),                    # losartan, valsartan, azilsartan medoxomil…
    ('CCB', r'дипин|верапамил|дилтиазем'),  # -dipines, verapamil, diltiazem
    ('beta-blocker', r'олол'),            # bisoprolol, metoprolol succinate…
    ('diuretic', r'индапамид|гидрохлоротиазид|хлорталидон|торасемид|фуросемид|спиронолактон|эплеренон|ксипамид'),
    ('central', r'моксонидин|клонидин|метилдопа|рилменидин'),       # moxonidine, clonidine, methyldopa, rilmenidine
    ('alpha-blocker', r'доксазозин|урапидил|празозин|теразозин'),    # doxazosin, urapidil, prazosin, terazosin
    ('other antihypertensive', r'алискирен|сакубитрил'),             # aliskiren, sacubitril
    ('statin', r'(аторва|розува|симва|флува|питава|права|лова)статин'),   # not nystatin
    ('antiplatelet/anticoagulant', r'ацетилсалицил|клопидогрел|тикагрелор|варфарин|апиксабан|ривароксабан|дабигатран'),
]
AH_CLASSES = ['ACE inhibitor', 'ARB', 'CCB', 'beta-blocker', 'diuretic', 'central', 'alpha-blocker', 'other antihypertensive']
CLASS_COL = {c: f'th_cls_{i}' for i, c in enumerate(AH_CLASSES)}


def load_therapy():
    """Prescriptions with patient id, the date the prescription became known, and drug classes.

    In the export, therapy is linked to the programme id, so the patient is taken from the "MO and patients" file.
    The prescription is known from the prescription date (if empty, the start date), but not before the programme start.
    Dates are daily: a prescription is used for cutoff t only if its day is strictly before t.
    """
    th = pd.read_csv(os.path.join(RAW_DIR, 'Медикаментозная терапия от 10.04.2025.csv'), low_memory=False)
    grp = pd.read_csv(os.path.join(RAW_DIR, 'МО и Пациенты 10.04.2025.csv'), sep=';')
    link = pd.DataFrame({'prog': parse_int(grp['id программы']), 'pid': parse_int(grp['id пациента'])}).dropna()
    link = link.drop_duplicates('prog').set_index('prog')['pid']
    th['prog'] = parse_int(th['id программы'])
    th['pid'] = link.reindex(th['prog'].values).values
    start_prog = np.floor(to_days(th['дата начала программы']))
    t_rx = to_days(th['дата назначения'])
    t_start = to_days(th['дата начала приёма'] if 'дата начала приёма' in th else th['дата начала приема'])
    th['t_known'] = np.fmax(np.where(np.isnan(t_rx), t_start, t_rx), start_prog)
    th['t_rx'] = t_rx
    th['t_end'] = to_days(th['дата окончания приёма'] if 'дата окончания приёма' in th else th['дата окончания приема'])
    th['t_prog_start'] = start_prog
    inn = th['МНН'].astype(str).str.lower()
    th['n_classes'] = 0
    for c, pat in DRUG_CLASSES:
        th[f'is_{c}'] = inn.str.contains(pat, regex=True)
        th['n_classes'] += th[f'is_{c}'].astype(int)
    th['is_AH'] = th[[f'is_{c}' for c in AH_CLASSES]].any(axis=1)
    th['taking'] = th['принимаемая терапия (да/нет)'].astype(str).str.lower().isin(['true', 'да', '1'])
    return th


def therapy_features(win, th):
    """Therapy features at the cutoff (only prescriptions known strictly before the cutoff day)."""
    path = os.path.join(DATA_DIR, f'features_therapy_{config_hash()}.parquet')   # hash of targets and windows in the file name
    if os.path.exists(path):
        f = pd.read_parquet(path)
        if len(f) == len(win) and (f['pid'].values == win['pid'].values).all() and (f['cut'].values == win['cut'].values).all():
            return f.drop(columns=['pid', 'cut'])
    th = th.dropna(subset=['pid', 't_known']).copy()
    th['pid'] = th['pid'].astype('int64')
    by_pid = {p: g for p, g in th.groupby('pid')}
    names = (['th_has_record', 'th_n_ah_active', 'th_n_any_active', 'th_n_ah_classes', 'th_new_ah_14',
              'th_new_ah_30', 'th_new_ah_60', 'th_stop_ah_30', 'th_days_since_ah_change', 'th_statin', 'th_antiplatelet']
             + list(CLASS_COL.values()))
    F = {n: np.full(len(win), np.nan) for n in names}
    cut_all = win['cut'].values
    for p, idx in win.groupby('pid').indices.items():
        g = by_pid.get(p)
        cut = cut_all[idx][:, None]
        if g is None:
            for n in names:
                F[n][idx] = 0.0 if n != 'th_days_since_ah_change' else np.nan
            continue
        tk = g['t_known'].values[None, :]
        te_ = g['t_end'].values[None, :]
        known = tk < cut                                    # prescription day strictly before the cutoff day
        active = known & ~(te_ < cut)
        ah = g['is_AH'].values[None, :]
        F['th_has_record'][idx] = known.any(1)
        F['th_n_ah_active'][idx] = (active & ah).sum(1)
        F['th_n_any_active'][idx] = active.sum(1)
        cls_active = []
        for c in AH_CLASSES:
            a = (active & g[f'is_{c}'].values[None, :]).any(1)
            F[CLASS_COL[c]][idx] = a
            cls_active.append(a)
        F['th_n_ah_classes'][idx] = np.sum(cls_active, axis=0)
        # "therapy change" — a new antihypertensive prescription after the programme start
        new = known & ah & (tk > g['t_prog_start'].values[None, :])
        for L in (14, 30, 60):
            F[f'th_new_ah_{L}'][idx] = (new & (tk >= cut - L)).sum(1)
        F['th_stop_ah_30'][idx] = (ah & known & (te_ < cut) & (te_ >= cut - 30)).sum(1)
        last = np.where(new, tk, -np.inf).max(1)
        F['th_days_since_ah_change'][idx] = np.where(np.isfinite(last), cut[:, 0] - last, np.nan)
        F['th_statin'][idx] = (active & g['is_statin'].values[None, :]).any(1)
        F['th_antiplatelet'][idx] = (active & g['is_antiplatelet/anticoagulant'].values[None, :]).any(1)
    f = pd.DataFrame(F).astype(float)
    pd.concat([win[['pid', 'cut']].reset_index(drop=True), f], axis=1).to_parquet(path)
    return f


# ---------------------------------------------------------------- keeping earlier results

SNAPSHOT_SUFFIX = '_before_eda_baselines'


def snapshot_results():
    """Copy RESULTS_DIR to RESULTS_DIR + SNAPSHOT_SUFFIX once, before the EDA and baseline notebooks add to it.

    The copy is made only if it does not exist yet, so rerunning the notebooks never overwrites it.
    """
    import shutil
    from clusterbp_lib import RESULTS_DIR
    dst = RESULTS_DIR + SNAPSHOT_SUFFIX
    if os.path.isdir(RESULTS_DIR) and not os.path.exists(dst):
        shutil.copytree(RESULTS_DIR, dst)
        print(f'Copy of {RESULTS_DIR}/ saved to {dst}/')
    elif os.path.exists(dst):
        print(f'Copy {dst}/ already exists and is kept as is')
    return dst


def upsert_rows(path, new, keys=('target', 'model')):
    """Add the rows of `new` to the CSV at `path`, replacing rows with the same keys; other rows are kept in order."""
    keys = list(keys)
    if os.path.exists(path):
        old = pd.read_csv(path)
        idx_new = pd.MultiIndex.from_frame(new[keys].astype(str))
        keep = ~pd.MultiIndex.from_frame(old[keys].astype(str)).isin(idx_new)
        out = pd.concat([old[keep], new], ignore_index=True)
        order = {t: i for i, t in enumerate(pd.unique(pd.concat([old[keys[0]], new[keys[0]]]).astype(str)))}
        out = out.sort_values(keys[0], key=lambda s: s.astype(str).map(order), kind='stable')
    else:
        out = new
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    out.to_csv(path, index=False)
    return out


# ---------------------------------------------------------------- daily series for MiniRocket

def daily_series(win, meas, days=14):
    """Daily means of SBP, DBP and HR over the `days` days before each cutoff: array [n_windows, 3, days] (float32).

    Day j covers [cut − days + j, cut − days + j + 1). Days without measurements are filled forward, then backward
    within the window (every window has measurements, so nothing stays empty).
    """
    m = meas[['pid', 'td', 'sbp', 'dbp', 'hr']].copy()
    m['day'] = np.floor(m['td']).astype('int64')
    d = m.groupby(['pid', 'day'])[['sbp', 'dbp', 'hr']].mean().reset_index()
    by_pid = {p: (g['day'].values, g[['sbp', 'dbp', 'hr']].values.astype(np.float32)) for p, g in d.groupby('pid')}
    out = np.full((len(win), 3, days), np.nan, dtype=np.float32)
    cut = np.round(win['cut'].values).astype('int64')
    offs = np.arange(-days, 0)
    for p, idx in win.groupby('pid').indices.items():
        if p not in by_pid:
            continue
        dd, vv = by_pid[p]
        grid = cut[idx][:, None] + offs[None, :]
        pos = np.clip(np.searchsorted(dd, grid), 0, len(dd) - 1)
        hit = dd[pos] == grid
        vals = np.where(hit[..., None], vv[pos], np.nan)            # [n, days, 3]
        out[idx] = np.transpose(vals, (0, 2, 1))
    s = pd.DataFrame(out.reshape(-1, days).T)                       # fill along time within each window and channel
    s = s.ffill().bfill()
    out = s.values.T.reshape(len(win), 3, days).astype(np.float32)
    return out


# ---------------------------------------------------------------- MiniRocket (Dempster, Schmidt & Webb, KDD 2021)

class MiniRocket:
    """MiniRocket transform for short multivariate series, following the reference implementation
    (github.com/angus924/minirocket, minirocket_multivariate.py), written with numpy only.

    84 fixed kernels of length 9 (weights −1, with +2 at three positions), dilations spaced exponentially up to
    (L − 1) / 8, padding alternating between kernel/dilation combinations, a random subset of channels for each
    combination, biases from quantiles (low-discrepancy sequence) of the convolution output of a random training
    example, and PPV (proportion of positive values) features. For a 14-day series only dilation 1 is possible and
    each kernel output has at most 15 distinct PPV values, so `num_features` defaults to 1,008 (12 biases per kernel)
    instead of 10,000.
    """

    def __init__(self, num_features=1008, max_dilations_per_kernel=32, seed=SEED):
        self.num_features = num_features
        self.max_dilations = max_dilations_per_kernel
        self.seed = seed

    def _conv(self, X, kernel_idx, dilation, channels, padding):
        """Convolution output [n, positions] of kernel `kernel_idx` summed over `channels`."""
        Xc = X[:, channels, :].sum(1, dtype=np.float64)              # weights are shared by channels
        n, L = Xc.shape
        pad = 4 * dilation
        Xp = np.zeros((n, L + 2 * pad))
        Xp[:, pad:pad + L] = Xc
        C = np.zeros((n, L))
        for j in range(9):                                            # −1 at every position
            C -= Xp[:, j * dilation:j * dilation + L]
        for j in self.indices[kernel_idx]:                            # +3 at three positions → net weight +2
            C += 3 * Xp[:, j * dilation:j * dilation + L]
        if padding == 0:                                              # positions that need no padding
            C = C[:, pad:L - pad] if L - 2 * pad > 0 else C[:, :0]
        return C

    def fit(self, X):
        from itertools import combinations
        rng = np.random.default_rng(self.seed)
        n, n_ch, L = X.shape
        self.indices = np.array(list(combinations(range(9), 3)))     # 84 kernels
        n_k = len(self.indices)
        per_kernel = self.num_features // n_k
        true_max = min(per_kernel, self.max_dilations)
        multiplier = per_kernel / true_max
        max_exp = np.log2(max(L - 1, 1) / 8)
        dil, counts = np.unique(np.floor(np.logspace(0, max(max_exp, 0), true_max, base=2)).astype(int),
                                return_counts=True)
        per_dil = (counts * multiplier).astype(int)
        rem, i = per_kernel - per_dil.sum(), 0
        while rem > 0:
            per_dil[i] += 1
            rem -= 1
            i = (i + 1) % len(per_dil)
        self.dilations, self.per_dilation = dil, per_dil
        n_comb = n_k * len(dil)
        max_ch = min(n_ch, 9)
        n_chan = np.floor(2 ** rng.uniform(0, np.log2(max_ch + 1), n_comb)).astype(int)
        self.channels = [rng.choice(n_ch, c, replace=False) for c in n_chan]
        phi = (np.sqrt(5) + 1) / 2
        quant = (np.arange(1, n_k * per_dil.sum() + 1) * phi) % 1
        self.combos = []                                              # (kernel, dilation, channels, padding, biases)
        q0, comb = 0, 0
        for di, dl in enumerate(dil):
            nf = per_dil[di]
            for k in range(n_k):
                padding = 1 if (di + k) % 2 == 0 else 0
                ex = X[rng.integers(n)][None]
                C = self._conv(ex, k, dl, self.channels[comb], 1)     # biases from the full (padded) output
                biases = np.quantile(C.ravel(), quant[q0:q0 + nf])
                self.combos.append((k, dl, self.channels[comb], padding, biases))
                q0 += nf
                comb += 1
        return self

    def transform(self, X, batch=50000):
        """PPV features [n, num_features] as float32."""
        n_out = sum(len(c[4]) for c in self.combos)
        F = np.empty((len(X), n_out), dtype=np.float32)
        for a in range(0, len(X), batch):
            Xb = X[a:a + batch]
            col = 0
            for k, dl, ch, padding, biases in self.combos:
                C = self._conv(Xb, k, dl, ch, padding)
                F[a:a + batch, col:col + len(biases)] = (C[:, :, None] > biases[None, None, :]).mean(1)
                col += len(biases)
        return F


def fit_ridge(F, y, tr, va, alphas=np.logspace(-1, 5, 13), chunk=50000):
    """Ridge classifier on standardised features (the usual MiniRocket classifier), with alpha chosen by AUPRC
    on the inner validation set and Platt scaling on the same set. Gram matrices are accumulated in chunks,
    so memory stays at about the size of F.

    Returns calibrated probabilities for all windows and the chosen alpha.
    """
    from sklearn.linear_model import LogisticRegression
    idx = np.flatnonzero(tr)
    n_f = F.shape[1]
    s1, s2 = np.zeros(n_f), np.zeros(n_f)
    for a in range(0, len(idx), chunk):
        x = F[idx[a:a + chunk]].astype(np.float64)
        s1 += x.sum(0)
        s2 += (x * x).sum(0)
    mu = s1 / len(idx)
    sd = np.sqrt(np.maximum(s2 / len(idx) - mu ** 2, 0))
    sd[sd < 1e-9] = 1.0
    t = np.where(y[idx] == 1, 1.0, -1.0)
    t -= t.mean()
    G, b = np.zeros((n_f, n_f)), np.zeros(n_f)
    for a in range(0, len(idx), chunk):
        z = (F[idx[a:a + chunk]].astype(np.float64) - mu) / sd
        G += z.T @ z
        b += z.T @ t[a:a + chunk]
    Zva = (F[va].astype(np.float64) - mu) / sd
    best = (-1.0, None, None)
    for al in alphas:
        w = np.linalg.solve(G + al * np.eye(n_f), b)
        ap = average_precision_score(y[va], Zva @ w)
        if ap > best[0]:
            best = (ap, w, al)
    _, w, alpha = best
    score = np.concatenate([((F[i:i + chunk].astype(np.float64) - mu) / sd) @ w for i in range(0, len(F), chunk)])
    platt = LogisticRegression(max_iter=1000).fit(score[va].reshape(-1, 1), y[va])
    return platt.predict_proba(score.reshape(-1, 1))[:, 1], alpha
