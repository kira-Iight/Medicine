"""Общий код дополнительных анализов ClusterBP v2 (ноутбуки clusterbp_v2_leakage / first_events / features / shift_subgroups).

Использует окна, построенные в clusterbp_v2.ipynb (cache_v2*/windows.parquet и ряды для фенотипов).
Разбиение на фолды берётся из oof_predictions.parquet, если он есть, иначе строится тем же
StratifiedGroupKFold, что и в clusterbp_v2.ipynb, поэтому фолды совпадают с основным ноутбуком.
Шаблоны фенотипов фолда берутся из models/templates_fold{k}.npy, если они уже есть, иначе строятся заново
(только по обучающим пациентам фолда) и сохраняются.
"""
import os
import re
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from clusterbp_lib import (RAW_DIR, DATA_DIR, CACHE_DIR, MODELS_DIR, SEED, N_FOLDS, N_CLUSTERS, TARGETS,
                           fit_phenotypes, assign_phenotypes, set_phenotype_columns, fit_xgb, best_threshold)

N_BOOT = int(os.environ.get('CLUSTERBP_N_BOOT', 1000))
DAY = 1.0


# ---------------------------------------------------------------- данные

def parse_int(s):
    return pd.to_numeric(s.astype(str).str.replace(r'\s', '', regex=True), errors='coerce')


def to_days(s):
    """Дни от 1970-01-01 (float); пустые и нераспознанные даты — NaN."""
    t = pd.to_datetime(s, errors='coerce')
    out = t.values.astype('datetime64[s]').astype('int64') / 86400.0
    return np.where(t.isna().values, np.nan, out)


def load_windows():
    """Окна и ряды для фенотипов из clusterbp_v2.ipynb; столбец fold — как в основном ноутбуке."""
    win = pd.read_parquet(os.path.join(DATA_DIR, 'windows.parquet'))
    pheno_win = np.load(os.path.join(DATA_DIR, 'pheno_win.npy'))
    pheno_first = np.load(os.path.join(DATA_DIR, 'pheno_first.npy'))
    pheno_first_pid = np.load(os.path.join(DATA_DIR, 'pheno_first_pid.npy'))
    oof_path = os.path.join(CACHE_DIR, 'oof_predictions.parquet')
    if os.path.exists(oof_path):
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
    """Предсказания clusterbp_v2.ipynb (если есть) — чтобы не переобучать XGB_full там, где он нужен как есть."""
    path = os.path.join(CACHE_DIR, 'oof_predictions.parquet')
    if not os.path.exists(path):
        return None
    oof = pd.read_parquet(path)
    assert (oof['pid'].values == win['pid'].values).all()
    return oof


def phenotypes(win, pheno_win, pheno_first, pheno_first_pid, k, all_patients=False):
    """Ставит в win столбцы фенотипа для фолда k и возвращает кластеры окон.

    all_patients=True — ошибка из аудита утечек: шаблоны строятся по всем пациентам, включая тестовых.
    Назначения окон кешируются, потому что GDTW по всем окнам — самая долгая часть.
    """
    key = 'all' if all_patients else f'fold{k}'
    cache = os.path.join(CACHE_DIR, f'pheno_assign_{key}.npz')
    if os.path.exists(cache):
        z = np.load(cache)
        cl, dist = z['cl'], z['dist']
    else:
        tpath = os.path.join(MODELS_DIR, f'templates_{key}.npy')
        if os.path.exists(tpath):
            templates = np.load(tpath)
        else:
            pids = win['pid'].values
            train_pids = np.unique(pids) if all_patients else np.unique(pids[win['fold'].values != k])
            templates, _, _ = fit_phenotypes(pheno_first, pheno_first_pid, train_pids, SEED + (0 if all_patients else k))
            np.save(tpath, templates)
        cl, dist = assign_phenotypes(pheno_win, templates)
        np.savez(cache, cl=cl, dist=dist)
    set_phenotype_columns(win, cl, dist)
    return cl


def event_target(win, events, codes, gap, horizon):
    """1, если у пациента окна есть событие из codes в [cut + gap, cut + gap + horizon)."""
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


def event_counts(win, events, codes, lo_off, hi_off=0.0):
    """Число событий из codes (None — все) в [cut − lo_off, cut − hi_off)."""
    e = events if codes is None else events[events.code.isin(codes)]
    e = e.sort_values(['pid', 'td'])
    by_pid = {p: g.values for p, g in e.groupby('pid')['td']}
    out = np.zeros(len(win))
    cut = win['cut'].values
    for p, idx in win.groupby('pid').indices.items():
        t = by_pid.get(p)
        if t is None:
            continue
        c = cut[idx]
        out[idx] = np.searchsorted(t, c - hi_off) - np.searchsorted(t, c - lo_off)
    return out


# ---------------------------------------------------------------- модели

def fit_predict(win, cols, y, tr, va, seed, X=None):
    """XGBoost с ранней остановкой на va; скор для всех окон и порог Macro F1, подобранный на va."""
    X = win[cols].values.astype(float) if X is None else X
    m = fit_xgb(X[tr], y[tr], X[va], y[va], seed)
    s = m.predict_proba(X)[:, 1]
    thr, _ = best_threshold(y[va], s[va])
    return s, thr, m


# ---------------------------------------------------------------- метрики

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
    """Бутстреп по пациентам для нескольких оценок сразу.

    entries — список (mask, y, score, pred); каждая оценка считается на своей маске окон, но веса пациентов
    в одной итерации общие, поэтому разности между оценками — парные (в том числе при разных масках).
    Возвращает массив [n_boot, len(entries), 3]: AUROC, AUPRC, Macro F1.
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
    out = {'окон': len(y), 'доля событий': y.mean() if len(y) else np.nan}
    if len(y) == 0 or y.min() == y.max():
        return {**out, 'AUROC': np.nan, 'AUPRC': np.nan, 'AUPRC / доля': np.nan, 'macro_F1': np.nan,
                'sensitivity': np.nan, 'specificity': np.nan, 'PPV': np.nan}
    ap = average_precision_score(y, s)
    out.update({'AUROC': roc_auc_score(y, s), 'AUPRC': ap, 'AUPRC / доля': ap / y.mean()})
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
    """entries: dict имя -> (mask, y, score, pred). Метрики, 95% ДИ (бутстреп по пациентам) и разности с ref."""
    names = list(entries)
    B = paired_boot([entries[n] for n in names], pid, n_boot=n_boot)
    rows = []
    for j, n in enumerate(names):
        mask, y, s, p = entries[n]
        r = {'вариант': n, **point_metrics(y[mask], s[mask], None if p is None else p[mask])}
        r['AUPRC 95% ДИ'] = ci(B[:, j, 1])
        r['macro_F1 95% ДИ'] = ci(B[:, j, 2])
        if ref is not None and n != ref:
            jr = names.index(ref)
            rr = point_metrics(entries[ref][1][entries[ref][0]], entries[ref][2][entries[ref][0]],
                               None if entries[ref][3] is None else entries[ref][3][entries[ref][0]])
            for mi, m in [(1, 'AUPRC'), (2, 'macro_F1'), (0, 'AUROC')]:
                d = B[:, j, mi] - B[:, jr, mi]
                r[f'Δ{m}'] = r[m] - rr[m]
                r[f'Δ{m} 95% ДИ'] = ci(d)
                r[f'p Δ{m}'] = pval(d)
        rows.append(r)
    return pd.DataFrame(rows).set_index('вариант')


def calib(y, p):
    """Калибровка в целом (средний прогноз / наблюдаемая доля) и ECE по децилям."""
    if len(y) < 20:
        return np.nan, np.nan
    q = pd.qcut(pd.Series(p).rank(method='first'), 10, labels=False).values
    obs = pd.Series(y).groupby(q).mean()
    pr = pd.Series(p).groupby(q).mean()
    wts = pd.Series(q).value_counts(normalize=True).sort_index()
    return p.mean() / max(y.mean(), 1e-12), float((np.abs(obs - pr) * wts).sum())


# ---------------------------------------------------------------- длинная история (30–90 дней)

LONG_SPANS = (30, 60, 90)


def long_history_features(win, meas, events, spans=LONG_SPANS):
    """Признаки за последние 30/60/90 дней до отсечки — по среднесуточным значениям (межсуточная вариабельность).

    По Rothwell прогностична долговременная вариабельность САД, поэтому кроме среднего считаются SD, ARV
    и наклон среднесуточного САД, доля дней со средним САД ≥ 140 и отклонение последних 14 дней от
    долговременного уровня. Счётчики событий A, B и всех КЗС — за те же периоды и за всё время в программе.
    Все признаки используют только данные до отсечки.
    """
    path = os.path.join(DATA_DIR, 'features_long.parquet')
    if os.path.exists(path):
        f = pd.read_parquet(path)
        if len(f) == len(win) and (f['pid'].values == win['pid'].values).all() and (f['cut'].values == win['cut'].values).all():
            return f.drop(columns=['pid', 'cut'])
    t0 = time.time()
    m = meas[['pid', 'td', 'sbp', 'dbp']].copy()
    m['day'] = np.floor(m['td'])
    daily = m.groupby(['pid', 'day'])[['sbp', 'dbp']].mean().reset_index()
    by_pid = {p: g for p, g in daily.groupby('pid')}
    cols = {}

    def put(name, idx, val):
        cols.setdefault(name, np.full(len(win), np.nan))[idx] = val

    cut_all = win['cut'].values
    for p, idx in win.groupby('pid').indices.items():
        g = by_pid.get(p)
        if g is None:
            continue
        d = g['day'].values
        s = g['sbp'].values
        db = g['dbp'].values
        cut = cut_all[idx]
        P = lambda x: np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
        Ps, Ps2, Pd, Pd2 = P(s), P(s * s), P(db), P(db * db)
        Phi = P((s >= 140).astype(float))
        Pa = P(np.concatenate([[0.0], np.abs(np.diff(s))]))
        Pu, Puu, Pus = P(d), P(d * d), P(d * s)
        hi = np.searchsorted(d, cut)                 # дни строго до отсечки (отсечка — полночь)
        lo14 = np.searchsorted(d, cut - 14)
        n14 = hi - lo14
        mean14 = np.where(n14 > 0, (Ps[hi] - Ps[lo14]) / np.maximum(n14, 1), np.nan)
        for L in spans:
            lo = np.searchsorted(d, cut - L)
            n = hi - lo
            ok = n > 0
            nn = np.maximum(n, 1)
            mean = (Ps[hi] - Ps[lo]) / nn
            var = np.maximum((Ps2[hi] - Ps2[lo]) / nn - mean ** 2, 0) * nn / np.maximum(nn - 1, 1)
            dmean = (Pd[hi] - Pd[lo]) / nn
            dvar = np.maximum((Pd2[hi] - Pd2[lo]) / nn - dmean ** 2, 0) * nn / np.maximum(nn - 1, 1)
            arv = np.where(n > 1, (Pa[hi] - Pa[np.minimum(lo + 1, hi)]) / np.maximum(n - 1, 1), np.nan)
            su, suu, sus, ss_ = Pu[hi] - Pu[lo], Puu[hi] - Puu[lo], Pus[hi] - Pus[lo], Ps[hi] - Ps[lo]
            den = n * suu - su ** 2
            slope = np.where(den > 1e-9, (n * sus - su * ss_) / np.where(den > 1e-9, den, 1), np.nan)
            put(f'lh{L}_days', idx, n)
            put(f'lh{L}_sbp_mean', idx, np.where(ok, mean, np.nan))
            put(f'lh{L}_sbp_sd', idx, np.where(n > 1, np.sqrt(var), np.nan))
            put(f'lh{L}_sbp_arv', idx, arv)
            put(f'lh{L}_sbp_slope', idx, slope)
            put(f'lh{L}_sbp_frac140', idx, np.where(ok, (Phi[hi] - Phi[lo]) / nn, np.nan))
            put(f'lh{L}_dbp_mean', idx, np.where(ok, dmean, np.nan))
            put(f'lh{L}_dbp_sd', idx, np.where(n > 1, np.sqrt(dvar), np.nan))
            put(f'lh{L}_sbp_mean14_minus', idx, np.where(ok, mean14 - mean, np.nan))
        cmax = np.array([np.nan] * len(cut))
        for i, c in enumerate(cut):
            j0, j1 = np.searchsorted(d, c - max(spans)), np.searchsorted(d, c)
            if j1 > j0:
                cmax[i] = s[j0:j1].max()
        put(f'lh{max(spans)}_sbp_daymax', idx, cmax)

    f = pd.DataFrame(cols)
    for g, codes in [('A', TARGETS['A']['codes']), ('B', TARGETS['B']['codes']), ('ALL', None)]:
        for L in spans:
            f[f'lh{L}_ev_{g}'] = event_counts(win, events, codes, L)
        total = event_counts(win, events, codes, 1e6)
        f[f'lh_ev_{g}_total'] = total
        f[f'lh_ev_{g}_per_week'] = total / np.maximum(win['days_in_program'].values, 1) * 7
    out = pd.concat([win[['pid', 'cut']].reset_index(drop=True), f], axis=1)
    out.to_parquet(path)
    print(f'Признаки длинной истории: {f.shape[1]} шт., {time.time() - t0:.0f} с')
    return f


# ---------------------------------------------------------------- терапия

# Классы препаратов по МНН (подстрока или регулярное выражение). Комбинированные препараты дают несколько классов.
DRUG_CLASSES = [
    ('ИАПФ', r'прил\b'),
    ('БРА', r'артан\b'),
    ('БКК', r'дипин|верапамил|дилтиазем'),
    ('ББ', r'лол\b'),
    ('диуретик', r'индапамид|гидрохлоротиазид|хлорталидон|торасемид|фуросемид|спиронолактон|эплеренон|ксипамид'),
    ('центральный', r'моксонидин|клонидин|метилдопа|рилменидин'),
    ('альфа-блокатор', r'доксазозин|урапидил|празозин|теразозин'),
    ('прочий гипотензивный', r'алискирен|сакубитрил'),
    ('статин', r'статин'),
    ('антиагрегант/антикоагулянт', r'ацетилсалицил|клопидогрел|тикагрелор|варфарин|апиксабан|ривароксабан|дабигатран'),
]
AH_CLASSES = ['ИАПФ', 'БРА', 'БКК', 'ББ', 'диуретик', 'центральный', 'альфа-блокатор', 'прочий гипотензивный']
CLASS_COL = {c: f'th_cls_{i}' for i, c in enumerate(AH_CLASSES)}


def load_therapy():
    """Назначения с id пациента, датой, когда назначение стало известно, и классами препарата.

    В выгрузке терапия привязана к id программы, поэтому пациент берётся из «МО и Пациенты».
    Момент, когда назначение известно: дата назначения (если пусто — дата начала приёма), но не раньше начала
    программы. Даты дневные: назначение учитывается для отсечки t, только если его день строго раньше t.
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
    """Признаки терапии на момент отсечки (только назначения, известные строго до дня отсечки)."""
    path = os.path.join(DATA_DIR, 'features_therapy.parquet')
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
        known = tk < cut                                    # день назначения строго раньше дня отсечки
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
        # «изменение терапии» — новое гипотензивное назначение после начала программы
        new = known & ah & (tk > g['t_prog_start'].values[None, :])
        for L in (14, 30, 60):
            F[f'th_new_ah_{L}'][idx] = (new & (tk >= cut - L)).sum(1)
        F['th_stop_ah_30'][idx] = (ah & known & (te_ < cut) & (te_ >= cut - 30)).sum(1)
        last = np.where(new, tk, -np.inf).max(1)
        F['th_days_since_ah_change'][idx] = np.where(np.isfinite(last), cut[:, 0] - last, np.nan)
        F['th_statin'][idx] = (active & g['is_статин'].values[None, :]).any(1)
        F['th_antiplatelet'][idx] = (active & g['is_антиагрегант/антикоагулянт'].values[None, :]).any(1)
    f = pd.DataFrame(F).astype(float)
    pd.concat([win[['pid', 'cut']].reset_index(drop=True), f], axis=1).to_parquet(path)
    return f
