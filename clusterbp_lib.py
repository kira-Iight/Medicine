"""Общий код ClusterBP v2 (clusterbp_v2*.ipynb): цели, признаки, фенотипы GDTW, модели, метрики.

Переменные окружения:
- CLUSTERBP_DEBUG=1 — быстрый прогон на 1 500 пациентах; модели, предсказания и таблицы пишутся
  в cache_v2_debug/ и results_v2_debug/, чтобы не затереть результаты полного прогона;
- CLUSTERBP_RAW_DIR — папка с выгрузкой (по умолчанию 2025_04_15);
- CLUSTERBP_TAG — метка другого набора данных (например, synthetic): все промежуточные данные и результаты
  пишутся в отдельные папки cache_v2_<метка>/ и results_v2_<метка>/.
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numba import njit, prange
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import average_precision_score, f1_score
from xgboost import XGBClassifier

DEBUG = os.environ.get('CLUSTERBP_DEBUG') == '1'
RAW_DIR = os.environ.get('CLUSTERBP_RAW_DIR', '2025_04_15')   # исходная выгрузка оператора (не в git)
_TAG = os.environ.get('CLUSTERBP_TAG', '')
_SUFFIX = f'_{_TAG}' if _TAG else ''
DATA_DIR = 'cache_v2' + _SUFFIX                                # очищенные данные и окна
CACHE_DIR = DATA_DIR + ('_debug' if DEBUG else '')             # модели и предсказания
RESULTS_DIR = 'results_v2' + _SUFFIX + ('_debug' if DEBUG else '')
MODELS_DIR = os.path.join(CACHE_DIR, 'models')
for _d in [DATA_DIR, CACHE_DIR, RESULTS_DIR, MODELS_DIR]:
    os.makedirs(_d, exist_ok=True)

SEED = 42
# Набор данных и шкала времени. 'pmp' — выгрузка оператора (время в днях). 'vitaldb' — открытый набор VitalDB,
# подготовленный datasets/vitaldb.py: время в минутах, поэтому все длительности ниже читаются в минутах
# (окно истории 14 мин, шаг 3 мин и т. д.), а цели — эпизоды гипо- и гипертензии.
PRESET = os.environ.get('CLUSTERBP_PRESET', 'pmp')
TIME_UNIT = {'pmp': 'день', 'vitaldb': 'минута'}[PRESET]
HISTORY_DAYS = 14      # окно истории
STRIDE_DAYS = 3        # шаг между отсечками
MIN_HIST_MEAS = 3      # минимум измерений в окне
MAX_POINTS = 20        # длина последовательности для GRU
N_FOLDS = 5
INNER_VAL_FRAC = 0.2
NEG_PER_POS = 3        # отрицательных на один положительный при обучении GRU
PHENO_DAYS = 30        # длина ряда для фенотипа
N_CLUSTERS = 3
N_CLUST_PATIENTS = 1500
GDTW_WINDOW = 14.0
GDTW_GAMMA = 1.0
LONG_SPANS = (30, 60, 90)   # длинная история: периоды до отсечки

if PRESET == 'pmp':
    # Состав целей выбран по данным (см. results_v2/shift_subgroups/B_codes_*.csv):
    # 33902 — красная зона пульса, а не давления; 33810 — «признаки достижения целевого АД», по смыслу противоположен B
    # и появляется только в отдельные месяцы, давая скачки доли B. Старые составы оставлены как проверки чувствительности.
    TARGETS = {
        'A':   dict(codes=[33012, 33901], gap=1, horizon=7, full=True,
                    title='A: острый подъём АД (красные зоны)'),
        'B':   dict(codes=[33601, 33801, 33802, 33911], gap=7, horizon=7, full=True,
                    title='B: превышение целевого АД (через неделю)'),
        'OLD': dict(codes=[33901, 33900, 33012, 33601, 33801], gap=0, horizon=7, full=True,
                    title='OLD: цель из первой версии статьи'),
        # Проверки чувствительности: только базовые правила, LR и XGBoost
        'A_with33902': dict(codes=[33012, 33901, 33902], gap=1, horizon=7, full=False,
                            title='A с 33902 (красная зона ЧП) — состав первой версии v2'),
        'B_with33810': dict(codes=[33601, 33801, 33802, 33810, 33911], gap=7, horizon=7, full=False,
                            title='B с 33810 (признаки достижения ЦУ) — состав первой версии v2'),
        'B_no33601':   dict(codes=[33801, 33802, 33911], gap=7, horizon=7, full=False,
                            title='B без 33601 (код есть только с ноября 2024)'),
        'B_gap0':      dict(codes=[33601, 33801, 33802, 33911], gap=0, horizon=7, full=False,
                            title='B без зазора'),
    }
elif PRESET == 'vitaldb':
    # Коды событий задаёт datasets/vitaldb.py: 90001 — минута со средним АД < 65, 90002 — минута с САД ≥ 160.
    TARGETS = {
        'A':   dict(codes=[90001], gap=5, horizon=5, full=True, title='A: гипотензия (САД ср. < 65) через 5–10 мин'),
        'B':   dict(codes=[90002], gap=5, horizon=5, full=True, title='B: гипертензия (САД ≥ 160) через 5–10 мин'),
        'OLD': dict(codes=[90001], gap=0, horizon=5, full=True, title='A без зазора'),
    }
else:
    raise ValueError(f'Неизвестный CLUSTERBP_PRESET={PRESET}')
FOLLOWUP_DAYS = max(cfg['gap'] + cfg['horizon'] for cfg in TARGETS.values())   # нужно наблюдение после отсечки
EVENT_GROUPS = {name: cfg['codes'] for name, cfg in TARGETS.items()}
EVENT_GROUPS['ALL'] = None   # все КЗС

# Ветвь программы (experience / control). Даже после выбора программы с указанной ветвью она «undefined»
# у 8 949 из 10 937 пациентов (82%), поэтому признаки grp_* в модели не входят, а анализ подгрупп по ветви пропущен.
# CLUSTERBP_USE_ARM=1 — вернуть их (например, для сравнения с первой версией v2).
USE_ARM_FEATURES = os.environ.get('CLUSTERBP_USE_ARM', '1') == '1'

BP_COLS = ['sbp_mean', 'sbp_std', 'sbp_min', 'sbp_max', 'sbp_last', 'sbp_slope', 'sbp_arv',
           'dbp_mean', 'dbp_std', 'dbp_min', 'dbp_max', 'dbp_last', 'dbp_slope', 'dbp_arv',
           'hr_mean', 'hr_std', 'hr_min', 'hr_max', 'hr_last', 'hr_slope', 'hr_arv',
           'pp_mean', 'pp_std', 'tir140', 'frac160', 'frac180', 'dbp_frac90', 'sbp_last3d_minus_prev',
           'n_meas', 'n_days', 'days_since_last_meas']
STATIC_COLS = ['age', 'bmi', 'days_in_program'] + (['grp_experience', 'grp_control'] if USE_ARM_FEATURES else [])
PHENO_COLS = ['cluster', 'pd0', 'pd1', 'pd2']
EV_COLS = [f'ev_{g}_{k}' for g in ['A', 'B', 'OLD', 'ALL'] for k in ['7', '14', 'days_since']]
LH_EV_GROUPS = ['A', 'B', 'ALL']
LH_BP_COLS = [f'lh{L}_{n}' for L in LONG_SPANS for n in ['days', 'sbp_mean', 'sbp_sd', 'sbp_arv', 'sbp_slope',
                                                          'sbp_frac140', 'dbp_mean', 'dbp_sd', 'sbp_mean14_minus']] \
             + [f'lh{max(LONG_SPANS)}_sbp_daymax']
LH_EV_COLS = [f'lh{L}_ev_{g}' for g in LH_EV_GROUPS for L in LONG_SPANS] \
             + [f'lh_ev_{g}_{k}' for g in LH_EV_GROUPS for k in ['total', 'per_week']]
LH_COLS = LH_BP_COLS + LH_EV_COLS
FEATURE_SETS = {
    'paper6': ['sbp_mean', 'sbp_std', 'tir140', 'age', 'bmi', 'cluster'],
    'bp': BP_COLS + STATIC_COLS + PHENO_COLS + LH_BP_COLS,          # всё про давление, без истории событий
    'full': BP_COLS + STATIC_COLS + PHENO_COLS + EV_COLS + LH_COLS,
}
LR_COLS = BP_COLS + STATIC_COLS + ['c0', 'c1', 'c2', 'pd0', 'pd1', 'pd2'] + EV_COLS + LH_COLS


# ---------------------------------------------------------------- кэш

def config_hash():
    """Хэш всего, от чего зависят окна, цели и признаки. Пишется рядом с кэшем и проверяется при загрузке."""
    cfg = {'preset': PRESET, 'targets': {k: [sorted(v['codes']), v['gap'], v['horizon']] for k, v in TARGETS.items()},
           'window': [HISTORY_DAYS, STRIDE_DAYS, MIN_HIST_MEAS, MAX_POINTS, FOLLOWUP_DAYS, PHENO_DAYS, list(LONG_SPANS)],
           'full': FEATURE_SETS['full'], 'debug': DEBUG}
    return hashlib.md5(json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:10]


def _meta_path(name, where):
    return os.path.join(where, f'{name}.meta.json')


def cache_is_current(name, where=None):
    """True, если кэш name построен при текущих целях и признаках."""
    p = _meta_path(name, where or CACHE_DIR)
    return os.path.exists(p) and json.load(open(p)).get('hash') == config_hash()


def mark_cache(name, where=None):
    json.dump({'hash': config_hash()}, open(_meta_path(name, where or CACHE_DIR), 'w'))


# ---------------------------------------------------------------- данные

def load_cached():
    """Окна, последовательности и ряды для фенотипов, построенные в clusterbp_v2.ipynb, и его предсказания."""
    check_windows()
    if not cache_is_current('cv'):
        raise RuntimeError('Предсказания кросс-валидации построены при других целях или признаках: '
                           'перезапустите clusterbp_v2.ipynb.')
    win = pd.read_parquet(os.path.join(DATA_DIR, 'windows.parquet'))
    X_seq = np.load(os.path.join(DATA_DIR, 'X_seq.npy'))
    seq_len = np.load(os.path.join(DATA_DIR, 'seq_len.npy'))
    pheno_win = np.load(os.path.join(DATA_DIR, 'pheno_win.npy'))
    oof = pd.read_parquet(os.path.join(CACHE_DIR, 'oof_predictions.parquet'))
    assert (oof['pid'].values == win['pid'].values).all() and (oof['cut'].values == win['cut'].values).all()
    win['fold'] = oof['fold'].values
    return win, X_seq, seq_len, pheno_win, oof


def check_windows():
    """Окна в DATA_DIR должны быть построены при текущих целях и признаках (их строит clusterbp_v2.ipynb)."""
    if not cache_is_current('windows', DATA_DIR):
        raise RuntimeError(f'Окна в {DATA_DIR}/ построены при других целях или признаках (или до их проверки): '
                           'перезапустите clusterbp_v2.ipynb — он перестроит их сам.')


def event_counts(win, events, codes, lo_off, hi_off=0.0):
    """Число событий из codes (None — все) в [cut − lo_off, cut − hi_off) для каждого окна."""
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


def long_history_features(win, meas, events, spans=LONG_SPANS):
    """Признаки за последние 30/60/90 дней до отсечки (столбцы LH_COLS) — по среднесуточным значениям.

    По Rothwell прогностична долговременная (межсуточная) вариабельность САД, поэтому кроме среднего считаются SD,
    ARV и наклон среднесуточного САД, доля дней со средним САД ≥ 140, отклонение последних 14 дней от долговременного
    уровня и максимум. Счётчики событий A, B и всех КЗС — за те же периоды и за всё время в программе.
    Используются только дни строго до отсечки (отсечка — полночь), поэтому признаки причинные.
    """
    m = meas[['pid', 'td', 'sbp', 'dbp']].copy()
    m['day'] = np.floor(m['td'])
    daily = m.groupby(['pid', 'day'])[['sbp', 'dbp']].mean().reset_index()
    by_pid = {p: g for p, g in daily.groupby('pid')}
    cols = {c: np.full(len(win), np.nan) for c in LH_BP_COLS}
    P = lambda x: np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
    cut_all = win['cut'].values
    for p, idx in win.groupby('pid').indices.items():
        g = by_pid.get(p)
        if g is None:
            continue
        d, s, db = g['day'].values, g['sbp'].values, g['dbp'].values
        cut = cut_all[idx]
        Ps, Ps2, Pd, Pd2 = P(s), P(s * s), P(db), P(db * db)
        Phi = P((s >= 140).astype(float))
        Pa = P(np.concatenate([[0.0], np.abs(np.diff(s))]))
        Pu, Puu, Pus = P(d), P(d * d), P(d * s)
        hi = np.searchsorted(d, cut)
        lo14 = np.searchsorted(d, cut - 14)
        n14 = hi - lo14
        mean14 = np.where(n14 > 0, (Ps[hi] - Ps[lo14]) / np.maximum(n14, 1), np.nan)
        for L in spans:
            lo = np.searchsorted(d, cut - L)
            n = hi - lo
            ok, nn = n > 0, np.maximum(n, 1)
            mean = (Ps[hi] - Ps[lo]) / nn
            var = np.maximum((Ps2[hi] - Ps2[lo]) / nn - mean ** 2, 0) * nn / np.maximum(nn - 1, 1)
            dmean = (Pd[hi] - Pd[lo]) / nn
            dvar = np.maximum((Pd2[hi] - Pd2[lo]) / nn - dmean ** 2, 0) * nn / np.maximum(nn - 1, 1)
            arv = np.where(n > 1, (Pa[hi] - Pa[np.minimum(lo + 1, hi)]) / np.maximum(n - 1, 1), np.nan)
            su, suu, sus, ss_ = Pu[hi] - Pu[lo], Puu[hi] - Puu[lo], Pus[hi] - Pus[lo], Ps[hi] - Ps[lo]
            den = n * suu - su ** 2
            slope = np.where(den > 1e-9, (n * sus - su * ss_) / np.where(den > 1e-9, den, 1), np.nan)
            vals = {'days': n, 'sbp_mean': np.where(ok, mean, np.nan), 'sbp_sd': np.where(n > 1, np.sqrt(var), np.nan),
                    'sbp_arv': arv, 'sbp_slope': slope, 'sbp_frac140': np.where(ok, (Phi[hi] - Phi[lo]) / nn, np.nan),
                    'dbp_mean': np.where(ok, dmean, np.nan), 'dbp_sd': np.where(n > 1, np.sqrt(dvar), np.nan),
                    'sbp_mean14_minus': np.where(ok, mean14 - mean, np.nan)}
            for k, v in vals.items():
                cols[f'lh{L}_{k}'][idx] = v
        cmax = np.full(len(cut), np.nan)
        for i, c in enumerate(cut):
            j0, j1 = np.searchsorted(d, c - max(spans)), np.searchsorted(d, c)
            if j1 > j0:
                cmax[i] = s[j0:j1].max()
        cols[f'lh{max(spans)}_sbp_daymax'][idx] = cmax
    f = pd.DataFrame(cols)
    for g in LH_EV_GROUPS:
        codes = EVENT_GROUPS[g]
        for L in spans:
            f[f'lh{L}_ev_{g}'] = event_counts(win, events, codes, L)
        total = event_counts(win, events, codes, 1e9)
        f[f'lh_ev_{g}_total'] = total
        f[f'lh_ev_{g}_per_week'] = total / np.maximum(win['days_in_program'].values, 1) * 7
    assert set(f.columns) == set(LH_COLS), 'столбцы long_history_features расходятся с LH_COLS'
    return f[LH_COLS]


def fold_masks(win, k):
    """Тест — фолд k; внутренняя валидация — 20% пациентов обучающей части (те же, что в clusterbp_v2.ipynb)."""
    patients = win['pid'].values
    te = win['fold'].values == k
    train_pids = np.unique(patients[~te])
    rng = np.random.default_rng(SEED + k)
    val_pids = rng.choice(train_pids, int(len(train_pids) * INNER_VAL_FRAC), replace=False)
    va = ~te & np.isin(patients, val_pids)
    return ~te & ~va, va, te


def load_regions():
    """Регион (субъект РФ) каждого пациента из справочника «МО и Пациенты»."""
    grp = pd.read_csv(os.path.join(RAW_DIR, 'МО и Пациенты 10.04.2025.csv'), sep=';')
    pid = pd.to_numeric(grp['id пациента'].astype(str).str.replace(r'\s', '', regex=True), errors='coerce')
    ok = pid.notna().values
    s = pd.Series(grp['название субъекта'].values[ok], index=pid[ok].astype('int64').values)
    return s[~s.index.duplicated()]


# ---------------------------------------------------------------- фенотипы GDTW

@njit(cache=True)
def gdtw(s1, t1, s2, t2, window, gamma):
    n, m = len(s1), len(s2)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dt = abs(t1[i - 1] - t2[j - 1])
            if dt > window:
                continue
            c = (s1[i - 1] - s2[j - 1]) ** 2 + gamma * (dt / window) ** 2
            D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return np.sqrt(D[n, m])


@njit(cache=True)
def zvalid(row):
    idx = np.where(~np.isnan(row))[0]
    v = row[idx].astype(np.float64)
    if len(v) >= 2 and v.std() > 0:
        v = (v - v.mean()) / v.std()
    else:
        v = np.zeros(len(v))
    return v, idx.astype(np.float64)


@njit(parallel=True, cache=True)
def pairwise_gdtw(P, window, gamma):
    n = P.shape[0]
    out = np.zeros((n, n))
    for i in prange(n):
        si, ti = zvalid(P[i])
        for j in range(i + 1, n):
            sj, tj = zvalid(P[j])
            out[i, j] = gdtw(si, ti, sj, tj, window, gamma)
    return out


@njit(parallel=True, cache=True)
def assign_gdtw(W, T, window, gamma):
    n, k = W.shape[0], T.shape[0]
    out = np.full((n, k), np.nan)
    for i in prange(n):
        s, t = zvalid(W[i])
        if len(s) == 0:
            continue
        a, b = int(t[0]), int(t[-1]) + 1
        for c in range(k):
            tt = np.arange(a, b).astype(np.float64)
            out[i, c] = gdtw(s, t, T[c, a:b], tt, window, gamma)
    return out


def fit_phenotypes(pheno_first, pheno_first_pid, train_pids, seed):
    """Шаблоны фенотипов по первым 30 дням обучающих пациентов: GDTW + Ward, k = 3, медианный z-профиль.

    Кластеры упорядочены по средней вариабельности САД участников: кластер 0 — самый вариабельный.
    """
    P = pheno_first[np.isin(pheno_first_pid, train_pids)]
    valid = ~np.isnan(P)
    P = P[valid[:, 0] & valid[:, 25:].any(1) & (valid.sum(1) >= 15)]
    rng = np.random.default_rng(seed)
    if len(P) > N_CLUST_PATIENTS:
        P = P[rng.choice(len(P), N_CLUST_PATIENTS, replace=False)]
    Dm = pairwise_gdtw(P, GDTW_WINDOW, GDTW_GAMMA)
    Dm = Dm + Dm.T
    Dm[~np.isfinite(Dm)] = np.nanmax(Dm[np.isfinite(Dm)])
    labels = fcluster(linkage(squareform(Dm, checks=False), method='ward'), N_CLUSTERS, criterion='maxclust') - 1
    Z = np.full(P.shape, np.nan)
    for i in range(len(P)):
        v, idx = zvalid(P[i])
        Z[i, idx.astype(int)] = v
    vol = np.array([np.nanmean(np.nanstd(P[labels == c], axis=1)) for c in range(N_CLUSTERS)])
    order = np.argsort(-vol)
    templates = np.array([pd.Series(np.nanmedian(Z[labels == c], axis=0)).interpolate(limit_direction='both').values
                          for c in order])
    sizes = np.array([(labels == c).sum() for c in order])
    return templates, sizes, vol[order]


def assign_phenotypes(pheno_win, templates):
    """Ближайший шаблон (1-NN по GDTW) по прошлым 30 дням окна."""
    dist = assign_gdtw(pheno_win, templates, GDTW_WINDOW, GDTW_GAMMA)
    cl = np.where(np.isnan(dist).all(1), -1, np.nanargmin(np.where(np.isnan(dist), np.inf, dist), axis=1))
    return cl, dist


def full_history_profiles(meas):
    """Ряд среднесуточного САД за ВСЁ наблюдение пациента, сжатый до PHENO_DAYS точек (нормированное время).

    Так фенотип строился в первой версии статьи: по всему ряду, включая будущее относительно прогноза.
    Нужен только аудиту утечек (переключатель «фенотип по всей истории»); в моделях не используется.
    Возвращает (матрица [пациенты × PHENO_DAYS], массив pid).
    """
    m = meas[['pid', 'td', 'sbp']].copy()
    m['day'] = np.floor(m['td'])
    daily = m.groupby(['pid', 'day'])['sbp'].mean().reset_index()
    rows, pids = [], []
    for p, g in daily.groupby('pid'):
        d = g['day'].values
        span = d[-1] - d[0] + 1
        b = np.minimum(((d - d[0]) / span * PHENO_DAYS).astype(int), PHENO_DAYS - 1)
        prof = np.full(PHENO_DAYS, np.nan, dtype=np.float32)
        sums, cnt = np.bincount(b, g['sbp'].values, PHENO_DAYS), np.bincount(b, minlength=PHENO_DAYS)
        prof[cnt > 0] = sums[cnt > 0] / cnt[cnt > 0]
        rows.append(prof)
        pids.append(p)
    return np.stack(rows), np.array(pids)


def set_phenotype_columns(win, cl, dist):
    win['cluster'] = cl
    for c in range(N_CLUSTERS):
        win[f'pd{c}'] = dist[:, c]
        win[f'c{c}'] = (cl == c).astype(float)


def add_phenotype_columns(win, pheno_win, k):
    """Кластеры окон по шаблонам фолда k, сохранённым в clusterbp_v2.ipynb."""
    templates = np.load(os.path.join(MODELS_DIR, f'templates_fold{k}.npy'))
    cl, dist = assign_phenotypes(pheno_win, templates)
    set_phenotype_columns(win, cl, dist)
    return cl, dist


# ---------------------------------------------------------------- модели

def best_threshold(y, s):
    """Порог, максимизирующий Macro F1 (перебор всех различающихся значений скора)."""
    o = np.argsort(-s, kind='stable')
    ys, ss = y[o], s[o]
    tp = np.cumsum(ys)
    k = np.arange(1, len(y) + 1)
    fp = k - tp
    P, N = tp[-1], len(y) - tp[-1]
    fn, tn = P - tp, N - fp
    f_pos = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    f_neg = 2 * tn / np.maximum(2 * tn + fn + fp, 1)
    macro = (f_pos + f_neg) / 2
    last_of_tie = np.r_[ss[1:] != ss[:-1], True]
    macro = np.where(last_of_tie, macro, -1)
    i = int(np.argmax(macro))
    thr = ss[i] if i == len(ss) - 1 else (ss[i] + ss[i + 1]) / 2
    return float(thr), float(macro[i])


def fit_xgb(Xtr, ytr, Xva, yva, seed, **kw):
    m = XGBClassifier(n_estimators=2000, max_depth=5, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, tree_method='hist', eval_metric='aucpr',
                      early_stopping_rounds=100, n_jobs=-1, random_state=seed)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False, **kw)
    return m


class GRUAttn(nn.Module):
    """GRU(64) с аддитивным вниманием по последним измерениям окна."""

    def __init__(self, n_in, hidden=64):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, batch_first=True)
        self.att = nn.Sequential(nn.Linear(hidden, 64), nn.Tanh(), nn.Linear(64, 1))
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, mask):
        h, _ = self.gru(x)
        a = self.att(h).squeeze(-1).masked_fill(~mask, -1e9)
        w = torch.softmax(a, dim=1).unsqueeze(-1)
        return self.out(self.drop((w * h).sum(1))).squeeze(-1)


def seq_mask(L):
    return np.arange(MAX_POINTS)[None, :] >= (MAX_POINTS - L)[:, None]


def gru_logits(model, X, L, mu, sd, bs=8192):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            m = seq_mask(L[i:i + bs])
            Z = np.where(m[..., None], (X[i:i + bs] - mu) / sd, 0).astype(np.float32)
            out.append(model(torch.from_numpy(Z), torch.from_numpy(m)).numpy())
    return np.concatenate(out)


class TCN(nn.Module):
    """Свёрточная сеть с причинными расширенными свёртками (dilation 1, 2, 4, 8) по последним измерениям окна."""

    def __init__(self, n_in, channels=64, kernel=3, dilations=(1, 2, 4, 8), dropout=0.2):
        super().__init__()
        self.inp = nn.Conv1d(n_in, channels, 1)
        self.blocks = nn.ModuleList()
        self.pads = list((kernel - 1) * d for d in dilations)
        for d in dilations:
            self.blocks.append(nn.Sequential(nn.Conv1d(channels, channels, kernel, dilation=d), nn.ReLU(),
                                             nn.Dropout(dropout)))
        self.out = nn.Linear(2 * channels, 1)

    def forward(self, x, mask):
        h = self.inp(x.transpose(1, 2))
        for pad, block in zip(self.pads, self.blocks):
            h = h + block(nn.functional.pad(h, (pad, 0)))
        h = h.transpose(1, 2)                                     # (B, T, C)
        m = mask.unsqueeze(-1).float()
        mean = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.out(torch.cat([mean, h[:, -1]], dim=1)).squeeze(-1)


class SeqTransformer(nn.Module):
    """Небольшой Transformer-энкодер (2 слоя, 4 головы) с маскированием паддинга и усреднением по измерениям."""

    def __init__(self, n_in, d_model=64, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(n_in, d_model)
        self.pos = nn.Parameter(torch.zeros(1, MAX_POINTS, d_model))
        layer = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=2 * d_model, dropout=dropout,
                                           batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Linear(d_model, 1)

    def forward(self, x, mask):
        h = self.enc(self.inp(x) + self.pos, src_key_padding_mask=~mask)
        m = mask.unsqueeze(-1).float()
        return self.out((h * m).sum(1) / m.sum(1).clamp(min=1)).squeeze(-1)


class TimeAwareGRU(nn.Module):
    """GRU с затуханием скрытого состояния по времени между измерениями (идея GRU-D / DATA-GRU)
    и аддитивным вниманием, как в GRUAttn. Канал 4 входа — log часов от предыдущего измерения."""

    def __init__(self, n_in, hidden=64, dt_channel=4):
        super().__init__()
        self.cell = nn.GRUCell(n_in, hidden)
        self.decay = nn.Linear(1, hidden)
        self.dt_channel = dt_channel
        self.att = nn.Sequential(nn.Linear(hidden, 64), nn.Tanh(), nn.Linear(64, 1))
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, mask):
        B, T, _ = x.shape
        h = x.new_zeros(B, self.cell.hidden_size)
        outs = []
        for t in range(T):
            gamma = torch.exp(-nn.functional.softplus(self.decay(x[:, t, self.dt_channel:self.dt_channel + 1])))
            h_new = self.cell(x[:, t], h * gamma)
            h = torch.where(mask[:, t:t + 1], h_new, h)
            outs.append(h)
        hs = torch.stack(outs, 1)
        a = self.att(hs).squeeze(-1).masked_fill(~mask, -1e9)
        w = torch.softmax(a, dim=1).unsqueeze(-1)
        return self.out(self.drop((w * hs).sum(1))).squeeze(-1)


SEQ_MODELS = {'GRU': GRUAttn, 'TimeAwareGRU': TimeAwareGRU, 'TCN': TCN, 'Transformer': SeqTransformer}


def train_gru(Xtr, Ltr, ytr, Xva, Lva, yva, seed, max_epochs=25, patience=3, bs=512):
    """Обучение GRU (см. train_seq_model)."""
    return train_seq_model(GRUAttn, Xtr, Ltr, ytr, Xva, Lva, yva, seed, max_epochs, patience, bs)


def train_seq_model(model_cls, Xtr, Ltr, ytr, Xva, Lva, yva, seed, max_epochs=25, patience=3, bs=512):
    """Обучение модели последовательностей: отрицательные окна прорежены до NEG_PER_POS:1,
    ранняя остановка по AUPRC на валидации.

    Возвращает модель, параметры стандартизации, поправку к logit на прореживание (log доли оставленных
    отрицательных) и AUPRC на валидации по эпохам.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    pos, neg = np.flatnonzero(ytr == 1), np.flatnonzero(ytr == 0)
    n_neg = min(len(neg), NEG_PER_POS * len(pos))
    keep_rate = n_neg / len(neg)
    idx = np.concatenate([pos, rng.choice(neg, n_neg, replace=False)])
    X, L, y = Xtr[idx], Ltr[idx], ytr[idx].astype(np.float32)
    m = seq_mask(L)
    mu = X[m].mean(0)
    sd = X[m].std(0) + 1e-6
    Z = np.where(m[..., None], (X - mu) / sd, 0).astype(np.float32)
    Zt, Mt, yt = torch.from_numpy(Z), torch.from_numpy(m), torch.from_numpy(y)

    model = model_cls(X.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    best, best_state, bad, log = -1.0, None, 0, []
    for _ in range(max_epochs):
        model.train()
        perm = torch.from_numpy(rng.permutation(len(y)))
        for i in range(0, len(y), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(Zt[b], Mt[b]), yt[b])
            loss.backward()
            opt.step()
        ap = average_precision_score(yva, gru_logits(model, Xva, Lva, mu, sd))
        log.append(ap)
        if ap > best:
            best, bad = ap, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, mu, sd, np.log(keep_rate), log


def load_gru(target, k, n_in=5):
    ck = torch.load(os.path.join(MODELS_DIR, f'gru_{target}_fold{k}.pt'), weights_only=False)
    model = GRUAttn(n_in)
    model.load_state_dict(ck['state'])
    return model, ck['mu'], ck['sd'], ck['corr']


def load_xgb(feature_set, target, k):
    m = XGBClassifier()
    m.load_model(os.path.join(MODELS_DIR, f'xgb_{feature_set}_{target}_fold{k}.json'))
    return m


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def tune_blend(y, pg, px, clusters=None, weights=np.round(np.arange(0, 1.0001, 0.05), 2), min_pos=30):
    """Подбор (w, τ) смеси w·p_GRU + (1−w)·p_XGB по Macro F1: глобально и, если даны кластеры, для каждого кластера."""
    def search(mask):
        best = (-1, 0.5, 0.5)
        for w in weights:
            s = w * pg[mask] + (1 - w) * px[mask]
            thr, f = best_threshold(y[mask], s)
            if f > best[0]:
                best = (f, w, thr)
        return best
    _, w, thr = search(np.ones(len(y), bool))
    params = {'global': (w, thr)}
    if clusters is not None:
        for c in np.unique(clusters):
            mask = clusters == c
            params[int(c)] = search(mask)[1:] if y[mask].sum() >= min_pos else (w, thr)
    return params


def apply_blend(pg, px, params, clusters=None):
    if clusters is None:
        w, thr = params['global']
        s = w * pg + (1 - w) * px
        return s, (s > thr).astype(np.int8)
    s = np.empty(len(pg))
    pred = np.empty(len(pg), dtype=np.int8)
    for c in np.unique(clusters):
        mask = clusters == c
        w, thr = params.get(int(c), params['global'])
        s[mask] = w * pg[mask] + (1 - w) * px[mask]
        pred[mask] = s[mask] > thr
    return s, pred


# ---------------------------------------------------------------- метрики

def boot_eval(y, pid, models, n_boot=1000, seed=SEED):
    """Бутстреп по пациентам: массив [n_boot, n_models, 3] (AUROC, AUPRC, Macro F1)."""
    _, pidx = np.unique(pid, return_inverse=True)
    n_pat = pidx.max() + 1
    prep = []
    for score, pred in models:
        o = np.argsort(-score, kind='stable')
        ss = score[o]
        g = np.concatenate([[0], np.cumsum(ss[1:] != ss[:-1])])
        prep.append((o, g, pred))
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, len(models), 3))
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, n_pat, n_pat), minlength=n_pat)[pidx].astype(float)
        for j, (o, g, pred) in enumerate(prep):
            wo, yo = w[o], y[o]
            pos = np.bincount(g, wo * yo)
            neg = np.bincount(g, wo * (1 - yo))
            P, N = pos.sum(), neg.sum()
            cpos, cneg = np.cumsum(pos), np.cumsum(neg)
            auroc = (pos * (N - cneg) + 0.5 * pos * neg).sum() / (P * N)
            ap = (pos * cpos / np.maximum(cpos + cneg, 1e-12)).sum() / P
            tp = (w * y * pred).sum(); fp = (w * (1 - y) * pred).sum()
            fn = (w * y * (1 - pred)).sum(); tn = (w * (1 - y) * (1 - pred)).sum()
            f1 = (2 * tp / (2 * tp + fp + fn) + 2 * tn / (2 * tn + fn + fp)) / 2
            out[b, j] = auroc, ap, f1
    return out


def ci(a):
    return f'[{np.percentile(a, 2.5):.3f}; {np.percentile(a, 97.5):.3f}]'


def balanced_f1(y, score, fold, idx, n_folds=N_FOLDS):
    """Macro F1 на сбалансированном тесте; порог подбирается на сбалансированных окнах остальных фолдов."""
    fs = []
    for k in range(n_folds):
        te, tr = idx[fold[idx] == k], idx[fold[idx] != k]
        thr, _ = best_threshold(y[tr], score[tr])
        fs.append(f1_score(y[te], score[te] > thr, average='macro'))
    return np.mean(fs), np.std(fs, ddof=1)
