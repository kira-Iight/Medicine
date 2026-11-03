"""Подготовка открытого набора VitalDB (https://vitaldb.net) для пайплайна ClusterBP v2.

VitalDB — интраоперационный мониторинг ~6 400 пациентов, открытый, без регистрации. Здесь он нужен, чтобы проверить
протокол и аудит утечек на данных, которые может скачать любой рецензент: клинические выводы статьи на нём не проверяются.

Скрипт пишет в cache_v2_vitaldb/ те же файлы, что основной ноутбук строит из выгрузки оператора:
- measurements.parquet — pid, td, sbp, dbp, hr (td — время в МИНУТАХ от начала записи; медианы за минуту);
- patients.parquet — pid, birth_days, bmi, group (birth_days подобран так, чтобы возраст в признаках был верным);
- events.parquet — pid, td, code: 90001 — минута со средним АД < 65 мм рт. ст. (гипотензия),
  90002 — минута с САД ≥ 160 мм рт. ст. (гипертензия).
Дальше clusterbp_v2.ipynb и clusterbp_v2_leakage.ipynb работают без изменений при
CLUSTERBP_PRESET=vitaldb CLUSTERBP_TAG=vitaldb: все длительности читаются в минутах (окно истории 14 мин, шаг 3 мин),
цели — эпизод в горизонте 5 мин после зазора 5 мин (A — гипотензия, B — гипертензия; OLD — гипотензия без зазора).

События, как и у оператора, задаются порогом по тому же сигналу, что и вход, поэтому зазор обязателен:
без него цель частично вычисляется из последних минут окна истории.

Использование (нужен доступ в интернет и пакет vitaldb: pip install vitaldb):
    python datasets/prepare_vitaldb.py --cases 600            # первые 600 подходящих операций
    CLUSTERBP_PRESET=vitaldb CLUSTERBP_TAG=vitaldb jupyter nbconvert --execute --to notebook clusterbp_v2.ipynb
    CLUSTERBP_PRESET=vitaldb CLUSTERBP_TAG=vitaldb jupyter nbconvert --execute --to notebook clusterbp_v2_leakage.ipynb
Остальные ноутбуки читают справочники оператора (регионы, терапия) и на VitalDB не запускаются.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

# Числовые тренды монитора Solar 8000: инвазивное АД (артериальная линия) и ЧСС, шаг 2 с
TRACKS = ['Solar8000/ART_SBP', 'Solar8000/ART_DBP', 'Solar8000/ART_MBP', 'Solar8000/HR']
CASES_URL = 'https://api.vitaldb.net/cases'
HYPO_CODE, HYPER_CODE = 90001, 90002
INTERVAL_S = 2


def clean_minutes(raw, interval_s=INTERVAL_S):
    """Массив [n × 4] (SBP, DBP, MBP, HR) с шагом interval_s → медианы за минуту с проверкой правдоподобия.

    Значения вне физиологических границ (артефакты промывки линии, отсоединение) удаляются до усреднения.
    Возвращает DataFrame: minute, sbp, dbp, mbp, hr (только минуты, где есть САД, ДАД и ЧСС).
    """
    df = pd.DataFrame(raw, columns=['sbp', 'dbp', 'mbp', 'hr'], dtype=float)
    ok = df.sbp.between(40, 250) & df.dbp.between(20, 150) & df.mbp.between(30, 200) & (df.sbp > df.dbp)
    df.loc[~ok, ['sbp', 'dbp', 'mbp']] = np.nan
    df.loc[~df.hr.between(30, 220), 'hr'] = np.nan
    df['minute'] = (np.arange(len(df)) * interval_s) // 60
    m = df.groupby('minute').median().dropna(subset=['sbp', 'dbp', 'hr']).reset_index()
    return m


def convert(cases, clinical):
    """cases: dict caseid -> DataFrame из clean_minutes; clinical: таблица api.vitaldb.net/cases.

    Возвращает (measurements, patients, events) в формате пайплайна.
    """
    clin = clinical.set_index('caseid')
    meas, pats, evs = [], [], []
    for cid, m in cases.items():
        if len(m) < 60:                      # меньше часа записи — окна не построить
            continue
        td = m['minute'].values.astype(float)
        meas.append(pd.DataFrame({'pid': cid, 'td': td, 'sbp': m.sbp.values, 'dbp': m.dbp.values, 'hr': m.hr.values}))
        mbp = m['mbp'].values
        for code, mask in [(HYPO_CODE, mbp < 65), (HYPER_CODE, m.sbp.values >= 160)]:
            mask = np.nan_to_num(mask, nan=0).astype(bool)
            evs.append(pd.DataFrame({'pid': cid, 'td': td[mask], 'code': code}))
        age = pd.to_numeric(clin.at[cid, 'age'], errors='coerce') if cid in clin.index else np.nan
        bmi = pd.to_numeric(clin.at[cid, 'bmi'], errors='coerce') if cid in clin.index else np.nan
        # age в признаках = (cut − birth_days) / 365.25; время в минутах, поэтому сдвиг за операцию пренебрежим
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
    import vitaldb   # pip install vitaldb (файл назван prepare_vitaldb.py, чтобы не перекрывать пакет)
    if not hasattr(vitaldb, 'find_cases'):
        raise ImportError(f'Импортирован не пакет vitaldb, а {vitaldb.__file__}: установите пакет (pip install vitaldb) '
                          'и проверьте, что рядом нет файла vitaldb.py')
    clinical = pd.read_csv(CASES_URL)
    ids = sorted(set(vitaldb.find_cases(TRACKS)))
    # только плановые операции под общей анестезией длительностью ≥ 1 ч
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
    print(f'Операций: {meas.pid.nunique()}, минут: {len(meas):,}; '
          f'гипотензия: {(events.code == HYPO_CODE).mean():.1%} событий, '
          f'доля минут с гипотензией: {(events.code == HYPO_CODE).sum() / len(meas):.1%}')


if __name__ == '__main__':
    sys.exit(main())
