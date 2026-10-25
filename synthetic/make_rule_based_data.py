"""
Синтетическая выгрузка для проверки пайплайна ClusterBP v2 без доступа к закрытым данным.

В отличие от make_synthetic_data.py (SDV), генератор не обучается на реальных данных вообще:
у каждого пациента моделируется ряд давления на несколько месяцев, а клинически значимые события
(КЗС) выставляются по правилам «зон», похожим на правила системы мониторинга. Структура и имена
файлов совпадают с выгрузкой оператора, поэтому ноутбуки clusterbp_v2*.ipynb работают на ней без изменений.

Модель ряда: базовый уровень САД + медленный тренд + дневная AR(1)-компонента + шум измерения +
эпизоды подъёма (нарастание 2–4 дня, плато, спад). ДАД и ЧП связаны с САД. События:
- 33012 / 33901 — «красная зона АД» (САД ≥ 180 или ДАД ≥ 110), автоматически и с подтверждением врача;
- 33902 / 33006 — «красная зона ЧП» (ЧП ≥ 110); 33004 — «синяя зона АД» (САД < 100);
- 33921 — «жёлтая зона АД» (САД 160–179); 33900 — «критическое отклонение» (САД ≥ 200);
- еженедельно по средним за 7 дней: 33911 «оранжевая зона» (САД ≥ 140), 33801 / 33802 «ускользание ЦУ»,
  33810 «признаки достижения ЦУ» (САД < 130), 33430 «отсутствие измерений»;
- 33601 — редкая «красная зона (э)»; 77710 — протокол первого месяца.

Цифры качества моделей на этих данных не имеют клинического смысла: данные нужны только для того,
чтобы весь пайплайн можно было запустить и проверить.

Использование:
    python synthetic/make_rule_based_data.py [--patients 1500] [--seed 42] [--out synthetic/rule_based]
Запуск пайплайна на результате (результаты пишутся в cache_v2_synthetic/ и results_v2_synthetic/):
    CLUSTERBP_RAW_DIR=synthetic/rule_based CLUSTERBP_TAG=synthetic jupyter nbconvert --execute ... clusterbp_v2.ipynb
"""
import argparse
import csv
import os

import numpy as np
import pandas as pd

PRIMARY_COLS = ['время измерения', 'время сохранения на сервере', 'часовой пояс пациента', 'САД', 'ДАД', 'ЧП',
                'название прибора', 'заводской номер', 'номер регистрации в организации', 'используется ИС ПМП',
                'id пациента', 'код пациента', 'id МО, где наблюдается пациент', 'название МО, где наблюдается пацие',
                'id субъекта', 'id врача', 'дата рождения пациента', 'рост', 'масса', 'основное заболевание',
                'сопутствующие заболевание', 'id программы']
DIAGNOSES = ['Гипертензивная [гипертоническая] болезнь с преимущественным поражением сердца без (застойной) сердечной '
             'недостаточности, I11/9', 'Эссенциальная [первичная] гипертензия, I10', 'Не указано']
DRUGS = ['периндоприл', 'амлодипин', 'лозартан', 'бисопролол', 'индапамид', 'валсартан']
START, END = pd.Timestamp('2023-11-15'), pd.Timestamp('2025-01-05')


def surge_profile(n_days, rng, rate):
    """Сумма эпизодов подъёма давления: нарастание 2–4 дня, плато 1–2 дня, спад 2–3 дня."""
    prof = np.zeros(n_days)
    for start in np.flatnonzero(rng.random(n_days) < rate):
        amp = rng.uniform(20, 50)
        up, flat, down = rng.integers(2, 5), rng.integers(1, 3), rng.integers(2, 4)
        shape = np.concatenate([np.linspace(amp / up, amp, up), np.full(flat, amp), np.linspace(amp, 0, down + 1)[1:]])
        end = min(n_days, start + len(shape))
        prof[start:end] += shape[:end - start]
    return prof


def simulate_patient(i, rng, regions):
    pid = 100000 + i
    prog = 500000 + i
    region_id, region_name = regions[rng.integers(len(regions))]
    mo_id = 1000 + region_id * 10 + rng.integers(0, 5)
    start = START + pd.Timedelta(days=int(rng.integers(0, 330)))
    n_days = int(min(rng.uniform(45, 330), (END - start).days))
    if n_days < 20:
        n_days = 20
    tz = int(rng.choice([2, 3, 4, 5, 7], p=[0.1, 0.45, 0.15, 0.15, 0.15]))

    base = np.clip(rng.normal(133, 11), 105, 175)
    trend = rng.normal(-0.03, 0.05)
    sd_day, sd_meas = rng.uniform(3, 8), rng.uniform(4, 10)
    ar = np.zeros(n_days)
    for d in range(1, n_days):
        ar[d] = 0.7 * ar[d - 1] + rng.normal(0, sd_day)
    rate = rng.gamma(1.0, 1 / 100)                     # эпизоды подъёма: в среднем раз в ~3 месяца, разброс между пациентами
    sbp_day = base + trend * np.arange(n_days) + ar + surge_profile(n_days, rng, rate)
    dbp_off, hr_base = rng.normal(12, 5), rng.normal(72, 8)
    tachy_rate = rng.gamma(0.5, 1 / 200)
    adherence = rng.beta(6, 2)

    rows = []
    for d in range(n_days):
        if rng.random() > adherence:
            continue
        hr_spike = rng.uniform(30, 50) if rng.random() < tachy_rate else 0.0
        for hour in sorted(rng.choice([8, 9, 20, 21], size=rng.integers(1, 4), replace=False)):
            t = start + pd.Timedelta(days=d, hours=int(hour) - tz, minutes=int(rng.integers(0, 60)),
                                     seconds=int(rng.integers(0, 60)))
            sbp = sbp_day[d] + rng.normal(0, sd_meas)
            dbp = 0.5 * sbp + dbp_off + rng.normal(0, 4)
            hr = hr_base + hr_spike + rng.normal(0, 6)
            rows.append((t, int(np.clip(round(sbp), 70, 240)), int(np.clip(round(dbp), 40, 140)),
                         int(np.clip(round(hr), 40, 180))))
    meas = pd.DataFrame(rows, columns=['t', 'sbp', 'dbp', 'hr'])
    return dict(pid=pid, prog=prog, region_id=region_id, region_name=region_name, mo_id=mo_id, tz=tz, start=start,
                n_days=n_days, meas=meas)


def zone_events(p, rng):
    """КЗС по правилам зон: по отдельным измерениям и по еженедельным средним."""
    ev = []
    m = p['meas']
    for t, sbp, dbp, hr in m.itertuples(index=False):
        srv = t + pd.Timedelta(seconds=int(rng.integers(2, 40)))
        if sbp >= 180 or dbp >= 110:
            ev.append((srv + pd.Timedelta(minutes=int(rng.integers(1, 30))), 33012))
            if rng.random() < 0.6:
                ev.append((srv + pd.Timedelta(hours=float(rng.uniform(1, 10))), 33901))
            if rng.random() < 0.02:
                ev.append((srv + pd.Timedelta(hours=float(rng.uniform(1, 24))), 33601))
        elif 160 <= sbp < 180 and rng.random() < 0.5:
            ev.append((srv + pd.Timedelta(hours=float(rng.uniform(1, 10))), 33921))
        if sbp >= 200:
            ev.append((srv + pd.Timedelta(minutes=int(rng.integers(1, 30))), 33900))
        if sbp < 100:
            ev.append((srv + pd.Timedelta(minutes=int(rng.integers(1, 30))), 33004))
        if hr >= 110:
            ev.append((srv + pd.Timedelta(minutes=int(rng.integers(1, 30))), 33006))
            if rng.random() < 0.6:
                ev.append((srv + pd.Timedelta(hours=float(rng.uniform(1, 10))), 33902))
    for d in range(7, p['n_days'], 7):
        check = p['start'] + pd.Timedelta(days=d, hours=9)
        week = m[(m.t >= check - pd.Timedelta(days=7)) & (m.t < check)]
        if (m.t >= check - pd.Timedelta(days=3)).sum() == (m.t >= check).sum():
            ev.append((check, 33430))
        if len(week) < 3:
            continue
        if week.sbp.mean() >= 145 and rng.random() < 0.7:
            ev.append((check + pd.Timedelta(hours=float(rng.uniform(1, 8))), 33911))
        if week.sbp.mean() >= 155 and rng.random() < 0.5:
            ev.append((check, 33801))
        if week.dbp.mean() >= 92 and rng.random() < 0.4:
            ev.append((check + pd.Timedelta(hours=float(rng.uniform(1, 24))), 33802))
        if week.sbp.mean() < 130 and rng.random() < 0.3:
            ev.append((check, 33810))
    if p['n_days'] > 30:
        ev.append((p['start'] + pd.Timedelta(days=30, hours=10), 77710))
    return ev


def fmt(ts):
    return ts.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--patients', type=int, default=1500)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'rule_based'))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    regions = [(i, f'Регион {i:02d}') for i in range(1, 20)]

    patients = [simulate_patient(i, rng, regions) for i in range(args.patients)]
    prim, events, links, therapy = [], [], [], []
    for p in patients:
        birth = (p['start'] - pd.Timedelta(days=int(365.25 * np.clip(rng.normal(64, 11), 30, 90)))).strftime('%Y-%m-%d')
        has_hw = rng.random() > 0.45
        height = round(rng.normal(167, 9)) if has_hw else ''
        weight = round(rng.normal(85, 15)) if has_hw else ''
        serial = int(rng.integers(10000, 99999))
        diag = DIAGNOSES[rng.integers(len(DIAGNOSES))]
        doctor = int(rng.integers(30000, 40000))
        for t, sbp, dbp, hr in p['meas'].itertuples(index=False):
            srv = t + pd.Timedelta(seconds=float(rng.uniform(2, 40)))
            prim.append([fmt(t), fmt(srv), p['tz'], sbp, dbp, hr, 'Тонометр', serial, f'reg-{p["pid"]}', 'true',
                         p['pid'], f'P{p["pid"]}', p['mo_id'], f'МО {p["mo_id"]}', p['region_id'], doctor, birth,
                         height, weight, diag, '', p['prog']])
        for t, code in zone_events(p, rng):
            events.append([p['prog'], p['pid'], fmt(t), code])
        group = rng.choice(['undefined', 'experience', 'control_1'], p=[0.8, 0.15, 0.05])
        links.append([p['region_name'], p['mo_id'], p['prog'], group, p['pid']])
        for drug in rng.choice(DRUGS, size=rng.integers(1, 4), replace=False):
            d0 = p['start'].strftime('%Y-%m-%d')
            therapy.append([p['prog'], fmt(p['start']), d0, 'true', 'true', drug, '{{утро}}', 'Постоянно, регулярно',
                            'Перорально', '', d0, ''])

    def write(name, header, rows, sep=','):
        with open(os.path.join(args.out, name), 'w', encoding='utf-8', newline='') as f:
            w = csv.writer(f, delimiter=sep, quoting=csv.QUOTE_MINIMAL)
            w.writerow(header)
            w.writerows(rows)

    write('Первичные данные от 10.04.2025.csv', PRIMARY_COLS, sorted(prim, key=lambda r: (r[10], r[0])))
    write('Клинически значимые события от 15.04.2025.csv',
          ['id программы', 'id пациента', 'дата, время формирования КЗС', 'код КЗС'], sorted(events, key=lambda r: (r[1], r[2])))
    write('МО и Пациенты 10.04.2025.csv', ['название субъекта', 'id МО', 'id программы', 'группа наблюдения', 'id пациента'],
          links, sep=';')
    write('Медикаментозная терапия от 10.04.2025.csv',
          ['id программы', 'дата начала программы', 'дата назначения', 'назначен очно (да/нет)',
           'принимаемая терапия (да/нет)', 'МНН', 'время приема', 'регулярность', 'способ применения',
           'продолж. приема на момент выгрузки', 'дата начала приема', 'дата окончания приема'], therapy)
    write('Идентификаторы МО 15.04.2025.csv', ['id МО', 'название МО'],
          sorted({(p['mo_id'], f'МО {p["mo_id"]}') for p in patients}))
    write('Идентификаторы Субъектов 10.04.2025.csv', ['id субъекта', 'название субъекта'], regions, sep=';')
    print(f'Пациентов: {len(patients)}, измерений: {len(prim):,}, событий: {len(events):,} -> {args.out}')


if __name__ == '__main__':
    main()
