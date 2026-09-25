"""
Генератор синтетических данных.
Читает РЕАЛЬНЫЕ CSV из ~/Downloads только для снятия схемы,
пишет синтетику в ~/Downloads/raw_data_synthetic/.
Реальные данные НЕ копируются и НЕ сохраняются.

Формат вывода максимально приближен к реальным файлам:
- все значения в кавычках (как у Excel/1С)
- разделитель: ';' для справочников и patient_program, ',' для остальных
"""
import csv
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)

HERE = os.path.dirname(os.path.abspath(__file__))
REAL = os.path.dirname(HERE)                       # ~/Downloads
OUT  = os.path.join(REAL, "raw_data_synthetic")
os.makedirs(OUT, exist_ok=True)

FILES = {
    "subjects":        ("Идентификаторы Субъектов 10.04.2025.csv",       ";"),
    "mo":              ("Идентификаторы МО 15.04.2025.csv",              ","),
    "patient_program": ("МО и Пациенты 10.04.2025.csv",                  ";"),
    "primary":         ("Первичные данные от 10.04.2025.csv",            ","),
    "therapy":         ("Медикаментозная терапия от 10.04.2025.csv",     ","),
    "kzs":             ("Клинически значимые события от 15.04.2025.csv", ","),
}

# ---------- 1. Снимок схемы реальных файлов ----------
print("Читаю схемы реальных файлов (только первые 3000 строк)...")
schema = {}
for key, (fname, sep) in FILES.items():
    path = os.path.join(REAL, fname)
    try:
        df = pd.read_csv(path, sep=sep, nrows=3000)
    except UnicodeDecodeError:
        df = pd.read_csv(path, sep=sep, nrows=3000, encoding="cp1251")
    schema[key] = df.columns.tolist()
    print(f"  {key:16s} → {len(df.columns)} колонок")

# ---------- 2. Справочники ----------
N_SUBJ, N_MO, N_PAT, N_PROG = 5, 20, 400, 40

subjects = pd.DataFrame({
    "id субъекта":      np.arange(1, N_SUBJ + 1),
    "название субъекта": [f"Region_{i:02d}" for i in range(1, N_SUBJ + 1)],
})
subjects.to_csv(f"{OUT}/Идентификаторы Субъектов.csv", sep=";",
                index=False, quoting=csv.QUOTE_ALL)

mo = pd.DataFrame({
    "id МО":       np.arange(1000, 1000 + N_MO),
    "название МО": [f"MO_{i:03d}" for i in range(1, N_MO + 1)],
})
mo.to_csv(f"{OUT}/Идентификаторы МО.csv", sep=",",
          index=False, quoting=csv.QUOTE_ALL)

# ---------- 3. Пациент-программа ----------
patient_program = pd.DataFrame({
    "название субъекта": RNG.choice(subjects["название субъекта"], N_PAT),
    "id МО":             RNG.choice(mo["id МО"], N_PAT),
    "id программы":      RNG.integers(1, N_PROG + 1, N_PAT),
    "группа наблюдения": RNG.choice(
        ["experience", "control_1", "undefined"], N_PAT, p=[.09, .03, .88]),
    "id пациента":       np.arange(100_000, 100_000 + N_PAT),
})
patient_program.to_csv(f"{OUT}/МО и Пациенты.csv", sep=";",
                       index=False, quoting=csv.QUOTE_ALL)

# карты для связи id пациента → прочие поля
pat2prog = dict(zip(patient_program["id пациента"], patient_program["id программы"]))
pat2mo   = dict(zip(patient_program["id пациента"], patient_program["id МО"]))
pat2grp  = dict(zip(patient_program["id пациента"], patient_program["группа наблюдения"]))
subj_name2id = dict(zip(subjects["название субъекта"], subjects["id субъекта"]))
pat2subj = {p: subj_name2id[s] for p, s in
            zip(patient_program["id пациента"], patient_program["название субъекта"])}

# ---------- 4. Первичные данные (22 колонки) ----------
print("\nГенерирую первичные данные...")
PRIMARY_COLS = [
    "время измерения", "время сохранения на сервере", "часовой пояс пациента",
    "САД", "ДАД", "ЧП", "название прибора", "заводской номер",
    "номер регистрации в организации", "используется ИС ПМП",
    "id пациента", "код пациента",
    "id МО, где наблюдается пациент", "название МО, где наблюдается пацие",
    "id субъекта", "id врача", "дата рождения пациента",
    "рост", "масса", "основное заболевание", "сопутствующие заболевание",
    "id программы",
]
assert PRIMARY_COLS == schema["primary"], "Порядок/имена колонок не совпадают!"

start = datetime(2023, 12, 22)
rows = []
for pid in patient_program["id пациента"]:
    grp = pat2grp[pid]
    n = min(int(RNG.gamma(2.0, 50)) + 5, 400)
    age    = float(np.clip(RNG.normal(63, 12), 18, 100))
    height = float(np.clip(RNG.normal(166, 9), 140, 210))
    weight = float(np.clip(RNG.normal(82, 16), 40, 150))
    tz     = int(RNG.choice([3, 4, 5, 7], p=[.45, .02, .31, .22]))
    if grp == "experience":   base_sad, base_dad = 130, 81
    elif grp == "control_1":  base_sad, base_dad = 129, 81
    else:                     base_sad, base_dad = 127, 80
    dob = (datetime(2024, 1, 1) - timedelta(days=int(age * 365.25))).strftime("%Y-%m-%d")
    device_id  = int(RNG.integers(1, 30))
    device_sn  = f"SN{RNG.integers(100000, 999999)}"
    device_reg = f"REG{RNG.integers(1000, 9999)}"
    doctor_id  = int(RNG.integers(1, 500))

    dates = sorted(start + timedelta(days=int(x))
                   for x in RNG.uniform(0, 365, n))
    for d in dates:
        rows.append({
            "время измерения": d.strftime("%Y-%m-%d %H:%M:%S"),
            "время сохранения на сервере":
                (d + timedelta(seconds=int(RNG.exponential(5)))).strftime("%Y-%m-%d %H:%M:%S"),
            "часовой пояс пациента": tz,
            "САД": int(np.clip(RNG.normal(base_sad, 14), 70, 220)),
            "ДАД": int(np.clip(RNG.normal(base_dad, 10), 40, 130)),
            "ЧП":  int(np.clip(RNG.normal(72, 11), 40, 180)),
            "название прибора": f"Device_{device_id:02d}",
            "заводской номер":  device_sn,
            "номер регистрации в организации": device_reg,
            "используется ИС ПМП": RNG.choice(["да", "нет"], p=[.7, .3]),
            "id пациента":  pid,
            "код пациента": pid,
            "id МО, где наблюдается пациент": pat2mo[pid],
            "название МО, где наблюдается пацие": f"MO_{pat2mo[pid]-999:03d}",
            "id субъекта": pat2subj[pid],
            "id врача":    doctor_id,
            "дата рождения пациента": dob,
            "рост":  height if RNG.random() > 0.44 else None,
            "масса": weight if RNG.random() > 0.44 else None,
            "основное заболевание":      "I10" if RNG.random() > 0.4 else None,
            "сопутствующие заболевание": "E11" if RNG.random() > 0.6 else None,
            "id программы": pat2prog[pid],
        })
primary_df = pd.DataFrame(rows, columns=PRIMARY_COLS)
primary_df.to_csv(f"{OUT}/Первичные данные.csv", sep=",",
                  index=False, quoting=csv.QUOTE_ALL)

# ---------- 5. Терапия (12 колонок) ----------
print("Генерирую терапию...")
THERAPY_COLS = [
    "id программы", "дата начала программы", "дата назначения",
    "назначен очно (да/нет)", "принимаемая терапия (да/нет)", "МНН",
    "время приема", "регулярность", "способ применения",
    "продолж. приема на момент выгрузки", "дата начала приема",
    "дата окончания приема",
]
assert THERAPY_COLS == schema["therapy"], "Порядок/имена колонок не совпадают!"

mnn_pool = ["бисопролол", "индапамид", "амлодипин", "лозартан",
            "периндоприл", "эналаприл", "аторвастатин", "лизиноприл",
            "моксонидин", "валсартан"]
N_TH = 2000
th_rows = []
for _ in range(N_TH):
    pid = int(RNG.choice(patient_program["id пациента"]))
    d_start = start + timedelta(days=int(RNG.uniform(0, 365)))
    d_end   = d_start + timedelta(days=int(RNG.uniform(30, 400)))
    th_rows.append({
        "id программы": pat2prog[pid],
        "дата начала программы": d_start.strftime("%Y-%m-%d"),
        "дата назначения":       d_start.strftime("%Y-%m-%d"),
        "назначен очно (да/нет)": RNG.choice(["да", "нет"], p=[.6, .4]),
        "принимаемая терапия (да/нет)": RNG.choice(["да", "нет"], p=[.8, .2]),
        "МНН": RNG.choice(mnn_pool),
        "время приема": RNG.choice(["утро", "вечер", "утро/вечер"], p=[.4, .3, .3]),
        "регулярность": RNG.choice(["ежедневно", "по требованию"], p=[.85, .15]),
        "способ применения": RNG.choice(["перорально", "подкожно"], p=[.95, .05]),
        "продолж. приема на момент выгрузки":
            int((d_end - d_start).days) if RNG.random() > .3 else None,
        "дата начала приема":    d_start.strftime("%Y-%m-%d"),
        "дата окончания приема": d_end.strftime("%Y-%m-%d"),
    })
therapy_df = pd.DataFrame(th_rows, columns=THERAPY_COLS)
therapy_df.to_csv(f"{OUT}/Медикаментозная терапия.csv", sep=",",
                  index=False, quoting=csv.QUOTE_ALL)

# ---------- 6. КЗС (4 колонки) ----------
print("Генерирую КЗС...")
KZS_COLS = ["id программы", "id пациента",
            "дата, время формирования КЗС", "код КЗС"]
assert KZS_COLS == schema["kzs"], "Порядок/имена колонок не совпадают!"

kzs_codes = [33921, 33923, 33911, 33430, 33922,
             33913, 33431, 33901, 33924, 77710]
kzs_p = [.30, .15, .13, .10, .08, .07, .05, .05, .04, .03]   # ровно 1.0
assert abs(sum(kzs_p) - 1.0) < 1e-9, f"Сумма вероятностей = {sum(kzs_p)}"

N_KZS = 5000
kzs_rows = []
for _ in range(N_KZS):
    pid = int(RNG.choice(patient_program["id пациента"]))
    d = start + timedelta(days=float(RNG.uniform(0, 365)),
                          hours=float(RNG.uniform(0, 24)))
    kzs_rows.append({
        "id программы": pat2prog[pid],
        "id пациента":  pid,
        "дата, время формирования КЗС": d.strftime("%Y-%m-%d %H:%M:%S"),
        "код КЗС": int(RNG.choice(kzs_codes, p=kzs_p)),
    })
kzs_df = pd.DataFrame(kzs_rows, columns=KZS_COLS)
kzs_df.to_csv(f"{OUT}/Клинически значимые события.csv", sep=",",
              index=False, quoting=csv.QUOTE_ALL)

# ---------- Итог ----------
print("\nГотово. Файлы в:", OUT)
for f in sorted(os.listdir(OUT)):
    n = sum(1 for _ in open(os.path.join(OUT, f), encoding="utf-8")) - 1
    print(f"  {f:45s} {n:>7d} строк")