"""
Генератор синтетических данных на основе SDV (Synthetic Data Vault).

Обучает GaussianCopulaSynthesizer на реальных CSV (или их подвыборках
для очень больших файлов), генерирует анонимизированные данные с той же
схемой и похожими распределениями. Реальные данные не сохраняются.

Требуется:
    pip install sdv

Использование:
    python make_synthetic_data.py

Входные файлы ищутся в родительской папке (../),
результат кладётся в ../raw_data_synthetic/.
"""
import os
import sys
import csv
import time
import warnings
import logging
import numpy as np
import pandas as pd

# ---------- тишина в консоли ----------
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sdv")
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("sdv").setLevel(logging.ERROR)

try:
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import GaussianCopulaSynthesizer
except ImportError:
    sys.exit(
        "Не установлен SDV. Выполните:\n"
        "    pip install sdv\n"
        "и запустите скрипт снова."
    )

RNG = np.random.default_rng(42)
np.random.seed(42)

HERE = os.path.dirname(os.path.abspath(__file__))
REAL = os.path.dirname(HERE)                # ~/Downloads
OUT  = os.path.join(REAL, "raw_data_synthetic")
os.makedirs(OUT, exist_ok=True)

# ---------- карта реальных файлов ----------
FILES = {
    "subjects":        ("Идентификаторы Субъектов 10.04.2025.csv",       ";"),
    "mo":              ("Идентификаторы МО 15.04.2025.csv",              ","),
    "patient_program": ("МО и Пациенты 10.04.2025.csv",                  ";"),
    "primary":         ("Первичные данные от 10.04.2025.csv",            ","),
    "therapy":         ("Медикаментозная терапия от 10.04.2025.csv",     ","),
    "kzs":             ("Клинически значимые события от 15.04.2025.csv", ","),
}

# ---------- сколько строк брать для обучения SDV ----------
# None = все строки. Для очень больших файлов — случайная подвыборка.
TRAIN_SAMPLE = {
    "subjects":        None,
    "mo":              None,
    "patient_program": None,
    "primary":         30_000,   # из 4.9M
    "therapy":         None,
    "kzs":             30_000,   # из 1.2M
}

# ---------- сколько строк генерировать ----------
GEN_SIZE = {
    "subjects":        5,
    "mo":              20,
    "patient_program": 400,
    "primary":         40_000,
    "therapy":         2_000,
    "kzs":             5_000,
}

WRITE_KW = dict(quoting=csv.QUOTE_ALL, encoding="utf-8")


# =====================================================================
#                          ВСПОМОГАТЕЛЬНЫЕ
# =====================================================================

def load_real(key, fname, sep):
    """Загружает реальный CSV или случайную подвыборку для обучения."""
    path = os.path.join(REAL, fname)
    n = TRAIN_SAMPLE[key]
    if n is None:
        df = pd.read_csv(path, sep=sep, low_memory=False)
    else:
        chunks, total = [], 0
        for chunk in pd.read_csv(path, sep=sep, chunksize=100_000,
                                 low_memory=False):
            chunks.append(chunk)
            total += len(chunk)
            if total >= n * 3:
                break
        df = pd.concat(chunks, ignore_index=True)
        if len(df) > n:
            df = df.sample(n=n, random_state=42).reset_index(drop=True)

    # парсим даты/время — без UserWarning
    for c in df.columns:
        if "дата" in c.lower() or "время" in c.lower():
            try:
                df[c] = pd.to_datetime(df[c], errors="coerce", format="mixed")
            except (TypeError, ValueError):
                try:
                    df[c] = pd.to_datetime(df[c], errors="coerce")
                except Exception:
                    pass
    return df


def fit_sample(df, n_rows, table_name):
    """Обучает GaussianCopulaSynthesizer и генерирует n_rows строк."""
    t0 = time.time()
    print(f"  → fit {table_name:16s} ({len(df):>7d} × {df.shape[1]})...",
          end=" ", flush=True)
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    synth = GaussianCopulaSynthesizer(metadata)
    synth.fit(df)
    print(f"{time.time()-t0:.1f} с")

    t0 = time.time()
    print(f"  → sample {table_name:16s} ({n_rows:>7d} строк)...",
          end=" ", flush=True)
    out = synth.sample(num_rows=n_rows)
    print(f"{time.time()-t0:.1f} с")
    return out


def weighted_patient_ids(ids, n):
    """
    Возвращает n синтетических id пациента так, чтобы распределение
    «сколько измерений на пациента» было правдоподобно неравномерным.
    """
    w = RNG.lognormal(mean=4.5, sigma=1.3, size=len(ids))
    w = w / w.sum()
    return RNG.choice(ids, size=n, replace=True, p=w)


# =====================================================================
#                            MAIN
# =====================================================================

print("Загружаю реальные данные...")
real = {}
for key, (fname, sep) in FILES.items():
    t0 = time.time()
    real[key] = load_real(key, fname, sep)
    print(f"  {key:16s}: {len(real[key]):>8d} строк, "
          f"{real[key].shape[1]} колонок  ({time.time()-t0:.1f} с)")

# ---------- 1. subjects ----------
print("\n1. Справочник субъектов")
subjects_syn = fit_sample(real["subjects"], GEN_SIZE["subjects"], "subjects")
subjects_syn["id субъекта"] = np.arange(1, len(subjects_syn) + 1)
subjects_syn["название субъекта"] = [f"Region_{i:02d}"
                                     for i in range(1, len(subjects_syn) + 1)]
subjects_syn.to_csv(f"{OUT}/Идентификаторы Субъектов.csv",
                    sep=";", index=False, **WRITE_KW)

# ---------- 2. mo ----------
print("\n2. Справочник МО")
mo_syn = fit_sample(real["mo"], GEN_SIZE["mo"], "mo")
mo_syn["id МО"] = np.arange(1000, 1000 + len(mo_syn))
mo_syn["название МО"] = [f"MO_{i:03d}" for i in range(1, len(mo_syn) + 1)]
mo_syn.to_csv(f"{OUT}/Идентификаторы МО.csv",
              sep=",", index=False, **WRITE_KW)

# ---------- 3. patient_program ----------
print("\n3. Пациент-программа")
pp_syn = fit_sample(real["patient_program"],
                    GEN_SIZE["patient_program"], "patient_program")

# — вымышленные id —
pp_syn["id пациента"] = np.arange(100_000, 100_000 + len(pp_syn))
pp_syn["id МО"]       = RNG.choice(mo_syn["id МО"].values, len(pp_syn))
pp_syn["название субъекта"] = RNG.choice(
    subjects_syn["название субъекта"].values, len(pp_syn))
pp_syn["id программы"] = RNG.integers(1, 41, len(pp_syn))
pp_syn["группа наблюдения"] = RNG.choice(
    ["experience", "control_1", "undefined"], len(pp_syn), p=[.09, .03, .88])

pp_syn.to_csv(f"{OUT}/МО и Пациенты.csv",
              sep=";", index=False, **WRITE_KW)

# карты для ремапа FK
syn_pat_ids    = pp_syn["id пациента"].values
pat2prog       = dict(zip(pp_syn["id пациента"], pp_syn["id программы"]))
pat2mo         = dict(zip(pp_syn["id пациента"], pp_syn["id МО"]))
pat2subj_name  = dict(zip(pp_syn["id пациента"], pp_syn["название субъекта"]))
subj_name2id   = dict(zip(subjects_syn["название субъекта"],
                          subjects_syn["id субъекта"]))

# ---------- 4. primary ----------
print("\n4. Первичные данные")
primary_syn = fit_sample(real["primary"], GEN_SIZE["primary"], "primary")

# ремап FK — с реалистичным распределением «много измерений на пациента»
new_pat = weighted_patient_ids(syn_pat_ids, len(primary_syn))
primary_syn["id пациента"]  = new_pat
primary_syn["код пациента"] = new_pat
primary_syn["id программы"] = [pat2prog[p] for p in new_pat]
primary_syn["id МО, где наблюдается пациент"] = [pat2mo[p] for p in new_pat]
primary_syn["название МО, где наблюдается пацие"] = [
    f"MO_{pat2mo[p]-999:03d}" for p in new_pat
]
primary_syn["id субъекта"] = [
    subj_name2id.get(pat2subj_name[p]) for p in new_pat
]
primary_syn.to_csv(f"{OUT}/Первичные данные.csv",
                   sep=",", index=False, **WRITE_KW)

# ---------- 5. therapy ----------
print("\n5. Терапия")
therapy_syn = fit_sample(real["therapy"], GEN_SIZE["therapy"], "therapy")
therapy_syn["id программы"] = RNG.choice(
    list(pat2prog.values()), len(therapy_syn))
therapy_syn.to_csv(f"{OUT}/Медикаментозная терапия.csv",
                   sep=",", index=False, **WRITE_KW)

# ---------- 6. kzs ----------
print("\n6. Клинически значимые события")
kzs_syn = fit_sample(real["kzs"], GEN_SIZE["kzs"], "kzs")
new_pat_kzs = weighted_patient_ids(syn_pat_ids, len(kzs_syn))
kzs_syn["id пациента"]  = new_pat_kzs
kzs_syn["id программы"] = [pat2prog[p] for p in new_pat_kzs]
kzs_syn.to_csv(f"{OUT}/Клинически значимые события.csv",
               sep=",", index=False, **WRITE_KW)

# ---------- итог ----------
print("\nГотово. Файлы в:", OUT)
for f in sorted(os.listdir(OUT)):
    n = sum(1 for _ in open(os.path.join(OUT, f), encoding="utf-8")) - 1
    print(f"  {f:45s} {n:>7d} строк")