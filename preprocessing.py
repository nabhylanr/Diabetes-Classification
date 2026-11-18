"""Preprocessing untuk dataset diabetes (Li et al., Scientific Data 2026).

dua tahap:
1. prepare()       : aturan tetap, tidak memakai statistik data, aman dijalankan sebelum split.
2. stats_cleaner   : langkah yang butuh statistik (kolom kosong, korelasi, skewness, imputasi).
                     fit di data latih saja (di dalam pipeline, per fold).

kode prepare / stats_cleaner / buat_split berasal dari eda-preprocessing.ipynb (branch feat/preprocess).
tambahan: turunan aritmetika GLO / AG / IDBIL dibuang, dan definisi feature set (non_lab / feature_sets).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GroupKFold, StratifiedKFold


kolom_teks = ["Data", "EscDrinkD", "EyeDia"]
target = "PGclass3"

# kolom gula darah dan turunannya (dihitung dari glukosa atau butuh tes OGTT)
bocor_glukosa = [
    "FPG", "PG2h", "GLU", "HomaIR", "Homaβ", "ISIGutt", "INS2h", "CP2h", "FINS", "FCP",
    "FPGover7", "P2PGover11",
]
tes_gula_lain = ["HbA1c", "HPLC", "GA", "A1cover6575"]
label_dan_turunan = ["DM", "DMnew3class", "PGclass6", "PGclass3", "DMhis"]
penanda_subjek = ["filter", "GAstudy", "GAnum"]
komplikasi_dm = ["DF", "MADPN", "TFDPN", "DR", "DRyd", "EyeDia"]
kuesioner_tidak_tahu = ["Known", "OPD", "ToChilren", "DietExcise", "Compl", "Check", "GlucoseLev"]

# kolom bocor yang sudah terbukti (bukan sifat pasien, melainkan turunan hasil tes / label)
flag_turunan_tes = ["VAR00001", "ThalussemiaAND"]
alasan_bocor = {
    "VAR00001": "bocor: bernilai 9 hanya bila FPG >= 6,1 (69% dari FPG >= 6,1; tidak pernah pada FPG < 6,1)",
    "ThalussemiaAND": "bocor: bernilai 1 bila MCV < 80 DAN glukosa abnormal (kelas >= 1); 99,7% cocok",
}
# belum terbukti bocor: kebijakan bisa diganti (drop / keep) untuk ablasi
kolom_mencurigakan = ["ThalussemiaOR", "ThalussemiaHPLC", "DMhighrisk"]

# kolom bendera: hanya berisi 1 atau kosong. kosong diasumsikan "tidak" (seperti notebook EDA).
# asumsi ini belum bisa diverifikasi dari kamus data, jadi bisa dimatikan lewat config.
bendera_isi_nol = ["HypertenHis", "DyslipideHis", "DMhighrisk", "ThalussemiaHPLC"]

alasan_mencurigakan = {
    "ThalussemiaOR": "ditahan: turunan hitung darah (MCV < 80), redundan dengan MCV/MCH",
    "ThalussemiaHPLC": "ditahan: bendera 1/kosong, hampir semua kosong; manfaat belum jelas",
    "DMhighrisk": "ditahan: komposit risiko, definisi tidak ada di kamus",
}

# satu konsep tampil dalam banyak versi kategori; versi numeriknya dipertahankan
turunan_kategori = {
    "BMI": ["BMI4Class", "overweight", "obesity", "BMI25", "WGOCobBMI"],
    "lingkar pinggang": ["WC9085", "WC8580", "WC9080"],
    "lemak tubuh": ["Fat2Class", "Fat3Class"],
    "umur": ["Age3Group", "AgeGroup"],
    "tekanan darah": ["BP13085", "WHO1999hbp"],
    "lipid": ["TGdis", "DysHDLIDFNCEP", "JCDHDLdis", "CDSdyelip"],
    "hati": ["ALTn", "TBILn", "FIB4n"],
    "ginjal": [
        "highCr", "HighUA", "HighUmALB", "ACR3degree", "HighACR", "GFR5lev", "GFR6levG5",
        "EUGFRabACR012", "EUGFR90UACR01", "EUGFR60abACR012", "UACRGFR", "DN",
    ],
    "darah": ["Hb4per", "Anemia"],
    "perilaku": ["Smoking2", "Drinking2"],
}

# bukan turunan kategori: duplikat transformasi dan ringkasan jawaban mentah
duplikat_transformasi = {
    "ALT.1": "duplikat: akar kuadrat dari ALT",
    "GLO": "turunan aritmetika: TP - ALB",
    "AG": "turunan aritmetika: ALB / GLO",
    "IDBIL": "turunan aritmetika: TBIL - DBIL",
}
riwayat_keluarga = ["FatherDM", "MotherDM", "BSDM", "ChildDM"]    # diringkas oleh DMfamilyHistory (OR keempatnya)

# kolom "kosong karena tidak ditanya": hanya ditanyakan ke perokok / peminum
# (kolom_nilai, kolom_penyaring, nilai_penyaring_yang_berarti_tidak)
kosong_struktural = [
    ("SmokingRange", "Smoking", 1),
    ("DrinkYears", "Drinking", 1),
    ("Drink50g", "Drinking", 1),
]
kosong_struktural_buang = [
    "SmokingAge", "DrinkingAge", "EscDrink", "EscDrinkM", "EscDrinkT", "EscDrinkR", "EscDrinkO",
]

# pertanyaan riwayat penyakit; kode 9 = "tidak tahu"
riwayat = [
    "Angina", "Arrhythmia", "PAD", "Hyper16", "Hyper13", "Blind", "CerebralHe", "CerebralIn",
    "TIA", "MI", "HF", "CAP", "Hypo14", "RF", "Amputation",
]
kode9_lain = ["WeightLoss", "Cancer"]


@dataclass
class config:
    kebijakan_mencurigakan: str = "drop"   # "drop" | "keep" (ablasi thalassemia dan DMhighrisk)
    buang_flag_turunan_tes: bool = True    # False hanya untuk ablasi (VAR00001, ThalussemiaAND)
    buang_turunan_kategori: bool = True    # False = pertahankan semua versi kategori (ablasi)
    nol_kan_tidak_wajar: bool = True
    populasi_skrining: bool = True         # keluarkan yang sudah tahu DM (DMhis == 1)
    isi_nol_bendera: bool = True           # kosong pada kolom bendera (HypertenHis, dst.) = 0

    def __post_init__(self):
        if self.kebijakan_mencurigakan not in ("drop", "keep"):
            raise ValueError(f"kebijakan_mencurigakan harus 'drop' atau 'keep': {self.kebijakan_mencurigakan!r}")


def load_raw(path: str) -> pd.DataFrame:
    """Baca CSV (encoding GBK) seperti notebook EDA: spasi jadi NaN, kolom angka jadi numerik."""
    df = pd.read_csv(path, encoding="gbk", dtype=str)
    df = df.replace(r"^\s*$", np.nan, regex=True)
    kolom_angka = df.columns.drop(kolom_teks)
    df[kolom_angka] = df[kolom_angka].apply(pd.to_numeric, errors="coerce")
    df["Data"] = pd.to_datetime(df["Data"], format="%m/%d/%Y")
    return df


def prepare(df: pd.DataFrame, cfg: config | None = None):
    """Aturan tetap. Mengembalikan (x, y, meta, laporan).

    x       : fitur (belum dibersihkan statistik)
    y       : PGclass3
    meta    : NO, Data (tanggal), glukosa referensi (FPG, PG2h) hanya untuk bukti kebocoran
    laporan : DataFrame kolom -> alasan dibuang, dan catatan perlakuan
    """
    cfg = cfg or config()
    wajib = (["NO", "Data", target, "DMhis", "FPG", "PG2h", "Gender", "SCRE", "GFR", "Height", "BMI",
              "Bpsys", "Bpdia", "LDL", "CHOL"] + riwayat + kode9_lain + bendera_isi_nol
             + [k for kolom, penyaring, _ in kosong_struktural for k in (kolom, penyaring)])
    hilang = sorted(set(wajib) - set(df.columns))
    if hilang:
        raise ValueError(f"kolom wajib tidak ada di data (versi dataset berbeda?): {hilang}")
    df = df.copy()
    alasan: dict[str, str] = {}
    catatan: list[str] = []

    if cfg.populasi_skrining:
        n0 = len(df)
        df = df[df["DMhis"] == 0].copy()
        catatan.append(f"populasi skrining: {n0} -> {len(df)} orang (DMhis == 1 dikeluarkan)")
    df = df.reset_index(drop=True)

    y = df[target].astype(int)
    meta = df[["NO", "Data", "FPG", "PG2h"]].copy()

    # jenis kelamin: kamus 0/1, data 1/2. di data 1 = laki-laki (tinggi, HGB lebih besar)
    kode_gender = set(df["Gender"].dropna().unique())
    if not kode_gender <= {1, 2}:
        raise ValueError(f"Gender harus berkode 1/2 (1 = laki-laki, seperti di data ini), ditemukan {sorted(kode_gender)}; kamus data menulis 0/1, periksa pengodean")
    df["male"] = (df["Gender"] == 1).astype(int)
    alasan["Gender"] = "diganti kolom male (1 = laki-laki)"

    # blok riwayat penyakit: jangan jadikan 15 kolom kosong, catat dalam satu penanda
    semua_9 = (df[riwayat] == 9).all(axis=1)
    df["riwayat_tidak_diisi"] = semua_9.astype(int)
    catatan.append(f"riwayat_tidak_diisi: {int(semua_9.sum())} orang menjawab 9 di semua {len(riwayat)} pertanyaan")
    semua_kosong = df[riwayat].isna().all(axis=1)
    catatan.append(f"riwayat semua kosong (bukan 9): {int(semua_kosong.sum())} orang, tidak ditandai (lihat sel Verifikasi Asumsi)")
    for c in riwayat + kode9_lain:
        df.loc[df[c] == 9, c] = np.nan

    # kolom bendera: hanya 1 atau kosong, kosong diasumsikan "tidak"
    if cfg.isi_nol_bendera:
        for c in bendera_isi_nol:
            n_kosong = int(df[c].isna().sum())
            df[c] = df[c].fillna(0)
            catatan.append(f"{c}: {n_kosong} kosong diisi 0 (asumsi: kosong = tidak)")

    # kosong struktural: tidak ditanya karena memang tidak merokok / minum -> 0
    for kolom, penyaring, nilai in kosong_struktural:
        m = df[kolom].isna() & (df[penyaring] == nilai)
        df.loc[m, kolom] = 0
        catatan.append(f"{kolom}: {int(m.sum())} kosong struktural ({penyaring} == {nilai}) diisi 0")

    # nilai tidak wajar -> NaN (dan kolom turunannya)
    if cfg.nol_kan_tidak_wajar:
        aturan = {
            "kreatinin < 35 (SCRE, GFR ikut)": (df["SCRE"] < 35, ["SCRE", "GFR"]),
            "eGFR > 200 (GFR)": (df["GFR"] > 200, ["GFR"]),
            "tinggi < 120 (Height, BMI)": (df["Height"] < 120, ["Height", "BMI"]),
            "tensi atas <= bawah": (df["Bpsys"] <= df["Bpdia"], ["Bpsys", "Bpdia"]),
            "LDL > kolesterol total (LDL dan CHOL)": (df["LDL"] > df["CHOL"], ["LDL", "CHOL"]),
        }
        for nama, (mask, kolom) in aturan.items():
            n = int(mask.sum())
            df.loc[mask, kolom] = np.nan
            catatan.append(f"tidak wajar: {nama}: {n} baris jadi NaN")

    # buang kolom berdasarkan alasan
    def buang(kolom, kenapa):
        for c in kolom:
            if c in df.columns and c not in alasan:
                alasan[c] = kenapa

    buang(["NO", "Data", target], "id / tanggal / target (disimpan terpisah)")
    buang(["EscDrinkD", "EyeDia"], "kolom teks")
    buang(bocor_glukosa, "bocor: gula darah atau dihitung dari gula darah")
    buang(tes_gula_lain, "tes gula darah lain")
    buang(label_dan_turunan, "label atau turunan label")
    buang(penanda_subjek, "penanda subjek studi")
    buang(komplikasi_dm, "komplikasi diabetes (setelah diagnosis)")
    buang(kuesioner_tidak_tahu, "kuesioner kesadaran diabetes, mayoritas tidak tahu")
    if cfg.buang_flag_turunan_tes:
        for c in flag_turunan_tes:
            buang([c], alasan_bocor[c])
    buang(kosong_struktural_buang, "kosong struktural, informasi ada di kolom penyaring")
    if cfg.kebijakan_mencurigakan == "drop":
        for c in kolom_mencurigakan:
            buang([c], alasan_mencurigakan[c])
    if cfg.buang_turunan_kategori:
        for konsep, kolom in turunan_kategori.items():
            buang(kolom, f"turunan kategori dari versi numerik ({konsep})")
        for c, kenapa in duplikat_transformasi.items():
            buang([c], kenapa)
        buang(riwayat_keluarga, "jawaban mentah per kerabat, diringkas oleh DMfamilyHistory")
    df = df.drop(columns=[c for c in alasan if c in df.columns])
    # kolom konstan sebelum statistik apa pun
    konstan = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    for c in konstan:
        if df[c].notna().any() and df[c].isna().any():
            alasan[c] = "hanya bernilai satu nilai atau kosong (bendera); terbuang karena kosong tidak diisi 0"
        else:
            alasan[c] = "kosong semua" if df[c].notna().sum() == 0 else "konstan"
    df = df.drop(columns=konstan)

    laporan = pd.DataFrame({"kolom": list(alasan), "alasan": list(alasan.values())})
    laporan.attrs["catatan"] = catatan
    return df, y, meta, laporan


class stats_cleaner(TransformerMixin, BaseEstimator):
    """Langkah yang butuh statistik data. Fit pada data latih saja.

    urutan : buang kolom terlalu kosong -> buang kolom biner sangat jarang -> buang pasangan
             berkorelasi tinggi -> log1p untuk kolom miring -> imputasi
    imputasi: "none" (NaN dibiarkan, mis. untuk TabPFN), "median", atau "iterative".
              kolom biner selalu diisi modus data latih (tetap pada nilai aslinya); kolom lain median atau iteratif.
    """

    def __init__(self, maks_kosong=0.5, min_minoritas=10, korelasi=0.95, skew=2.0, imputasi="none", min_periods_korelasi=100):
        self.maks_kosong = maks_kosong
        self.min_minoritas = min_minoritas
        self.korelasi = korelasi
        self.skew = skew
        self.imputasi = imputasi
        self.min_periods_korelasi = min_periods_korelasi      # pasangan yang tumpang tindih kurang dari ini tidak dinilai (rho dari 3 baris tak berarti)

    def fit(self, x: pd.DataFrame, y=None):
        if self.imputasi not in ("none", "median", "iterative"):
            raise ValueError(f"imputasi tidak dikenal: {self.imputasi!r}")
        x = x.copy()
        self.dibuang_: dict[str, str] = {}

        kosong = x.isna().mean()
        for c in kosong[(kosong > self.maks_kosong) | (kosong >= 1.0)].index:      # kolom kosong semua selalu dibuang
            self.dibuang_[c] = f"kosong {kosong[c]:.0%} (> {self.maks_kosong:.0%})" if kosong[c] < 1.0 else "kosong semua di data latih"
        x = x.drop(columns=list(self.dibuang_))

        for c in x.columns:
            nilai = x[c].dropna()
            if 0 < nilai.nunique() <= 2:
                minoritas = int(nilai.value_counts().min()) if nilai.nunique() == 2 else 0   # konstan = minoritas 0
                if minoritas < self.min_minoritas:
                    self.dibuang_[c] = f"biner terlalu jarang (minoritas = {minoritas})"
        x = x.drop(columns=[c for c in self.dibuang_ if c in x.columns])

        # pasangan sangat berkorelasi (spearman): buang yang lebih banyak kosong
        rho = x.corr(method="spearman", min_periods=self.min_periods_korelasi).abs()
        atas = rho.where(np.triu(np.ones(rho.shape, dtype=bool), k=1))
        for c in atas.columns:
            for lain in atas.index[atas[c] >= self.korelasi]:
                if lain in self.dibuang_:
                    continue
                korban = c if x[c].isna().mean() > x[lain].isna().mean() else lain
                keep = lain if korban == c else c
                self.dibuang_[korban] = f"|rho| = {atas.loc[lain, c]:.3f} dengan {keep}"
                if korban == c:
                    break          # c sudah dibuang: jangan buang kolom lain karena c
        x = x.drop(columns=[c for c in self.dibuang_ if c in x.columns])

        self.kolom_ = list(x.columns)
        self.kolom_log_ = [
            c for c in x.columns
            if x[c].nunique() > 10 and x[c].min() >= 0 and x[c].skew() > self.skew
        ]
        if self.kolom_log_:
            x[self.kolom_log_] = np.log1p(x[self.kolom_log_])

        self.kolom_biner_ = [c for c in x.columns if 0 < x[c].dropna().nunique() <= 2]
        self.modus_ = {c: x[c].mode(dropna=True).iloc[0] for c in self.kolom_biner_}

        self.imputer_ = None
        if self.imputasi == "median":
            from sklearn.impute import SimpleImputer
            self.imputer_ = SimpleImputer(strategy="median").fit(x)
        elif self.imputasi == "iterative":
            import importlib
            importlib.import_module("sklearn.experimental.enable_iterative_imputer")   # mengaktifkan IterativeImputer
            from sklearn.impute import IterativeImputer
            self.imputer_ = IterativeImputer(max_iter=10, random_state=0).fit(x)
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        x = x[self.kolom_].copy()
        if self.kolom_log_:
            x[self.kolom_log_] = np.log1p(x[self.kolom_log_].clip(lower=0))
        if self.imputer_ is not None:
            kosong_biner = x[self.kolom_biner_].isna()
            x = pd.DataFrame(self.imputer_.transform(x), columns=self.kolom_, index=x.index)
            # kolom biner tetap pada nilai aslinya: sel yang tadinya kosong diisi modus data latih
            # (median dari 0/1 bisa 0,5; iterative memberi nilai kontinu)
            for c in self.kolom_biner_:
                x.loc[kosong_biner[c], c] = self.modus_[c]
        return x


def buat_split(y: pd.Series, meta: pd.DataFrame, skema: str, n_lipatan: int = 5, seed: int = 0):
    """Kembalikan list (idx_latih, idx_uji).

    acak     : stratified k-fold
    tanggal  : GroupKFold per tanggal survei (hari yang sama tidak terbagi)
    temporal : 70% awal berdasar urutan ID untuk latih, 30% akhir untuk uji (n_lipatan dan seed tidak berlaku;
               skema tanggal juga tidak memakai seed).
               NB: skema tanggal hanya mencegah satu hari terbagi antara latih dan uji; ia mengacak urutan hari,
               jadi bukan uji pergeseran waktu. holdout temporal satu-satunya yang menguji pergeseran waktu. urutan ID hampir identik dengan
               urutan tanggal (spearman 0,9995; 2 pasangan mundur sampai 2 hari)
    """
    n = len(y)
    if skema == "acak":
        return list(StratifiedKFold(n_lipatan, shuffle=True, random_state=seed).split(np.zeros(n), y))
    if skema == "tanggal":
        return list(GroupKFold(n_lipatan).split(np.zeros(n), y, groups=meta["Data"].dt.strftime("%Y-%m-%d")))
    if skema == "temporal":
        if "Data" in meta and meta["Data"].notna().all():
            arah = meta["NO"].rank().corr(meta["Data"].rank())
            if not arah >= 0.9:      # NO dipakai sebagai urutan waktu: harus sejalan dengan tanggal
                raise ValueError(f"urutan NO tidak sejalan dengan tanggal (spearman {arah:.2f}); split temporal tidak bermakna")
        urut = np.argsort(meta["NO"].to_numpy())
        potong = int(n * 0.7)
        return [(urut[:potong], urut[potong:])]
    raise ValueError(f"skema tidak dikenal: {skema}")


def terapkan_lipatan(x: pd.DataFrame, idx_latih, idx_uji, imputasi: str = "none", **kwargs):
    """Fit stats_cleaner pada data latih, lalu transform data latih dan uji.

    imputasi: "none" (CatBoost, TabPFN), "median" atau "iterative" (Logistic Regression, MLP, FT-Transformer).
    kembalian: (x_latih, x_uji, cleaner)
    """
    cleaner = stats_cleaner(imputasi=imputasi, **kwargs).fit(x.iloc[idx_latih])
    return cleaner.transform(x.iloc[idx_latih]), cleaner.transform(x.iloc[idx_uji]), cleaner


def metrik_tiga_kelas(yt, p):
    """Metrik untuk tiga kelas (0 = NGR, 1 = prediabetes, 2 = DM) dari probabilitas p (n x 3).

    macro_auc : AUC one-vs-rest, rata-rata tiga kelas
    auc_dm    : AUC DM vs sisanya
    ap_dm     : PR-AUC (average precision) DM
    auc_pre_vs_normal : AUC prediabetes vs normal (DM dikeluarkan)
    """
    from sklearn.metrics import average_precision_score, roc_auc_score
    yt = np.asarray(yt)
    p = np.asarray(p)
    out = {
        "macro_auc": roc_auc_score(yt, p, multi_class="ovr", average="macro"),
        "auc_dm": roc_auc_score(yt == 2, p[:, 2]),
        "ap_dm": average_precision_score(yt == 2, p[:, 2]),
    }
    m = yt <= 1
    pembagi = p[m, 0] + p[m, 1]
    out["auc_pre_vs_normal"] = roc_auc_score(yt[m] == 1, np.divide(p[m, 1], pembagi, out=np.full(int(m.sum()), 0.5), where=pembagi > 0))
    return out


# feature set untuk RQ1: yang bisa didapat tanpa ambil darah (demografi, antropometri, tensi,
# gaya hidup, riwayat) vs ditambah lab rutin non-glikemik
non_lab = (
    ["male", "Age", "Height", "Weight", "BMI", "waist1", "Waist2", "hip", "WHR", "leg", "Fat", "HR",
     "Bpsys", "Bpdia", "HypertenHis", "DyslipideHis", "Smoking", "SmokingRange", "Drinking", "DrinkYears",
     "Drink50g", "Tea", "WeightLoss", "DMfamilyHistory", "cataract", "Cancer", "riwayat_tidak_diisi"]
    + riwayat
)
kolom_cdrs = ["Age", "male", "BMI", "waist1", "Bpsys", "DMfamilyHistory"]


def feature_sets(x: pd.DataFrame) -> dict:
    """Kembalikan feature set dari kolom hasil prepare(). F1 = non-lab, F2 = semua (non-lab + lab rutin)."""
    f1 = [c for c in x.columns if c in non_lab]
    return {
        "F1_nonlab": f1,
        "F2_combined": list(x.columns),
        "lab_only": [c for c in x.columns if c not in non_lab],
        "cdrs": kolom_cdrs,
    }
