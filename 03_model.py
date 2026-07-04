"""
Modeller: lojistik regresyon, XGBoost, LightGBM.

02'deki walk-forward iskeletine takilirlar, hepsi ayni sizintisiz terazide
olculur. Sinif dengesizligi (~%31 pozitif) class_weight / scale_pos_weight
ile dengelenir.

    python 03_model.py --model xgboost
    python 03_model.py --model hepsi
"""

import argparse
import importlib.util
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")  # kucuk-veri uyarilari


def _motoru_yukle():
    # 02 rakamla basladigi icin normal import calismaz, importlib ile aliyoruz
    yol = Path(__file__).with_name("02_backtest.py")
    spec = importlib.util.spec_from_file_location("backtest_motoru", yol)
    modul = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modul)
    return modul


bt = _motoru_yukle()


# Modeller — hepsi ayni imza. Tek yerde kurulur; hem 0/1 tahmin hem olasilik uretir.
def _model_kur(ad: str, y_egitim):
    if ad == "lojistik":
        # olceklemeye duyarli, o yuzden StandardScaler'li pipeline
        return Pipeline([
            ("olcek", StandardScaler()),
            ("lr", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ])
    if ad == "xgboost":
        # max_depth=3: kucuk veride asiri ogrenmeyi engelle
        pozitif = max(int((y_egitim == 1).sum()), 1)
        negatif = int((y_egitim == 0).sum())
        return xgb.XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=negatif / pozitif,
            eval_metric="logloss", n_jobs=-1, verbosity=0,
        )
    if ad == "lightgbm":
        return lgb.LGBMClassifier(
            n_estimators=200, max_depth=3, num_leaves=15,
            learning_rate=0.05, min_child_samples=20,
            subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced", n_jobs=-1, verbose=-1,
        )
    raise ValueError(f"bilinmeyen model: {ad}")


def _tahminci(ad):
    # (X_egitim, y_egitim, X_test) -> 0/1 dizisi  (walk_forward'in bekledigi imza)
    def f(X_egitim, y_egitim, X_test) -> np.ndarray:
        model = _model_kur(ad, y_egitim)
        model.fit(X_egitim, y_egitim)
        return model.predict(X_test)
    return f


def _olasilikci(ad):
    # (X_egitim, y_egitim, X_test) -> P(yukselir) olasilik dizisi  (portfoy siralamasi icin)
    def f(X_egitim, y_egitim, X_test) -> np.ndarray:
        model = _model_kur(ad, y_egitim)
        model.fit(X_egitim, y_egitim)
        return model.predict_proba(X_test)[:, 1]
    return f


MODELLER = {ad: _tahminci(ad) for ad in ("lojistik", "xgboost", "lightgbm")}
OLASILIKLAR = {ad: _olasilikci(ad) for ad in ("lojistik", "xgboost", "lightgbm")}


def modeli_kosur(ad, tahminci, df, egitim, test, adim, genisleyen) -> None:
    # secilen modeli butun hisselerde walk-forward kosar, sureyi de olcer
    print(f"\n{'#' * 60}\n### MODEL: {ad}\n{'#' * 60}")
    baslangic = time.perf_counter()

    tum_oos = []
    for hisse, grup in df.groupby("hisse"):
        oos = bt.walk_forward(grup, tahminci, egitim, test, adim, genisleyen)
        if oos.empty:
            print(f"\n  ! {hisse}: yeterli veri yok.")
            continue
        bt.rapor_yazdir(hisse, oos)
        tum_oos.append(oos)

    if len(tum_oos) > 1:
        bt.rapor_yazdir("TUM HISSELER (birlesik seri — hisseler art arda)", pd.concat(tum_oos, ignore_index=True))

    sure = time.perf_counter() - baslangic
    print(f"\n  >> {ad} egitim+test suresi (tum walk-forward): {sure:.2f} saniye")


def main() -> None:
    a = argparse.ArgumentParser(description="BIST ML modelleri — walk-forward")
    a.add_argument("--veri", default=bt.VERI_DOSYASI, help="Girdi CSV (01'in ciktisi)")
    a.add_argument("--model", default="lojistik",
                   choices=list(MODELLER) + ["hepsi"], help="Egitilecek model")
    a.add_argument("--egitim", type=int, default=500, help="Egitim penceresi (gun)")
    a.add_argument("--test", type=int, default=60, help="Test penceresi (gun)")
    a.add_argument("--adim", type=int, default=60, help="Kaydirma adimi (gun)")
    a.add_argument("--genisleyen", action="store_true", help="Genisleyen egitim penceresi")
    args = a.parse_args()

    df = pd.read_csv(args.veri, parse_dates=["tarih"])
    df = bt.ertesi_getiri_ekle(df)
    print(f"Veri: {len(df)} satir, {df['hisse'].nunique()} hisse")
    print(f"Pencere: egitim={args.egitim} test={args.test} adim={args.adim} "
          f"{'(genisleyen)' if args.genisleyen else '(kayan)'}")
    bt.sizinti_kontrol(df)

    secilenler = list(MODELLER) if args.model == "hepsi" else [args.model]
    for ad in secilenler:
        modeli_kosur(ad, MODELLER[ad], df, args.egitim, args.test, args.adim, args.genisleyen)


if __name__ == "__main__":
    main()
