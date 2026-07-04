"""
Walk-forward backtest iskeleti.

borsa_veri.csv'yi okuyup sizintisiz bir degerlendirme catisi kurar: yakin
gecmisle egit -> hemen sonrasini test et -> kaydir. train_test_split
kullanmiyoruz, zaman serisinde gelecekten sizinti yaratir.

Maliyet (binde 1.5) her pozisyon degisiminde dusulur, sonuc her zaman
al-tut ile karsilastirilir.

    python 02_backtest.py --strateji momentum --egitim 500 --test 60
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

VERI_DOSYASI = "borsa_veri.csv"
HEDEF_ESIK = 0.01          # ertesi gun > %1 ise "yukseldi"
ISLEM_MALIYETI = 0.0015    # tek yon, binde 1.5
YILLIK_GUN = 252           # Sharpe yilliklamasi
META_KOLONLAR = ["tarih", "hisse", "hedef", "ertesi_getiri"]


def siniflandirma_metrikleri(gercek: np.ndarray, tahmin: np.ndarray) -> dict:
    # taban_oran = "hep cogunluk de" dogrulugu, kiyas icin
    dogruluk = float((gercek == tahmin).mean())
    taban_oran = float(max((gercek == 0).mean(), (gercek == 1).mean()))
    return {
        "dogruluk": dogruluk,
        "taban_oran": taban_oran,
        "f1": float(f1_score(gercek, tahmin, zero_division=0)),
        "kesinlik": float(precision_score(gercek, tahmin, zero_division=0)),
        "duyarlilik": float(recall_score(gercek, tahmin, zero_division=0)),
    }


def strateji_serisi(pozisyon: pd.Series, ertesi_getiri: pd.Series,
                    maliyet: float = ISLEM_MALIYETI) -> pd.Series:
    # gunluk maliyet-sonrasi strateji getirisi (sermaye egrisi ve metrikler icin)
    pozisyon = pozisyon.reset_index(drop=True)
    getiri = ertesi_getiri.reset_index(drop=True)
    degisim = pozisyon.diff().abs()
    degisim.iloc[0] = abs(pozisyon.iloc[0])  # ilk gun girise de maliyet
    return pozisyon * getiri - degisim * maliyet


def strateji_metrikleri(pozisyon: pd.Series, ertesi_getiri: pd.Series,
                        maliyet: float = ISLEM_MALIYETI) -> dict:
    # pozisyon: 1=long, 0=nakit. Maliyet sonrasi PnL metrikleri.
    strat_getiri = strateji_serisi(pozisyon, ertesi_getiri, maliyet)
    pozisyon = pozisyon.reset_index(drop=True)
    getiri = ertesi_getiri.reset_index(drop=True)
    degisim = pozisyon.diff().abs()
    degisim.iloc[0] = abs(pozisyon.iloc[0])

    return {
        "kumulatif_getiri": float((1 + strat_getiri).prod() - 1),
        "al_tut_getiri": float((1 + getiri).prod() - 1),
        "yillik_sharpe": _sharpe(strat_getiri),
        "maks_dusus": _maks_dusus(strat_getiri),
        "islem_sayisi": int(degisim.sum()),
        "long_gun_orani": float((pozisyon == 1).mean()),
    }


def _sharpe(getiri: pd.Series) -> float:
    std = getiri.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(getiri.mean() / std * np.sqrt(YILLIK_GUN))


def _maks_dusus(getiri: pd.Series) -> float:
    sermaye = (1 + getiri).cumprod()
    tepe = sermaye.cummax()
    return float((sermaye / tepe - 1).min())


# Baseline tahminciler (gercek model gelene kadar kiyas)
def tahmin_cogunluk(X_tr, y_tr, X_te) -> np.ndarray:
    cogunluk = int(y_tr.mode().iloc[0])
    return np.full(len(X_te), cogunluk, dtype=int)


def tahmin_momentum(X_tr, y_tr, X_te) -> np.ndarray:
    # bugun yukseldiyse yarin da yukselir varsayimi
    return (X_te["getiri"] > 0).astype(int).to_numpy()


def tahmin_hep_long(X_tr, y_tr, X_te) -> np.ndarray:
    return np.ones(len(X_te), dtype=int)


STRATEJILER = {
    "cogunluk": tahmin_cogunluk,
    "momentum": tahmin_momentum,
    "hep_long": tahmin_hep_long,
}


def walk_forward(df, tahminci, egitim_penceresi, test_penceresi, adim, genisleyen):
    # tek hisse, zaman sirali. Test daima egitimin geleceginde -> sizinti yok.
    df = df.reset_index(drop=True)
    ozellik_kolonlari = [k for k in df.columns if k not in META_KOLONLAR]
    n = len(df)
    parcalar = []

    baslangic = egitim_penceresi
    while baslangic + test_penceresi <= n:
        egitim_basi = 0 if genisleyen else baslangic - egitim_penceresi
        egitim = df.iloc[egitim_basi:baslangic]
        test = df.iloc[baslangic:baslangic + test_penceresi]

        tahmin = tahminci(
            egitim[ozellik_kolonlari], egitim["hedef"], test[ozellik_kolonlari]
        )
        sonuc = test[["tarih", "hisse", "hedef", "ertesi_getiri"]].copy()
        sonuc["tahmin"] = tahmin
        parcalar.append(sonuc)

        baslangic += adim

    # Son tam pencereden sonra kalan gunler (kisa kuyruk) kapsanmazdi -> OOS bugune
    # kadar ulassin diye bir artik pencere ekle (test tam boyuta ulasmasa da).
    if baslangic < n:
        egitim_basi = 0 if genisleyen else baslangic - egitim_penceresi
        egitim = df.iloc[egitim_basi:baslangic]
        test = df.iloc[baslangic:n]
        if len(egitim) >= 50 and len(test) > 0:
            tahmin = tahminci(egitim[ozellik_kolonlari], egitim["hedef"], test[ozellik_kolonlari])
            sonuc = test[["tarih", "hisse", "hedef", "ertesi_getiri"]].copy()
            sonuc["tahmin"] = tahmin
            parcalar.append(sonuc)

    if not parcalar:
        return pd.DataFrame()
    return pd.concat(parcalar, ignore_index=True)


def ertesi_getiri_ekle(df: pd.DataFrame) -> pd.DataFrame:
    # ertesi gunun gercek getirisi — sadece PnL hesabi icin, asla ozellik degil
    df = df.sort_values(["hisse", "tarih"]).copy()
    df["ertesi_getiri"] = df.groupby("hisse")["getiri"].shift(-1)
    return df.dropna(subset=["ertesi_getiri"]).reset_index(drop=True)


def sizinti_kontrol(df: pd.DataFrame) -> None:
    # geri kurulan ertesi_getiri ile kayitli hedef tutuyor mu? (akil saglik)
    yeniden = (df["ertesi_getiri"] > HEDEF_ESIK).astype(int)
    uyum = float((yeniden == df["hedef"]).mean())
    if uyum < 0.99:
        print(f"  ! UYARI: hedef/ertesi_getiri uyumu dusuk ({uyum:.1%}) — kontrol et.")
    else:
        print(f"  hedef tutarlilik kontrolu: %{uyum * 100:.1f} (saglikli)")


def rapor_yazdir(hisse, oos: pd.DataFrame) -> None:
    sinif = siniflandirma_metrikleri(oos["hedef"].to_numpy(), oos["tahmin"].to_numpy())
    strat = strateji_metrikleri(oos["tahmin"], oos["ertesi_getiri"])

    print(f"\n=== {hisse}  ({len(oos)} ornek-disi gun) ===")
    print("  -- Yon dogrulugu --")
    print(f"    Dogruluk : {sinif['dogruluk']:.3f}   (taban oran: {sinif['taban_oran']:.3f})")
    print(f"    F1       : {sinif['f1']:.3f}   Kesinlik: {sinif['kesinlik']:.3f}   Duyarlilik: {sinif['duyarlilik']:.3f}")
    print("  -- Strateji (maliyet sonrasi) --")
    print(f"    Kumulatif getiri : {strat['kumulatif_getiri']:+.1%}")
    print(f"    Al-tut (kiyas)   : {strat['al_tut_getiri']:+.1%}")
    print(f"    Yillik Sharpe    : {strat['yillik_sharpe']:.2f}")
    print(f"    Maks dusus       : {strat['maks_dusus']:.1%}")
    print(f"    Islem sayisi     : {strat['islem_sayisi']}   Long gun orani: {strat['long_gun_orani']:.1%}")


def main() -> None:
    a = argparse.ArgumentParser(description="BIST walk-forward backtest iskeleti")
    a.add_argument("--veri", default=VERI_DOSYASI, help="Girdi CSV (01'in ciktisi)")
    a.add_argument("--strateji", default="momentum", choices=list(STRATEJILER),
                   help="Baseline tahminci (model gelene kadar)")
    a.add_argument("--egitim", type=int, default=500, help="Egitim penceresi (gun)")
    a.add_argument("--test", type=int, default=60, help="Test penceresi (gun)")
    a.add_argument("--adim", type=int, default=60, help="Kaydirma adimi (gun)")
    a.add_argument("--genisleyen", action="store_true",
                   help="Sabit pencere yerine genisleyen (expanding) egitim kullan")
    args = a.parse_args()

    df = pd.read_csv(args.veri, parse_dates=["tarih"])
    df = ertesi_getiri_ekle(df)
    print(f"Veri: {len(df)} satir, {df['hisse'].nunique()} hisse  |  strateji: {args.strateji}")
    print(f"Pencere: egitim={args.egitim} test={args.test} adim={args.adim} "
          f"{'(genisleyen)' if args.genisleyen else '(kayan)'}")
    sizinti_kontrol(df)

    tahminci = STRATEJILER[args.strateji]
    tum_oos = []
    for hisse, grup in df.groupby("hisse"):
        oos = walk_forward(grup, tahminci, args.egitim, args.test, args.adim, args.genisleyen)
        if oos.empty:
            print(f"\n  ! {hisse}: yeterli veri yok (egitim+test > satir sayisi).")
            continue
        rapor_yazdir(hisse, oos)
        tum_oos.append(oos)

    if len(tum_oos) > 1:
        birlesik = pd.concat(tum_oos, ignore_index=True)
        rapor_yazdir("TUM HISSELER (birlesik seri — hisseler art arda)", birlesik)


if __name__ == "__main__":
    main()
