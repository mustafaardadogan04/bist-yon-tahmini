"""
Veri cekme + ozellik uretimi.

yfinance'ten fiyat ceker, teknik gostergeler uretir, sizintisiz hedef ekler
ve borsa_veri.csv'ye yazar.

Ozellikler yalnizca gecmise bakar. shift(-1) SADECE hedef icin: "yarin
yukselecek mi?" etiketi. Bu, bir ozellik olarak modele verilmez.

    python 01_veri_ve_ozellikler.py --hisseler THYAO.IS GARAN.IS --usdtry
    python 01_veri_ve_ozellikler.py --usdtry --endeks --cikti deney_endeksli.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

VARSAYILAN_HISSELER = ["THYAO.IS"]
BASLANGIC_TARIHI = "2018-01-01"
HEDEF_ESIK = 0.01            # ertesi gun > %1 ise 1
USDTRY_TICKER = "TRY=X"
ENDEKS_TICKER = "XU100.IS"   # BIST 100 — piyasa ruzgari (--endeks)
CIKTI_DOSYASI = "borsa_veri.csv"


def rsi_hesapla(kapanis: pd.Series, periyot: int = 14) -> pd.Series:
    degisim = kapanis.diff()
    kazanc = degisim.clip(lower=0)
    kayip = -degisim.clip(upper=0)
    ort_kazanc = kazanc.ewm(alpha=1 / periyot, min_periods=periyot).mean()
    ort_kayip = kayip.ewm(alpha=1 / periyot, min_periods=periyot).mean()
    rs = ort_kazanc / ort_kayip
    return 100 - (100 / (1 + rs))


def macd_hesapla(kapanis: pd.Series, hizli=12, yavas=26, sinyal=9):
    ema_hizli = kapanis.ewm(span=hizli, adjust=False).mean()
    ema_yavas = kapanis.ewm(span=yavas, adjust=False).mean()
    macd = ema_hizli - ema_yavas
    sinyal_cizgisi = macd.ewm(span=sinyal, adjust=False).mean()
    histogram = macd - sinyal_cizgisi
    return macd, histogram


def fiyat_cek(ticker: str) -> pd.DataFrame:
    ham = yf.download(
        ticker,
        start=BASLANGIC_TARIHI,
        auto_adjust=True,
        progress=False,
    )
    if ham.empty:
        return ham

    # yfinance tek ticker'da bile bazen MultiIndex donduruyor
    if isinstance(ham.columns, pd.MultiIndex):
        ham.columns = ham.columns.get_level_values(0)

    return ham[["Open", "High", "Low", "Close", "Volume"]].copy()


def usdtry_degisimi_cek() -> pd.Series:
    # tum hisseler icin ortak ozellik; ayni gunun degisimi o gun bilinir, sizinti yok
    ham = yf.download(
        USDTRY_TICKER, start=BASLANGIC_TARIHI, auto_adjust=True, progress=False
    )
    if ham.empty:
        return pd.Series(dtype="float64", name="usdtry_degisim")

    if isinstance(ham.columns, pd.MultiIndex):
        ham.columns = ham.columns.get_level_values(0)

    degisim = ham["Close"].pct_change()
    degisim.name = "usdtry_degisim"
    return degisim


def endeks_ozellikleri_cek() -> pd.DataFrame | None:
    # XU100: piyasa ruzgari. Hepsi ayni gun bilinen degerler -> sizinti yok
    ham = yf.download(ENDEKS_TICKER, start=BASLANGIC_TARIHI, auto_adjust=True, progress=False)
    if ham.empty:
        return None
    if isinstance(ham.columns, pd.MultiIndex):
        ham.columns = ham.columns.get_level_values(0)

    kapanis = ham["Close"]
    getiri = kapanis.pct_change()
    df = pd.DataFrame(index=ham.index)
    df["endeks_getiri"] = getiri
    df["endeks_getiri5"] = kapanis.pct_change(5)     # yalniz goreli guc hesabi icin
    df["endeks_getiri20"] = kapanis.pct_change(20)   # yalniz goreli guc hesabi icin
    df["endeks_vol5"] = getiri.rolling(5).std()
    df["endeks_vol20"] = getiri.rolling(20).std()
    sma20 = kapanis.rolling(20).mean()
    df["endeks_sma20_fark"] = (kapanis - sma20) / sma20
    return df


def kur_volatilitesi_cek() -> pd.DataFrame | None:
    # USD/TRY oynakligi: kur soku gunlerinde hisse davranisi degisir (rejim sinyali)
    degisim = usdtry_degisimi_cek()
    if degisim.empty:
        return None
    df = pd.DataFrame(index=degisim.index)
    df["usdtry_vol5"] = degisim.rolling(5).std()
    df["usdtry_vol20"] = degisim.rolling(20).std()
    return df


def ozellik_uret(fiyat: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(index=fiyat.index)
    kapanis = fiyat["Close"]
    hacim = fiyat["Volume"]

    getiri = kapanis.pct_change()
    df["getiri"] = getiri
    for gecikme in (1, 2, 3, 5):
        df[f"getiri_lag{gecikme}"] = getiri.shift(gecikme)

    # SMA farklari: fiyatin ortalamaya gore konumu
    for pencere in (5, 10, 20):
        sma = kapanis.rolling(pencere).mean()
        df[f"sma{pencere}_fark"] = (kapanis - sma) / sma

    sma_kisa = kapanis.rolling(5).mean()
    sma_uzun = kapanis.rolling(20).mean()
    df["sma_kisa_uzun_fark"] = (sma_kisa - sma_uzun) / sma_uzun

    for pencere in (5, 10, 20):
        df[f"volatilite{pencere}"] = getiri.rolling(pencere).std()

    df["hacim_degisim"] = hacim.pct_change()
    df["hacim_orani"] = hacim / hacim.rolling(20).mean()

    df["rsi14"] = rsi_hesapla(kapanis, 14)
    macd, macd_hist = macd_hesapla(kapanis)
    df["macd"] = macd
    df["macd_histogram"] = macd_hist

    return df


def hedef_ekle(fiyat: pd.DataFrame) -> pd.Series:
    # tek shift(-1) burada: ertesi gun getirisi esigi asarsa 1
    getiri = fiyat["Close"].pct_change()
    ertesi_gun_getirisi = getiri.shift(-1)
    hedef = (ertesi_gun_getirisi > HEDEF_ESIK).astype(float)
    # son gunun yarini bilinmez: NaN > esik sessizce False donerdi ve son satir
    # sahte hedef=0 olurdu; NaN birak ki asagidaki dropna o satiri dussun
    hedef[ertesi_gun_getirisi.isna()] = np.nan
    return hedef


def hisseyi_isle(ticker: str, usdtry: pd.Series | None,
                 endeks: pd.DataFrame | None = None,
                 kur_vol: pd.DataFrame | None = None,
                 tahmin_satiri: bool = False) -> pd.DataFrame:
    fiyat = fiyat_cek(ticker)
    if fiyat.empty:
        print(f"  ! {ticker}: veri bulunamadi, atlaniyor.")
        return pd.DataFrame()

    ozellikler = ozellik_uret(fiyat)
    ozellikler["hedef"] = hedef_ekle(fiyat)
    ozellikler.insert(0, "hisse", ticker)

    if usdtry is not None:
        ozellikler = ozellikler.join(usdtry, how="left")

    if endeks is not None:
        # piyasa ruzgari + hissenin ruzgara gore konumu (goreli guc)
        ozellikler = ozellikler.join(endeks, how="left")
        ozellikler["goreli_guc1"] = ozellikler["getiri"] - ozellikler["endeks_getiri"]
        ozellikler["goreli_guc5"] = fiyat["Close"].pct_change(5) - ozellikler["endeks_getiri5"]
        ozellikler["goreli_guc20"] = fiyat["Close"].pct_change(20) - ozellikler["endeks_getiri20"]
        # ham 5/20g endeks getirileri yalniz goreli guc icin gerekliydi
        ozellikler = ozellikler.drop(columns=["endeks_getiri5", "endeks_getiri20"])

    if kur_vol is not None:
        ozellikler = ozellikler.join(kur_vol, how="left")

    # inf'leri (orn. onceki hacim 0) NaN yap
    ozellikler = ozellikler.replace([np.inf, -np.inf], np.nan)
    if tahmin_satiri:
        # Canli tahmin: EN SON barin hedefi (ertesi gun yonu) henuz bilinmiyor
        # (yarinin kapanisi yok) ama ozellikleri tam. O bari TAHMIN icin tut;
        # yalniz gosterge isinmasi yuzunden ozelligi eksik satirlari at.
        # Aksi halde son bar dusup sinyal bir gun eski kalirdi.
        ozellik_kol = [k for k in ozellikler.columns if k != "hedef"]
        ozellikler = ozellikler.dropna(subset=ozellik_kol)
    else:
        # Egitim verisi: hedef kaymasi + isinma NaN'larini birlikte at
        ozellikler = ozellikler.dropna()
        ozellikler["hedef"] = ozellikler["hedef"].astype(int)   # NaN'li seri float olmustu
    ozellikler.index.name = "tarih"
    print(f"  + {ticker}: {len(ozellikler)} satir hazir.")
    return ozellikler.reset_index()


def main() -> None:
    ayristirici = argparse.ArgumentParser(description="BIST veri ve ozellik uretimi")
    ayristirici.add_argument(
        "--hisseler", nargs="+", default=VARSAYILAN_HISSELER,
        help="BIST ticker listesi (orn: THYAO.IS GARAN.IS)",
    )
    ayristirici.add_argument(
        "--usdtry", action="store_true",
        help="USD/TRY gunluk degisimini ortak ozellik olarak ekle",
    )
    ayristirici.add_argument(
        "--endeks", action="store_true",
        help="XU100 + goreli guc + kur volatilitesi ozelliklerini ekle (Deney 1)",
    )
    ayristirici.add_argument(
        "--cikti", default=CIKTI_DOSYASI, help="Cikti CSV dosyasinin adi",
    )
    args = ayristirici.parse_args()

    usdtry = usdtry_degisimi_cek() if args.usdtry else None
    if args.usdtry and (usdtry is None or usdtry.empty):
        print("! USD/TRY verisi cekilemedi, bu ozellik olmadan devam ediliyor.")
        usdtry = None

    endeks = kur_vol = None
    if args.endeks:
        endeks = endeks_ozellikleri_cek()
        kur_vol = kur_volatilitesi_cek()
        if endeks is None:
            print("! XU100 verisi cekilemedi, endeks ozellikleri olmadan devam ediliyor.")
        if kur_vol is None:
            print("! Kur verisi cekilemedi, kur volatilitesi olmadan devam ediliyor.")

    print(f"Isleniyor: {', '.join(args.hisseler)}")
    parcalar = [hisseyi_isle(t, usdtry, endeks, kur_vol) for t in args.hisseler]
    parcalar = [p for p in parcalar if not p.empty]

    if not parcalar:
        print("Hicbir hisse islenemedi. Cikti uretilmedi.")
        return

    tablo = pd.concat(parcalar, ignore_index=True)
    cikti_yolu = Path(args.cikti)
    tablo.to_csv(cikti_yolu, index=False)
    print(
        f"\nBitti -> {cikti_yolu.resolve()}\n"
        f"  Toplam satir: {len(tablo)}  |  Kolon: {tablo.shape[1]}\n"
        f"  Hedef dagilimi (1=yukseldi):\n{tablo['hedef'].value_counts().to_string()}"
    )


if __name__ == "__main__":
    main()
