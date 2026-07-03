"""
Backtest motorunun kritik fonksiyonlari icin birim testler.

Calistirma (proje kokunde):  python -m pytest
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

KOK = Path(__file__).resolve().parent.parent


def _yukle(dosya_adi, modul_adi):
    # dosyalar rakamla basladigi icin normal import calismaz (03/04'tekiyle ayni yontem)
    spec = importlib.util.spec_from_file_location(modul_adi, KOK / dosya_adi)
    modul = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modul)
    return modul


bt = _yukle("02_backtest.py", "bt_test")
veri = _yukle("01_veri_ve_ozellikler.py", "veri_test")


# --- 02: metrikler -----------------------------------------------------------

def test_sharpe_sabit_seride_sifir():
    assert bt._sharpe(pd.Series([0.0] * 30)) == 0.0


def test_sharpe_isareti_ortalamayi_izler():
    artan = pd.Series([0.01, 0.02] * 30)
    azalan = pd.Series([-0.01, -0.02] * 30)
    assert bt._sharpe(artan) > 0
    assert bt._sharpe(azalan) < 0


def test_maks_dusus_bilinen_seri():
    # 1.0 -> 1.1 -> 0.55: tepe 1.1'den %50 dusus
    dusus = bt._maks_dusus(pd.Series([0.10, -0.50]))
    assert dusus == pytest.approx(-0.50)


def test_siniflandirma_mukemmel_tahmin():
    gercek = np.array([0, 1, 1, 0])
    m = bt.siniflandirma_metrikleri(gercek, gercek.copy())
    assert m["dogruluk"] == 1.0
    assert m["f1"] == 1.0
    assert m["taban_oran"] == 0.5


def test_strateji_giris_maliyeti_dusuluyor():
    pozisyon = pd.Series([1, 1])
    getiri = pd.Series([0.01, 0.01])
    m = bt.strateji_metrikleri(pozisyon, getiri)
    # pozisyon degismedigi icin maliyet yalnizca ilk gun girisinde kesilir
    beklenen = (1 + 0.01 - bt.ISLEM_MALIYETI) * (1 + 0.01) - 1
    assert m["kumulatif_getiri"] == pytest.approx(beklenen)
    assert m["islem_sayisi"] == 1
    assert m["long_gun_orani"] == 1.0


def test_strateji_nakitte_getiri_sifir():
    m = bt.strateji_metrikleri(pd.Series([0, 0, 0]), pd.Series([0.05, -0.03, 0.02]))
    assert m["kumulatif_getiri"] == 0.0
    assert m["islem_sayisi"] == 0


def test_strateji_maliyet_parametresi():
    # maliyet=0 verildiginde kesinti olmamali (pano kaydiricisinin kullandigi yol)
    pozisyon = pd.Series([1, 1])
    getiri = pd.Series([0.01, 0.01])
    m = bt.strateji_metrikleri(pozisyon, getiri, maliyet=0.0)
    assert m["kumulatif_getiri"] == pytest.approx(1.01 * 1.01 - 1)


# --- 02: walk-forward sizintisizlik -----------------------------------------

def test_walk_forward_test_hep_egitimin_geleceginde():
    n = 200
    df = pd.DataFrame({
        "tarih": pd.date_range("2020-01-01", periods=n, freq="D"),
        "hisse": "TEST",
        "ozellik": np.arange(n, dtype=float),
        "hedef": np.zeros(n, dtype=int),
        "ertesi_getiri": np.zeros(n),
    })

    def tahminci(X_tr, y_tr, X_te):
        # sizintisizlik: egitim satirlari her zaman testten once gelmeli
        assert X_tr.index.max() < X_te.index.min()
        return np.zeros(len(X_te), dtype=int)

    oos = bt.walk_forward(df, tahminci, 100, 20, 20, genisleyen=False)
    assert len(oos) == 100                                    # 5 pencere x 20 gun
    assert oos["tarih"].iloc[0] == df["tarih"].iloc[100]      # ilk test gunu 101. gun


# --- 01: gostergeler ve hedef ------------------------------------------------

def test_rsi_uclarda_dogru_davranir():
    artan = pd.Series(np.linspace(100, 200, 60))
    azalan = pd.Series(np.linspace(200, 100, 60))
    assert veri.rsi_hesapla(artan).iloc[-1] > 99
    assert veri.rsi_hesapla(azalan).iloc[-1] < 1


def test_hedef_son_gunu_etiketlemez():
    fiyat = pd.DataFrame({"Close": [100.0, 100.0, 102.0, 102.0, 110.0]})
    hedef = veri.hedef_ekle(fiyat)
    assert hedef.iloc[:-1].tolist() == [0.0, 1.0, 0.0, 1.0]   # %1 esigine gore
    assert pd.isna(hedef.iloc[-1])                            # yarin bilinmiyor
