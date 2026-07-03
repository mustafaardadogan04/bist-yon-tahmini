"""
Streamlit panosu — 01/02/03'u import edip uzerine bir arayuz koyar.

Hisse + model + pencere sec, guncel sinyali, maliyet sonrasi metrikleri
ve sermaye egrisini gor. Onceki scriptlere dokunmaz.

    streamlit run 04_dashboard.py

Ogrenme/portfoy projesidir, yatirim tavsiyesi degildir.
"""

import importlib.util
from pathlib import Path

import pandas as pd
import streamlit as st


def _yukle(dosya_adi: str, modul_adi: str):
    # rakamla baslayan dosyalar normal import edilemez
    yol = Path(__file__).with_name(dosya_adi)
    spec = importlib.util.spec_from_file_location(modul_adi, yol)
    modul = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modul)
    return modul


bt = _yukle("02_backtest.py", "backtest_motoru")
ml = _yukle("03_model.py", "model_motoru")
veri = _yukle("01_veri_ve_ozellikler.py", "veri_motoru")

VERI_DOSYASI = Path(__file__).with_name("borsa_veri.csv")


@st.cache_data(show_spinner=False)
def veri_yukle() -> pd.DataFrame:
    if not VERI_DOSYASI.exists():
        return pd.DataFrame()
    return pd.read_csv(VERI_DOSYASI, parse_dates=["tarih"])


@st.cache_data(show_spinner="Hisse yfinance'ten cekiliyor...")
def hisse_cek(ticker: str) -> pd.DataFrame:
    # yeni bir BIST kodunu canli cek (01'in ozellik hattindan gecir).
    # USD/TRY de dahil: CSV'deki hisselerle ayni ozellik seti olsun
    usdtry = veri.usdtry_degisimi_cek()
    return veri.hisseyi_isle(ticker, usdtry if not usdtry.empty else None)


@st.cache_data(show_spinner=False)
def backtest_calistir(df_tek: pd.DataFrame, model_ad: str,
                      egitim: int, test: int, adim: int) -> pd.DataFrame:
    df2 = bt.ertesi_getiri_ekle(df_tek)
    return bt.walk_forward(df2, ml.MODELLER[model_ad], egitim, test, adim, False)


def guncel_sinyal(df_tek: pd.DataFrame, model_ad: str):
    # sonucu bilinen gunlerle egit, verideki EN SON gunu tahmin et
    df2 = bt.ertesi_getiri_ekle(df_tek)   # yarini bilinmeyen son gun burada duser
    if len(df2) < 50:
        return None, None
    ozellik = [k for k in df2.columns if k not in bt.META_KOLONLAR]
    son = df_tek.sort_values("tarih").iloc[[-1]]   # gercek en son gun
    tahmin = ml.MODELLER[model_ad](df2[ozellik], df2["hedef"], son[ozellik])
    return int(tahmin[0]), son["tarih"].iloc[-1]


st.set_page_config(page_title="BIST Hibrit Tahmin", page_icon="📈", layout="wide")
st.title("📈 BIST Hibrit Tahmin Panosu")
st.caption("Sizintisiz walk-forward backtest ile teknik gostergelere dayali yon tahmini. "
           "Ogrenme/portfoy projesidir — yatirim tavsiyesi degildir.")

# Streamlit'in kosan-adam gostergesi yerine kendi st.spinner'imizi kullaniyoruz
st.markdown(
    """
    <style>
      [data-testid="stStatusWidget"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

tum_veri = veri_yukle()

# Kenar cubugu
st.sidebar.header("Ayarlar")

mevcut_hisseler = sorted(tum_veri["hisse"].unique()) if not tum_veri.empty else []
secim = st.sidebar.selectbox(
    "Hisse",
    mevcut_hisseler + ["+ Yeni hisse cek..."] if mevcut_hisseler else ["+ Yeni hisse cek..."],
)

if secim == "+ Yeni hisse cek...":
    yeni = st.sidebar.text_input("BIST kodu (orn: GARAN.IS)", value="GARAN.IS").strip().upper()
    df_tek = hisse_cek(yeni) if yeni else pd.DataFrame()
    hisse_adi = yeni
else:
    df_tek = tum_veri[tum_veri["hisse"] == secim].copy()
    hisse_adi = secim

model_ad = st.sidebar.selectbox("Model", list(ml.MODELLER), index=list(ml.MODELLER).index("xgboost"))
st.sidebar.caption("Varsayilan XGBoost: olculen en verimli model (Sharpe 1.35).")

egitim = st.sidebar.slider("Egitim penceresi (gun)", 250, 1000, 500, 50)
test = st.sidebar.slider("Test penceresi (gun)", 20, 120, 60, 10)
adim = st.sidebar.slider("Kaydirma adimi (gun)", 20, 120, 60, 10)
maliyet = st.sidebar.slider("Islem maliyeti (binde)", 0.0, 5.0, 1.5, 0.5) / 1000

if df_tek.empty or len(df_tek) < egitim + test:
    st.warning("Yeterli veri yok. Farkli bir hisse sec ya da egitim/test penceresini kucult.")
    st.stop()

with st.spinner("Hesaplaniyor..."):
    oos = backtest_calistir(df_tek, model_ad, egitim, test, adim)
    if oos.empty:
        st.warning("Backtest sonucu bos — pencere ayarlarini kucult.")
        st.stop()

    strat = bt.strateji_metrikleri(oos["tahmin"], oos["ertesi_getiri"], maliyet)
    sinif = bt.siniflandirma_metrikleri(oos["hedef"].to_numpy(), oos["tahmin"].to_numpy())
    sinyal, sinyal_tarih = guncel_sinyal(df_tek, model_ad)

st.subheader(f"{hisse_adi} — {model_ad}")

s1, s2 = st.columns([1, 3])
with s1:
    if sinyal == 1:
        st.success(f"### Sinyal: AL\n{sinyal_tarih:%d.%m.%Y} verisine gore")
    elif sinyal == 0:
        st.info(f"### Sinyal: BEKLE\n{sinyal_tarih:%d.%m.%Y} verisine gore")
    else:
        st.write("Sinyal hesaplanamadi.")
with s2:
    st.caption(f"Ornek-disi (out-of-sample) deger: {len(oos)} gun  ·  "
               f"Yon dogrulugu {sinif['dogruluk']:.1%} (taban oran {sinif['taban_oran']:.1%})")

# Metrik kartlari
k = st.columns(5)
k[0].metric("Strateji getirisi", f"{strat['kumulatif_getiri']:+.0%}")
k[1].metric("Al-tut (kiyas)", f"{strat['al_tut_getiri']:+.0%}")
k[2].metric("Yillik Sharpe", f"{strat['yillik_sharpe']:.2f}")
k[3].metric("Maks dusus", f"{strat['maks_dusus']:.1%}")
k[4].metric("Islem sayisi", f"{strat['islem_sayisi']}")

# Sermaye egrisi
strat_get = bt.strateji_serisi(oos["tahmin"], oos["ertesi_getiri"], maliyet)
egri = pd.DataFrame({
    "tarih": pd.to_datetime(oos["tarih"]).reset_index(drop=True),
    "Strateji": (1 + strat_get).cumprod(),
    "Al-tut": (1 + oos["ertesi_getiri"].reset_index(drop=True)).cumprod(),
}).set_index("tarih")

st.markdown("##### Sermaye egrisi — 1₺ baslangic (maliyet sonrasi)")
st.line_chart(egri)

with st.expander("Yon dogrulugu detayi (F1 / kesinlik / duyarlilik)"):
    d = st.columns(3)
    d[0].metric("F1", f"{sinif['f1']:.3f}")
    d[1].metric("Kesinlik", f"{sinif['kesinlik']:.3f}")
    d[2].metric("Duyarlilik", f"{sinif['duyarlilik']:.3f}")
    st.caption("Yon dogrulugunu tek basina okuma: taban oran (hep ayni sinifi soylemek) bile "
               "yuksek cikabilir. Olcut, maliyet sonrasi getiri ve Sharpe. "
               "%90 dogruluk gorursen kod hatasi ara.")

st.divider()
st.caption("⚠️ Sonuclar ornek-disi gunlerde, secilen islem maliyeti dusulerek hesaplanir. "
           "Gecmis performans gelecegi garanti etmez. Yatirim tavsiyesi degildir.")
