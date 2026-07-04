"""
Streamlit panosu — 01/02/03'u import edip uzerine bir arayuz koyar.

Hisse + model + pencere sec, guncel sinyali, maliyet sonrasi metrikleri
ve sermaye egrisini gor. Onceki scriptlere dokunmaz.

    streamlit run 04_dashboard.py

Ogrenme/portfoy projesidir, yatirim tavsiyesi degildir.
"""

import importlib.util
from datetime import date
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
def hisse_cek(ticker: str, gun: str) -> pd.DataFrame:
    # bir BIST kodunu canli cek (01'in ozellik hattindan gecir).
    # 'gun' cache anahtari: ayni gun icinde tekrar cekmez, ertesi gun yeniler.
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


def gecmis_tarih_testi(df_tek: pd.DataFrame, model_ad: str, kesim, ileri_gun: int = 20):
    # ZAMAN MAKINESI: kesim tarihine kadarki veriyle egit, o gun tahmin et, SONRASINI
    # ortaya cikar. Egitim yalniz kesim'den ONCEKI (hedefi o gun bilinen) satirlarla
    # yapilir -> gelecekten sizinti yok.
    df = df_tek.sort_values("tarih").reset_index(drop=True)
    kesim = pd.Timestamp(kesim)
    ozellik = [k for k in df.columns if k not in bt.META_KOLONLAR]
    egitim = df[df["tarih"] < kesim]
    tahmin_satiri = df[df["tarih"] <= kesim].tail(1)   # kesim gunu (ya da en yakin onceki)
    sonraki = df[df["tarih"] > kesim].head(ileri_gun)  # ortaya cikarilacak gunler
    if len(egitim) < 100 or tahmin_satiri.empty or sonraki.empty:
        return None
    tahmin = ml.MODELLER[model_ad](egitim[ozellik], egitim["hedef"], tahmin_satiri[ozellik])
    ertesi_getiri = float(sonraki["getiri"].iloc[0])   # kesim+1 gununun gercek getirisi
    return {
        "tarih": tahmin_satiri["tarih"].iloc[0],
        "sinyal": int(tahmin[0]),
        "ertesi_getiri": ertesi_getiri,
        "hedef_tuttu": ertesi_getiri > bt.HEDEF_ESIK,   # ertesi gun gercekten %1 asti mi
        "kumulatif": float((1 + sonraki["getiri"]).prod() - 1),   # sonraki ~ay
        "ileri_gun": len(sonraki),
        "sonraki_egri": sonraki[["tarih", "getiri"]].copy(),
    }


@st.cache_data(show_spinner=False)
def zaman_makinesi(df_tek: pd.DataFrame, model_ad: str, kesim_iso: str):
    # cache'li sarmal: ayni hisse/model/tarih icin her rerun'da yeniden egitmez
    return gecmis_tarih_testi(df_tek, model_ad, pd.Timestamp(kesim_iso))


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

bugun = date.today().isoformat()   # canli cache anahtari (gun basina bir cekim)

mevcut_hisseler = sorted(tum_veri["hisse"].unique()) if not tum_veri.empty else []
secim = st.sidebar.selectbox(
    "Hisse",
    mevcut_hisseler + ["+ Yeni hisse cek..."] if mevcut_hisseler else ["+ Yeni hisse cek..."],
)

canli = st.sidebar.checkbox(
    "🔄 Canli veri (yfinance)", value=False,
    help="Acikken secili hisse yfinance'ten guncel veriyle cekilir ve model o an "
         "yeniden egitilir. Kapaliyken repodaki donmus ornek veri kullanilir.")

if secim == "+ Yeni hisse cek...":
    yeni = st.sidebar.text_input("BIST kodu (orn: GARAN.IS)", value="GARAN.IS").strip().upper()
    df_tek = hisse_cek(yeni, bugun) if yeni else pd.DataFrame()
    hisse_adi = yeni
elif canli:
    df_tek = hisse_cek(secim, bugun)   # listedeki hisseyi de canli cek
    hisse_adi = secim
else:
    df_tek = tum_veri[tum_veri["hisse"] == secim].copy()
    hisse_adi = secim

if canli:
    st.sidebar.caption("⚠️ Canli mod: sonuclar guncel veriyle hesaplanir; her gun "
                       "degisir ve README'deki donmus (2020–2026) sayilarla birebir "
                       "tutmayabilir. yfinance bulut IP'lerini ara sira engelleyebilir.")

model_ad = st.sidebar.selectbox("Model", list(ml.MODELLER), index=list(ml.MODELLER).index("xgboost"))
st.sidebar.caption("Varsayilan XGBoost: olculen en verimli model (Sharpe 1.35).")

egitim = st.sidebar.slider("Egitim penceresi (gun)", 250, 1000, 500, 50)
test = st.sidebar.slider("Test penceresi (gun)", 20, 120, 60, 10)
adim = st.sidebar.slider("Kaydirma adimi (gun)", 20, 120, 60, 10)
maliyet = st.sidebar.slider("Islem maliyeti (binde)", 0.0, 5.0, 1.5, 0.5) / 1000
sermaye = st.sidebar.number_input("Baslangic sermayesi (₺)", 1000, 10_000_000, 10_000, 1000,
                                  help="Sermaye egrisi ve sonuc bu tutar uzerinden ₺ olarak gosterilir.")

# --- Zaman makinesi kontrolleri (kenar cubugu) ---
st.sidebar.divider()
st.sidebar.subheader("🕰 Geçmiş tarih testi")
zm_secim = st.sidebar.selectbox(
    "Modeli hangi güne götürelim?",
    ["(kapalı)", "1 ay önce", "2 ay önce", "3 ay önce", "6 ay önce", "Tarih seç..."],
    help="Model yalnızca o tarihe kadarki veriyle eğitilir (sızıntı yok), o gün AL/BEKLE "
         "der; sonra gerçekte ne olduğunu ortaya çıkarırız.")

zm_kesim = None
if zm_secim != "(kapalı)" and not df_tek.empty:
    _son = pd.to_datetime(df_tek["tarih"]).max()
    _ilk = pd.to_datetime(df_tek["tarih"]).min()
    if zm_secim == "Tarih seç...":
        _tmin = (_ilk + pd.Timedelta(days=400)).date()   # egitim icin yer birak
        _tmax = (_son - pd.Timedelta(days=35)).date()     # ortaya cikarma icin yer birak
        if _tmin < _tmax:
            zm_kesim = st.sidebar.date_input("Kesim tarihi",
                                             value=(_son - pd.Timedelta(days=150)).date(),
                                             min_value=_tmin, max_value=_tmax)
        else:
            st.sidebar.info("Veri aralığı bu test için çok kısa.")
    else:
        _ay = {"1 ay önce": 1, "2 ay önce": 2, "3 ay önce": 3, "6 ay önce": 6}[zm_secim]
        zm_kesim = (_son - pd.DateOffset(months=_ay)).date()

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

# --- Zaman makinesi sonucu (kenar cubugundan secilince ust kisimda gorunur) ---
if zm_kesim is not None:
    st.divider()
    st.markdown(f"### 🕰 Model {pd.Timestamp(zm_kesim):%d.%m.%Y} tarihinde ne derdi?")
    with st.spinner("Model o tarihe kadarki veriyle eğitiliyor..."):
        zm = zaman_makinesi(df_tek, model_ad, pd.Timestamp(zm_kesim).isoformat())
    if zm is None:
        st.warning("Bu tarih için yeterli veri yok — daha ortada bir tarih seç.")
    else:
        z1, z2, z3 = st.columns([1, 1, 2])
        with z1:
            if zm["sinyal"] == 1:
                st.success(f"### AL\n{zm['tarih']:%d.%m.%Y} verisine göre")
            else:
                st.info(f"### BEKLE\n{zm['tarih']:%d.%m.%Y} verisine göre")
        with z2:
            dogru = (zm["sinyal"] == 1) == zm["hedef_tuttu"]
            st.metric("Ertesi gün gerçekte", f"{zm['ertesi_getiri']:+.2%}",
                      "✅ model haklı" if dogru else "✗ model yanıldı")
        with z3:
            st.metric(f"Sonraki {zm['ileri_gun']} işlem günü (al-tut)", f"{zm['kumulatif']:+.1%}")
            st.caption("Model yalnızca **ertesi günü** tahmin eder; aylık getiri bağlam içindir.")
        _zegri = pd.DataFrame({
            "tarih": pd.to_datetime(zm["sonraki_egri"]["tarih"]).values,
            "Fiyat (kesim=1₺)": (1 + zm["sonraki_egri"]["getiri"]).cumprod().values,
        }).set_index("tarih")
        st.line_chart(_zegri, height=200)

# Sermaye egrisi — secilen baslangic sermayesi uzerinden ₺
strat_get = bt.strateji_serisi(oos["tahmin"], oos["ertesi_getiri"], maliyet)
altut_get = oos["ertesi_getiri"].reset_index(drop=True)
egri = pd.DataFrame({
    "tarih": pd.to_datetime(oos["tarih"]).reset_index(drop=True),
    "Strateji (₺)": sermaye * (1 + strat_get).cumprod(),
    "Al-tut (₺)": sermaye * (1 + altut_get).cumprod(),
}).set_index("tarih")

son_strat = sermaye * float((1 + strat_get).prod())
son_altut = sermaye * float((1 + altut_get).prod())
def _tl(x):
    return f"{x:,.0f}".replace(",", ".") + "₺"

st.markdown(f"##### Sermaye eğrisi — {_tl(sermaye)} başlangıç (maliyet sonrası)")
st.line_chart(egri)
mp1, mp2, mp3 = st.columns(3)
mp1.metric("Başlangıç", _tl(sermaye))
mp2.metric("Strateji sonu", _tl(son_strat), f"{son_strat/sermaye-1:+.0%}")
mp3.metric("Al-tut sonu", _tl(son_altut), f"{son_altut/sermaye-1:+.0%}")

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
