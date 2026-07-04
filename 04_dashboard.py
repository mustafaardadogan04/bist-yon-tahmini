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


@st.cache_data(show_spinner=False)
def _zm_tahmin(df_tek: pd.DataFrame, model_ad: str, kesim_iso: str):
    # ZAMAN MAKINESI (pahali kisim, cache'li): modeli kesim tarihine kadarki veriyle
    # BIR KEZ egit, sonra o gunden itibaren gunluk sinyalleri uret. Egitim yalniz
    # kesim'den ONCEKI veriyle -> gelecekten sizinti yok. (sermaye/maliyetten bagimsiz)
    df2 = bt.ertesi_getiri_ekle(df_tek.sort_values("tarih").reset_index(drop=True))
    kesim = pd.Timestamp(kesim_iso)
    ozellik = [k for k in df2.columns if k not in bt.META_KOLONLAR]
    egitim = df2[df2["tarih"] < kesim]
    pencere = df2[df2["tarih"] >= kesim].reset_index(drop=True)   # kesim gununden bugune
    if len(egitim) < 100 or len(pencere) < 5:
        return None
    tahmin = ml.MODELLER[model_ad](egitim[ozellik], egitim["hedef"], pencere[ozellik])
    return {
        "tarih": pencere["tarih"],
        "pozisyon": pd.Series(tahmin),
        "ertesi_getiri": pencere["ertesi_getiri"].reset_index(drop=True),
        "hedef": pencere["hedef"].reset_index(drop=True),
    }


def zaman_makinesi(df_tek, model_ad, kesim_iso, sermaye, maliyet):
    # ucuz kisim: egitilmis tahminlere maliyet + sermaye uygula (portfoy senaryosu)
    r = _zm_tahmin(df_tek, model_ad, kesim_iso)
    if r is None:
        return None
    strat_get = bt.strateji_serisi(r["pozisyon"], r["ertesi_getiri"], maliyet)
    altut_get = r["ertesi_getiri"]
    return {
        "baslangic": r["tarih"].iloc[0],
        "gun": len(r["pozisyon"]),
        "al_gunu": int(r["pozisyon"].sum()),   # kac gun AL (pozisyonda) kalindi
        "strat_son": sermaye * float((1 + strat_get).prod()),
        "altut_son": sermaye * float((1 + altut_get).prod()),
        "strat_getiri": float((1 + strat_get).prod() - 1),
        "altut_getiri": float((1 + altut_get).prod() - 1),
        "egri": pd.DataFrame({
            "tarih": r["tarih"].values,
            "Strateji": (sermaye * (1 + strat_get).cumprod()).values,
            "Al-tut": (sermaye * (1 + altut_get).cumprod()).values,
        }),
    }


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

sermaye = st.sidebar.number_input("Başlangıç sermayesi (₺)", 1000, 10_000_000, 10_000, 1000,
                                  help="Sermaye eğrisi ve sonuç bu tutar üzerinden ₺ gösterilir.")
maliyet = st.sidebar.slider("İşlem maliyeti (binde)", 0.0, 5.0, 1.5, 0.5) / 1000

with st.sidebar.expander("⚙️ Pencere ayarları (gelişmiş)"):
    egitim = st.slider("Eğitim penceresi (gün)", 250, 1000, 500, 50)
    test = st.slider("Test penceresi (gün)", 20, 120, 60, 10)
    adim = st.slider("Kaydırma adımı (gün)", 20, 120, 60, 10)

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

# --- Strateji getiri serileri ---
strat_get = bt.strateji_serisi(oos["tahmin"], oos["ertesi_getiri"], maliyet)
altut_get = oos["ertesi_getiri"].reset_index(drop=True)


def _tl(x):   # ₺ bicimleme — yalnizca zaman makinesi (gecmis yatirim) icin
    return f"{x:,.0f}".replace(",", ".") + "₺"


st.subheader(f"{hisse_adi} — {model_ad}")

# Manset: guncel sinyal
if sinyal == 1:
    st.success(f"### 📈 Sinyal: AL  ·  {sinyal_tarih:%d.%m.%Y} verisine göre")
elif sinyal == 0:
    st.info(f"### ⏸️ Sinyal: BEKLE  ·  {sinyal_tarih:%d.%m.%Y} verisine göre")
else:
    st.write("Sinyal hesaplanamadı.")

sekme_ozet, sekme_zaman, sekme_detay = st.tabs(
    ["📊 Özet", "🕰 Geçmiş tarih testi", "🔍 Model detayı"])

# --- SEKME 1: OZET ---
with sekme_ozet:
    m = st.columns(4)
    m[0].metric("Strateji getirisi", f"{strat['kumulatif_getiri']:+.0%}")
    m[1].metric("Al-tut (kıyas)", f"{strat['al_tut_getiri']:+.0%}")
    m[2].metric("Yıllık Sharpe", f"{strat['yillik_sharpe']:.2f}")
    m[3].metric("Maks düşüş", f"{strat['maks_dusus']:.1%}")

    egri = pd.DataFrame({
        "tarih": pd.to_datetime(oos["tarih"]).reset_index(drop=True),
        "Strateji": (1 + strat_get).cumprod(),
        "Al-tut": (1 + altut_get).cumprod(),
    }).set_index("tarih")
    st.markdown("##### Sermaye eğrisi — başlangıç = 1 kat (maliyet sonrası)")
    st.line_chart(egri)
    st.caption(f"{len(oos)} örnek-dışı gün · {strat['islem_sayisi']} işlem · "
               "al-tut ile aynı dönemde karşılaştırıldı. Somut ₺ karşılığı için "
               "→ **🕰 Geçmiş tarih testi** sekmesi.")

# --- SEKME 2: ZAMAN MAKINESI ---
with sekme_zaman:
    if zm_kesim is None:
        st.info("← Kenar çubuğundaki **🕰 Geçmiş tarih testi** menüsünden bir zaman seç "
                "(örn. *2 ay önce*). Model o güne kadarki veriyle eğitilip ne diyeceğini, "
                "sonra gerçekte ne olduğunu gösterir.")
    else:
        st.markdown(f"##### Bu modelle {pd.Timestamp(zm_kesim):%d.%m.%Y} tarihinde başlasaydın")
        with st.spinner("Model o tarihe kadarki veriyle eğitiliyor..."):
            zm = zaman_makinesi(df_tek, model_ad, pd.Timestamp(zm_kesim).isoformat(),
                                sermaye, maliyet)
        if zm is None:
            st.warning("Bu tarih için yeterli veri yok — daha ortada bir zaman seç.")
        else:
            z1, z2, z3 = st.columns(3)
            z1.metric(f"Strateji ({zm['gun']} işlem günü)", _tl(zm["strat_son"]),
                      f"{zm['strat_getiri']:+.1%}")
            z2.metric("Al-tut (kıyas)", _tl(zm["altut_son"]), f"{zm['altut_getiri']:+.1%}")
            z3.metric("Pozisyonda", f"{zm['al_gunu']} / {zm['gun']} gün",
                      help="Modelin AL dediği (hisse tuttuğu) gün sayısı. Kalan günler nakitte.")
            st.line_chart(zm["egri"].set_index("tarih"))
            _not = ""
            if zm["al_gunu"] == 0:
                _not = " Model bu dönemde hiç AL demedi, yani hep nakitte kaldı — o yüzden strateji düz."
            elif zm["strat_getiri"] < zm["altut_getiri"]:
                _not = " Bu kısa pencerede al-tut'u geçemedi; kısa dönem gürültülüdür, genel resim → Özet."
            st.caption(f"💰 {_tl(sermaye)} ile {zm['baslangic']:%d.%m.%Y}'te başlayıp modelin günlük "
                       "sinyalleriyle yönetildi (AL → tut, BEKLE → nakit). Nakitte geçen gün ne "
                       "kazandırır ne kaybettirir." + _not +
                       " Model yalnızca kesim tarihine kadarki veriyle eğitildi — sızıntı yok.")

# --- SEKME 3: MODEL DETAYI ---
with sekme_detay:
    st.caption(f"Örnek-dışı (out-of-sample) değer: {len(oos)} gün  ·  "
               f"Yön doğruluğu {sinif['dogruluk']:.1%} (taban oran {sinif['taban_oran']:.1%})")
    d = st.columns(3)
    d[0].metric("F1", f"{sinif['f1']:.3f}")
    d[1].metric("Kesinlik", f"{sinif['kesinlik']:.3f}")
    d[2].metric("Duyarlılık", f"{sinif['duyarlilik']:.3f}")
    st.caption("Bu değerlerin düşük olması **beklenendir**: 'yükselecek' günleri azınlıkta "
               "(dengesiz sınıf) ve yön tahmini doğası gereği zor. Ölçüt bunlar değil — maliyet "
               "sonrası getiri ve Sharpe. Yüksek F1 / %90 doğruluk görürsen sızıntı/kod hatası ara.")

st.divider()
st.caption("⚠️ Sonuçlar örnek-dışı günlerde, seçilen işlem maliyeti düşülerek hesaplanır. "
           "Geçmiş performans geleceği garanti etmez. Yatırım tavsiyesi değildir.")
