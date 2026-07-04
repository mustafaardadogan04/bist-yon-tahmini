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
def _portfoy_ham(df_coklu: pd.DataFrame, model_ad: str, kesim_iso: str, k: int):
    # PORTFOY (pahali kisim, cache'li): her hisseyi kesim tarihine kadarki veriyle egit,
    # sonra o gunden bugune her gun yukselis olasiligi uret. Egitim yalniz kesim'den
    # ONCEKI veriyle -> sizinti yok. (sermaye/maliyetten bagimsiz)
    kesim = pd.Timestamp(kesim_iso)
    gunluk = {}   # tarih -> {hisse: (olasilik, ertesi_getiri)}
    for h, d in df_coklu.groupby("hisse"):
        d2 = bt.ertesi_getiri_ekle(d.sort_values("tarih").reset_index(drop=True))
        ozellik = [c for c in d2.columns if c not in bt.META_KOLONLAR]
        egitim = d2[d2["tarih"] < kesim]
        pencere = d2[d2["tarih"] >= kesim]
        if len(egitim) < 100 or len(pencere) < 5:
            continue
        proba = ml.OLASILIKLAR[model_ad](egitim[ozellik], egitim["hedef"], pencere[ozellik])
        for t, p, g in zip(pencere["tarih"], proba, pencere["ertesi_getiri"]):
            gunluk.setdefault(t, {})[h] = (float(p), float(g))
    if not gunluk:
        return None
    tarihler = sorted(gunluk)
    onceki = set()
    port_ham, turnover, bench = [], [], []
    for t in tarihler:
        adaylar = sorted(gunluk[t].items(), key=lambda x: -x[1][0])   # olasiliga gore
        secili = [(h, pg) for h, pg in adaylar if pg[0] > 0.5][:k]     # en iyi k AL
        sset = {h for h, _ in secili}
        port_ham.append(sum(pg[1] for _, pg in secili) / len(secili) if secili else 0.0)
        turnover.append(len(sset ^ onceki) / max(k, 1))   # degisen pozisyon orani
        onceki = sset
        gt = [pg[1] for pg in gunluk[t].values()]
        bench.append(sum(gt) / len(gt))                    # hepsini esit tut (al-tut)
    return {"tarih": [pd.Timestamp(t) for t in tarihler],
            "port_ham": port_ham, "turnover": turnover, "bench": bench}


def portfoy_gecmis(df_coklu, model_ad, kesim_iso, sermaye, maliyet, k):
    # ucuz kisim: maliyet (turnover) + sermaye uygula
    r = _portfoy_ham(df_coklu, model_ad, kesim_iso, k)
    if r is None:
        return None
    port = pd.Series(r["port_ham"]) - maliyet * pd.Series(r["turnover"])
    bench = pd.Series(r["bench"])
    return {
        "gun": len(r["tarih"]),
        "baslangic": r["tarih"][0],
        "port_son": sermaye * float((1 + port).prod()),
        "bench_son": sermaye * float((1 + bench).prod()),
        "port_getiri": float((1 + port).prod() - 1),
        "bench_getiri": float((1 + bench).prod() - 1),
        "egri": pd.DataFrame({
            "tarih": r["tarih"],
            f"Portföy (en iyi {k})": (sermaye * (1 + port).cumprod()).values,
            "Al-tut (hepsi eşit)": (sermaye * (1 + bench).cumprod()).values,
        }).set_index("tarih"),
    }


@st.cache_data(show_spinner=False)
def bugun_tarama(df_coklu: pd.DataFrame, model_ad: str):
    # her hisse icin TUM veriyle egit, EN SON gunu tahmin et; olasiliga gore sirala
    sonuc = []
    for h, d in df_coklu.groupby("hisse"):
        d = d.sort_values("tarih").reset_index(drop=True)
        d2 = bt.ertesi_getiri_ekle(d)   # hedefi bilinmeyen son gun burada duser
        ozellik = [c for c in d2.columns if c not in bt.META_KOLONLAR]
        if len(d2) < 100:
            continue
        son = d.iloc[[-1]]
        p = float(ml.OLASILIKLAR[model_ad](d2[ozellik], d2["hedef"], son[ozellik])[0])
        sonuc.append({"Hisse": h.replace(".IS", ""),
                      "Yükseliş olasılığı": p,
                      "Sinyal": "🟢 AL" if p > 0.5 else "⚪ BEKLE"})
    if not sonuc:
        return pd.DataFrame()
    return (pd.DataFrame(sonuc)
            .sort_values("Yükseliş olasılığı", ascending=False)
            .reset_index(drop=True))


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
_secenekler = (mevcut_hisseler + ["+ Yeni hisse cek..."]) if mevcut_hisseler else ["+ Yeni hisse cek..."]
_vars_idx = _secenekler.index("THYAO.IS") if "THYAO.IS" in _secenekler else 0
secim = st.sidebar.selectbox("Hisse (Özet sekmesi için)", _secenekler, index=_vars_idx)

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

# --- Portfoy kontrolleri (kenar cubugu; cok hisse) ---
st.sidebar.divider()
st.sidebar.subheader("💼 Portföy (çok hisse)")
port_k = st.sidebar.slider("Kaç hisse tutulsun (en iyi K)", 1, 8, 3,
                           help="Model her gün en güvendiği K hisseyi eşit ağırlıkla tutar.")
port_secim = st.sidebar.selectbox(
    "Geçmiş senaryo: hangi güne?",
    ["(kapalı)", "1 ay önce", "2 ay önce", "3 ay önce", "6 ay önce", "1 yıl önce"],
    help="Seçilen tarihe kadar eğit, o günden bugüne portföyü modelin sinyalleriyle yürüt.")

port_kesim = None
if port_secim != "(kapalı)" and not tum_veri.empty:
    _son = tum_veri["tarih"].max()
    _ay = {"1 ay önce": 1, "2 ay önce": 2, "3 ay önce": 3, "6 ay önce": 6, "1 yıl önce": 12}[port_secim]
    port_kesim = (_son - pd.DateOffset(months=_ay)).date()

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

sekme_ozet, sekme_portfoy, sekme_detay = st.tabs(
    ["📊 Özet (tek hisse)", "💼 Portföy (çok hisse)", "🔍 Model detayı"])

# --- SEKME 1: OZET (tek hisse) ---
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
               "al-tut ile aynı dönemde karşılaştırıldı. Para ile çok-hisse senaryosu "
               "→ **💼 Portföy** sekmesi.")

# --- SEKME 2: PORTFOY (cok hisse) ---
with sekme_portfoy:
    if tum_veri.empty or tum_veri["hisse"].nunique() < 2:
        st.info("Portföy için birden çok hisse gerekir. `borsa_veri.csv`'yi çok hisseli üret "
                "(`01_veri_ve_ozellikler.py --hisseler THYAO.IS GARAN.IS ... --usdtry`).")
    else:
        # A) Bugun tarama
        st.markdown("##### 📅 Bugün — model hangi hisselerde AL diyor?")
        with st.spinner(f"{tum_veri['hisse'].nunique()} hisse taranıyor..."):
            tar = bugun_tarama(tum_veri, model_ad)
        if tar.empty:
            st.warning("Tarama üretilemedi.")
        else:
            al_sayi = int(tar["Sinyal"].str.contains("AL").sum())
            st.dataframe(tar, use_container_width=True, hide_index=True,
                         column_config={"Yükseliş olasılığı": st.column_config.ProgressColumn(
                             "Yükseliş olasılığı", format="%.0f%%", min_value=0, max_value=1)})
            st.caption(f"Yarın için tahmin · **{al_sayi} hissede AL** sinyali. En üsttekiler "
                       "modelin en güvendiği (= en çok artması beklenen).")

        st.divider()
        # B) Gecmis senaryo
        st.markdown("##### 🕰 Geçmiş senaryo — bu portföyle başlasaydın")
        if port_kesim is None:
            st.info("← Kenar çubuğundaki **💼 Portföy → Geçmiş senaryo** menüsünden bir zaman "
                    "seç (örn. *6 ay önce*). Model her hisse için o güne kadar eğitilir, sonra "
                    "her gün en güvendiği hisseler tutulur.")
        else:
            with st.spinner("Model her hisse için o güne kadar eğitiliyor..."):
                pg = portfoy_gecmis(tum_veri, model_ad, pd.Timestamp(port_kesim).isoformat(),
                                    sermaye, maliyet, port_k)
            if pg is None:
                st.warning("Bu tarih için yeterli veri yok — daha yakın bir zaman seç.")
            else:
                p1, p2, p3 = st.columns(3)
                p1.metric(f"Portföy (en iyi {port_k})", _tl(pg["port_son"]),
                          f"{pg['port_getiri']:+.1%}")
                p2.metric("Al-tut (hepsi eşit)", _tl(pg["bench_son"]), f"{pg['bench_getiri']:+.1%}")
                p3.metric("Süre", f"{pg['gun']} işlem günü")
                st.line_chart(pg["egri"])
                _fark = ("Portföy al-tut'u geçti 👍" if pg["port_getiri"] > pg["bench_getiri"]
                         else "Bu dönemde al-tut'u geçemedi — geniş boğada seçicilik çoğu zaman "
                              "geniş tutmayı yenmez (projenin tezine uygun).")
                st.caption(f"💰 {_tl(sermaye)} ile {pg['baslangic']:%d.%m.%Y}'te başlayıp model her "
                           f"gün en güvendiği {port_k} AL hissesini eşit tuttu. Al-tut = 15 hisseyi "
                           f"eşit tutmak. {_fark} Model kesim öncesi veriyle eğitildi — sızıntı yok.")

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
