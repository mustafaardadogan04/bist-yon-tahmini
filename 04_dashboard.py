"""
Streamlit panosu — 01/02/03'u import edip uzerine bir arayuz koyar.

Iki sekme:
- Tek hisse: sayfadan bir hisse sec, guncel sinyali + walk-forward sermaye egrisini gor.
- Gecmis test: bir para + hisseler sec, buton'a bas; model her gun secili hisseler
  icinden en cok guvendigine yuklenip parayi buyutmeye calisir (sizinti yok).

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


def _tl(x):
    return f"{x:,.0f}".replace(",", ".") + "₺"


@st.cache_data(show_spinner=False)
def veri_yukle() -> pd.DataFrame:
    if not VERI_DOSYASI.exists():
        return pd.DataFrame()
    return pd.read_csv(VERI_DOSYASI, parse_dates=["tarih"])


@st.cache_data(show_spinner="Hisse yfinance'ten cekiliyor...")
def hisse_cek(ticker: str, gun: str) -> pd.DataFrame:
    # bir BIST kodunu canli cek; 'gun' cache anahtari (gunde bir cekim)
    # tahmin_satiri=True: en son bar (hedefi henuz bilinmeyen) tahmin icin tutulur
    usdtry = veri.usdtry_degisimi_cek()
    return veri.hisseyi_isle(ticker, usdtry if not usdtry.empty else None,
                             tahmin_satiri=True)


@st.cache_data(show_spinner="Hisseler yfinance'ten güncel çekiliyor...")
def coklu_canli(hisseler: tuple, gun: str) -> pd.DataFrame:
    # birden cok hisseyi canli cekip birlestir (portfoy icin). Gunde bir cekim.
    parcalar = [hisse_cek(h, gun) for h in hisseler]
    parcalar = [p for p in parcalar if not p.empty]
    return pd.concat(parcalar, ignore_index=True) if parcalar else pd.DataFrame()


@st.cache_data(show_spinner=False)
def backtest_calistir(df_tek: pd.DataFrame, model_ad: str,
                      egitim: int, test: int, adim: int) -> pd.DataFrame:
    df2 = bt.ertesi_getiri_ekle(df_tek)
    return bt.walk_forward(df2, ml.MODELLER[model_ad], egitim, test, adim, False)


def guncel_sinyal(df_tek: pd.DataFrame, model_ad: str):
    # sonucu bilinen gunlerle egit, verideki EN SON gunu tahmin et
    df2 = bt.ertesi_getiri_ekle(df_tek)
    if len(df2) < 50:
        return None, None
    ozellik = [k for k in df2.columns if k not in bt.META_KOLONLAR]
    son = df_tek.sort_values("tarih").iloc[[-1]]
    tahmin = ml.MODELLER[model_ad](df2[ozellik], df2["hedef"], son[ozellik])
    return int(tahmin[0]), son["tarih"].iloc[-1]


# ── Portfoy (gecmis test) motoru ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _portfoy_ham(df_secili: pd.DataFrame, model_ad: str, kesim_iso: str):
    # Her secili hisseyi kesim tarihine kadar egit, sonra o gunden bugune her gun
    # yukselis olasiligi + gercek getiri uret. Egitim yalniz kesim'den ONCEKI
    # veriyle -> gelecekten sizinti yok. (strateji/sermaye/maliyetten bagimsiz)
    kesim = pd.Timestamp(kesim_iso)
    getiri_tab, proba_tab = {}, {}
    for h, d in df_secili.groupby("hisse"):
        d_sirali = d.sort_values("tarih").reset_index(drop=True)
        # d2'deki "tarih" KARAR gunu (o gunun kapanisinda tahmin uretilir); ama
        # getiri bir sonraki islem gununde GERCEKLESIR. Grafikte/logda karar
        # gunu yerine gerceklesme gununu gostermeliyiz, yoksa son nokta hep bir
        # islem gunu geriden gelir (en taze kapanisi hic gostermez).
        gerceklesme = pd.Series(d_sirali["tarih"].shift(-1).values,
                                index=d_sirali["tarih"].values)
        d2 = bt.ertesi_getiri_ekle(d_sirali)
        ozellik = [c for c in d2.columns if c not in bt.META_KOLONLAR]
        egitim = d2[d2["tarih"] < kesim]
        pencere = d2[d2["tarih"] >= kesim]
        if len(egitim) < 100 or len(pencere) < 5:
            continue
        proba = ml.OLASILIKLAR[model_ad](egitim[ozellik], egitim["hedef"], pencere[ozellik])
        idx = pd.to_datetime(pencere["tarih"].map(gerceklesme).values)
        getiri_tab[h.replace(".IS", "")] = pd.Series(pencere["ertesi_getiri"].values, index=idx)
        proba_tab[h.replace(".IS", "")] = pd.Series(proba, index=idx)
    if not getiri_tab:
        return None
    G = pd.DataFrame(getiri_tab).sort_index()
    P = pd.DataFrame(proba_tab).reindex(G.index)
    return {"tarihler": [pd.Timestamp(t) for t in G.index],
            "getiri": {h: G[h].fillna(0).tolist() for h in G.columns},
            "proba": {h: P[h].fillna(0.0).tolist() for h in P.columns}}


def portfoy_test(df_secili, model_ad, kesim_iso, sermaye, maliyet, k=None):
    # AL VE TUT: kesim gununde modelin secitigi hisseleri al, sona kadar esit tut.
    # k=None -> modelin kendisi karar verir (guvendigi tum hisseler). k=int -> sabit K.
    # Satis yok (sadece bir kez giris maliyeti). Grafik yalniz tutulan hisseleri gosterir.
    r = _portfoy_ham(df_secili, model_ad, kesim_iso)
    if r is None:
        return None
    tarihler = pd.to_datetime(r["tarihler"])
    G = pd.DataFrame(r["getiri"], index=tarihler)
    P = pd.DataFrame(r["proba"], index=tarihler)

    ilk = P.iloc[0].sort_values(ascending=False)     # kesim gunundeki guven sirasi
    if k is None:
        # MODEL KARAR VERSIN: guvendigi (>0.5) tum hisseler; hicbiri yoksa en iyi 1
        secilen = list(ilk[ilk > 0.5].index) or [ilk.index[0]]
    else:
        secilen = list(ilk.head(k).index)            # kullanici K'yi sabitledi
    port = G[secilen].mean(axis=1).copy()            # esit agirlik, tutulur
    port.iloc[0] -= maliyet                           # tek giris maliyeti
    bench = G.mean(axis=1)                            # kiyas: tum secilenleri esit tut

    egri = {h: (sermaye * (1 + G[h]).cumprod()).values for h in secilen}   # tutulan hisseler
    egri["★ Model portföy"] = (sermaye * (1 + port.values).cumprod())
    if len(G.columns) > len(secilen):                # kiyas ancak fazladan hisse varsa anlamli
        egri["Al-tut (tümü)"] = (sermaye * (1 + bench.values).cumprod())
    egri_df = pd.DataFrame(egri, index=tarihler)

    # Gunluk log: gun gun ne oldu (portfoy getirisi, para, her hissenin hareketi)
    gunluk = pd.DataFrame(index=tarihler.strftime("%d.%m.%Y"))
    gunluk.index.name = "Tarih"
    gunluk["Model günlük %"] = (port.values * 100).round(2)
    gunluk["Para (₺)"] = (sermaye * (1 + port.values).cumprod()).round(0).astype(int)
    for h in secilen:
        gunluk[f"{h} %"] = (G[h].values * 100).round(2)

    return {
        "gun": len(tarihler), "baslangic": tarihler[0], "secilen": secilen,
        "model_son": sermaye * float((1 + port.values).prod()),
        "model_getiri": float((1 + port.values).prod() - 1),
        "bench_son": sermaye * float((1 + bench.values).prod()),
        "bench_getiri": float((1 + bench.values).prod() - 1),
        "egri": egri_df, "gunluk": gunluk,
    }


# ── Arayuz ───────────────────────────────────────────────────────────────────
st.set_page_config(page_title="BIST Tahmin Panosu", page_icon="📈", layout="wide")
st.markdown("<style>[data-testid='stStatusWidget']{display:none;}</style>",
            unsafe_allow_html=True)

st.title("📈 BIST Tahmin Panosu")
st.caption("Sızıntısız walk-forward backtest ile teknik göstergelere dayalı yön tahmini. "
           "Öğrenme/portföy projesidir — yatırım tavsiyesi değildir.")

tum_veri = veri_yukle()
bugun = date.today().isoformat()

# Kenar cubugu: minimal
st.sidebar.header("Ayarlar")
model_ad = st.sidebar.selectbox("Model", list(ml.MODELLER),
                                index=list(ml.MODELLER).index("xgboost"))
st.sidebar.caption("XGBoost: ölçülen en verimli model.")
with st.sidebar.expander("⚙️ Gelişmiş"):
    maliyet = st.slider("İşlem maliyeti (binde)", 0.0, 5.0, 1.5, 0.5) / 1000
    egitim = st.slider("Eğitim penceresi (gün)", 250, 1000, 500, 50)
    test = st.slider("Test penceresi (gün)", 20, 120, 60, 10)
    adim = st.slider("Kaydırma adımı (gün)", 20, 120, 60, 10)

hisse_listesi = sorted(tum_veri["hisse"].unique()) if not tum_veri.empty else []

sekme_tek, sekme_gecmis = st.tabs(["📈 Tek hisse", "💼 Geçmiş test (portföy)"])


# ── SEKME 1: TEK HISSE ───────────────────────────────────────────────────────
with sekme_tek:
    secenekler = ["— Hisse seç —"] + hisse_listesi + ["+ Başka hisse (canlı)"]
    secim = st.selectbox("Hisse", secenekler, index=0, key="tek_hisse")
    st.caption("🔄 Veri otomatik olarak yfinance'ten güncel çekilir; başarısız olursa "
               "repodaki donmuş veriye düşülür.")

    df_tek, hisse_adi = pd.DataFrame(), None
    if secim == "+ Başka hisse (canlı)":
        yeni = st.text_input("BIST kodu (örn: GARAN.IS)", value="").strip().upper()
        if yeni:
            df_tek = hisse_cek(yeni, bugun)
            hisse_adi = yeni
    elif secim != "— Hisse seç —":
        hisse_adi = secim
        df_tek = hisse_cek(secim, bugun)
        if df_tek.empty:
            st.warning("Canlı veri çekilemedi, donmuş veriye düşüldü.")
            df_tek = tum_veri[tum_veri["hisse"] == secim].copy()

    if hisse_adi is None:
        st.info("👆 Bir hisse seç. Model o hissenin yarınki yön sinyalini ve geçmiş "
                "başarısını (sızıntısız backtest) gösterir.")
    elif df_tek.empty or len(df_tek) < egitim + test:
        st.warning("Yeterli veri yok. Farklı bir hisse seç ya da Gelişmiş'ten pencereyi küçült.")
    else:
        with st.spinner("Hesaplanıyor..."):
            oos = backtest_calistir(df_tek, model_ad, egitim, test, adim)
            strat = bt.strateji_metrikleri(oos["tahmin"], oos["ertesi_getiri"], maliyet)
            sinif = bt.siniflandirma_metrikleri(oos["hedef"].to_numpy(), oos["tahmin"].to_numpy())
            sinyal, sinyal_tarih = guncel_sinyal(df_tek, model_ad)

        # tahmin, son veri gununun ERTESI islem gunu icin -> hafta sonunu atla
        hedef_gun = pd.Timestamp(sinyal_tarih) + pd.offsets.BDay(1)
        if sinyal == 1:
            st.success(f"### 📈 {hisse_adi}: **{hedef_gun:%d.%m.%Y}** için sinyal **AL**  ·  {sinyal_tarih:%d.%m.%Y} verisine göre")
        elif sinyal == 0:
            st.info(f"### ⏸️ {hisse_adi}: **{hedef_gun:%d.%m.%Y}** için sinyal **BEKLE**  ·  {sinyal_tarih:%d.%m.%Y} verisine göre")

        m = st.columns(4)
        m[0].metric("Strateji getirisi", f"{strat['kumulatif_getiri']:+.0%}")
        m[1].metric("Al-tut (kıyas)", f"{strat['al_tut_getiri']:+.0%}")
        m[2].metric("Yıllık Sharpe", f"{strat['yillik_sharpe']:.2f}")
        m[3].metric("Maks düşüş", f"{strat['maks_dusus']:.1%}")

        strat_get = bt.strateji_serisi(oos["tahmin"], oos["ertesi_getiri"], maliyet)
        altut_get = oos["ertesi_getiri"].reset_index(drop=True)
        egri = pd.DataFrame({
            "tarih": pd.to_datetime(oos["tarih"]).reset_index(drop=True),
            "Strateji": (1 + strat_get).cumprod(),
            "Al-tut": (1 + altut_get).cumprod(),
        }).set_index("tarih")
        st.markdown("##### Sermaye eğrisi — başlangıç = 1 kat (maliyet sonrası)")
        st.line_chart(egri)
        st.caption(f"{len(oos)} örnek-dışı gün ({oos['tarih'].min():%m.%Y} – "
                   f"{oos['tarih'].max():%m.%Y}) · {strat['islem_sayisi']} işlem. "
                   "Para ile çok-hisse senaryosu → **💼 Geçmiş test** sekmesi.")

        with st.expander("🔍 Model detayı (F1 / kesinlik / duyarlılık)"):
            d = st.columns(3)
            d[0].metric("F1", f"{sinif['f1']:.3f}")
            d[1].metric("Kesinlik", f"{sinif['kesinlik']:.3f}")
            d[2].metric("Duyarlılık", f"{sinif['duyarlilik']:.3f}")
            st.caption("Bu değerlerin düşük olması beklenendir: dengesiz sınıf + yön tahmini zor. "
                       "Ölçüt bunlar değil, maliyet sonrası getiri ve Sharpe.")


# ── SEKME 2: GECMIS TEST (PORTFOY) ───────────────────────────────────────────
with sekme_gecmis:
    if len(hisse_listesi) < 2:
        st.info("Bu özellik için `borsa_veri.csv` çok hisseli olmalı "
                "(`01_veri_ve_ozellikler.py --hisseler THYAO.IS GARAN.IS ... --usdtry`).")
    else:
        st.markdown("#### Geçmişe git, model senin için en iyi hisseleri seçsin")
        st.caption(f"Model, {len(hisse_listesi)} hissenin hepsine bakar; hangilerini ve "
                   "**kaç tanesini** alacağına kendisi karar verir. Sen sadece para ve zamanı söylersin.")

        st.caption("🔄 Tüm hisseler otomatik olarak yfinance'ten güncel çekilir "
                   "(biraz sürebilir); başarısız olursa donmuş veriye düşülür.")
        pf_veri = coklu_canli(tuple(hisse_listesi), bugun)
        if pf_veri.empty:
            st.warning("Canlı veri çekilemedi (yfinance engellemiş olabilir). Donmuş "
                       "veriye dönülüyor.")
            pf_veri = tum_veri

        temizle = lambda s: s.replace(".IS", "")
        g1, g2 = st.columns(2)
        para = g1.number_input("Para (₺)", 1000, 10_000_000, 10_000, 1000)
        zaman = g2.selectbox("Ne zaman başlasın?",
                             ["60 gün önce", "3 ay önce", "6 ay önce", "1 yıl önce", "2 yıl önce"],
                             index=0)

        # Varsayilan: model kendi karar verir + tum hisseler havuz. Isteyen degistirebilir.
        with st.expander("🔧 İsteğe bağlı ayarlar"):
            secili = st.multiselect("Model yalnızca bunlar arasından seçsin (varsayılan: tümü)",
                                    hisse_listesi, default=hisse_listesi, format_func=temizle)
            k_sabit = st.checkbox("Kaç hisse tutacağını ben belirleyeyim")
            k_deger = st.slider("En iyi K hisse", 1, 8, 3, disabled=not k_sabit)
        if not secili:
            secili = hisse_listesi
        port_k = k_deger if k_sabit else None      # None -> model kendi karar verir

        if st.button("▶ Hesapla", type="primary"):
            st.session_state["pf_calistir"] = True

        if st.session_state.get("pf_calistir"):
            _son = pf_veri["tarih"].max()
            _teklif = {
                "60 gün önce": pd.DateOffset(days=60),
                "3 ay önce": pd.DateOffset(months=3),
                "6 ay önce": pd.DateOffset(months=6),
                "1 yıl önce": pd.DateOffset(months=12),
                "2 yıl önce": pd.DateOffset(months=24),
            }[zaman]
            kesim = (_son - _teklif).date()
            df_sec = pf_veri[pf_veri["hisse"].isin(secili)]
            kullan_k = None if port_k is None else min(port_k, len(secili))
            with st.spinner(f"Model {len(secili)} hisseye bakıp en iyilerini seçiyor..."):
                pf = portfoy_test(df_sec, model_ad, pd.Timestamp(kesim).isoformat(),
                                  para, maliyet, kullan_k)
            if pf is None:
                st.warning("Bu zaman aralığı için yeterli veri yok — daha yakın bir zaman seç.")
            else:
                k = st.columns(3)
                k[0].metric("💰 Model portföy", _tl(pf["model_son"]), f"{pf['model_getiri']:+.1%}")
                k[1].metric("Al-tut (tüm hisseler)", _tl(pf["bench_son"]), f"{pf['bench_getiri']:+.1%}")
                k[2].metric("Süre", f"{pf['gun']} işlem günü")

                st.markdown("##### Para eğrisi — modelin seçtiği hisseler + **★ Model portföy**")
                st.line_chart(pf["egri"])

                secilen_txt = ", ".join(pf["secilen"])
                kiyas = ("**Model, hepsini tutmaktan iyiydi 👍**" if pf["model_getiri"] > pf["bench_getiri"]
                         else "Bu dönemde hepsini tutmayı geçemedi.")
                karar = "kendi karar verip" if port_k is None else "senin belirlediğin sayıda"
                st.caption(
                    f"💡 {_tl(para)} ile {pf['baslangic']:%d.%m.%Y}'te başlandı. Model {karar} "
                    f"**{len(pf['secilen'])} hisse seçip tuttu: {secilen_txt}** (satış yok). "
                    f"★ Model portföy = bu hisselerin eşit ağırlığı. "
                    f"Al-tut (tümü) = tüm hisseleri körü körüne tutmak. {kiyas} "
                    "Model yalnızca geçmiş veriyle seçti — geleceği görmedi, sızıntı yok.")

                with st.expander(f"📋 Günlük log — {pf['gun']} gün, gün gün ne oldu"):
                    st.dataframe(pf["gunluk"], use_container_width=True)
                    _csv = pf["gunluk"].to_csv().encode("utf-8-sig")
                    st.download_button("📥 Logu indir (CSV)", _csv,
                                       "portfoy_gunluk_log.csv", "text/csv")

st.divider()
st.caption("⚠️ Sonuçlar örnek-dışı günlerde, seçilen işlem maliyeti düşülerek hesaplanır. "
           "Geçmiş performans geleceği garanti etmez. Yatırım tavsiyesi değildir.")
