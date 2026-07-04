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
    usdtry = veri.usdtry_degisimi_cek()
    return veri.hisseyi_isle(ticker, usdtry if not usdtry.empty else None)


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
        d2 = bt.ertesi_getiri_ekle(d.sort_values("tarih").reset_index(drop=True))
        ozellik = [c for c in d2.columns if c not in bt.META_KOLONLAR]
        egitim = d2[d2["tarih"] < kesim]
        pencere = d2[d2["tarih"] >= kesim]
        if len(egitim) < 100 or len(pencere) < 5:
            continue
        proba = ml.OLASILIKLAR[model_ad](egitim[ozellik], egitim["hedef"], pencere[ozellik])
        idx = pd.to_datetime(pencere["tarih"].values)
        getiri_tab[h.replace(".IS", "")] = pd.Series(pencere["ertesi_getiri"].values, index=idx)
        proba_tab[h.replace(".IS", "")] = pd.Series(proba, index=idx)
    if not getiri_tab:
        return None
    G = pd.DataFrame(getiri_tab).sort_index()
    P = pd.DataFrame(proba_tab).reindex(G.index)
    return {"tarihler": [pd.Timestamp(t) for t in G.index],
            "getiri": {h: G[h].fillna(0).tolist() for h in G.columns},
            "proba": {h: P[h].fillna(0.0).tolist() for h in P.columns}}


def portfoy_test(df_secili, model_ad, kesim_iso, sermaye, maliyet, tarz, k):
    # ucuz kisim: secili tarza gore portfoyu kur (aktif = gunluk gecis, al_tut = bastaki en iyi K)
    r = _portfoy_ham(df_secili, model_ad, kesim_iso)
    if r is None:
        return None
    tarihler = pd.to_datetime(r["tarihler"])
    G = pd.DataFrame(r["getiri"], index=tarihler)
    P = pd.DataFrame(r["proba"], index=tarihler)

    if tarz == "aktif":
        # her gun en guvendigi AL hisseye yuklen; ayni kalirsa maliyet yok
        al = P.where(P > 0.5)
        gecerli = al.notna().any(axis=1)
        en_iyi = pd.Series(pd.NA, index=al.index, dtype=object)
        en_iyi.loc[gecerli] = al.loc[gecerli].idxmax(axis=1)
        port_get, turnover, secili = [], [], []
        onceki = None
        for t in G.index:
            h = en_iyi.loc[t]
            if pd.isna(h):
                port_get.append(0.0); turnover.append(1 if onceki else 0); onceki = None
                secili.append("Nakit")
            else:
                port_get.append(float(G.loc[t, h]))
                turnover.append(0 if h == onceki else 1); onceki = h
                secili.append(h)
        port = pd.Series(port_get, index=G.index) - maliyet * pd.Series(turnover, index=G.index)
        tutulan = pd.Series(secili).value_counts()
        aciklama = f"Model her gün en güvendiği AL hissesine geçti. En çok: "
    else:
        # AL VE TUT: kesimdeki (ilk gun) en guvendigi K hisseyi al, sona kadar esit tut
        ilk = P.iloc[0].sort_values(ascending=False)
        secilen = list(ilk.head(k).index)
        port = G[secilen].mean(axis=1).copy()
        port.iloc[0] -= maliyet          # sadece bir kez giris maliyeti
        tutulan = pd.Series({h: len(G) for h in secilen})
        aciklama = f"Kesim gününde modelin en güvendiği {len(secilen)} hisse alınıp tutuldu: "

    egri = {h: (sermaye * (1 + pd.Series(gl)).cumprod()).values for h, gl in r["getiri"].items()}
    egri["★ Model portföy"] = (sermaye * (1 + port.values).cumprod())
    egri_df = pd.DataFrame(egri, index=tarihler)
    bench = pd.DataFrame(r["getiri"]).mean(axis=1)
    return {
        "gun": len(tarihler), "baslangic": tarihler[0],
        "model_son": sermaye * float((1 + port.values).prod()),
        "model_getiri": float((1 + port.values).prod() - 1),
        "bench_son": sermaye * float((1 + bench.values).prod()),
        "bench_getiri": float((1 + bench.values).prod() - 1),
        "egri": egri_df, "tutulan": tutulan, "aciklama": aciklama,
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
    c1, c2 = st.columns([2, 1])
    secenekler = ["— Hisse seç —"] + hisse_listesi + ["+ Başka hisse (canlı)"]
    secim = c1.selectbox("Hisse", secenekler, index=0, key="tek_hisse")
    canli = c2.checkbox("🔄 Canlı veri", value=False,
                        help="Seçili hisseyi yfinance'ten güncel veriyle çeker.")

    df_tek, hisse_adi = pd.DataFrame(), None
    if secim == "+ Başka hisse (canlı)":
        yeni = c1.text_input("BIST kodu (örn: GARAN.IS)", value="").strip().upper()
        if yeni:
            df_tek = hisse_cek(yeni, bugun)
            hisse_adi = yeni
    elif secim != "— Hisse seç —":
        hisse_adi = secim
        df_tek = hisse_cek(secim, bugun) if canli else tum_veri[tum_veri["hisse"] == secim].copy()

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

        if sinyal == 1:
            st.success(f"### 📈 {hisse_adi}: yarın için sinyal **AL**  ·  {sinyal_tarih:%d.%m.%Y}")
        elif sinyal == 0:
            st.info(f"### ⏸️ {hisse_adi}: yarın için sinyal **BEKLE**  ·  {sinyal_tarih:%d.%m.%Y}")

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
        st.markdown("#### Geçmişe git, parayı en iyi hisselerle büyütmeyi dene")
        g1, g2, g3 = st.columns([2, 1, 1])
        temizle = lambda s: s.replace(".IS", "")
        varsayilan = [h for h in ["THYAO.IS", "GARAN.IS", "ASELS.IS", "BIMAS.IS", "TUPRS.IS"]
                      if h in hisse_listesi]
        secili = g1.multiselect("Hisseler (istediğin kadar)", hisse_listesi,
                                default=varsayilan, format_func=temizle)
        para = g2.number_input("Para (₺)", 1000, 10_000_000, 10_000, 1000)
        zaman = g3.selectbox("Ne zaman başlasın?",
                             ["1 ay önce", "3 ay önce", "6 ay önce", "1 yıl önce", "2 yıl önce"],
                             index=2)

        h1, h2 = st.columns([2, 1])
        tarz_etiket = h1.radio(
            "İşlem tarzı",
            ["🔄 Günlük aktif — her gün en güvendiğine geç",
             "📌 Al ve tut — baştaki en iyi K hisseyi tut"],
            help="Aktif: sık al-sat (maliyet yüksek). Al-tut: baştan seçip tutar (maliyet ~yok).")
        tarz = "aktif" if tarz_etiket.startswith("🔄") else "al_tut"
        port_k = h2.slider("Al-tut: kaç hisse?", 1, 8, 3, disabled=(tarz == "aktif"))

        if st.button("▶ Hesapla", type="primary"):
            st.session_state["pf_calistir"] = True

        if st.session_state.get("pf_calistir"):
            if len(secili) < 1:
                st.warning("En az bir hisse seç.")
            else:
                _son = tum_veri["tarih"].max()
                _ay = {"1 ay önce": 1, "3 ay önce": 3, "6 ay önce": 6,
                       "1 yıl önce": 12, "2 yıl önce": 24}[zaman]
                kesim = (_son - pd.DateOffset(months=_ay)).date()
                df_sec = tum_veri[tum_veri["hisse"].isin(secili)]
                with st.spinner(f"{len(secili)} hisse için model o güne kadar eğitiliyor..."):
                    pf = portfoy_test(df_sec, model_ad, pd.Timestamp(kesim).isoformat(),
                                      para, maliyet, tarz, port_k)
                if pf is None:
                    st.warning("Bu zaman aralığı için yeterli veri yok — daha yakın bir zaman seç.")
                else:
                    k = st.columns(3)
                    k[0].metric("💰 Model portföy", _tl(pf["model_son"]),
                                f"{pf['model_getiri']:+.1%}")
                    k[1].metric("Al-tut (seçilenler eşit)", _tl(pf["bench_son"]),
                                f"{pf['bench_getiri']:+.1%}")
                    k[2].metric("Süre", f"{pf['gun']} işlem günü")

                    st.markdown("##### Para eğrisi — her hisse farklı renk, **★ Model portföy** kalın")
                    st.line_chart(pf["egri"])

                    en_cok = pf["tutulan"].drop(labels=["Nakit"], errors="ignore")
                    tutulan_txt = ", ".join(f"{h} ({n}g)" for h, n in en_cok.head(4).items()) or "—"
                    kiyas = ("Model al-tut'u geçti 👍" if pf["model_getiri"] > pf["bench_getiri"]
                             else "Bu dönemde al-tut'u geçemedi.")
                    st.caption(
                        f"💡 {_tl(para)} ile {pf['baslangic']:%d.%m.%Y}'te başlandı. {pf['aciklama']}"
                        f"{tutulan_txt}. {kiyas} Model yalnızca geçmiş veriyle karar verdi — "
                        "geleceği görmedi, sızıntı yok.")

st.divider()
st.caption("⚠️ Sonuçlar örnek-dışı günlerde, seçilen işlem maliyeti düşülerek hesaplanır. "
           "Geçmiş performans geleceği garanti etmez. Yatırım tavsiyesi değildir.")
