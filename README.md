# BIST Hibrit Tahmin

> 🇬🇧 **English summary** — Next-day direction prediction for Borsa Istanbul stocks from technical indicators, with an honest evaluation setup: walk-forward backtesting (no look-ahead leakage), transaction costs included, and every result benchmarked against buy & hold. Out-of-sample, XGBoost reaches Sharpe 1.35 with ~56% directional accuracy — and still does not beat buy & hold in a strong bull market, which the project openly reports. Includes a Streamlit dashboard and unit tests. Educational project, not financial advice.

Borsa İstanbul hisseleri için teknik göstergelerle "yarın yükselir mi?" tahmini yapan bir proje. Derdim yüksek getiri değildi; **dürüst bir değerlendirme** kurmaktı. Çoğu borsa/ML projesi veri sızıntısı yüzünden gerçekçi olmayan sonuçlar gösteriyor, ben bundan kaçınmaya çalıştım.

Öğrenme amaçlı bir portföy projesi — yatırım tavsiyesi değil.

![Pano — walk-forward sonuçları ve sermaye eğrisi](pano.png)

## Ne yapıyor

Fiyat, hacim, RSI, MACD, volatilite gibi göstergelerden özellik çıkarıp "ertesi gün getirisi %1'i geçecek mi?" sorusunu bir sınıflandırma problemi olarak çözüyor.

Değerlendirmeyi walk-forward ile yapıyorum: yakın geçmişle eğit, hemen sonrasını test et, pencereyi kaydır. Böylece test hep eğitimin geleceğinde kalıyor ve sızıntı olmuyor. Bilerek `train_test_split` kullanmadım — zaman serisinde gelecekten geçmişe bilgi sızdırır.

Sonuçlara accuracy ile değil; işlem maliyeti (binde ~1.5) düşülmüş getiri, Sharpe ve maksimum düşüş ile bakıyorum, hepsini de al-tut (buy & hold) ile karşılaştırıyorum.

## Sonuç (THYAO, 2020–2026, örnek-dışı)

| | Strateji | Al-tut |
|---|---|---|
| Getiri | +633% | +2663% |
| Sharpe | 1.35 | — |
| Maks düşüş | −27.8% | daha derin |
| Yön isabeti | %55.9 | — |

Dürüst olmak gerekirse: teknik göstergeler tek başına, böyle güçlü bir boğa piyasasında al-tut'un getirisini yenmiyor. Ama makul bir Sharpe, daha sınırlı düşüş ve borsa için gerçekçi sayılan ~%56 yön isabeti veriyor.

Karar eşiğini düşürünce backtest getirisi artıyordu, ama bunu yapmadım: sonucu görüp ona göre ayar çekmek test'e uydurmak (overfitting) olurdu. Varsayılan eşiği 0.5'te bıraktım.

## Hangi model?

Üç modeli aynı koşulda denedim:

| Model | Doğruluk | Sharpe | Getiri |
|---|---|---|---|
| Lojistik | 0.519 | 0.81 | +256% |
| XGBoost | 0.603 | 1.35 | +633% |
| LightGBM | 0.567 | 0.98 | +316% |

XGBoost her metrikte önde, o yüzden panoda varsayılan o.

## Dosyalar

```
01_veri_ve_ozellikler.py   veri çekme + özellikler (yfinance)
02_backtest.py             walk-forward backtest motoru
03_model.py                modeller (lojistik, XGBoost, LightGBM)
04_dashboard.py            Streamlit arayüzü
borsa_veri.csv             örnek işlenmiş veri
```

## Çalıştırma

```bash
pip install -r requirements.txt
python 01_veri_ve_ozellikler.py   # borsa_veri.csv üretir (hazır olanı da kullanabilirsin)
streamlit run 04_dashboard.py     # panoyu açar
```

Panoda hisse, model ve eğitim/test penceresini seçip güncel sinyali, maliyet sonrası metrikleri ve sermaye eğrisini görebilirsin. İstersen yeni bir BIST kodu girip canlı veri de çekebilirsin.

## Testler

Backtest motorunun kritik parçaları (Sharpe, maksimum düşüş, işlem maliyeti, sızıntısız pencere kaydırma, hedef etiketleme) birim testlidir; her push'ta GitHub Actions üzerinde otomatik koşarlar.

```bash
python -m pytest
```

## Notlar

- Geçmiş performans geleceği bağlamaz; bu bir öğrenme projesi.
- Geliştirirken ayrıca eşik taraması, para simülasyonu gibi analiz scriptleri de yazdım; repoyu sade tutmak için onları eklemedim.
- Sıradaki fikir: Türkçe haber/yorum verisinden duygu skoru çıkarıp (BERTurk) teknik veriyle birleştirmek. Teknik tarafın tek başına tavanına geldim.
