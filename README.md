# Applied Data Science - Sınıflandırma Projesi

## Proje Hakkında

Bu proje, **i2i Academy Applied Data Science** eğitimi kapsamında geliştirilmiştir.

Projede, makine öğrenmesi kullanılarak bir sınıflandırma problemi çözülmüştür. Hazır olarak sunulan **Breast Cancer Wisconsin** veri seti kullanılmış ve farklı sınıflandırma algoritmalarının performansları karşılaştırılmıştır.

---

## Projenin Amacı

Bu projenin amacı;

- Makine öğrenmesi iş akışını uygulamak,
- Veri setini analiz etmek,
- Veriyi eğitim ve test olarak ayırmak,
- Farklı sınıflandırma algoritmalarını eğitmek,
- Modelleri karşılaştırarak en başarılı algoritmayı belirlemektir.

---

## Kullanılan Veri Seti

Projede **Breast Cancer Wisconsin Dataset** kullanılmıştır.

Veri seti;

- 569 örnekten oluşmaktadır.
- 30 adet sayısal özellik içermektedir.
- İkili sınıflandırma problemidir.

Hedef değişken:

- **0 → Malignant (Kötü Huylu)**
- **1 → Benign (İyi Huylu)**

---

## Kullanılan Teknolojiler

- Python
- Pandas
- Scikit-learn

---

## Projede Uygulanan Adımlar

- Veri setinin yüklenmesi
- Veri setinin incelenmesi
- Eksik veri kontrolü
- Özellikler (Feature) ve hedef değişkenin (Target) ayrılması
- Train-Test Split işlemi
- StandardScaler ile Feature Scaling uygulanması
- Logistic Regression modelinin eğitilmesi
- Random Forest modelinin eğitilmesi
- Accuracy Score hesaplanması
- Confusion Matrix oluşturulması
- Modellerin karşılaştırılması

---

## Kullanılan Modeller

### Logistic Regression

- Model eğitildi.
- Test verisi üzerinde tahmin yapıldı.
- Accuracy Score hesaplandı.
- Confusion Matrix oluşturuldu.

### Random Forest

- Model eğitildi.
- Test verisi üzerinde tahmin yapıldı.
- Accuracy Score hesaplandı.
- Confusion Matrix oluşturuldu.

---

## Sonuç

Her iki model de veri seti üzerinde yüksek doğruluk oranı elde etmiştir.

Yapılan karşılaştırma sonucunda **Logistic Regression** modeli, bu veri setinde daha yüksek doğruluk oranı göstererek en başarılı model olmuştur.

---

## Proje Yapısı

```
i2i-Academy-AppliedDataScience-1
│
├── main.py
└── README.md
```

---

## Projeyi Çalıştırma

### Gerekli kütüphaneleri yükleyin

```bash
pip install pandas scikit-learn
```

### Programı çalıştırın

```bash
python main.py
```

---

## Geliştirici

**Gamze Canpolat**

Bilgisayar Mühendisliği Öğrencisi
