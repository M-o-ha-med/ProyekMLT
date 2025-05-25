# Laporan Proyek Machine Learning - Mohamed

## Domain Proyek

Dalam dunia medis, pemantauan kesehatan janin selama kehamilan merupakan aspek penting untuk mencegah komplikasi yang dapat membahayakan ibu maupun janin. Salah satu metode pemantauan yang umum dilakukan adalah melalui kardiotokografi (CTG), yang merekam denyut jantung janin dan kontraksi rahim. Namun, interpretasi data CTG secara manual oleh tenaga medis bisa sangat subjektif dan bervariasi. Oleh karena itu, penerapan machine learning dalam klasifikasi kesehatan janin dapat membantu meningkatkan akurasi diagnosis dan efisiensi kerja dokter.

Menurut World Health Organization (WHO), sekitar 2,4 juta bayi meninggal pada bulan pertama kehidupan setiap tahunnya, banyak di antaranya dapat dicegah dengan pemantauan dan penanganan kehamilan yang lebih baik \[1]. Model klasifikasi otomatis berbasis machine learning dapat membantu mengidentifikasi kondisi janin sebagai "normal", "suspect", atau "pathological", sehingga dokter dapat mengambil tindakan yang tepat lebih cepat.

Referensi:
\[1] World Health Organization. (2020). "Newborns: reducing mortality." [https://www.who.int/news-room/fact-sheets/detail/newborns-reducing-mortality](https://www.who.int/news-room/fact-sheets/detail/newborns-reducing-mortality)

## Business Understanding

### Problem Statements

* Bagaimana mengklasifikasikan kondisi kesehatan janin berdasarkan data hasil CTG?
* Apakah model klasifikasi dapat mengidentifikasi kasus janin yang "suspect" atau "pathological" secara akurat dan konsisten?

### Goals

* Mengembangkan model machine learning yang mampu memprediksi label "fetal\_health" dengan akurasi tinggi.
* Meningkatkan recall dan precision untuk kelas minoritas (suspect dan pathological) agar tidak terjadi false negative yang berbahaya.

### Solution Statements

* Membangun model baseline menggunakan algoritma seperti Logistic Regression dan Random Forest.
* Melakukan data preprocessing termasuk scaling dan balancing menggunakan SMOTE.
* Melakukan hyperparameter tuning untuk Random Forest dan membandingkan hasilnya dengan XGBoost dan SVM.

## Data Understanding

Dataset yang digunakan adalah [Fetal Health Classification Dataset](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification) dari Kaggle.

Dataset ini berisi 2126 sampel dengan 21 fitur numerik hasil kardiotokografi dan satu target label bernama "fetal\_health" yang berisi tiga kelas: 1.0 (normal), 2.0 (suspect), dan 3.0 (pathological).

Contoh variabel:

* baseline value: nilai detak jantung dasar janin
* accelerations: jumlah akselerasi detak jantung
* fetal\_movement: frekuensi gerakan janin
* uterine\_contractions: kekuatan kontraksi uterus
* abnormal\_short\_term\_variability: variasi denyut jantung abnormal jangka pendek

EDA dilakukan untuk memahami distribusi data dan menemukan adanya outlier pada beberapa fitur numerik. Boxplot dan pairplot digunakan untuk eksplorasi visual.

## Data Preparation

Beberapa tahapan yang dilakukan:

1. Menghapus outlier dengan metode IQR (beberapa iterasi diperlukan karena penghapusan outlier mengubah distribusi).
2. Memisahkan data fitur (X) dan target (y).
3. Melakukan train-test split dengan stratifikasi label.
4. Melakukan scaling dengan MinMaxScaler untuk menjaga jarak antar fitur.
5. Menyeimbangkan data pelatihan menggunakan SMOTE untuk menangani class imbalance.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
smote = SMOTE()
X_train, y_train = smote.fit_resample(X_train, y_train)
```

## Modeling

Tiga model diuji:

1. Logistic Regression
2. Random Forest
3. XGBoost

Random Forest menunjukkan hasil terbaik setelah tuning dengan parameter berikut:

```python
RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
```

Kelebihan Random Forest:

* Robust terhadap outlier
* Dapat menangani data imbalanced dengan baik

XGBoost digunakan sebagai pembanding, dan meskipun akurat, memerlukan tuning lebih kompleks.

## Evaluation

Metrik evaluasi yang digunakan:

* **Accuracy**: proporsi prediksi yang benar
* **Precision**: kemampuan model menghindari false positive
* **Recall**: kemampuan model menangkap semua true positive
* **F1-Score**: harmonic mean dari precision dan recall

```text
Accuracy: 0.944794952681388
Classification Report:
              precision    recall  f1-score   support

        1.0       0.97      0.97      0.97       494
        2.0       0.81      0.78      0.80        88
        3.0       0.96      0.96      0.96        52
```

Hasil menunjukkan bahwa model mampu mengenali semua kelas dengan baik. Terutama recall pada kelas 2.0 dan 3.0 meningkat berkat balancing menggunakan SMOTE.

---

*Catatan tambahan:*

* Outlier dihapus hanya dari fitur numerik, bukan label.
* Label tidak ikut di-scale karena merupakan target klasifikasi.
* Visualisasi confusion matrix dan ROC-AUC juga bisa ditambahkan untuk evaluasi lebih mendalam.
