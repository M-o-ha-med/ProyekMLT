# **Laporan Proyek Machine Learning: Klasifikasi Kesehatan Janin dari Data CTG - Mohamed**


---

## **1. Domain Proyek**

### **Latar Belakang**

Pemantauan kondisi janin selama kehamilan sangat krusial untuk mencegah komplikasi serius, baik bagi ibu maupun bayi. Salah satu metode pemantauan yang paling umum digunakan dalam dunia medis adalah **kardiotokografi (CTG)**, yaitu teknik yang merekam detak jantung janin dan kontraksi uterus.

Namun, interpretasi CTG masih mengandalkan pengamatan manual oleh tenaga medis, yang bisa bersifat subjektif dan bervariasi antar praktisi. Menurut data dari **World Health Organization (WHO)**, sekitar 2,4 juta bayi meninggal pada bulan pertama kehidupannya setiap tahun, dan banyak dari kematian ini dapat dicegah dengan deteksi dini dan perawatan yang tepat \[1].

Oleh karena itu, dibutuhkan sistem klasifikasi otomatis berbasis **machine learning (ML)** untuk membantu dokter dalam menginterpretasi data CTG secara objektif, cepat, dan akurat. Beberapa penelitian sebelumnya, seperti oleh Mehbodniya et al. (2022) dan Salini et al. (2024), menunjukkan bahwa model machine learning dapat meningkatkan akurasi klasifikasi kesehatan janin secara signifikan \[2]\[3].

---

### **Referensi**

\[1] World Health Organization. (2020). *Newborns: reducing mortality*. [https://www.who.int/news-room/fact-sheets/detail/newborns-reducing-mortality](https://www.who.int/news-room/fact-sheets/detail/newborns-reducing-mortality)
\[2] Mehbodniya, A. et al. (2022). *Fetal health classification from cardiotocographic data using machine learning*. Expert Systems, 39(6), e12899.
\[3] Salini, Y. et al. (2024). *Cardiotocography data analysis for fetal health classification using machine learning models*. IEEE Access, 12, 26005–26022.

---

## **2. Business Understanding**

### **Problem Statements**

* Bagaimana mengklasifikasikan kondisi kesehatan janin berdasarkan data CTG menjadi “normal”, “suspect”, dan “pathological”?
* Apakah model machine learning mampu mengenali janin dalam kondisi “suspect” dan “pathological” dengan akurasi dan konsistensi yang tinggi?

### **Goals**

* Membangun model machine learning yang mampu memprediksi label `fetal_health` dengan akurasi tinggi.
* Mengidentifikasi model terbaik yang dapat digunakan sebagai alat bantu diagnosis kondisi janin.

### **Solution Statements**

Untuk mencapai tujuan, solusi yang diterapkan meliputi:

1. **Penerapan Multi-Model**:
   Membangun dan membandingkan performa dari **4 algoritma**:

   * Support Vector Machine (SVM)
   * K-Nearest Neighbors (KNN)
   * Random Forest
   * Gradient Boosting

2. **Evaluasi Terukur**:
   Setiap model dievaluasi berdasarkan metrik:

   * **Accuracy**
   * **Precision**
   * **Recall**
   * **F1-Score**

3. **Balancing & Preprocessing**:
   Mengatasi class imbalance dengan **SMOTE**, serta melakukan **scaling dan outlier removal** agar model lebih robust.

---

## **3. Data Understanding**

### **Sumber Data**

Dataset: [Fetal Health Classification Dataset – Kaggle](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification)
Jumlah data: **2.126 sampel**
Jumlah fitur: **21 fitur numerik + 1 target**

### **Deskripsi Target**

* **fetal\_health**:

  * `1.0` = Normal
  * `2.0` = Suspect
  * `3.0` = Pathological

### **Daftar Fitur**

* `baseline_value`, `accelerations`, `fetal_movement`, `uterine_contractions`
* `light_decelerations`, `severe_decelerations`, `prolongued_decelerations`
* `abnormal_short_term_variability`, `mean_value_of_short_term_variability`
* `percentage_of_time_with_abnormal_long_term_variability`, `mean_value_of_long_term_variability`
* `histogram_width`, `histogram_min`, `histogram_max`, `histogram_number_of_peaks`
* `histogram_number_of_zeroes`, `histogram_mode`, `histogram_mean`, `histogram_median`
* `histogram_variance`, `histogram_tendency`

### **EDA (Exploratory Data Analysis)**

* Visualisasi distribusi tiap fitur untuk deteksi outlier.
* Korelasi antar fitur untuk menghindari multikolinearitas.
* Distribusi kelas target menunjukkan ketidakseimbangan kelas.

![Visualisasi Pie Chart](assets/Bar_chart_Fetal_Health.png)

---

## **4. Data Preparation**

Data independen dan dependen atau label dipisahkan terlebih dahulu, dimana variabel X memuat data independen sedangkan variabel y memuat data dependen atau label.
```python
X = df.drop(['fetal_health'],axis = 1)
y = df['fetal_health']
```

Selanjutnya dilakukan splitting data menjadi data latih  dan data tes, dengan perbandingan 80 : 20 , 80 % untuk data latih dan 20 % untuk data tes. dengan menambahkan argumen stratify pada argumen fungsi supaya menjaga distribusi dari kelas.
```python
X_train , X_test , y_train , y_test = train_test_split(X,y , test_size = 0.2 , stratify=y , random_state = 42)
```

Setelah dilakukan data splitting, dilakukan proses normalisasi pada data latih guna mempermudah model dalam melakukan prekdisi data.

```python
Scaler = MinMaxScaler()
X_train = Scaler.fit_transform(X_train)
X_test = Scaler.transform(X_test)
```

Terakhir, dilakukan augmentasi data pada data latih menggunakan SMOTE untuk bisa menyeimbangkan data pada masing-masing kelas terutama pada data dengan label 2.0 dan 3.0
```python
smote = SMOTE()
X_train_resampled , y_train_resampled = smote.fit_resample(X_train , y_train)
```

## **5. Modeling**

## 1. K-Nearest Neighbors (KNN)

### Deskripsi:
KNN adalah algoritma klasifikasi (atau regresi) berbasis instance-based learning. Saat memprediksi, KNN mencari `k` tetangga terdekat dari data uji dan menggunakan mayoritas kelas dari tetangga tersebut.

### Rumus (jarak Euclidean):
$$d(x, x_i) = \sqrt{\sum_{j=1}^{n} (x_j - x_{ij})^2}$$

**Keterangan:**
- $$d(x, x_i)$$: Jarak Euclidean antara titik uji $$x$$ dan data latih ke-$$i$$
- $$x$$: Vektor fitur dari data uji  
- $$x_i$$: Vektor fitur dari data latih ke-$$i$$  
- $$x_j$$: Nilai fitur ke-$$j$$ pada data uji  
- $$x_{ij}$$: Nilai fitur ke-$$j$$ pada data latih ke-$$i$$  
- $$n$$: Jumlah fitur (dimensi data)

### Parameter:
- `n_neighbors=5`: jumlah tetangga yang dipertimbangkan  
- `weights='uniform'`: semua tetangga memiliki bobot yang sama  
- `metric='minkowski'`, `p=2`: menggunakan jarak Euclidean  

### Kelebihan:
- Sederhana dan mudah dipahami  
- Cocok untuk dataset kecil  
- Tidak memerlukan pelatihan (lazy learning)  

### Kekurangan:
- Sensitif terhadap outlier  
- Lambat pada dataset besar  
- Sensitif terhadap skala fitur (butuh normalisasi)  

---

## 2. Random Forest

### Deskripsi:
Random Forest adalah metode ensemble berbasis pohon keputusan. Algoritma ini membangun banyak decision tree secara acak, lalu menggabungkan hasilnya untuk prediksi (voting atau rata-rata).

### Rumus (voting mayoritas):
$$\hat{y} = \text{mode}(T_1(x), T_2(x), \ldots, T_k(x))$$

**Keterangan:**
- $$T_i(x)$$: prediksi dari decision tree ke-\( i \)  
- $$\hat{y}$$ : hasil prediksi akhir  

### Parameter:
- `n_estimators=100`: jumlah pohon  
- `max_depth=None`: kedalaman maksimum pohon  
- `bootstrap=True`: sampling dengan pengembalian  
- `random_state=42`: agar hasil bisa direproduksi  

### Kelebihan:
- Tahan terhadap overfitting  
- Menangani data tidak seimbang dengan baik  
- Memberikan informasi pentingnya fitur  
- Bekerja baik pada data numerik maupun kategorikal  

### Kekurangan:
- Lambat untuk pelatihan data besar  
- Tidak sebaik gradient boosting untuk prediksi presisi tinggi  
- Agak sulit untuk interpretasi individual tree  

---

## 3. Support Vector Machine (SVM)

### Deskripsi:
SVM mencari hyperplane optimal yang memisahkan kelas dengan margin terbesar. Cocok untuk data berdimensi tinggi dan linear/non-linear tergantung kernel yang digunakan.

### Rumus (untuk SVM linear):
$$\min_{w,b} \frac{1}{2} \|w\|^2 \quad \text{dengan syarat: } y_i(w \cdot x_i + b) \geq 1$$

**Keterangan:**
- $$w$$: vektor bobot  
- $$b$$: bias  
- $$x_i$$: data ke-$$i$$  
- $$y_i$$: label data ke-$$i$$  
- Margin = $$\frac{2}{\|w\|}$$

### Parameter:
- `C=1.0`: parameter regulasi  
- `kernel='rbf'`: kernel radial basis function  
- `gamma='scale'`: parameter skala untuk kernel  

### Kelebihan:
- Akurat untuk data berdimensi tinggi  
- Bisa menangani data non-linear dengan kernel  
- Efektif jika data punya margin pemisahan yang jelas  

### Kekurangan:
- Lambat untuk dataset besar  
- Boros memori  
- Parameter kernel dan regulasi harus disetel dengan hati-hati  

---

## 4. Gradient Boosting

### Deskripsi:
Gradient Boosting adalah metode ensemble yang membangun model secara bertahap, di mana setiap model baru mengoreksi kesalahan dari model sebelumnya.

### Rumus (prediksi iteratif):
$$F_{m}(x) = F_{m-1}(x) + \gamma_m h_m(x)$$

**Keterangan:**
- $$F_{m}(x)$$: model pada iterasi ke-$$m$$ 
- $$h_m(x)$$: decision tree kecil (weak learner) ke-$$m$$
- $$\gamma_m$$: learning rate  
- Model meminimalkan loss: $$\mathcal{L}(y, F(x))$$

### Parameter:
- `n_estimators=100`: jumlah boosting stage  
- `learning_rate=0.1`: ukuran langkah koreksi tiap model  
- `max_depth=3`: kedalaman pohon  

### Kelebihan:
- Akurasi tinggi di banyak kompetisi dan masalah real-world  
- Bisa menangani data kategorikal dan numerik  
- Bisa dikontrol dengan tuning (learning rate, max depth)  

### Kekurangan:
- Overfitting jika tidak dikontrol  
- Pelatihan lambat  
- Sensitif terhadap parameter  


---

## **6. Evaluation**

### **Metrik Evaluasi**

1. **Accuracy**
   $$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

2. **Precision**
   $$Precision = \frac{TP}{TP + FP}$$

3. **Recall**
   $$Recall = \frac{TP}{TP + FN}$$

4. **F1-Score**
   $$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

### **Hasil Evaluasi**

| Model             | Accuracy | Precision | Recall   | F1-Score |
| ----------------- | -------- | --------- | -------- | -------- |
| SVM               | 0.88     | 0.91      | 0.88     | 0.89     |
| KNN               | 0.88     | 0.90      | 0.88     | 0.88     |
| Random Forest     | 0.86     | 0.87      | 0.86     | 0.86     |
| Gradient Boosting | **0.93** | **0.93**  | **0.93** | **0.93** |

### **Model Terbaik**

Model terbaik adalah **Gradient Boosting**, karena memiliki skor evaluasi tertinggi secara konsisten di semua metrik. Model ini dipilih karena:

* Performa prediksi terbaik
* Kemampuan menangani kompleksitas non-linear
* Meningkatkan recall pada kelas minoritas berkat SMOTE

---

## **Kesimpulan**

Dengan menerapkan serangkaian preprocessing, balancing, dan evaluasi metrik yang tepat, model Gradient Boosting terbukti paling akurat dalam mengklasifikasikan kesehatan janin berdasarkan data CTG. Model ini dapat digunakan sebagai sistem pendukung keputusan untuk meningkatkan diagnosis dini dalam dunia medis, khususnya dalam bidang obstetri.

---
