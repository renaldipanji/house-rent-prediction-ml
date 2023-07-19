# Proyek Pertama MLT (House Rent Prediction)

**Dibuat oleh : Renaldi Panji Wibowo**

Proyek ini adalah proyek pertama _predictive analytics_ untuk memenuhi submission MLT Dicoding. Proyek ini membangun model __machine learning__ yang dapat memprediksi harga sewa rumah dan apartemen.

## Domain Proyek

### **Latar Belakang**
Tempat tinggal, seperti rumah atau apartemen, adalah kebutuhan utama manusia untuk melindungi dan menetap. Nilai tempat tinggal ditentukan oleh berbagai faktor, seperti lokasi, ukuran, jumlah kamar, jumlah kamar mandi, perabotan, dan fitur lainnya [1].

![dataset-cover (4)](https://github.com/renaldipanji/house-rent-prediction-ml/assets/75974146/94737eba-73a2-4c3b-888e-2c6c599bb150)

[Referensi Gambar](https://storage.googleapis.com/kaggle-datasets-images/2644747/4525552/2b4663622a47936b6984513d9e377964/dataset-cover.png?t=2022-11-16-19-36-24)

Harga rumah mencerminkan nilai yang dimiliki oleh properti tersebut. Namun, harga rumah tidak selalu dapat diprediksi dengan akurat secara manual. Oleh karena itu, perusahaan penyewaan perlu mengurangi ketidakpastian dengan membangun sistem prediksi menggunakan _machine learning_ [3]. Tujuan dari sistem ini adalah dapat memperkirakan harga sewa yang wajar untuk setiap karakteristik rumah.

Melalui penelitian ini, diharapkan model _machine learning_ mampu memprediksi harga sewa rumah yang sesuai dengan harga pasar. Prediksi ini kemudian akan menjadi panduan bagi perusahaan dalam menentukan harga sewa yang dapat menghasilkan keuntungan [2].

Dengan menggunakan model prediksi harga sewa, perusahaan penyewaan properti dapat mengoptimalkan keputusan bisnis mereka. Berikut beberapa contoh skenario penggunaan model ini oleh perusahaan penyewaan properti:

1. Penentuan Harga Sewa yang Kompetitif: Model _machine learning_ dapat membantu perusahaan dalam menentukan harga sewa yang kompetitif berdasarkan karakteristik rumah. Dengan memperhitungkan faktor-faktor seperti lokasi, ukuran, dan fitur-fitur lainnya, perusahaan dapat menentukan harga yang sesuai dengan nilai pasar dan mengoptimalkan pendapatan mereka.
2. Identifikasi Potensi Profitabilitas: Dengan menggunakan model prediksi, perusahaan dapat mengidentifikasi properti-properti dengan potensi profitabilitas yang tinggi. Model dapat membantu mengidentifikasi karakteristik yang berkontribusi terhadap harga sewa yang lebih tinggi, seperti lokasi yang strategis atau fitur-fitur yang dicari oleh penyewa. Hal ini memungkinkan perusahaan untuk fokus pada properti-properti yang memiliki potensi pendapatan yang lebih tinggi.
3. Pengambilan Keputusan Investasi: Model prediksi harga sewa dapat digunakan sebagai alat bantu dalam pengambilan keputusan investasi oleh perusahaan penyewaan properti. Dengan memprediksi harga sewa di masa depan berdasarkan karakteristik rumah, perusahaan dapat mengevaluasi potensi pengembalian investasi pada properti baru atau pengambilan keputusan terkait pembelian atau penjualan properti.
4. Penyesuaian Strategi Pemasaran: Model prediksi dapat memberikan wawasan tentang faktor-faktor yang paling berpengaruh terhadap harga sewa. Perusahaan dapat menggunakan informasi ini untuk menyusun strategi pemasaran yang lebih efektif, seperti menyoroti fitur-fitur yang paling dicari oleh penyewa atau menyesuaikan kampanye pemasaran dengan preferensi target pasar.

Dengan menggunakan model _machine learning_ untuk memprediksi harga sewa, perusahaan penyewaan properti dapat memperoleh keuntungan yang signifikan, termasuk penentuan harga yang optimal, identifikasi properti yang potensial menghasilkan profit, pengambilan keputusan investasi yang lebih baik, dan penyesuaian strategi pemasaran yang efektif.

## **_Business Understanding_**
Proyek ini ditujukan untuk perusahaan yang memiliki model bisnis sebagai berikut:

* Perusahaan memiliki atau mengakuisisi properti rumah dan apartemen, dan kemudian menyewakannya kepada konsumen.

* Perusahaan menyediakan layanan konsultasi harga sewa rumah dan apartemen kepada konsumen

Model _machine learning_ yang dikembangkan dalam proyek ini dapat memberikan banyak manfaat bagi perusahaan dalam menentukan harga sewa yang lebih optimal dan berkontribusi pada peningkatan keuntungan. Berikut beberapa contoh bagaimana model ini dapat membantu perusahaan:

1. Optimasi Harga Sewa: Model prediksi harga sewa membantu perusahaan mengoptimalkan harga sewa untuk setiap properti berdasarkan karakteristiknya. Misalnya, properti dengan lokasi strategis dan fitur premium diberi harga lebih tinggi, sedangkan properti dengan karakteristik sederhana diberi harga lebih rendah. Dengan demikian, perusahaan dapat memaksimalkan pendapatan dan menghindari kesalahan penentuan harga.

2. Identifikasi Potensi Profitabilitas: Model prediksi harga sewa membantu perusahaan mengidentifikasi properti dengan potensi profitabilitas tinggi. Misalnya, properti dengan luas yang lebih besar, jumlah kamar yang banyak, dan fitur-fitur premium memiliki potensi untuk menghasilkan harga sewa yang lebih tinggi. Perusahaan dapat fokus pada pengembangan atau akuisisi properti dengan karakteristik tersebut untuk meningkatkan profitabilitas mereka.

3. Model prediksi harga sewa membantu perusahaan dalam menyesuaikan strategi pemasaran mereka. Misalnya, jika lokasi dan keberadaan fasilitas umum berpengaruh signifikan terhadap harga sewa, perusahaan dapat menekankan keunggulan lokasi tersebut dalam kampanye pemasaran. Hal ini akan menarik penyewa yang mencari properti dengan akses mudah ke fasilitas tersebut, meningkatkan tingkat okupansi, dan meningkatkan keuntungan perusahaan.

Dengan menggunakan model _machine learning_ untuk memprediksi harga sewa, perusahaan dapat mengambil keputusan yang lebih cerdas dan efektif dalam menentukan harga sewa, mengelola portofolio properti, dan merancang strategi pemasaran. Hal ini akan berkontribusi pada peningkatan keuntungan perusahaan dan membantu mereka mempertahankan posisi yang kompetitif di pasar penyewaan properti.

### **_Problem Statements_**

1. Apa parameter yang paling signifikan dalam menentukan harga sewa rumah atau apartemen di India ?
 
2. Bagaimana cara yang efektif untuk memproses data agar dapat digunakan untuk melatih model dengan baik ?

3. Berapa perkiraan harga sewa rumah di India berdasarkan karakteristik tertentu ?

### **_Goals_**

1. Mengetahui parameter apa yang paling berpengaruh dalam menentukan harga sewa rumah atau apartemen di India.

2. Melakukan persiapan data untuk dapat dilatih oleh model.

3. Membuat model _machine learning_ yang dapat memprediksi harga sewa rumah di India seakurat mungkin berdasarkan karakteristik tertentu.

### **Solution Statement**

1. Menganalisis data dengan melakukan _univariate analysis_ dan _multivariate analysis_. Memahami data juga dapat dilakukan dengan visualisasi. Memahami data dapat membantu untuk mengetahui korelasi antar fitur dan mendeteksi outlier.
2. Menyiapkan data agar bisa digunakan dalam membangun model.
3. Melakukan hyperparameter tuning menggunakan grid search dan membangun model regresi yang dapat memprediksi bilangan kontinu. Algoritma yang dipakai dalam proyek ini adalah K-Nearest Neighbour, Random Forest, dan AdaBoost.

## **Data Understanding & Removing Outlier**
Dataset yang digunakan dalam proyek ini merupakan data harga sewa rumah dengan berbagai karakteristik di India. Dataset ini dapat diunduh di Kaggle : [House Rent Prediction Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset)

Berikut informasi yang ada pada dataset :
* Dataset memiliki format CSV.
* Dataset memiliki 4746 sampel dengan 12 fitur.
* Dataset memiliki 4 fitur bertipe int64 dan 8 fitur bertipe object.
* Tidak ada missing value dalam dataset.

### **Variabel-variabel pada dataset**
* Posted On: Tanggal data diposting.
* BHK: Jumlah dari kamar tidur, aula, dan dapur.
* Rent: Harga sewa dari rumah/apartemen.
* Size: Luas dari rumah/apartemen dalam square feet (sqft).
* Floor: Letak dan jumlah lantai rumah/apartemen.
* Area Type: Ukuran dari rumah dalam kategori Super Area atau Carpet Area atau Build Area.
* Area Locality: Lokasi rumah/apartemen.
* City: Kota dimana rumah/apartemen berada.
* Furnishing Status: Status perabotan rumah/apartemen, baik Furnished atau Semi-Furnished atau Unfurnished.
* Tenant Preferred: Jenis penyewa yang diutamakan pemilik atau agen.
* Bathroom: Jumlah kamar mandi.
* Point of Contact: Kontak yang dihubungi untuk informasi lebih lanjut mengenai rumah/apartemen.

Fitur Point of Contract dan Posted On tidak digunakan karena tidak mempengaruhi harga sewa rumah sehingga akan dihapus. Hal tersebut dikarenakan kedua fitur tersebut tidak diperlukan dalam membangun model prediksi harga sewa.

## **_Univariate Analysis_**
_Univariate Analysis_ adalah menganalisis masing-masing fitur.

### **Analisis jumlah nilai unique pada setiap fitur kategorik**
  
Pada fitur area type terdapat sample yang tidak merata sebagai berikut :

* Area Type
  
  ![Area Type](https://github.com/renaldipanji/house-rent-prediction-ml/assets/75974146/987d7684-d7bb-45e9-983d-3d160efbcb5b)

  **Gambar 1. Jumlah nilai unik pada fitur area type**

  Hanya ada 2 data dari Built Area pada fitur Area Type. Untuk menghindari data dengan dimensi tinggi (high dimensional data), kedua data ini akan dihapus.
**Tabel 1. Jumlah nilai unik dari masing-masing fitur**

| Fitur | Jumlah Nilai Unik |
|----------|-------------:|
| Area Type |  3 |
| City |    6   |
| Furnishing Status |    3   |
| Floor |    480   |
| Area Localicity |    2234   |
  
  Pada tabel 1 fitur Floor dan Area Locality memiliki banyak nilai unik yang menyebabkan dimensi data menjadi tinggi. Untuk menghindari data dengan dimensi tinggi (high dimensional data), kedua fitur ini akan dihapus.

### **Analisis sebaran pada setiap fitur numerik**

![visualisasi data fitur numerik](https://github.com/renaldipanji/house-rent-prediction-ml/assets/75974146/cfb5b474-bf5d-4eb9-8465-a9f4bdfa46ff)

**Gambar 2. Visualisasi Sebaran data pada fitur numerik**

Berikut analisis dari grafik di atas :
* Sebagian besar rumah memiliki 1 sampai 3 BHK dan 1 sampai 3 kamar mandi.
* Sebagian besar rumah memiliki luas di bawah 2000 sqft.
* Rentang harga sewa cukup tinggi, yaitu dari 1200 hingga 3500000. Namun, rata-rata harga rumah hanya 35003. Distribusi harga yang kurang bagus seperti ini dapat berimplikasi pada model.

## **_Multivariate Analysis_**

_Multivariate Analysis_ menunjukkan hubungan antara dua atau lebih fitur dalam data.

### **Analisis fitur numerik**

* Fitur Size dan BHK akan dianalisis, dan dilakukan penghapusan outlier pada fitur BHK. Hal ini karena tidak biasa bagi rumah dengan 1 BHK memiliki luas 100 sqft. Kami akan menetapkan batas treshold atau batas 300 sqft/BHK. Data yang berada di bawah batas ini akan dihapus. Sebagai hasilnya, akan ada pengurangan jumlah sampel sebesar 548.

* Fitur Size dan Rent (Menghapus Price per sqft Outlier) Untuk memudahkan dalam mendeteksi outlier, maka dibuat fitur baru 'Price_per_sqft' dari kedua fitur tersebut untuk menganalisis harga sewa per luas sqft.

  **Tabel 2. Statistik deskriptif fitur per sqft**

  | count | 4196.000000 |
  | ----------- | :---------: |
  | mean | 32827.385605 |
  | std | 41300.048982 |
  | min | 571.428571 |
  | 25% | 13000.000000 |
  | 50% | 18511.595708 |
  | 75% | 34896.788991 |
  | max | 1400000.000000 |
  
  Dari tabel 2, terlihat bahwa harga 571 per sqft sangat rendah dan harga 1400000 per sqft sangat tinggi. Oleh karena itu, dilakukan penghapusan outlier pada harga per sqft menggunakan metode mean dan satu standar deviasi, yang dikelompokkan berdasarkan kota. Hal ini menyebabkan pengurangan jumlah sampel sebesar 497.

* Fitur Bathroom dan BHK akan dianalisis, dan dilakukan penghapusan outlier pada fitur Bathroom. Hal ini karena tidak biasa bagi rumah dengan 2 BHK memiliki 4 kamar mandi. Kami akan menetapkan batas bahwa jumlah kamar mandi tidak boleh melebihi jumlah BHK + 2. Sebagai hasilnya, akan ada pengurangan jumlah sampel sebesar 3.

* Melihat kolerasi antara semua fitur numerik
  ![Korelasi Fitur Numerik](https://github.com/renaldipanji/house-rent-prediction-ml/assets/75974146/19374161-877e-46f2-9980-a45d4f97854d)

  **Gambar 3. Korelasi fitur numerik**

  Dari analisis yang dilakukan, terlihat bahwa fitur BHK, Size, dan Bathroom tidak memiliki korelasi yang signifikan dengan fitur target (Rent). Hal ini mungkin disebabkan oleh kurangnya data dalam penelitian ini. Namun, fitur BHK dan Bathroom tetap memiliki korelasi yang signifikan dengan fitur Size. Hal ini sesuai dengan harapan setelah dilakukannya penghapusan outlier sebelumnya.

### **Analisis fitur kategorik**
Analisis ini dilakukan untuk melihat kolerasi antara fitur kategorik dengan fitur target (Rent).

* Fitur Area Type

  ![Fitur Area Type](https://github.com/renaldipanji/house-rent-prediction-ml/assets/75974146/b065ece8-7c63-4fe1-92cd-cd67d0bbbf08)

  **Gambar 4. Korelasi fitur area type**
  
  Fitur Area Type memiliki pengaruh yang kecil terhadap rata-rata harga sewa.
  
* Fitur City
  
  ![Fitur City](https://github.com/renaldipanji/house-rent-prediction-ml/assets/75974146/129cee6f-6f55-489f-a5ce-2805c451ec53)

  **Gambar 5. Korelasi fitur city**
  
  Dari analisis tersebut, terlihat bahwa fitur City memiliki pengaruh yang cukup besar terhadap rata-rata harga sewa, terutama jika rumah berada di kota Mumbai. Faktanya, sebaran harga rumah mencapai tingkat tertinggi di kota Mumbai. Mumbai memang dikenal sebagai kota yang memiliki biaya hidup yang tinggi di India, diikuti oleh Delhi. Ini menunjukkan bahwa faktor lokasi, terutama kota tempat rumah berada, memiliki dampak signifikan pada harga sewa rumah.

* Fitur Furnishing Status
   
  ![Fitur Furnishing Status](https://github.com/renaldipanji/house-rent-prediction-ml/assets/75974146/14c09152-12e8-4bd9-84f4-3b30f4de752f)

  **Gambar 6. Korelasi fitur furnishing status**
  
  Dari analisis yang dilakukan, terlihat bahwa fitur Furnishing Status memiliki pengaruh yang cukup besar terhadap rata-rata harga sewa. Hal ini wajar, karena rumah yang dilengkapi dengan perabotan lengkap umumnya akan diberi harga sewa yang lebih tinggi daripada rumah yang tidak dilengkapi perabotan. Furnishing Status menjadi salah satu faktor penting yang memengaruhi penentuan harga sewa rumah, karena keberadaan perabotan dapat memberikan nilai tambah dan kenyamanan bagi penyewa.

* Tenant Preferred

  ![Fitur Tenant](https://github.com/renaldipanji/house-rent-prediction-ml/assets/75974146/f4616bf4-cfa5-4005-a627-02068d5d49b2)

  **Gambar 7. Korelasi fitur tenant preferred**
  
  Dari analisis yang dilakukan, terlihat bahwa fitur Tenant Preferred memiliki pengaruh yang cukup signifikan terhadap rata-rata harga sewa. Dari grafik yang ditampilkan, terlihat bahwa rumah yang sangat disarankan untuk disewa oleh keluarga memiliki rata-rata harga sewa yang lebih tinggi dibandingkan dengan preferensi penyewa lainnya. Hal ini dapat disebabkan oleh faktor-faktor seperti ukuran rumah yang lebih besar, kebutuhan ruang yang lebih luas, atau fasilitas tambahan yang cocok untuk kebutuhan keluarga. Oleh karena itu, Tenant Preferred menjadi faktor penting yang memengaruhi penentuan harga sewa rumah.


## **Data preparation**

* One Hot Encoding
  one-hot encoding adalah teknik yang digunakan untuk mengubah data kategorikal menjadi data numerik dengan menciptakan kolom baru untuk setiap kategori yang ada dan mengisinya dengan nilai 0 atau 1.

  Dalam proyek ini, fitur-fitur yang akan diubah menjadi numerik menggunakan one-hot encoding adalah:

  * Area Type: Fitur ini akan dipecah menjadi kolom baru berdasarkan jenis-jenis Area Type yang ada, seperti Super Area, Carpet Area, dan Build Area.
  * City: Fitur ini akan dipecah menjadi kolom baru berdasarkan nama-nama kota yang ada, seperti Mumbai, Delhi, dan sebagainya.
  * Furnishing Status: Fitur ini akan dipecah menjadi kolom baru berdasarkan status perabotan, seperti Furnished, Semi-Furnished, dan   Unfurnished.
  * Tenant Preferred: Fitur ini akan dipecah menjadi kolom baru berdasarkan jenis-jenis preferensi penyewa yang ada, seperti Family, Bachelor, dan sebagainya.

  Dengan menggunakan one-hot encoding, fitur-fitur ini akan diubah menjadi data numerik yang dapat digunakan oleh model _machine learning_ untuk proses pelatihan dan prediksi.

* Train Test Split
  Train-test split adalah proses membagi data menjadi data latih (train data) dan data uji (test data). Data latih digunakan untuk melatih model, sementara data uji digunakan untuk menguji kinerja model yang telah dilatih.

  Pada proyek ini, dataset yang memiliki ukuran 3696 akan dibagi menjadi 3326 untuk data latih dan 370 untuk data uji. Dalam hal ini, sekitar 90% dari data akan digunakan sebagai data latih, sedangkan sekitar 10% akan digunakan sebagai data uji. Pembagian ini dapat dilakukan dengan menggunakan teknik random sampling untuk memastikan representativitas data dalam kedua subset.

  Data latih akan digunakan untuk melatih model _machine learning_ dan menyesuaikan parameter model. Setelah model dilatih, data uji akan digunakan untuk menguji performa dan mengukur akurasi model yang dihasilkan.

  Dengan membagi data menjadi data latih dan data uji, ini memungkinkan evaluasi yang obyektif terhadap kinerja model di luar data yang digunakan untuk melatihnya, sehingga memberikan perkiraan yang lebih baik tentang bagaimana model tersebut akan berperforma pada data baru yang belum pernah dilihat sebelumnya.

* Normalization
  normalisasi data dapat membantu meningkatkan performa dan kecepatan algoritma _machine learning_, terutama ketika data memiliki skala yang berbeda-beda. Salah satu teknik normalisasi yang umum digunakan adalah Standarisasi, yang dapat dilakukan dengan menggunakan kelas StandardScaler dari modul sklearn.preprocessing.

  Pada proyek ini, Standarisasi dengan StandardScaler digunakan untuk memastikan bahwa skala data pada fitur-fitur yang digunakan dalam model _machine learning_ adalah relatif sama. Dengan menggunakan StandardScaler, fitur-fitur akan diubah sedemikian rupa sehingga memiliki mean 0 dan variansi 1. Hal ini membantu algoritma _machine learning_ dalam menginterpretasikan dan memproses data dengan lebih baik, serta menghindari dominasi fitur-fitur dengan skala yang lebih besar.

  Dengan melakukan Standarisasi pada data, model _machine learning_ dapat memberikan hasil yang lebih baik dan konsisten dalam melakukan prediksi pada data yang belum pernah dilihat sebelumnya.


## **Modeling**
* Algoritma Penelitian ini melakukan pemodelan dengan 3 algoritma, yaitu K-Nearest Neighbour, Random Forest, dan Adaboost. Ketiga algoritma (K-Nearest Neighbors, Random Forest, dan Adaboost) dipilih karena keunggulan masing-masing dalam konteks prediksi harga sewa rumah.

1. K-Nearest Neighbors (KNN): Algoritma KNN dipilih karena kemampuannya dalam melakukan prediksi berdasarkan jarak terdekat ke tetangga-tetangga terdekat dalam ruang fitur. Dalam konteks prediksi harga sewa rumah, KNN dapat memberikan perkiraan yang baik berdasarkan rumah-rumah dengan karakteristik serupa. Dalam proyek ini, KNN digunakan untuk membandingkan jarak antara satu sampel dengan sampel pelatihan lainnya dan memilih sejumlah k tetangga terdekat untuk melakukan prediksi harga sewa.

2. Random Forest: Algoritma Random Forest dipilih karena kemampuannya dalam mengatasi masalah overfitting dan meningkatkan akurasi prediksi. Dalam konteks prediksi harga sewa rumah, Random Forest membangun banyak pohon keputusan secara paralel dan menggabungkan hasil prediksi dari setiap pohon untuk menghasilkan prediksi akhir. Ini membantu mengatasi variabilitas dan kecenderungan overfitting yang mungkin terjadi dengan pohon keputusan tunggal. Dalam proyek ini, Random Forest digunakan untuk membangun ensemble dari banyak pohon keputusan dalam upaya untuk memprediksi harga sewa rumah dengan lebih akurat.

3. Adaboost: Algoritma Adaboost dipilih karena kemampuannya untuk menggabungkan beberapa model sederhana menjadi model yang lebih kuat. Dalam konteks prediksi harga sewa rumah, Adaboost membangun rangkaian model sederhana secara berurutan dan mengurangi bias kesalahan dari model sebelumnya dengan memberikan penekanan lebih pada sampel yang sulit diprediksi. Dalam proyek ini, Adaboost digunakan untuk membangun rangkaian model sederhana (decision stumps) secara berurutan untuk memprediksi harga sewa rumah.
   
    * Pada proyek ini, model K-Nearest Neighbors (KNN) digunakan dengan menggunakan kelas KNeighborsRegressor dari modul sklearn.neighbors. Model ini membandingkan jarak antara satu sampel dengan sampel pelatihan lainnya dan memilih sejumlah k tetangga terdekat untuk melakukan prediksi.

      Parameter yang dapat digunakan pada KNeighborsRegressor adalah:
        * n_neighbors = Jumlah k tetangga tedekat. 

    * Pada proyek ini, algoritma Random Forest digunakan dengan menggunakan kelas RandomForestRegressor dari modul sklearn.ensemble. Random Forest adalah metode ensemble yang membangun banyak pohon keputusan (decision tree) pada saat pelatihan.

      Beberapa parameter yang digunakan pada proyek ini adalah:

      * n_estimators: Parameter ini menentukan jumlah maksimum estimator di mana boosting akan dihentikan. Estimator mengacu pada jumlah pohon keputusan yang akan dibangun dalam ensemble. Semakin banyak estimator, semakin kompleks modelnya. Nilai yang umum digunakan adalah 100, 200, atau lebih.

      * max_depth: Parameter ini menentukan kedalaman maksimum dari setiap pohon keputusan dalam ensemble. Kedalaman pohon mempengaruhi kompleksitas dan kemampuan model untuk mempelajari pola pada data. Jika tidak ditentukan, pohon akan tumbuh secara penuh dan dapat menyebabkan overfitting. Nilai yang lebih kecil untuk max_depth dapat membantu menghindari overfitting.

      * random_state: Parameter ini digunakan untuk mengontrol seed (benih) acak yang diberikan pada setiap base estimator pada setiap iterasi boosting. Dengan menetapkan random_state dengan nilai tertentu, hasil yang dihasilkan akan konsisten setiap kali kode dijalankan. Ini berguna untuk memastikan reproduktibilitas hasil yang sama jika parameter dan data yang sama digunakan.

      Dalam proyek ini, n_estimators, max_depth, dan random_state dapat diatur sesuai kebutuhan dan karakteristik data untuk membangun model Random Forest yang optimal. Anda dapat menyesuaikan nilai-nilai ini untuk mengeksplorasi berbagai konfigurasi dan menemukan kombinasi yang menghasilkan kinerja terbaik.

    * Pada proyek ini, algoritma Adaboost (Adaptive Boosting) digunakan dengan menggunakan kelas AdaBoostRegressor dari modul sklearn.ensemble. Adaboost adalah metode ensemble yang bertujuan untuk meningkatkan performa atau akurasi prediksi dengan menggabungkan beberapa model sederhana yang dianggap lemah secara berurutan, sehingga membentuk suatu model yang kuat.

      Beberapa parameter yang digunakan pada proyek ini adalah:

      * n_estimators: Parameter ini menentukan jumlah maksimum estimator di mana proses boosting akan dihentikan. Estimator merujuk pada jumlah model sederhana (weak learners), seperti pohon keputusan dengan hanya 1 split (decision stumps), yang akan digunakan dalam ensemble. Semakin banyak estimator, semakin kompleks modelnya. Nilai yang umum digunakan adalah 50, 100, atau lebih.

      * learning_rate: Parameter ini mengontrol kontribusi setiap regressor (model sederhana) terhadap ensemble. Learning rate memperkuat atau melemahkan kontribusi setiap model sederhana dalam pembentukan model kuat. Nilai yang lebih kecil untuk learning rate akan memberikan kontribusi yang lebih kecil untuk setiap regressor, sedangkan nilai yang lebih besar akan memberikan kontribusi yang lebih besar. Nilai yang umum digunakan adalah 0,1, 0,01, atau lebih kecil.

      * random_state: Parameter ini digunakan untuk mengontrol seed (benih) acak yang diberikan pada setiap base estimator pada setiap iterasi boosting. Dengan menetapkan random_state dengan nilai tertentu, hasil yang dihasilkan akan konsisten setiap kali kode dijalankan. Ini berguna untuk memastikan reproduktibilitas hasil yang sama jika parameter dan data yang sama digunakan.

      Dalam proyek ini, n_estimators, learning_rate, dan random_state dapat diatur sesuai kebutuhan dan karakteristik data untuk membangun model Adaboost yang optimal. Anda dapat menyesuaikan nilai-nilai ini untuk mengeksplorasi berbagai konfigurasi dan menemukan kombinasi yang menghasilkan kinerja terbaik.

    * Untuk mencapai keputusan tentang parameter terbaik untuk masing-masing algoritma, digunakan teknik Grid Search. Grid Search merupakan metode untuk mencari kombinasi terbaik dari hyperparameter dalam suatu model dengan mencoba semua kemungkinan kombinasi nilai hyperparameter yang didefinisikan sebelumnya. Dalam proyek ini, Grid Search digunakan untuk mencari kombinasi terbaik dari parameter-parameter seperti jumlah tetangga terdekat (n_neighbors) pada KNN, jumlah estimator (n_estimators) dan kedalaman maksimum (max_depth) pada Random Forest, serta jumlah estimator (n_estimators) dan learning rate pada Adaboost.

      Dengan menggunakan Grid Search, model dapat dievaluasi dengan berbagai kombinasi parameter dan dipilih parameter terbaik yang menghasilkan performa model yang optimal, seperti akurasi yang tinggi atau MSE yang rendah. Grid Search memungkinkan eksplorasi sistematis dari ruang parameter dan membantu dalam menghindari pemilihan parameter secara acak yang dapat menghasilkan model yang tidak optimal.

       Berikut adalah hasil dari Grid Search pada proyek ini :

       **Tabel 3. Parameter terbaik dari masing-masing algoritma**
      
      | Model | best_params |
      | ----------- | :---------: |
      | knn | {'n_neighbors': 3} | 
      | boosting | {'learning_rate': 0.001, 'n_estimators': 25, 'random_state': 33} | 
      | random_forest | {'max_depth': 8, 'n_estimators': 25, 'random_state': 77} |

      Hasil Grid Search pada tabel 3 menunjukkan kombinasi parameter terbaik untuk masing-masing algoritma, yaitu K-Nearest Neighbors dengan 3 tetangga terbaik, Adaboost dengan learning_rate=0.001, n_estimators=25, dan random_state=33, serta Random Forest dengan max_depth=8, n_estimators=25, dan random_state=77. Penggunaan parameter terbaik ini diharapkan dapat meningkatkan kinerja dan akurasi model.

## **Evaluation**

Dalam proyek ini, metrik evaluasi yang digunakan adalah akurasi dan mean squared error (MSE). Akurasi mengukur sejauh mana hasil prediksi cocok dengan nilai sebenarnya (y_test). Sedangkan MSE mengukur kesalahan dalam model statistik dengan menghitung rata-rata dari kuadrat selisih antara nilai aktual dan nilai prediksi. Berikut adalah rumus MSE:

$$ MSE = \sum \frac{(\hat{y} - y)^2}{n}$$

$\hat{y}$ = Nilai Prediksi

$y$ = Nilai Sebenarnya

$n$ = Jumlah Data

Berikut hasil evaluasi pada proyek ini :
* Akurasi

  **Tabel 4. Akurasi dari masing-masing algoritma**
  
  | Model | Akurasi |
  | ----------- | :---------: |
  | knn | 0.825527 | 
  | boosting | 0.902328 | 
  | random_forest | 0.933445 |

  Pada tabel 4. Nilai akurasi menunjukkan sejauh mana model dapat memprediksi dengan benar harga sewa rumah berdasarkan karakteristik yang diberikan. Semakin tinggi nilai akurasi, semakin tinggi tingkat kesesuaian antara hasil prediksi model dengan nilai sebenarnya. Dalam konteks proyek ini, model Random Forest memiliki akurasi tertinggi (0.933445), yang berarti model ini mampu memprediksi harga sewa dengan tingkat kecocokan yang lebih tinggi dibandingkan dengan model KNN dan Adaboost.

* Mean Squared Error (MSE)

  ![hasil pengujian](https://github.com/renaldipanji/house-rent-prediction-ml/assets/75974146/30056f77-030f-4ad2-98dd-d12f82e043d4)

  **Gambar 8. Visualisasi Perbandingan MSE dari masing-masing algoritma**

  Pada gambar 8 Nilai MSE mengukur seberapa dekat hasil prediksi dengan nilai sebenarnya dalam bentuk rata-rata error kuadrat. Semakin rendah nilai MSE, semakin kecil kesalahan prediksi yang dilakukan oleh model. Dalam konteks proyek ini, model Random Forest memiliki nilai MSE terendah, menunjukkan bahwa model ini memiliki tingkat kesalahan prediksi yang lebih kecil dan lebih mendekati nilai sebenarnya dibandingkan dengan model KNN dan Adaboost.

Hasil evaluasi yang menunjukkan bahwa model Random Forest memiliki akurasi yang tinggi dan MSE yang rendah berarti model tersebut mampu memprediksi harga sewa rumah dengan tingkat kecocokan yang tinggi dan tingkat kesalahan yang rendah. Hal ini sesuai dengan tujuan proyek untuk membangun model _machine learning_ yang dapat memprediksi harga sewa yang sesuai dengan harga pasar. Dengan menggunakan model Random Forest, perusahaan penyewaan dapat menentukan harga sewa yang optimal, meningkatkan keuntungan, dan menghindari underpricing atau overpricing properti.

Dengan demikian, hasil evaluasi yang menunjukkan keunggulan model Random Forest memberikan dukungan yang kuat terhadap tujuan proyek dan memberikan keyakinan bahwa penggunaan model ini akan memberikan manfaat yang signifikan bagi perusahaan penyewaan properti.

## **Kesimpulan**

Dalam proyek ini, model _machine learning_ telah dikembangkan untuk memprediksi harga sewa rumah berdasarkan karakteristiknya. Model ini memberikan manfaat yang signifikan bagi perusahaan penyewaan properti dalam mengoptimalkan harga sewa, mengidentifikasi properti dengan potensi profitabilitas tinggi, dan menyesuaikan strategi pemasaran. Evaluasi model menunjukkan bahwa algoritma Random Forest memberikan performa yang lebih baik dengan akurasi yang tinggi dan tingkat error yang rendah.

Dengan menggunakan model ini, perusahaan dapat menentukan harga sewa yang lebih optimal, meningkatkan keuntungan, dan membuat keputusan yang lebih informasi. Model ini juga memberikan wawasan berharga dalam identifikasi properti yang memiliki potensi profitabilitas tinggi dan membantu perusahaan dalam merancang strategi pemasaran yang efektif.

Untuk pengembangan masa depan, disarankan untuk memperluas dataset, menambahkan fitur tambahan yang relevan, melakukan penyetelan hyperparameter yang lebih baik, dan terus memantau serta meningkatkan model sesuai kebutuhan. Dengan melakukan langkah-langkah ini, perusahaan dapat terus meningkatkan kinerja model dan mengoptimalkan manfaat yang diberikan.

Secara keseluruhan, penggunaan model _machine learning_ ini memberikan kontribusi yang berarti bagi perusahaan penyewaan properti dalam meningkatkan efisiensi, keuntungan, dan pengambilan keputusan yang lebih baik dalam bisnis penyewaan properti.

## **Referensi**

[1] E. F. Rahayuningtyas, F. N. Rahayu, and Y. Azhar, “Prediksi Harga rumah menggunakan general regression neural network,” Jurnal Informatika, vol. 8, no. 1, pp. 59–66, 2021. doi:10.31294/ji.v8i1.9036

[2] I. Maula, L. U. Hasanah, and A. Tholib, “Analisis prediksi Harga Rumah di Jabodetabek Menggunakan multiple linear regression,” Jurnal Informatika Kaputama (JIK), vol. 7, no. 2, pp. 216–224, 2023. doi:10.59697/jik.v7i2.135 

[3] M. L. Mu’tashim, T. Muhayat, S. A. Damayanti, H. N. Zaki, and R. Wirawan, “Analisis Prediksi Harga Rumah Sesuai spesifikasi Menggunakan Multiple Linear Regression,” Informatik : Jurnal Ilmu Komputer, vol. 17, no. 3, p. 238, 2021. doi:10.52958/iftk.v17i3.3635 
