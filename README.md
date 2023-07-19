# Proyek Pertama MLT (House Rent Prediction)

**Dibuat oleh : Renaldi Panji Wibowo**

Proyek ini adalah proyek pertama predictive analytics untuk memenuhi submission MLT Dicoding. Proyek ini membangun model machine learning yang dapat memprediksi harga sewa rumah dan apartemen.

## Domain Proyek

### **Latar Belakang**
Tempat tinggal, seperti rumah atau apartemen, adalah kebutuhan utama manusia untuk melindungi dan menetap. Nilai tempat tinggal ditentukan oleh berbagai faktor, seperti lokasi, ukuran, jumlah kamar, jumlah kamar mandi, perabotan, dan fitur lainnya.

![dataset-cover (4)](https://github.com/renaldipanji/house-rent-prediction-ml/assets/75974146/94737eba-73a2-4c3b-888e-2c6c599bb150)

[Referensi Gambar](https://storage.googleapis.com/kaggle-datasets-images/2644747/4525552/2b4663622a47936b6984513d9e377964/dataset-cover.png?t=2022-11-16-19-36-24)

Harga rumah biasanya mencerminkan nilai yang dimiliki oleh properti tersebut. Namun, harga rumah tidak selalu dapat diprediksi dengan akurat secara manual. Oleh karena itu, perusahaan penyewaan perlu mengurangi ketidakpastian dengan membangun sistem prediksi menggunakan machine learning. Tujuan dari sistem ini adalah dapat memperkirakan harga sewa yang wajar untuk setiap karakteristik rumah.

Melalui penelitian ini, diharapkan model machine learning mampu memprediksi harga sewa rumah yang sesuai dengan harga pasar. Prediksi ini kemudian akan menjadi panduan bagi perusahaan dalam menentukan harga sewa yang dapat menghasilkan keuntungan.

Referensi : https://ejournal.bsi.ac.id/ejurnal/index.php/ji/article/view/9036

# Business Understanding
Proyek ini ditujukan untuk perusahaan yang memiliki model bisnis sebagai berikut:

* Perusahaan memiliki atau mengakuisisi properti rumah dan apartemen, dan kemudian menyewakannya kepada konsumen.

* Perusahaan menyediakan layanan konsultasi harga sewa rumah dan apartemen kepada konsumen

### **Problem Statements**

1. Apa parameter yang paling signifikan dalam menentukan harga sewa rumah atau apartemen di India ?
 
2. Bagaimana cara yang efektif untuk memproses data agar dapat digunakan untuk melatih model dengan baik ?

3. Berapa perkiraan harga sewa rumah di India berdasarkan karakteristik tertentu ?

### **Goals**

1. Mengetahui parameter apa yang paling berpengaruh dalam menentukan harga sewa rumah atau apartemen di India.

2. Melakukan persiapan data untuk dapat dilatih oleh model.

3. Membuat model machine learning yang dapat memprediksi harga sewa rumah di India seakurat mungkin berdasarkan karakteristik tertentu.

### **Solution Statement**

1. Menganalisis data dengan melakukan univariate analysis dan multivariate analysis. Memahami data juga dapat dilakukan dengan visualisasi. Memahami data dapat membantu untuk mengetahui korelasi antar fitur dan mendeteksi outlier.
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

## **Univariate Analysis**
Univariate Analysis adalah menganalisis masing-masing fitur.

### **Analisis jumlah nilai unique pada setiap fitur kategorik**
Fitur kategorik City, Furnishing Status, dan Tenant Preferred memiliki sebaran sample yang cukup merata.

* Fitur City

  ![City](https://github.com/renaldipanji/house-rent-prediction-ml/assets/75974146/276b14eb-60ae-4ce8-9e5d-a59d3c3bd361)

* Fitur Furnishing Status
  
  ![Furnishing Status](https://github.com/renaldipanji/house-rent-prediction-ml/assets/75974146/1eb2af4e-2c2d-4f4c-89e5-a3e41e2c9e39)

* Fitur Tenant Preferred
  
  ![Tenant](https://github.com/renaldipanji/house-rent-prediction-ml/assets/75974146/26a3f0b7-3794-46dc-b9d3-c9df80593ba3)

Untuk Fitur dengan sample yang tidak merata sebagai berikut :

* Area Type
  
  ![Area Type](https://github.com/renaldipanji/house-rent-prediction-ml/assets/75974146/987d7684-d7bb-45e9-983d-3d160efbcb5b)

  Hanya ada 2 data dari Built Area pada fitur Area Type. Untuk menghindari data dengan dimensi tinggi (high dimensional data), kedua data ini akan dihapus.

* Floor dan Area Locality
  
  ![Floor](https://github.com/renaldipanji/house-rent-prediction-ml/assets/75974146/7099ea45-46da-430f-a95a-730035133dec)
  
  ![Area Locality](https://github.com/renaldipanji/house-rent-prediction-ml/assets/75974146/bd5eed51-f6d8-4a7d-bdea-ebd42e2baa0d)

  Fitur Floor dan Area Locality memiliki banyak nilai unik yang menyebabkan dimensi data menjadi tinggi. Untuk menghindari data dengan dimensi tinggi (high dimensional data), kedua fitur ini akan dihapus.

### **Analisis sebaran pada setiap fitur numerik**

![visualisasi data fitur numerik](https://github.com/renaldipanji/house-rent-prediction-ml/assets/75974146/cfb5b474-bf5d-4eb9-8465-a9f4bdfa46ff)

Berikut analisis dari grafik di atas :
* Sebagian besar rumah memiliki 1 sampai 3 BHK dan 1 sampai 3 kamar mandi.
* Sebagian besar rumah memiliki luas di bawah 2000 sqft.
* Rentang harga sewa cukup tinggi, yaitu dari 1200 hingga 3500000. Namun, rata-rata harga rumah hanya 35003. Distribusi harga yang kurang bagus seperti ini dapat berimplikasi pada model.

## **Multivariate Analysis**

Multivariate Analysis menunjukkan hubungan antara dua atau lebih fitur dalam data.

### **Analisis fitur numerik**

* Fitur Size dan BHK akan dianalisis, dan dilakukan penghapusan outlier pada fitur BHK. Hal ini karena tidak biasa bagi rumah dengan 1 BHK memiliki luas 100 sqft. Kami akan menetapkan batas treshold atau batas 300 sqft/BHK. Data yang berada di bawah batas ini akan dihapus. Sebagai hasilnya, akan ada pengurangan jumlah sampel sebesar 548.

* Fitur Size dan Rent (Menghapus Price per sqft Outlier) Untuk memudahkan dalam mendeteksi outlier, maka dibuat fitur baru 'Price_per_sqft' dari kedua fitur tersebut untuk menganalisis harga sewa per luas sqft.

  ![outlier sqft](https://github.com/renaldipanji/house-rent-prediction-ml/assets/75974146/b75dfda7-2b94-4f50-8db6-5aaae9f4f8bc)

  Dari analisis tersebut, terlihat bahwa harga 571 per sqft sangat rendah dan harga 1400000 per sqft sangat tinggi. Oleh karena itu,   dilakukan penghapusan outlier pada harga per sqft menggunakan metode mean dan satu standar deviasi, yang dikelompokkan berdasarkan kota. Hal ini menyebabkan pengurangan jumlah sampel sebesar 497.

* Fitur Bathroom dan BHK akan dianalisis, dan dilakukan penghapusan outlier pada fitur Bathroom. Hal ini karena tidak biasa bagi rumah dengan 2 BHK memiliki 4 kamar mandi. Kami akan menetapkan batas bahwa jumlah kamar mandi tidak boleh melebihi jumlah BHK + 2. Sebagai hasilnya, akan ada pengurangan jumlah sampel sebesar 3.

* Melihat kolerasi antara semua fitur numerik
  ![Korelasi Fitur Numerik](https://github.com/renaldipanji/house-rent-prediction-ml/assets/75974146/19374161-877e-46f2-9980-a45d4f97854d)
  Dari analisis yang dilakukan, terlihat bahwa fitur BHK, Size, dan Bathroom tidak memiliki korelasi yang signifikan dengan fitur target (Rent). Hal ini mungkin disebabkan oleh kurangnya data dalam penelitian ini. Namun, fitur BHK dan Bathroom tetap memiliki korelasi yang signifikan dengan fitur Size. Hal ini sesuai dengan harapan setelah dilakukannya penghapusan outlier sebelumnya.

### **Analisis fitur kategorik**
Analisis ini dilakukan untuk melihat kolerasi antara fitur kategorik dengan fitur target (Rent).

* Fitur Area Type

  ![Fitur Area Type](https://github.com/renaldipanji/house-rent-prediction-ml/assets/75974146/b065ece8-7c63-4fe1-92cd-cd67d0bbbf08)

  Fitur Area Type memiliki pengaruh yang kecil terhadap rata-rata harga sewa.
  
* Fitur City
  
  ![Fitur City](https://github.com/renaldipanji/house-rent-prediction-ml/assets/75974146/129cee6f-6f55-489f-a5ce-2805c451ec53)

  Dari analisis tersebut, terlihat bahwa fitur City memiliki pengaruh yang cukup besar terhadap rata-rata harga sewa, terutama jika rumah berada di kota Mumbai. Faktanya, sebaran harga rumah mencapai tingkat tertinggi di kota Mumbai. Mumbai memang dikenal sebagai kota yang memiliki biaya hidup yang tinggi di India, diikuti oleh Delhi. Ini menunjukkan bahwa faktor lokasi, terutama kota tempat rumah berada, memiliki dampak signifikan pada harga sewa rumah.

* Fitur Furnishing Status
   
  ![Fitur Furnishing Status](https://github.com/renaldipanji/house-rent-prediction-ml/assets/75974146/14c09152-12e8-4bd9-84f4-3b30f4de752f)

  Dari analisis yang dilakukan, terlihat bahwa fitur Furnishing Status memiliki pengaruh yang cukup besar terhadap rata-rata harga sewa. Hal ini wajar, karena rumah yang dilengkapi dengan perabotan lengkap umumnya akan diberi harga sewa yang lebih tinggi daripada rumah yang tidak dilengkapi perabotan. Furnishing Status menjadi salah satu faktor penting yang memengaruhi penentuan harga sewa rumah, karena keberadaan perabotan dapat memberikan nilai tambah dan kenyamanan bagi penyewa.

* Tenant Preferred

  ![Fitur Tenant](https://github.com/renaldipanji/house-rent-prediction-ml/assets/75974146/f4616bf4-cfa5-4005-a627-02068d5d49b2)

  Dari analisis yang dilakukan, terlihat bahwa fitur Tenant Preferred memiliki pengaruh yang cukup signifikan terhadap rata-rata harga sewa. Dari grafik yang ditampilkan, terlihat bahwa rumah yang sangat disarankan untuk disewa oleh keluarga memiliki rata-rata harga sewa yang lebih tinggi dibandingkan dengan preferensi penyewa lainnya. Hal ini dapat disebabkan oleh faktor-faktor seperti ukuran rumah yang lebih besar, kebutuhan ruang yang lebih luas, atau fasilitas tambahan yang cocok untuk kebutuhan keluarga. Oleh karena itu, Tenant Preferred menjadi faktor penting yang memengaruhi penentuan harga sewa rumah.


## **Data preparation**

* One Hot Encoding
  one-hot encoding adalah teknik yang digunakan untuk mengubah data kategorikal menjadi data numerik dengan menciptakan kolom baru untuk setiap kategori yang ada dan mengisinya dengan nilai 0 atau 1.

  Dalam proyek ini, fitur-fitur yang akan diubah menjadi numerik menggunakan one-hot encoding adalah:

  * Area Type: Fitur ini akan dipecah menjadi kolom baru berdasarkan jenis-jenis Area Type yang ada, seperti Super Area, Carpet Area, dan Build Area.
  * City: Fitur ini akan dipecah menjadi kolom baru berdasarkan nama-nama kota yang ada, seperti Mumbai, Delhi, dan sebagainya.
  * Furnishing Status: Fitur ini akan dipecah menjadi kolom baru berdasarkan status perabotan, seperti Furnished, Semi-Furnished, dan   Unfurnished.
  * Tenant Preferred: Fitur ini akan dipecah menjadi kolom baru berdasarkan jenis-jenis preferensi penyewa yang ada, seperti Family, Bachelor, dan sebagainya.

  Dengan menggunakan one-hot encoding, fitur-fitur ini akan diubah menjadi data numerik yang dapat digunakan oleh model machine learning untuk proses pelatihan dan prediksi.

* Train Test Split
  Train-test split adalah proses membagi data menjadi data latih (train data) dan data uji (test data). Data latih digunakan untuk melatih model, sementara data uji digunakan untuk menguji kinerja model yang telah dilatih.

  Pada proyek ini, dataset yang memiliki ukuran 3696 akan dibagi menjadi 3326 untuk data latih dan 370 untuk data uji. Dalam hal ini, sekitar 90% dari data akan digunakan sebagai data latih, sedangkan sekitar 10% akan digunakan sebagai data uji. Pembagian ini dapat dilakukan dengan menggunakan teknik random sampling untuk memastikan representativitas data dalam kedua subset.

  Data latih akan digunakan untuk melatih model machine learning dan menyesuaikan parameter model. Setelah model dilatih, data uji akan digunakan untuk menguji performa dan mengukur akurasi model yang dihasilkan.

  Dengan membagi data menjadi data latih dan data uji, ini memungkinkan evaluasi yang obyektif terhadap kinerja model di luar data yang digunakan untuk melatihnya, sehingga memberikan perkiraan yang lebih baik tentang bagaimana model tersebut akan berperforma pada data baru yang belum pernah dilihat sebelumnya.

* Normalization
  normalisasi data dapat membantu meningkatkan performa dan kecepatan algoritma machine learning, terutama ketika data memiliki skala yang berbeda-beda. Salah satu teknik normalisasi yang umum digunakan adalah Standarisasi, yang dapat dilakukan dengan menggunakan kelas StandardScaler dari modul sklearn.preprocessing.

  Pada proyek ini, Standarisasi dengan StandardScaler digunakan untuk memastikan bahwa skala data pada fitur-fitur yang digunakan dalam model machine learning adalah relatif sama. Dengan menggunakan StandardScaler, fitur-fitur akan diubah sedemikian rupa sehingga memiliki mean 0 dan variansi 1. Hal ini membantu algoritma machine learning dalam menginterpretasikan dan memproses data dengan lebih baik, serta menghindari dominasi fitur-fitur dengan skala yang lebih besar.

  Dengan melakukan Standarisasi pada data, model machine learning dapat memberikan hasil yang lebih baik dan konsisten dalam melakukan prediksi pada data yang belum pernah dilihat sebelumnya.


## **Modeling**
* Algoritma Penelitian ini melakukan pemodelan dengan 3 algoritma, yaitu K-Nearest Neighbour, Random Forest, dan Adaboost.
   
    * Pada proyek ini, model K-Nearest Neighbors (KNN) digunakan dengan menggunakan kelas KNeighborsRegressor dari modul     sklearn.neighbors. Model ini membandingkan jarak antara satu sampel dengan sampel pelatihan lainnya dan memilih sejumlah k tetangga terdekat untuk melakukan prediksi.

      Parameter yang dapat digunakan pada KNeighborsRegressor adalah:
        * n_neighbors = Jumlah k tetangga tedekat. 

    * Pada proyek ini, algoritma Random Forest digunakan dengan menggunakan kelas RandomForestRegressor dari modul sklearn.ensemble. Random Forest adalah metode ensemble yang membangun banyak pohon keputusan (decision tree) pada saat pelatihan.

      Beberapa parameter yang digunakan pada proyek ini adalah:

      * n_estimators: Parameter ini menentukan jumlah maksimum estimator di mana boosting akan dihentikan. Estimator mengacu pada jumlah pohon keputusan yang akan dibangun dalam ensemble. Semakin banyak estimator, semakin kompleks modelnya. Biasanya, nilai yang umum digunakan adalah 100, 200, atau lebih.

      * max_depth: Parameter ini menentukan kedalaman maksimum dari setiap pohon keputusan dalam ensemble. Kedalaman pohon mempengaruhi kompleksitas dan kemampuan model untuk mempelajari pola pada data. Jika tidak ditentukan, pohon akan tumbuh secara penuh dan dapat menyebabkan overfitting. Nilai yang lebih kecil untuk max_depth dapat membantu menghindari overfitting.

      * random_state: Parameter ini digunakan untuk mengontrol seed (benih) acak yang diberikan pada setiap base estimator pada setiap iterasi boosting. Dengan menetapkan random_state dengan nilai tertentu, hasil yang dihasilkan akan konsisten setiap kali kode dijalankan. Ini berguna untuk memastikan reproduktibilitas hasil yang sama jika parameter dan data yang sama digunakan.

      Dalam proyek ini, n_estimators, max_depth, dan random_state dapat diatur sesuai kebutuhan dan karakteristik data untuk membangun model Random Forest yang optimal. Anda dapat menyesuaikan nilai-nilai ini untuk mengeksplorasi berbagai konfigurasi dan menemukan kombinasi yang menghasilkan kinerja terbaik.

    * Pada proyek ini, algoritma Adaboost (Adaptive Boosting) digunakan dengan menggunakan kelas AdaBoostRegressor dari modul sklearn.ensemble. Adaboost adalah metode ensemble yang bertujuan untuk meningkatkan performa atau akurasi prediksi dengan menggabungkan beberapa model sederhana yang dianggap lemah secara berurutan, sehingga membentuk suatu model yang kuat.

      Beberapa parameter yang digunakan pada proyek ini adalah:

      * n_estimators: Parameter ini menentukan jumlah maksimum estimator di mana proses boosting akan dihentikan. Estimator merujuk pada jumlah model sederhana (weak learners), seperti pohon keputusan dengan hanya 1 split (decision stumps), yang akan digunakan dalam ensemble. Semakin banyak estimator, semakin kompleks modelnya. Biasanya, nilai yang umum digunakan adalah 50, 100, atau lebih.

      * learning_rate: Parameter ini mengontrol kontribusi setiap regressor (model sederhana) terhadap ensemble. Learning rate memperkuat atau melemahkan kontribusi setiap model sederhana dalam pembentukan model kuat. Nilai yang lebih kecil untuk learning rate akan memberikan kontribusi yang lebih kecil untuk setiap regressor, sedangkan nilai yang lebih besar akan memberikan kontribusi yang lebih besar. Biasanya, nilai yang umum digunakan adalah 0,1, 0,01, atau lebih kecil.

      * random_state: Parameter ini digunakan untuk mengontrol seed (benih) acak yang diberikan pada setiap base estimator pada setiap iterasi boosting. Dengan menetapkan random_state dengan nilai tertentu, hasil yang dihasilkan akan konsisten setiap kali kode dijalankan. Ini berguna untuk memastikan reproduktibilitas hasil yang sama jika parameter dan data yang sama digunakan.

      Dalam proyek ini, n_estimators, learning_rate, dan random_state dapat diatur sesuai kebutuhan dan karakteristik data untuk membangun model Adaboost yang optimal. Anda dapat menyesuaikan nilai-nilai ini untuk mengeksplorasi berbagai konfigurasi dan menemukan kombinasi yang menghasilkan kinerja terbaik.

    * Hyperparameter Tuning (Grid Search) Hyperparameter tuning adalah cara untuk mendapatkan parameter terbaik dari algoritma dalam membangun model. Salah satu teknik dalam hyperparameter tuning yang digunakan dalam proyek ini adalah grid search. Berikut adalah hasil dari Grid Search pada proyek ini :

      | Model | best_params |
      | ----------- | :---------: |
      | knn | {'n_neighbors': 3} | 
      | boosting | {'learning_rate': 0.001, 'n_estimators': 25, 'random_state': 33} | 
      | random_forest | {'max_depth': 8, 'n_estimators': 25, 'random_state': 77} |

## **Evaluation**

Dalam proyek ini, metrik evaluasi yang digunakan adalah akurasi dan mean squared error (MSE). Akurasi mengukur sejauh mana hasil prediksi cocok dengan nilai sebenarnya (y_test). Sedangkan MSE mengukur kesalahan dalam model statistik dengan menghitung rata-rata dari kuadrat selisih antara nilai aktual dan nilai prediksi. Berikut adalah rumus MSE:

![MSE](https://github.com/renaldipanji/house-rent-prediction-ml/assets/75974146/ed033d35-9cf0-42b8-a290-40cb590c983b)

Berikut hasil evaluasi pada proyek ini :
* Akurasi
    
  | Model | Akurasi |
  | ----------- | :---------: |
  | knn | 0.825527 | 
  | boosting | 0.902328 | 
  | random_forest | 0.933445 |

* Mean Squared Error (MSE)

  ![hasil pengujian](https://github.com/renaldipanji/house-rent-prediction-ml/assets/75974146/30056f77-030f-4ad2-98dd-d12f82e043d4)

Dari hasil evaluasi, dapat disimpulkan bahwa model dengan algoritma Random Forest memiliki akurasi yang lebih tinggi dan tingkat error (MSE) yang lebih rendah dibandingkan dengan algoritma lain yang digunakan dalam proyek ini. Hal ini menunjukkan bahwa Random Forest memberikan performa yang lebih baik dalam memprediksi nilai target dibandingkan dengan algoritma lainnya yang telah diuji.

Akurasi yang lebih tinggi menunjukkan bahwa model Random Forest memiliki tingkat kecocokan atau kesesuaian yang lebih tinggi antara hasil prediksi dan nilai sebenarnya. Sebagai hasilnya, model ini dapat memberikan hasil prediksi yang lebih akurat. Selain itu, tingkat error (MSE) yang lebih rendah menunjukkan bahwa model Random Forest cenderung memiliki kesalahan prediksi yang lebih kecil, sehingga lebih mendekati nilai sebenarnya.

Dengan demikian, berdasarkan evaluasi yang dilakukan, model dengan algoritma Random Forest dapat dianggap sebagai pilihan terbaik dalam proyek ini, karena mampu memberikan performa yang lebih baik dalam memprediksi harga sewa rumah.
