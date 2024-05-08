# OCR-KTP
This project aims to create an API that can scan and convert important data (NIK, Name, Place and Date of Birth) from a KTP image into text using PyTesseract Optical Character Recognition (OCR). In addition there is also a deep learning (YOLO) based KTP detector that can automatically crop the KTP image to improve OCR's ability to read text on images (this feature is still very imperfect and requires further development). Thanks to the developers who have developed most of the contents of this system before.

## Prerequisites
* Flask
```
pip install flask
```
* Numpy
```
pip install numpy
```
* OpenCV
```
pip install opencv-contrib-python==4.5.1.48
```
* Pandas
```
pip install pandas
```
* PIL
```
pip install pillow
```
* PyTesseract
```
sudo apt-get install tesseract-ocr
sudo apt-get install tesseract-ocr-ind
pip install pytesseract
```
* TextDistance
```
pip install textdistance
```

## Running the Program
Create virtual environment:

```sh
python -m venv env
```

Run virtual environment (for Windows user):

```sh
env\Scripts\activate.bat
```

Run virtual environment (for Unix user):

```sh
source ./env/bin/activate
```

To run the program, use the command below:
```
export FLASK_APP=app
flask run
```
or alternatively using this command:
```
python app.py
```

### Request Parameter
Parameter | Data Type | Mandatory | Notes
--- | --- | :---: | ---
image | Image Files | M | Foto KTP

### Response Parameter

Parameter | Description
--- | ---
nik | NIK dari hasil OCR
nama | Nama dari hasil OCR
tempat_lahir | Nama tempat lahir dari hasil OCR
tgl_lahir | Tanggal lahir dari hasil OCR (DD-MM-YYYY)
time_elapsed | Waktu yang pemrosesan yang dibutuhkan (detik)

### Success Response Example
```
{
    "error": false,
    "message": "Proses OCR Berhasil",
    "result": {
        "nik": "1234567890123456",
        "nama": "RIJAL MUHYIDIN",
        "tempat_lahir": "PALEMBANG",
        "tgl_lahir": "10-10-1999",
        "jenis_kelamin": "LAKI-LAKI",
        "agama": "ISLAM",
        "status_perkawinan": "BELUM KAWIN",
        "pekerjaan": "PELAJAR/MAHASISWA",
        "kewarganegaraan": "WNI",
        "alamat": {
            "name": "DUSUN 1 OGAN 5",
            "rt_rw": "001/002",
            "kel_desa": "SUNGAI ARE",
            "kecamatan": "ALANG-ALANG LEBAR",
            "kabupaten": "OGAN ILIR",
            "provinsi": "SUMATERA SELATAN"
        },
        "time_elapsed": "6.306"
    }
}
```

### Notes for KTP Detection using CNN
1. Create new folder, data/cnn
2. Run the program

## Acknowledgments
* https://github.com/enningxie/KTP-OCR
* https://github.com/jeffreyevan/OCR