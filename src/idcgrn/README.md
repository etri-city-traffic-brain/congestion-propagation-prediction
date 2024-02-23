# traffic_forecasting
An implementation of Improved MegaCRN algorithm utilizing Pytorch.

### Getting startetd
It will be upload requirements.txt file later.

Clone repo and install requirements.txt

```sh
git clone https://github.com/~~
cd 2023_etri_source_code
pip install -r requirements.txt
```

### How to run the code

[//]: # (- preprocessing)

[//]: # (We have datasets of dunsan.)

[//]: # ()
[//]: # (1. ts_202107-09 file to csv file.)

[//]: # ()
[//]: # (```sh)

[//]: # (python ts_to_csv.py --dir tsFolderDirectory #ts folder directory)

[//]: # (```)

[//]: # ()
[//]: # (2. Make the necessary input files in STGCM, and MW-TGC.)

[//]: # ()
[//]: # (```sh)

[//]: # (python preprocessing.py --loc front # run front)

[//]: # (```)

[//]: # ()
[//]: # (3. Make the necessary input files in DCRNN.)

[//]: # (```sh)

[//]: # (cd dtgrnn)

[//]: # (python generate_training_data_npz.py --loc front # run front)

[//]: # (				--output_dir dataset/ #Output directory)

[//]: # (```)

- Performance measurement (Execute following prompt in the 'uniq_tcp' directory.)  

	+ Pre-trained Improved MegaCRN (without-training)
		```sh
		python uniq_tcp.py --loc DUNSAN
		```
    + Train-test Improved MegaCRN
		```sh
		python traintest_MegaCRN.py --loc DUNSAN --gpu 0
		```

- Visualization

	+ Generate visualization .h5 data (Execute this prompt in the '2023_etri_source_code' directory.)
		```sh
		python visualization_h5_data_generate.py --from_date_time 20210929063500 --to_date_time 20210929083000
		```
    + Generate visualization .npz data (Excute this prompt in the '2023_etri_source_code' directory after executing the previous prompt.)
		```sh
		python visualization_npz_data_generate.py --from_date_time 20210929063500 --to_date_time 20210929083000
		```
    + Visualization (Excute this prompt in the 'uniq_tcp' directory.)
		```sh
		python visualization_case_study.py
		```
