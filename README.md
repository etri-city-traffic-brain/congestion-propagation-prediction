# Traffic Congestion Propagation Prediction

## Predicting Congestion Propagation Based on Traffic Prediction Algorithms
An implementation of existing DCRNN, STGCN, MW-TGC algorithm utilizing dgl and Pytorch.

### Getting started
It will be upload requirements.txt file later.

Clone repo and install requirements.txt

```sh
git clone https://github.com/~~
cd traffic_forecasting
pip install -r requirements.txt
```

### How to run the code
- preprocessing
We have datasets of front, doan, dunsan, and wolpyeong.

1. ts_202107-09 file to csv file.

```sh
python ts_to_csv.py --dir tsFolderDirectory #ts folder directory
```

2. Make the necessary input files in STGCM, and MW-TGC.

```sh
python preprocessing.py --loc front # run front
```

3. Make the necessary input files in DCRNN.
```sh
python preprocessing.py --loc front # run front
			--output_dir dataset/ #Output directory
```

- Usage

	+ DCRNN
	```sh
	python train.py --loc front # run front
	```


	+ STGCN
	```sh
	python main_tmap.py --loc front # run front
	```

	+ MW-TGC 
	```sh
	python main.py --loc front # run front
	```
	
