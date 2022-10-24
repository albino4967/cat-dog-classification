if [[ "$#" -gt 1 ||  "$#" == 0 ]]; then
  echo "Usage: $0 TRAIN_CLASSIFICATION_CODE_DIR"
  exit
fi

DATA_DIR=$1

echo 'env settings ... '
pip install -U pip
pip install -r requirements.txt

echo 'download mask dataset ... '
python data_load.py

echo 'unzip dataset ... '
unzip cat-dog-dataset.zip

if [[ "$DATA_DIR" = "cnn" ]];
then
  echo "$DATA_DIR"
  python cat-dog-classification.py \
    --num_layers 5 \
    --epoch 10 \
    --dropout 0.75 \
    -learning_rate 0.001 \
    --momentum 0.9

elif [[ "$DATA_DIR" = "resnet" ]];
then
  echo "$DATA_DIR"
  python resnet.py \
    --epoch 1

else
  echo "exception error"
fi

echo "finished ..."