DATASET_PATH=$1


colmap_sparse() {
  sh ./colmap_sparse.sh $DATASET_PATH
}

# 1) start with first images
colmap_sparse

# 2) add new images iteratively
for FILE in $DATASET_PATH/images_new/*; do
  cp $FILE $DATASET_PATH/images/

  echo "$(basename $FILE)"
  echo "$(basename $FILE)" >> $DATASET_PATH/images/image-list.txt

  colmap_sparse

done
