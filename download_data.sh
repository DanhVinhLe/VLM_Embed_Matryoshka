wget https://huggingface.co/datasets/TIGER-Lab/MMEB-train/resolve/main/images_zip/MSCOCO.zip
unzip MSCOCO.zip -d ./vlm2vec_train/MMEB-train/images/
rm MSCOCO.zip

wget https://huggingface.co/datasets/TIGER-Lab/MMEB-train/resolve/main/images_zip/OK-VQA.zip
unzip OK-VQA.zip -d ./vlm2vec_train/MMEB-train/images/
rm OK-VQA.zip

wget https://huggingface.co/datasets/TIGER-Lab/MMEB-train/resolve/main/images_zip/VOC2007.zip
unzip VOC2007.zip -d ./vlm2vec_train/MMEB-train/images/
rm VOC2007.zip