# Download and unpack Crossmodal-3600 dataset
# mkdir -p data/xm3600/images
# curl -L https://google.github.io/crossmodal-3600/web-data/captions.zip > captions.zip
# mv captions.zip data/xm3600
# unzip data/xm3600/captions.zip -d data/xm3600
# rm data/xm3600/captions.zip
# curl -L https://open-images-dataset.s3.amazonaws.com/crossmodal-3600/images.tgz > images.tgz
# mv images.tgz data/xm3600
# tar -xvzf images.tgz -C data/xm3600/images
# rm data/xm3600/images.tgz

# Download and unpack COCO CN images
curl -L "https://nllb-data.com/test/coco-cn/images.tar.gz" > data/coco_cn/images.tar.gz
tar -xvzf data/coco_cn/images.tar.gz -C data/coco_cn

# Unpack XTD10 images
curl -L "https://nllb-data.com/test/xtd10/images.tar.gz" > data/xtd10/images.tar.gz
tar -xvzf data/xtd10/images.tar.gz -C data/xtd10