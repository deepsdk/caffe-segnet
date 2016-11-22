set -e

make all -j4

INPUT=$@
INPUT="../Seq05VD_f01740.png"
MODEL="./models/segnet_basic_camvid/segnet_basic_camvid.prototxt"
WEIGHTS="./models/segnet_basic_camvid/segnet_basic_camvid.caffemodel"
LABEL="./models/segnet_basic_camvid/colors12.txt"
echo $WEIGHTS
build/tools/classification.bin -model $MODEL -weights $WEIGHTS -label $LABEL -output segnet_basic.png $INPUT

MODEL="./models/segnet_driving_web_demo/segnet_model_driving_webdemo.prototxt"
WEIGHTS="./models/segnet_driving_web_demo/segnet_weights_driving_webdemo.caffemodel"
LABEL="./models/segnet_driving_web_demo/colors13.txt"
echo $WEIGHTS
build/tools/classification.bin -model $MODEL -weights $WEIGHTS -label $LABEL -output seg_driving.png  $INPUT

##MODEL="./models/bayesian_segnet_camvid/bayesian_segnet_camvid.prototxt"
##WEIGHTS="./models/bayesian_segnet_camvid/bayesian_segnet_camvid.caffemodel"
##LABEL="./models/segnet_basic_camvid/colors12.txt"
##echo $WEIGHTS
##build/tools/classification.bin -model $MODEL -weights $WEIGHTS -label $LABEL -output bay_seg_camvid.png  $INPUT
#
#
#MODEL="./models/segnet_sun/segnet_sun.prototxt"
#WEIGHTS="./models/segnet_sun/segnet_sun.caffemodel"
#LABEL="./models/segnet_sun/colors37.txt"
#echo $WEIGHTS
#build/tools/classification.bin -model $MODEL -weights $WEIGHTS -label $LABEL -output segnet_sun.png  $INPUT


