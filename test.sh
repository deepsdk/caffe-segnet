set -e

make all -j4

MODEL="./models/test_full.prototxt"
build/tools/test.bin -model $MODEL -weights full $@

#echo "========================================================="
#
#MODEL="./models/test_half1.prototxt"
#build/tools/test.bin -model $MODEL -weights half1 $@
#
#MODEL="./models/test_half2.prototxt"
#build/tools/test.bin -model $MODEL -weights half2 $@

#MODEL="./models/test_single.prototxt"
#build/tools/test.bin -model $MODEL -weights single $@
