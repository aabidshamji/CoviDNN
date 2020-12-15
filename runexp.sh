for m in default VGG16 VGG19 ResNet50 DenseNet121 MobileNet InceptionV3 ResNet50V2 Xception;
do
  python3 train.py $m > out/$m.out;
done
