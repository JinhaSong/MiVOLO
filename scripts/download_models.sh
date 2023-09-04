DIR="/workspace/weights"

# Check if the directory exists
if [ ! -d "$DIR" ]; then
  # If the directory does not exist, create it
  mkdir -p "$DIR"
  echo "Directory $DIR created."
else
  echo "Directory $DIR already exists."
fi

echo "======================================="
echo "Download start(yolov8x_person_face)"
wget -q ftp://mldisk.sogang.ac.kr/aixcon/mivolo/yolov8x_person_face.pt -O /workspace/weights/yolov8x_person_face.pt \
&& echo "Download successful(yolov8x_person_face)" \
|| echo "\e[31mDownload failed(yolov8x_person_face)\e[0m"
echo "======================================="
echo "Download start(imdb_cross_person)"
wget -q ftp://mldisk.sogang.ac.kr/aixcon/mivolo/model_imdb_cross_person_4.22_99.46.pth.tar -O /workspace/weights/model_imdb_cross_person_4.22_99.46.pth.tar \
&& echo "Download successful(imdb_cross_person)" \
|| echo "\e[31mDownload failed(imdb_cross_person)\e[0m"
echo "======================================="