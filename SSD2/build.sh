CXX=${CXX:-g++}


$CXX -std=c++11 -I. -o ssd_test test_ssd_ADAS_PEDESTRIAN_640x360.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpssd  -pthread -lglog
