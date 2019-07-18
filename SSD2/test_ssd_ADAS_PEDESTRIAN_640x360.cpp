/*
-- (c) Copyright 2018 Xilinx, Inc. All rights reserved.
--
-- This file contains confidential and proprietary information
-- of Xilinx, Inc. and is protected under U.S. and
-- international copyright and other intellectual property
-- laws.
--
-- DISCLAIMER
-- This disclaimer is not a license and does not grant any
-- rights to the materials distributed herewith. Except as
-- otherwise provided in a valid license issued to you by
-- Xilinx, and to the maximum extent permitted by applicable
-- law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
-- WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
-- AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
-- BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
-- INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and
-- (2) Xilinx shall not be liable (whether in contract or tort,
-- including negligence, or under any other theory of
-- liability) for any loss or damage of any kind or nature
-- related to, arising under or in connection with these
-- materials, including for any direct, or any indirect,
-- special, incidental, or consequential loss or damage
-- (including loss of data, profits, goodwill, or any type of
-- loss or damage suffered as a result of any action brought
-- by a third party) even if such damage or loss was
-- reasonably foreseeable or Xilinx had been advised of the
-- possibility of the same.
--
-- CRITICAL APPLICATIONS
-- Xilinx products are not designed or intended to be fail-
-- safe, or for use in any application requiring fail-safe
-- performance, such as life-support or safety devices or
-- systems, Class III medical devices, nuclear facilities,
-- applications related to the deployment of airbags, or any
-- other applications that could lead to death, personal
-- injury, or severe property or environmental damage
-- (individually and collectively, "Critical
-- Applications"). Customer assumes the sole risk and
-- liability of any use of Xilinx products in Critical
-- Applications, subject only to applicable laws and
-- regulations governing limitations on product liability.
--
-- THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
-- PART OF THIS FILE AT ALL TIMES.
*/
#include <xilinx/ssd/ssd.hpp>
#include <iostream>
#include <xilinx/ssd/ssd.hpp>
#include <iostream>
#include <memory>
#include <glog/logging.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <xilinx/demo/demo.hpp>

#include <opencv2/opencv.hpp>
#include <map>
#include<time.h>
#include <string>
using namespace std;
using namespace cv;
#define B_ 255
#define G_ 255
#define R_ 255
typedef struct
{	int label;
	char names[20];
	int B;
	int G;
	int R;
} parameter_struct;
char info[30];
parameter_struct parameter[20]=
{
	/* if uncommon we set the colors all white*/
	{0,"aeroplane",B_,G_,R_},
	{1,"person",128,0,128},//purple   bicycle ==> person
	{2,"bird",B_,G_,R_},
	{3,"boat",B_,G_,R_},
	{4,"bottle",0,97,255},//lemon
	{5,"bus",255,0,0},//blue
	{6,"car",0,255,255},//yellow
	{7,"cat",B_,G_,R_},
	{8,"chair",255,0,0},//green
	{9,"cow",B_,G_,R_},
	{10,"diningtable",128,0,128},//purple
	{11,"dog",B_,G_,R_},
	{12,"horse",B_,G_,R_},
	{13,"motorbike",B_,G_,R_},
	{14,"bicycle",0,0,255},//red     person ==> bicycle
	{15,"pottedplant",0,0,0},//black
	{16,"sheep",B_,G_,R_},
	{17,"sofa",0,255,255},//yellow
	{18,"train",B_,G_,R_},
	{18,"tvmonitor",255,0,0}//blue
};

/*
  This is an example on how to use customer-provided model for yolov3 lib.
  Below parameter of "yolov3_voc_416" in create_ex() is assumed as 
  the model provided by customer. 
  This parameter must be same as the kernel name in the configuration file 
  of prototxt in /etc/XILINX_AI_SDK.conf.d/.
  That means, the file name of the .prototxt, the kernel name, and the parameter
  must be same.  

  below is the detailed steps of HowTo.
    Note: 
      * replace the ... in below dir to your own directory
      * please set correct parameter when running below tool.
        You need refer to corresponding document for detailed information
        of the tool in SDK which mentioned below.

  1. prepare your own customer-provided model (maybe trained by caffe or tensorflow);
  2. convert your own model to Xilinx model format via convert tool provided by SDK.
  3. use dnnc tool to build your model into .elf file
     3.1. caffe model.    
      dnnc --prototxt=/home/.../yolov3_voc_416/deploy.prototxt 
        --caffemodel=/home/..../deploy.caffemodel --output_dir=/home/.... 
        --net_name=yolov3_voc_416 --dpu=4096FA --cpu_arch=arm64 --mode=normal
     3.2. tensorflow model
      dnnc --parser=tensorflow --frozen_pb=/home/.../deploy.pb --output_dir=/home/... 
        --net_name=ssd_your_kern_name --dpu=4096FA --cpu_arch=arm64 --mode=normal
  4. build your model into library. This step need cross-compiling tool.
      /home/.../aarch64-linux-gnu-g++ -nostdlib -fPIC -shared 
        /home/.../dpu_your_own_model.elf -o 
        libdpumodelssd_your_own_model.so
  5. place the built lib in /usr/lib or other library path which can be accessed.
  6. prepare your own prototxt file and place it in etc/XILINX_AI_SDK.conf.d/ .
     Please refer to the document on how to modify this file.

  test pic: use "sample_yolov3.jpg" for this test.

*/

int main(int argc, char *argv[])
{


  //Mat img = cv::imread(argv[1]);
 // if(img.empty()) {
 //     cerr << "cannot load " << argv[1] << endl;
 //     abort();
 // }
  cv::VideoCapture cap(0);
//printf("CV_CAP_PROP_FRAME_WIDTH=%d\n",cap.get(CV_CAP_PROP_FRAME_WIDTH));
//printf("CV_CAP_PROP_FRAME_HEIGHT=%d\n",cap.get(CV_CAP_PROP_FRAME_HEIGHT));
//printf("CV_CAP_PROP_FPS=%d\n",cap.get(CV_CAP_PROP_FPS));
//printf("CV_CAP_PROP_BRIGHTNESS=%d\n",cap.get(CV_CAP_PROP_BRIGHTNESS));
//printf("CV_CAP_PROP_CONTRAST=%d\n",cap.get(CV_CAP_PROP_CONTRAST));
//printf("CV_CAP_PROP_SATURATION=%d\n",cap.get(CV_CAP_PROP_SATURATION));
//printf("CV_CAP_PROP_HUE=%d\n",cap.get(CV_CAP_PROP_HUE));
//printf("CV_CAP_PROP_EXPOSURE=%d\n",cap.get(CV_CAP_PROP_EXPOSURE));
cap.set(CV_CAP_PROP_FRAME_WIDTH,640);
cap.set(CV_CAP_PROP_FRAME_HEIGHT,480);
cap.set(CV_CAP_PROP_FPS,30);
float fps2;
  auto ssd = xilinx::ssd::SSD::create(xilinx::ssd::ADAS_PEDESTRIAN_640x360,true);
  
int font=CV_FONT_HERSHEY_SIMPLEX;
//cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC);
while(1)
{int begintime,endtime;
	Mat img;
	cap>>img;
	begintime=clock();
  auto results = ssd->run(img);

/*  
std::cout << "results.size " << results.bboxes.size() << " " //
            << std::endl;
*/
  for(auto &box : results.bboxes){
      int label = box.label;
      float xmin = box.x * img.cols + 1;
      float ymin = box.y * img.rows + 1;
      float xmax = xmin + box.width * img.cols;
      float ymax = ymin + box.height * img.rows;
      if(xmin < 0.) xmin = 1.;
      if(ymin < 0.) ymin = 1.;
      if(xmax > img.cols) xmax = img.cols;
      if(ymax > img.rows) ymax = img.rows;
      float confidence = box.score;
//	printf("img.cols=%d\n",img.cols);
//	printf("img.rows=%d\n",img.rows);
//      cout << "RESULT: " << label << "\t" << xmin << "\t" << ymin << "\t"
/*	
	cout << "RESULT: " << parameter[label].names<< "\t" << xmin << "\t" << ymin << "\t"
           << xmax << "\t" << ymax << "\t" << confidence << "\n";
*/
      rectangle(img, Point(xmin, ymin), Point(xmax, ymax), Scalar(parameter[label].B, 			parameter[label].G, parameter[label].R),
                  1, 1, 0);
	char buf[20];
	sprintf(buf,"  %0.4f",confidence);
	strcpy(info,parameter[label].names);
	strcat(info,buf);
        putText(img,info,cvPoint(xmin,ymin),CV_FONT_HERSHEY_SIMPLEX,0.5,Scalar(parameter[label].B,parameter[label].G, parameter[label].R),0,0,false);
  //imwrite("sample_yolov3_customer_provided_result.jpg", img);
	}
	//imshow("current",img);
//waitKey(2);
	endtime=clock();
	fps2=1/((endtime-begintime)/1000000.0);
	char buf2[20];
	sprintf(buf2,"  %3.2ffps",fps2);
        putText(img,buf2,cvPoint(5,15),CV_FONT_HERSHEY_SIMPLEX,0.5,Scalar(0,0,255),0,0,false);
	//printf("%3.2ffps\n",fps2);
	imshow("current",img);
	waitKey(2);

}	
  return 0;
}
