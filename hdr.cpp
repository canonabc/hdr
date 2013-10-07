#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <highgui.h>
#include <FreeImage.h>
#include <omp.h>
#include "tonemap.h"
#include "hdr.h"
//#define NUM_IMAGES 9
#define GAMMA 0.95
#define BF_SCALE 0.9
#define KEY 0.6
//#define  DEBUG
using namespace std;

void cvmat2ipl(int w, int h, CvMat **m, IplImage *img)
{
	for(int y=0 ; y<h ; y++){
		for(int x=0 ; x<w ; x++){
			CvScalar s;
			for(int ch=0 ; ch<3 ; ch++)
				s.val[ch] = cvmGet(m[ch], x, y);
			cvSet2D(img, y, x, s);
		}
	}
}

int main(int argv, char *argc[])
{
	string path = argc[1];
	//string path = "in";
	int num_img = atoi(argc[2]);
	//int num_img = 11;
	//cout << num_img << endl;
	IplImage **images = new IplImage*[20];
	for(int i=1 ; i<=num_img ; i++){
		char buf[20];
		sprintf(buf, "\/img (%d)", i);
		string filename = string(buf);
		filename = path + filename + ".jpg";
		cout << filename << endl;
		images[i-1] = new IplImage();
		images[i-1] = cvLoadImage(filename.c_str(), CV_LOAD_IMAGE_COLOR);
		if(images[i-i]==NULL){
			cout << "Loading error" << endl;
			system("pause");
			exit(0);
		}
	}
	int width = images[0]->width;
	int height = images[0]->height;
	double time[20] = {128, 64, 32, 16, 8, 4, 2, 1, 0.5, 0.25, 0.125, 0.0625, 1.0/32, 1.0/64, 1.0/128, 1.0/256, 1.0/512, 1.0/1024, 1.0/2048};
	Robertson hdr = Robertson(images, time, num_img); // Initial the solver
	CvMat **exposure = new CvMat*[3];
	double **G = new double*[3];
	for(int i=0 ; i<3 ; i++){
		exposure[i] = cvCreateMat(width, height, CV_64FC1);
		G[i] = new double[256];
	}
	for(int channel = 0 ; channel<3 ; channel++){
		for(int i=0 ; i<height ; i++){
			for(int j=0 ; j<width ; j++){
				cvmSet(exposure[channel], j, i, 1.0);
			}	
		}
		for(int i=0 ; i<256 ; i++)
			G[channel][i] = i;
	}
	cout << "Initialization done!" << endl;
	int iteration = 6;
	for(int i=0 ; i<iteration ; i++){
		hdr.update_parameter(exposure, G);
		cout << "iteration = " << i << endl;
	}
	
	FIBITMAP *final_hdr = FreeImage_AllocateT(FIT_RGBF, width, height);
	for(int y = 0; y < FreeImage_GetHeight(final_hdr); y++) {
		FIRGBF *bits = (FIRGBF *)FreeImage_GetScanLine(final_hdr, y);
		for(int x = 0; x < FreeImage_GetWidth(final_hdr); x++) {
				bits[x].red = cvmGet(exposure[2], x, height-1-y);
				bits[x].green = cvmGet(exposure[1], x, height-1-y);
				bits[x].blue = cvmGet(exposure[0], x, height-1-y);
			//bits[x].alpha = 128;
		}
	}
	string output = path + "\/output.hdr";
	FreeImage_Save(FIF_HDR, final_hdr, output.c_str(), 0);
	//ToneMapping tonemapping = ToneMapping(width, height, KEY);
	//for(int i=0 ; i<3 ; i++)
	//	tonemapping.global_operator(exposure[i]);
	/*for(int ch=0 ; ch<3 ; ch++){
		for(int y=0 ; y<height ; y++){
			for(int x=0 ; x<width ; x++){
				cvmSet(exposure[ch], x, y, pow(cvmGet(exposure[ch], x, y), GAMMA));
			}
		}
	}*/
	

#ifndef DEBUG
	
	IplImage *hdr_img = cvCreateImage(cvSize(width, height), IPL_DEPTH_32F, 3);
	IplImage *tone_img = cvCreateImage(cvSize(width, height), IPL_DEPTH_32F, 3);
	cvmat2ipl(width, height, exposure, hdr_img);
	BilateralTonemapping bf(width, height, BF_SCALE);
	bf.tonemapping(hdr_img, tone_img);
	
	IplImage *final = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);
	bf.trans_to_8u(tone_img, final);
	cout << "done!!! " << endl;
	output = path + "\/output.jpg";
	cvSaveImage(output.c_str(), final);
	
	/*
	IplImage *final = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);
	hdr.trans_to_image(exposure, final);
	cout << "done!!! " << endl;
	cvSaveImage("in.jpg", final);*/
	
#endif
	system("pause");

}