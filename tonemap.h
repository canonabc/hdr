#include "bilateral.h"
using namespace std;

class ToneMapping{
public:
	ToneMapping(int w, int h, double a){
		width = w;
		height = h;
		key = a;
	}
	void global_operator(CvMat *E){
		double Lbar = 0.0;
		double max = -1;
		for(int y=0 ; y<height ; y++){
			for(int x=0 ; x<width ; x++){
				if(cvmGet(E, x, y) > max)
					max = cvmGet(E, x, y);
				Lbar += log(0.0000001 + cvmGet(E, x, y));
			}
		}
		Lbar /= (width*height);
		Lbar = exp(Lbar);
		double Lwhite = max*key/Lbar;
		for(int y=0 ; y<height ; y++){
			for(int x=0 ; x<width ; x++){
				double Lm = cvmGet(E, x, y)*key/Lbar;
				double Ld = Lm*(1 + (Lm/Lwhite/Lwhite))/(1.0+Lm);
				cvmSet(E, x, y, Ld);
			}
		}
	}
private:
	int width;
	int height;
	double key;
};

class BilateralTonemapping{
public:
	BilateralTonemapping(int w, int h, double a){
		width = w;
		height = h;
		key = a;
	}
	void tonemapping(IplImage *src, IplImage *tar){
		IplImage *yuv_image = cvCreateImage(cvGetSize(src), src->depth, src->nChannels);
		IplImage *yuv_image2 = cvCreateImage(cvGetSize(src), src->depth, src->nChannels);
		//cvCvtColor(src, yuv_image, CV_BGR2YCrCb);
		//cvCvtColor(src, yuv_image2, CV_BGR2YCrCb);
		double logmin = 10000000;
		double logmax = -10000000;
		for(int y=0 ; y<height ; y++){
			for(int x=0 ; x<width ; x++){
				CvScalar scalar = cvGet2D(src, y, x);
				CvScalar tmp;
				tmp.val[0] = ((scalar.val[0]+scalar.val[1]*40+scalar.val[2]*20)/61);
				cvSet2D(yuv_image, y, x, tmp);
				//CvScalar scalar2 = cvGet2D(yuv_image2, y, x);
				//scalar2.val[0] = log(scalar2.val[0]);
				cvSet2D(yuv_image2, y, x, tmp);
				if(scalar.val[0] > logmax)
					logmax = scalar.val[0];
				if(scalar.val[0] < logmin)
					logmin = scalar.val[0];
			}
		}
		key = log(70.0)/log(logmax/logmin);
		FastBilateral fastbilateral(32, 16, 16, 8);
		fastbilateral.filter(yuv_image, BGR); // filter done
		cout << "Bilateral filter done! Begining tonemapping." << endl;
		for(int y=0 ; y<height ; y++){
			for(int x=0 ; x<width ; x++){
				CvScalar bfscalar = cvGet2D(yuv_image, y, x);
				CvScalar origscalar = cvGet2D(yuv_image2, y, x);
				double logdetail = log(origscalar.val[0]) - log(bfscalar.val[0]);
				double out = exp((log(bfscalar.val[0])*key + logdetail - log(logmax)*key));
				CvScalar tar_s;
				CvScalar src_s = cvGet2D(src, y, x);;
				for(int ch=0 ; ch<3 ; ch++){
					tar_s.val[ch] = src_s.val[ch]/(origscalar.val[0])*out;
				}
				cvSet2D(tar, y, x, tar_s);
			}
		}
		//cvCvtColor(yuv_image2, tar, CV_YCrCb2BGR);
	}

	void trans_to_8u(IplImage *src, IplImage *img){
		for(int ch=0 ; ch<3 ; ch++){
			double max = -1;
			double min = 10000;
			for(int y=0 ; y<height ; y++){
				for(int x=0 ; x<width ; x++){
					CvScalar s = cvGet2D(src, y, x);
					double intensity = s.val[ch];
					if(intensity>max)
						max = intensity;
					if(intensity<min)
						min = intensity;
				}
			}
			for(int y=0 ; y<height ; y++){
				for(int x=0 ; x<width ; x++){
					CvScalar target = cvGet2D(img, y, x);
					CvScalar s = cvGet2D(src, y, x);
					double intensity = s.val[ch];
					target.val[ch] = unsigned int((intensity-min)/(max-min)*255);
					cvSet2D(img, y, x, target);
				}
			}
		}
	}
private:
	int width;
	int height;
	double key;
};

