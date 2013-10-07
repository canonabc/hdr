#include <cmath>
#include <cv.h>
using namespace std;

class Robertson{
public:
	Robertson(IplImage **img, double *t, int num){
		images = img;
		time = t;
		num_images = num;
		w = images[0]->width;
		h = images[0]->height;
		weight_table = new double[256];
		s = 50;
		mean = 127;
		for(int i=0 ; i<256 ; i++){
			weight_table[i] = exp(-(double(i-mean)*double(i-mean)/s/s));
		}
	}
	void update_parameter(CvMat **E, double **G){
#pragma omp parallel for
		for(int ch=0 ; ch<3 ; ch++){
			// update G function
			long int count[256] = {0};
			double sum_up[256] = {0.0};
			for(int i=0 ; i<num_images ; i++){
				for(int y=0 ; y<h ; y++){
					for(int x=0 ; x<w ; x++){
						CvScalar scalar = cvGet2D(images[i], y, x);
						int intensity = scalar.val[ch];
						assert(0<=intensity && intensity<=255);
						sum_up[intensity] += (cvmGet(E[ch], x, y)*time[i]);
						count[intensity]++;
					}
				}
			}
			for(int i=0 ; i<256 ; i++)
				G[ch][i] = sum_up[i]/(double)count[i]*(double)count[mean]/sum_up[mean];			
			// update exposure
			for(int y=0 ; y<h ; y++){
				for(int x=0 ; x<w ; x++){
					double sum_up = 0.0;
					double sum_down = 0.0;
					for(int i=0 ; i<num_images ; i++){
						CvScalar scalar = cvGet2D(images[i], y, x);
						int intensity = scalar.val[ch];
						assert(0<=intensity && intensity<=255);
						double weight = weight_table[intensity];
						sum_up += (weight*G[ch][intensity]*time[i]);
						sum_down += (weight*time[i]*time[i]);
					}
					
					cvmSet(E[ch], x, y, (sum_up/sum_down));
				}
			}
			
			cout << "channel " << ch << " done "<< endl;
		}
	}
	void trans_to_image(CvMat **E, IplImage *img){
		for(int ch=0 ; ch<3 ; ch++){
			double max = -1;
			double min = 10000;
			for(int y=0 ; y<h ; y++){
				for(int x=0 ; x<w ; x++){
					double intensity = cvmGet(E[ch], x, y);
					if(intensity>max)
						max = intensity;
					if(intensity<min)
						min = intensity;
				}
			}
			for(int y=0 ; y<h ; y++){
				for(int x=0 ; x<w ; x++){
					double intensity = cvmGet(E[ch], x, y);
					CvScalar scalar = cvGet2D(img, y, x);
					scalar.val[ch] = unsigned int((intensity-min)/(max-min)*255);
					cvSet2D(img, y, x, scalar);
				}
			}
		}
	}
	void trans_to_hdr(CvMat **E, IplImage *img){
		for(int y=0 ; y<h ; y++){
			for(int x=0 ; x<w ; x++){
				CvScalar scalar;
				for(int ch=0 ; ch<3 ; ch++){
					scalar.val[ch] = cvmGet(E[ch], x, y);
				}
				cvSet2D(img, y, x, scalar);
			}
		}
	}

	

private:
	IplImage **images;
	double *time;
	int num_images;
	int w, h;
	double *weight_table;
	double s;
	int mean;
};
