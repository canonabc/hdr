#include <iostream>
#include <cv.h>
#include <highgui.h>
#include <cmath>
#include <ctime>
#define LAB 0
#define BGR 1
using namespace std;
class Bilateral{
	public:
		Bilateral(double sigmas, double sigmar){
			_sigmas = sigmas;
			_sigmar = sigmar;
		};
		virtual void filter(IplImage *img, int color_space) = 0;
	protected:
		double _sigmas, _sigmar;
		double gaussian(double x, double mean, double sigma){
			double normalize = double(x-mean)/sigma;
			return exp(-0.5 * normalize*normalize);
		}
		double gaussian(int x, int y, int meanx, int meany, double sigma){
			double normalize_x = double(x-meanx)/sigma;
			double normalize_y = double(y-meany)/sigma;
			return exp(-0.5*(normalize_x*normalize_x + normalize_y*normalize_y));
		}
};

inline int rounding(double num)
{
	return num+0.5 > int(num)+1 ? int(num)+1 : int(num); 
}

struct Channel{
	Channel(){
		for(int i=0 ; i<3 ; i++)
			color[i] = 0.0;
	}
	double color[3];
};


class FastBilateral : public Bilateral{
	public:
		FastBilateral(double sigmas, double sigmar, double rates, double rater) : Bilateral(sigmas, sigmar){
			_rates = rates;
			_rater = rater;
		}
		void filter(IplImage *img, int color_space){
			if(color_space==LAB)
				cvCvtColor(img, img, CV_BGR2Lab);
			int height = rounding((double)img->height/_rates)+2;
			int width = rounding((double)img->width/_rates)+2;
			int depth = rounding(256.0/_rater)+1;
			Channel ***wi = new Channel**[height];
			Channel ***w = new Channel**[height];
			Channel ***conv_wi = new Channel**[height];
			Channel ***conv_w = new Channel**[height];
			for(int i=0 ; i<height ; i++){    //Initialization
				wi[i] = new Channel*[width];
				w[i] = new Channel*[width];
				conv_wi[i] = new Channel*[width];
				conv_w[i] = new Channel*[width];
				for(int j=0 ; j<width ; j++){
					wi[i][j] = new Channel[depth];
					w[i][j] = new Channel[depth];
					conv_wi[i][j] = new Channel[depth];
					conv_w[i][j] = new Channel[depth];
					for(int k=0 ; k<depth ; k++){
						wi[i][j][k] = Channel();
						w[i][j][k] = Channel();
						conv_wi[i][j][k] = Channel();
						conv_w[i][j][k] = Channel();
					}
				}
			}
			update_downsample_space(wi, w, img);
			convolution(wi, w, conv_wi, conv_w, height, width, depth);
			upsampling(img, conv_wi, conv_w, height, width, depth);
			if(color_space==LAB)
				cvCvtColor(img, img, CV_Lab2BGR);
		}
	private:
		double _rates, _rater;
		double gaussian(int x, int y, int z, int meanx, int meany, int meanz, double sigmas, double sigmar){
			double normalize_x = double(x-meanx)/sigmas;
			double normalize_y = double(y-meany)/sigmas;
			double normalize_z = double(z-meanz)/sigmar;
			return exp(-0.5*(normalize_x*normalize_x + normalize_y*normalize_y + normalize_z*normalize_z));
		}
		void update_downsample_space(Channel ***wi, Channel ***w, IplImage *img){
			for(int i=0 ; i<img->height ; i++){
				for(int j=0 ; j<img->width ; j++){
					CvScalar s = cvGet2D(img, i, j);
					int x = rounding(i/_rates);
					int y = rounding(j/_rates);
					for(int channel=0 ; channel<3 ; channel++){	
						int delta = rounding(s.val[channel]/_rater);
						wi[x][y][delta].color[channel] += s.val[channel];
						w[x][y][delta].color[channel] += 1.0;
					}
				}
			}
			cerr << "update done" << endl;
		}
		void convolution(Channel ***wi, Channel ***w, Channel ***conv_wi, Channel ***conv_w, int height, int width, int depth){
			double sigmas = _sigmas/_rates;
			double sigmar = _sigmar/_rater;
			double range = 2;
			double ***table = new double **[2*int(sigmas)+2];
			for(int i=0 ; i<2*int(sigmas)+2 ; i++){
				table[i] = new double *[2*int(sigmas)+2];
				for(int j=0 ; j<2*int(sigmas)+2 ; j++)
					table[i][j] = new double [2*int(sigmar)+2];
			}
			for(int i=0 ; i<2*int(sigmas)+2 ; i++)
				for(int j=0 ; j<2*int(sigmas)+2 ; j++)
					for(int k=0 ; k<2*int(sigmar)+2 ; k++)
						table[i][j][k] = gaussian(i, j, k, 0, 0, 0, sigmas, sigmar);
			cout << "table done..." << endl;
			int count[3] = {0};
			for(int y=0 ; y<height ; y++){
				cerr <<  "row = "  << y << "///" << height << ", " << width << ", " << depth << endl;
				for(int x=0 ; x<width ; x++){
					for(int z=0 ; z<depth ; z++){
						one_pixel_conv(wi, w, conv_wi, conv_w, y, x, z, height, width, depth, sigmas, sigmar, range, table);
					}
				}
			}
		}
		void one_pixel_conv(Channel ***wi, Channel ***w, Channel ***conv_wi, Channel ***conv_w, 
							int y, int x, int z, int height, int width, int depth, 
							double sigmas, double sigmar, double range, double ***table){
			int down = y - range*sigmas < 0 ? 0 : int(y - range*sigmas);
			int up = y + range*sigmas > height-1 ? height-1 : int(y + range*sigmas);
			int left = x - range*sigmas < 0 ? 0 : int(x - range*sigmas);
			int right = x + range*sigmas > width-1 ? width-1 : int(x + range*sigmas);
			int low = z - range*sigmar < 0 ? 0 : int(z - range*sigmar);
			int high = z + range*sigmar > depth-1 ? depth-1 : int(z + range*sigmar);
			double conv_pixel_wi[3] = {0.0}, conv_pixel_w[3] = {0.0};
			// convolve for one pixel
			for(int i=down ; i<=up ; i++){
				for(int j=left ; j<=right ; j++){
					for(int k=low ; k<=high ; k++){
						if(wi[i][j][k].color[0]+wi[i][j][k].color[1]+wi[i][j][k].color[2] < 0.1)
							continue;
						double weight = table[abs(i-y)][abs(j-x)][abs(k-z)];
						for(int channel=0 ; channel<3 ; channel++){
							if(wi[i][j][k].color[channel] < 0.1)
								continue;
							conv_pixel_wi[channel] += (weight*wi[i][j][k].color[channel]);
							conv_pixel_w[channel] += (weight*w[i][j][k].color[channel]);
						}
					}
				}
			}
			// update
			for(int i=0 ; i<3 ; i++){
				conv_wi[y][x][z].color[i] = conv_pixel_wi[i];
				conv_w[y][x][z].color[i] = conv_pixel_w[i];
			}
		}

		void upsampling(IplImage *img, Channel ***wi, Channel ***w, int height, int width, int depth)
		{
			for(int i=0 ; i<img->height ; i++){
				for(int j=0 ; j<img->width ; j++){
					CvScalar s = cvGet2D(img, i, j);
					for(int channel=0 ; channel<3 ; channel++){
						double x = (double)i/_rates;
						double y = (double)j/_rates;
						double z = s.val[channel]/_rater;
						int qx = int(x), qy = int(y), qz = int(z);
						double dx = x-qx, dy = y-qy, dz = z-qz;
						double c00 = wi[qx][qy][qz].color[channel]*(1-dx) + wi[qx+1][qy][qz].color[channel]*dx;
						double c10 = wi[qx][qy+1][qz].color[channel]*(1-dx) + wi[qx+1][qy+1][qz].color[channel]*dx;
						double c01 = wi[qx][qy][qz+1].color[channel]*(1-dx) + wi[qx+1][qy][qz+1].color[channel]*dx;
						double c11 = wi[qx][qy+1][qz+1].color[channel]*(1-dx) + wi[qx+1][qy+1][qz+1].color[channel]*dx;
						double c0 = c00*(1-dy) + c10*dy;
						double c1 = c01*(1-dy) + c11*dy;
						s.val[channel] = c0*(1-dz) + c1*dz;
						c00 = w[qx][qy][qz].color[channel]*(1-dx) + w[qx+1][qy][qz].color[channel]*dx;
						c10 = w[qx][qy+1][qz].color[channel]*(1-dx) + w[qx+1][qy+1][qz].color[channel]*dx;
						c01 = w[qx][qy][qz+1].color[channel]*(1-dx) + w[qx+1][qy][qz+1].color[channel]*dx;
						c11 = w[qx][qy+1][qz+1].color[channel]*(1-dx) + w[qx+1][qy+1][qz+1].color[channel]*dx;
						c0 = c00*(1-dy) + c10*dy;
						c1 = c01*(1-dy) + c11*dy;
						s.val[channel] = s.val[channel]/(c0*(1-dz) + c1*dz);
					}
					cvSet2D(img, i, j, s);
				}
			}
		}
};