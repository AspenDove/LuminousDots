#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <random>
//#include <iostream>
//#include <algorithm>
//#include <ctype.h>
#include "tbb/tbb.h"
#include "tbb/task_scheduler_init.h"
//
using namespace tbb;
using namespace cv;
using namespace std;

const char* szWinName = "PerlinNoise";

auto f = [](double t) {return t * t*t*(t*(t * 6 - 15) + 10); };
double Interpolate(double a, double b, double t, std::function <double(double)> f)
{
	return a * (1 - f(t)) + b * f(t);
}

Mat rotation(float theta)
{
	return Mat_<float>(2, 2) <<
		cos(theta), -sin(theta),
		sin(theta), +cos(theta);
}

void PeilinNoise2D(Mat& img, const vector<vector<Vec2f>>& GridGradient, const size_t& ngird)
{
	for (size_t x = 0; x != ngird; ++x)
	{
		for (int ix = 0; ix < (img.cols + ngird / 2.) / ngird; ix++)
		{
			const float tx = float(ix)*ngird / float(img.cols);
			for (size_t y = 0; y != ngird; ++y)
			{
				Vec2f g00 = GridGradient[y][x];
				Vec2f g01 = GridGradient[y][x + 1];
				Vec2f g10 = GridGradient[y + 1][x];
				Vec2f g11 = GridGradient[y + 1][x + 1];
				for (int iy = 0; iy < (img.rows + ngird / 2) / ngird; iy++)
				{
					const float ty = float(iy)*ngird / float(img.rows);
					float n00 = g00.dot(Mat{ tx,ty });
					float n10 = g01.dot(Mat{ tx - 1,ty });
					float n01 = g10.dot(Mat{ tx,ty - 1 });
					float n11 = g11.dot(Mat{ tx - 1,ty - 1 });
					float n0 = Interpolate(n00, n10, tx, f);
					float n1 = Interpolate(n01, n11, tx, f);
					img.at<float>(y* img.rows / ngird + iy, x* img.cols / ngird + ix) = abs((float)Interpolate(n0, n1, ty, f));
				}
			}
		}
	}
}

//int main(int argc, char** argv)
//{
//	size_t width = 800, height = 600;
//	size_t ngird = 10;
//	std::default_random_engine generator(time(nullptr));
//	std::uniform_real_distribution<double> distribution(-1.0, 1.0);
//
//	//task_scheduler_init init;
//	namedWindow(szWinName);
//
//	vector<vector<Vec2f>> GridGradient(ngird + 1);
//
//	for (auto &row : GridGradient)
//	{
//		row.resize(ngird + 1);
//		for (auto &i : row)
//		{
//			float nx = distribution(generator);
//			float ny = distribution(generator);
//			i = Vec2f(nx, ny) / sqrt(nx *nx + ny * ny);
//		}
//	}
//
//	float theta = 0;
//	for(;;)
//	{
//		theta += 0.05;
//		//	size_t Ndata = 200;
//		Mat rot = rotation(theta);
//		Mat img = Mat::zeros(Size(width, height), CV_32FC1);
//		//Mat sum = Mat::zeros(Size(width, height), CV_32FC1);
//		//for (int i = 0; i != 5; ++i)
//		//{
//		//	//Mat img = Mat::zeros(Size(width, height), CV_32FC1);
//		//	
//		//	ngird = pow(2, i) * 3;
//		//	vector<vector<Vec2f>> GridGradient(ngird + 1);
//
//		//	for (auto &row : GridGradient)
//		//	{
//		//		row.resize(ngird + 1);
//		//		for (auto &i : row)
//		//		{
//		//			float nx = distribution(generator);
//		//			float ny = distribution(generator);
//		//			i = Vec2f(nx, ny) / sqrt(nx *nx + ny * ny);
//		//		}
//		//	}
//
//		//	PeilinNoise2D(img, GridGradient, ngird);
//		//	//convertScaleAbs(img, img);
//		//	sum += img / ngird;
//		//}
//		//PeilinNoise2D(img, GridGradient, ngird);
//
//		
//		parallel_for(blocked_range<size_t>(0, GridGradient[0].size() - 1, 1),
//			[&](const blocked_range<size_t> &r)
//		{
//			for(size_t x = r.begin(); x != r.end(); ++x)
//			{
//				for(int ix = 0; ix < ngird; ix++)
//				{
//					const float tx = float(ix) / float(ngird);
//					parallel_for(blocked_range<size_t>(0, GridGradient.size() - 1, 1),
//						[&](const blocked_range<size_t> &r)
//					{
//						for(size_t y = r.begin(); y != r.end(); ++y)
//						{
//							//for(int y = 0; y != GridGradient.size() - 1; ++y)
//						//	{
//							Mat g00 = rot*GridGradient[y][x];
//							Mat g01 = rot*GridGradient[y][x + 1];
//							Mat g10 = rot*GridGradient[y + 1][x];
//							Mat g11 = rot*GridGradient[y + 1][x + 1];
//							for(int iy = 0; iy < ngird; iy++)
//							{
//								const float ty = float(iy) / float(ngird);
//								float n00 = g00.dot(Mat{ tx,ty });
//								float n10 = g01.dot(Mat{ tx - 1,ty });
//								float n01 = g10.dot(Mat{ tx,ty - 1 });
//								float n11 = g11.dot(Mat{ tx - 1,ty - 1 });
//								float n0 = Interpolate(n00, n10, tx, f);
//								float n1 = Interpolate(n01, n11, tx, f);
//								img.at<float>(y*ngird + iy, x*ngird + ix) = (float)Interpolate(n0, n1, ty, f);// 255 * min(max(Interpolate(n0, n1, ty, f), 0.0), 1.0);
//							}
//						}
//					});
//				}
//			}
//		});
//		
//		/*for(int x = 0; x != GridGradient[0].size() - 1; ++x)
//		{
//		}*/
//		//	img.at<uchar>(x, y) = 255;
//		//img = sum;
//		normalize(img, img, 0.0, 1.0, NORM_MINMAX);
//
//		img = (img * 255);
//		img.convertTo(img, CV_8UC1);
//		applyColorMap(img, img, COLORMAP_RAINBOW);
//
//		imshow(szWinName, img);
//		char c = (char)waitKey(1);
//	}
//
//	return 0;
//}

int main(int argc, char** argv)
{
	size_t width = 800, height = 600;
	size_t ngird = 50;
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution(-1.0, 1.0);

	task_scheduler_init init;
	namedWindow(szWinName);

	vector<vector<Vec2f>> GridGradient(height / ngird + 1);
	for (auto &row : GridGradient)
	{
		row.resize(width / ngird + 1);
		for (auto &i : row)
		{
			float nx = distribution(generator);
			float ny = distribution(generator);
			i = Vec2f(nx, ny) / sqrt(nx *nx + ny * ny);
		}
	}
	parallel_for(blocked_range<size_t>(0, GridGradient[0].size() - 1, 1),
		[](const blocked_range<size_t> &r)
	{
		for (size_t x = r.begin(); x != r.end(); ++x)
		{

		}
	});
	float theta = 0;
	for (;;)
	{
		theta += 0.05;
		//	size_t Ndata = 200;
		Mat rot = rotation(theta);
		Mat img = Mat::zeros(Size(width, height), CV_32FC1);


		parallel_for(blocked_range<size_t>(0, GridGradient[0].size() - 1, 1),
			[&](const blocked_range<size_t> &r)
		{
			for (size_t x = r.begin(); x != r.end(); ++x)
			{
				for (int ix = 0; ix < ngird; ix++)
				{
					const float tx = float(ix) / float(ngird);
					parallel_for(blocked_range<size_t>(0, GridGradient.size() - 1, 1),
						[&](const blocked_range<size_t> &r)
					{
						for (size_t y = r.begin(); y != r.end(); ++y)
						{

							//for(int y = 0; y != GridGradient.size() - 1; ++y)
						//	{
							Mat g00 = rot * GridGradient[y][x];
							Mat g01 = rot * GridGradient[y][x + 1];
							Mat g10 = rot * GridGradient[y + 1][x];
							Mat g11 = rot * GridGradient[y + 1][x + 1];
							for (int iy = 0; iy < ngird; iy++)
							{
								const float ty = float(iy) / float(ngird);
								float n00 = g00.dot(Mat{ tx,ty });
								float n10 = g01.dot(Mat{ tx - 1,ty });
								float n01 = g10.dot(Mat{ tx,ty - 1 });
								float n11 = g11.dot(Mat{ tx - 1,ty - 1 });
								float n0 = Interpolate(n00, n10, tx, f);
								float n1 = Interpolate(n01, n11, tx, f);
								img.at<float>(y*ngird + iy, x*ngird + ix) = (float)Interpolate(n0, n1, ty, f);// 255 * min(max(Interpolate(n0, n1, ty, f), 0.0), 1.0);
							}
						}
					});
				}
			}
		});

		/*for(int x = 0; x != GridGradient[0].size() - 1; ++x)
		{
		}*/
		//	img.at<uchar>(x, y) = 255;
		normalize(img, img, 0.0, 1.0, NORM_MINMAX);
		//cout << img;
		img = (img * 255);
		img.convertTo(img, CV_8UC1);
		applyColorMap(img, img, COLORMAP_RAINBOW);


		imshow(szWinName, img);
		char c = (char)waitKey(1);
		if (c == 27)
			break;
	}

	return 0;
}