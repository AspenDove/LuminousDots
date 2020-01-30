﻿#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <random>
#include <iostream>
//#include <iostream>
//#include <algorithm>
//#include <ctype.h>
//
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
					img.at<float>(y* img.rows / ngird + iy, x* img.cols / ngird + ix) = (float)Interpolate(n0, n1, ty, f);
				}
			}
		}
	}
}

inline void rounded_rectangle(Mat& src,const Point& topLeft,const Point& bottomRight, const Scalar lineColor, Scalar fillColor,
						const int thickness, const int cornerRadius, const int lineType = LINE_8)
{
	/* corners:
	 * p1 - p2
	 * |     |
	 * p4 - p3
	 */
	Point p1 = topLeft;
	Point p2 = Point(bottomRight.x, topLeft.y);
	Point p3 = bottomRight;
	Point p4 = Point(topLeft.x, bottomRight.y);

	// draw straight lines
	line(src, Point(p1.x + cornerRadius, p1.y), Point(p2.x - cornerRadius, p2.y), lineColor, thickness, lineType);
	line(src, Point(p2.x, p2.y + cornerRadius), Point(p3.x, p3.y - cornerRadius), lineColor, thickness, lineType);
	line(src, Point(p4.x + cornerRadius, p4.y), Point(p3.x - cornerRadius, p3.y), lineColor, thickness, lineType);
	line(src, Point(p1.x, p1.y + cornerRadius), Point(p4.x, p4.y - cornerRadius), lineColor, thickness, lineType);

	// draw arcs
	ellipse(src, p1 + Point(cornerRadius, cornerRadius), Size(cornerRadius, cornerRadius), 180.0, 0, 90, lineColor, thickness, lineType);
	ellipse(src, p2 + Point(-cornerRadius, cornerRadius), Size(cornerRadius, cornerRadius), 270.0, 0, 90, lineColor, thickness, lineType);
	ellipse(src, p3 + Point(-cornerRadius, -cornerRadius), Size(cornerRadius, cornerRadius), 0.0, 0, 90, lineColor, thickness, lineType);
	ellipse(src, p4 + Point(cornerRadius, -cornerRadius), Size(cornerRadius, cornerRadius), 90.0, 0, 90, lineColor, thickness, lineType);

	Point fillFrom(topLeft.x + 10, topLeft.y + 10);
	floodFill(src, fillFrom, fillColor);

	//line(src, Point(p1.x + cornerRadius, p1.y), Point(p2.x - cornerRadius, p2.y), lineColor, thickness, LINE_AA);
	//line(src, Point(p2.x, p2.y + cornerRadius), Point(p3.x, p3.y - cornerRadius), lineColor, thickness, LINE_AA);
	//line(src, Point(p4.x + cornerRadius, p4.y), Point(p3.x - cornerRadius, p3.y), lineColor, thickness, LINE_AA);
	//line(src, Point(p1.x, p1.y + cornerRadius), Point(p4.x, p4.y - cornerRadius), lineColor, thickness, LINE_AA);
}

template <typename T = std::size_t>
constexpr T generate_ith_number(const std::size_t index) {
	static_assert(std::is_integral<T>::value, "T must to be an integral type");

	//if (index == 0) return 2.0;
	return 190 - index * 3;
}

template <std::size_t... Is>
constexpr auto make_sequence_impl(std::index_sequence<Is...>)
{
	return std::index_sequence<generate_ith_number(Is)...>{};
}

template <std::size_t N>
constexpr auto make_sequence()
{
	return make_sequence_impl(std::make_index_sequence<N>{});
}
template <std::size_t... Is>
constexpr auto make_array_from_sequence_impl(std::index_sequence<Is...>)
{
	return std::array<double, sizeof...(Is)>{Is...};
}

template <typename Seq>
constexpr auto make_array_from_sequence(Seq)
{
	return make_array_from_sequence_impl(Seq{});
}
int main(int argc, char** argv)
{
	const size_t width = 800, height = 600;
	const size_t ngird = 8;
	std::default_random_engine generator(time(nullptr));
	std::uniform_real_distribution<double> distribution(-1.0, 1.0);
	vector<vector<Vec2f>> GridGradient(ngird + 1);

	for (auto &row : GridGradient)
	{
		row.resize(ngird + 1);
		for (auto &i : row)
		{
			float nx = distribution(generator);
			float ny = distribution(generator);
			i = Vec2f(nx, ny) / sqrt(nx *nx + ny * ny);
		}
	}

	Mat substrate = Mat::zeros(Size(width, height), CV_32FC1);
	PeilinNoise2D(substrate, GridGradient, ngird);

	normalize(substrate, substrate, 0.0, 1.0, NORM_MINMAX);
	substrate = (substrate*255);
	substrate.convertTo(substrate, CV_8UC1);
	applyColorMap(substrate, substrate, COLORMAP_OCEAN);//COLORMAP_RAINBOW

	//size_t nx = 32, ny = 24;
	const size_t lbox = 20;
	const size_t lgap = 4;
	Mat img = Mat::zeros(Size(width, height), CV_8UC3);

	for(size_t x = 0;x<=width-lbox-lgap;x+=lbox+lgap)
	{
		for(size_t y = 0;y<=height-lbox-lgap;y+=lbox+lgap)
		{
			Point lt(x, y);
			Point rb = lt + Point(lbox, lbox);
			Mat roi = substrate(Rect(lt, rb));
			Scalar color(mean(roi));
			rounded_rectangle(img, lt, rb, color*1.5, color, 2, 5);
		}
	}

	constexpr auto magnification = make_array_from_sequence(make_sequence<30>());

	size_t pos = 0;
	size_t row = 8;
	Mat temp = img.clone();
	for(;;)
	{
		//substrate = img.clone();
		img = temp.clone();

		for (size_t i = 0; i != magnification.size() && pos>=i; ++i)
		{
			if(pos-i>= width / (lbox + lgap))continue;
			Point lt((pos-i)*(lbox+lgap), row*(lbox+lgap));
			Point rb = lt + Point(lbox, lbox);
			Mat roi = substrate(Rect(lt, rb));
			//roi *= magnification[i];
			Scalar color(mean(roi)*magnification[i]/100.f);
			rounded_rectangle(img, lt, rb, color*1.5, color, 2, 5);
		}

		pos++;
		if (pos == width/(lbox+lgap)+magnification.size())pos = 0;

		//GaussianBlur(img, img, Size(3,3), 0.5);
		imshow(szWinName, img);
		waitKey(80);
	}


	return 0;
}

