#include "BiCubic_Interpolation.h"

BiCubic_Interpolation::BiCubic_Interpolation(double a)
{
	this->a = a;
}

double BiCubic_Interpolation::BiCubic(double x)
{
	if (abs(x) <= 1.0)
		return (a + 2) * pow(abs(x), 3) - (a + 2) * pow(abs(x), 2) + 1;
	else if (abs(x) < 2.0)
		return a * pow(abs(x), 3) - 5 * a * pow(abs(x), 2) + 8 * a * abs(x) - 4 * a;
	else
		return 0.0;
}

void BiCubic_Interpolation::Run(const Mat& _img, int Mul)
{
	//获取新图像的长与宽
	int iTotalColsNew = _img.cols * Mul;
	int iTotalRowsNew = _img.rows * Mul;
	//创建全0结果图像
	result = Mat::zeros(iTotalRowsNew, iTotalColsNew, CV_8UC3);

	//双三次插值
	//遍历结果图的每一个像素
	for (int i = 0; i < iTotalRowsNew; i++)
	{
		for (int j = 0; j < iTotalColsNew; j++)
		{
			//获取当前像素点在原图中的小数坐标
			double row = 1.0 * i / Mul;
			double col = 1.0 * j / Mul;

			//获取当前像素点在原图中的整数坐标
			int row_ = (int)row;
			int col_ = (int)col;

			//获取提供信息的16个像素点的位置
			int x0 = row_ - 1;
			int y0 = col_ - 1;
			int x1 = row_;
			int y1 = col_;
			int x2 = row_ + 1;
			int y2 = col_ + 1;
			int x3 = row_ + 2;
			int y3 = col_ + 2;
			
			//普通情况
			if ((x0 >= 0) && (x3 < _img.rows) && (y0 >= 0) && (y3 < _img.cols))
			{
				/*
				//分别求出x，y方向对应的BiCubic系数
				double Bic_x0 = BiCubic(row - x0);
				double Bic_x1 = BiCubic(row - x1);
				double Bic_x2 = BiCubic(row - x2);
				double Bic_x3 = BiCubic(row - x3);
				//double Bic_x3 = 1.0 - Bic_x0 - Bic_x1 - Bic_x2;
				double Bic_y0 = BiCubic(col - y0);
				double Bic_y1 = BiCubic(col - y1);
				double Bic_y2 = BiCubic(col - y2);
				double Bic_y3 = BiCubic(col - y3);
				//double Bic_y3 = 1.0 - Bic_y0 - Bic_y1 - Bic_y2;
				*/
				double Bic_x0 = ((a * ((row-row_) + 1) - 5 * a) * ((row-row_) + 1) + 8 * a) * ((row-row_) + 1) - 4 * a;
				double Bic_x1 = ((a + 2) * (row-row_) - (a + 3)) * (row-row_) * (row-row_) + 1;
				double Bic_x2 = ((a + 2) * (1 - (row-row_)) - (a + 3)) * (1 - (row-row_)) * (1 - (row-row_)) + 1;
				double Bic_x3 = 1.f - Bic_x0 - Bic_x1 - Bic_x2;

				double Bic_y0 = ((a * ((col-col_) + 1) - 5 * a) * ((col-col_) + 1) + 8 * a) * ((col-col_) + 1) - 4 * a;
				double Bic_y1 = ((a + 2) * (col-col_) - (a + 3)) * (col-col_) * (col-col_) + 1;
				double Bic_y2 = ((a + 2) * (1 - (col-col_)) - (a + 3)) * (1 - (col-col_)) * (1 - (col-col_)) + 1;
				double Bic_y3 = 1.f - Bic_x0 - Bic_x1 - Bic_x2;

				//（xi，yi）处像素的权重就是xi，yi的系数之积，由此可以乘出相应权重
				double W_x0y0 = Bic_x0 * Bic_y0;
				double W_x0y1 = Bic_x0 * Bic_y1;
				double W_x0y2 = Bic_x0 * Bic_y2;
				double W_x0y3 = Bic_x0 * Bic_y3;
				double W_x1y0 = Bic_x1 * Bic_y0;
				double W_x1y1 = Bic_x1 * Bic_y1;
				double W_x1y2 = Bic_x1 * Bic_y2;
				double W_x1y3 = Bic_x1 * Bic_y3;
				double W_x2y0 = Bic_x2 * Bic_y0;
				double W_x2y1 = Bic_x2 * Bic_y1;
				double W_x2y2 = Bic_x2 * Bic_y2;
				double W_x2y3 = Bic_x2 * Bic_y3;
				double W_x3y0 = Bic_x3 * Bic_y0;
				double W_x3y1 = Bic_x3 * Bic_y1;
				double W_x3y2 = Bic_x3 * Bic_y2;
				double W_x3y3 = Bic_x3 * Bic_y3;

				//为了总亮度不变化，需要标准化权重矩阵
				//double sum = W_x0y0 + W_x0y1 + W_x0y2 + W_x0y3 + W_x1y0 + W_x1y1 + W_x1y2 + W_x1y3 + W_x2y0 + W_x2y1 + W_x2y2 + W_x2y3 + W_x3y0 + W_x3y1 + W_x3y2 + W_x3y3;
				double sum = 1;
				//计算当前像素点的具体值
				result.at<Vec3b>(i, j) = (
					_img.at<Vec3b>(x0, y0) * W_x0y0 / sum +
					_img.at<Vec3b>(x0, y1) * W_x0y1 / sum +
					_img.at<Vec3b>(x0, y2) * W_x0y2 / sum +
					_img.at<Vec3b>(x0, y3) * W_x0y3 / sum +
					_img.at<Vec3b>(x1, y0) * W_x1y0 / sum +
					_img.at<Vec3b>(x1, y1) * W_x1y1 / sum +
					_img.at<Vec3b>(x1, y2) * W_x1y2 / sum +
					_img.at<Vec3b>(x1, y3) * W_x1y3 / sum +
					_img.at<Vec3b>(x2, y0) * W_x2y0 / sum +
					_img.at<Vec3b>(x2, y1) * W_x2y1 / sum +
					_img.at<Vec3b>(x2, y2) * W_x2y2 / sum +
					_img.at<Vec3b>(x2, y3) * W_x2y3 / sum +
					_img.at<Vec3b>(x3, y0) * W_x3y0 / sum +
					_img.at<Vec3b>(x3, y1) * W_x3y1 / sum +
					_img.at<Vec3b>(x3, y2) * W_x3y2 / sum +
					_img.at<Vec3b>(x3, y3) * W_x3y3 / sum);

			}

		}
	}

	
}

Mat3b BiCubic_Interpolation::ShowResult()
{
	return result;
}
