#pragma once
#include "stdax.h"
#include <math.h>

class BiCubic_Interpolation
{
private:
    double a;  // Bicubic函数的系数
    Mat3b result;  // 存储结果图像
public:
    BiCubic_Interpolation(double a);
	double BiCubic(double f);   //用于插值的BuCubic函数
	void Run(const Mat& _img, int Mul);
	Mat3b ShowResult();
};