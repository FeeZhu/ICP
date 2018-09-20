#pragma once
#include<time.h>//时间相关头文件，可用其中函数计算图像处理速度    
#include <iostream>  
#include <vector>
#include <Eigen/Dense>
#define N 35947
//#define N 761
#define M_PI 3.1415926

struct Iter_para //Interation paraments
{
	int ControlN;//控制点个数
	int Maxiterate;//最大迭代次数
	double threshold;//阈值
	double acceptrate;//接收率

};

using namespace std;
using Eigen::Map;

void icp(const Eigen::MatrixXd cloud_target,
	const Eigen::MatrixXd cloud_source,
	const Iter_para Iter, Eigen::Matrix4d &transformation_matrix);
void Getinfo();