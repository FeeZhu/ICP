#pragma once
/*
*ICP算法的GPU实现
*/
#include <fstream> 
#include "header.h"

/******************************************************
函数描述：点云变换
输入参数：ConP点云控制点3*N，transformation_matrix点云之间变换参数4*4
输出参数：NewP新的点云控制点3*N
********************************************************/
Eigen::MatrixXd Transform(const Eigen::MatrixXd ConP, const Eigen::MatrixXd Transmatrix)
{
	Eigen::MatrixXd R = Transmatrix.block(0, 0, 3, 3);
	Eigen::VectorXd T = Transmatrix.block(0, 3, 3, 1);

	Eigen::MatrixXd NewP = (R*ConP).colwise() + T;
	return NewP;
}

void print4x4Matrix(const Eigen::Matrix4d & matrix)
{
	printf("Rotation matrix :\n");
	printf("    | %6.3f %6.3f %6.3f | \n", matrix(0, 0), matrix(0, 1), matrix(0, 2));
	printf("R = | %6.3f %6.3f %6.3f | \n", matrix(1, 0), matrix(1, 1), matrix(1, 2));
	printf("    | %6.3f %6.3f %6.3f | \n", matrix(2, 0), matrix(2, 1), matrix(2, 2));
	printf("Translation vector :\n");
	printf("t = < %6.3f, %6.3f, %6.3f >\n\n", matrix(0, 3), matrix(1, 3), matrix(2, 3));
}

Eigen::MatrixXd ReadFile(std::string FileName)
{

	Eigen::MatrixXd cloud(3,N);
	
	std::ifstream fin(FileName);
	if (!fin.is_open())
	{
		cout << "Error:can not open the file!";
		exit(1);
	}
	int i = 0;
	while (!fin.eof())
	{
		fin >> cloud(0,i) >> cloud(1,i) >> cloud(2,i);
		i++;
	}

	return cloud;
}

void main()
{
	// read point cloud from txt file
	Eigen::MatrixXd cloud_in = ReadFile("../data/bunny.txt");
	Eigen::MatrixXd cloud_icp;

	// Defining a rotation matrix and translation vector
	Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity();

	// A rotation matrix (see https://en.wikipedia.org/wiki/Rotation_matrix)
	double theta = M_PI / 8;  // The angle of rotation in radians
	transformation_matrix(0, 0) = cos(theta);
	transformation_matrix(0, 1) = -sin(theta);
	transformation_matrix(1, 0) = sin(theta);
	transformation_matrix(1, 1) = cos(theta);

	// A translation on Z axis (0.4 meters)
	transformation_matrix(2, 3) = 0.2;
	transformation_matrix(1, 3) = 0;
	// Display in terminal the transformation matrix
	std::cout << "Applying this rigid transformation to: cloud_in -> cloud_icp" << std::endl;
	print4x4Matrix(transformation_matrix);

	cloud_icp=Transform(cloud_in, transformation_matrix);
	
	//icp algorithm
	Eigen::Matrix4d matrix_icp;
	Iter_para iter{ N,20,0.001,0.8 };//迭代参数
	Getinfo();
	long begin = clock();//存开始时间    
	icp(cloud_in, cloud_icp, iter, matrix_icp);
	std::cout << "GPU运行时间为: " << int(((double)(clock() - begin)) / CLOCKS_PER_SEC * 1000) << "ms " << std::endl;
	//cout << matrix_icp << endl;
	getchar();
}
