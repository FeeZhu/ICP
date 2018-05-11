#pragma once

#include <string>
#include <iostream>

#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/console/time.h>   // TicToc
#include <omp.h>

//#include "kdTree.cpp"

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

struct Iter_para //Interation paraments
{
	int ControlN;//控制点个数
	int Maxiterate;//最大迭代次数
	double threshold;//阈值
	double acceptrate;//接收率

};

double distance(Eigen::Vector3d x, Eigen::Vector3d y)
{
	return (x - y).norm();
}

/******************************************************
函数描述：计算两个点云之间最近点的距离误差，没有使用高级的数据结构
输入参数：cloud_target目标点云矩阵，cloud_source原始点云矩阵
输出参数：error 最近点距离误差和误差函数的值,ConQ与P对应的控制点矩阵
********************************************************/
double FindClosest(const Eigen::MatrixXd  cloud_target,
	const Eigen::MatrixXd cloud_source, Eigen::MatrixXd &ConQ)
{
	double error = 0.0;
	int *Index = new int[cloud_target.cols()];
	//时间复杂度爆炸,需要使用kd-tree这种结构
#pragma omp parallel for
	for (int i = 0; i < cloud_target.cols(); i++)
	{
		double maxdist = 100.0;
		for (int j = 0; j < cloud_source.cols(); j++)
		{
			double dist = distance(cloud_target.col(i), cloud_source.col(j));
			if (dist<maxdist)
			{
				maxdist = dist;
				Index[i] = j;
			}
		}
		Eigen::Vector3d v = cloud_source.col(Index[i]);
		ConQ.col(i) = v;
		error += maxdist;
	}
	return error;
}

/******************************************************
函数描述：求两个点云之间的变换矩阵
输入参数：ConP目标点云控制点3*N，ConQ原始点云控制点3*N
输出参数：transformation_matrix点云之间变换参数4*4
********************************************************/
Eigen::Matrix4d GetTransform(const Eigen::MatrixXd ConP, const Eigen::MatrixXd ConQ)
{
	int nsize = ConP.cols();
	//求点云中心并移到中心点
	Eigen::VectorXd MeanP = ConP.rowwise().mean();
	Eigen::VectorXd MeanQ = ConQ.rowwise().mean();
	//cout << MeanP<< MeanQ<<endl;
	Eigen::MatrixXd ReP = ConP.colwise() - MeanP;
	Eigen::MatrixXd ReQ  = ConQ.colwise() - MeanQ;
	//求解旋转矩阵
	//Eigen::MatrixXd H = ReQ*(ReP.transpose());
	Eigen::MatrixXd H = ReP*(ReQ.transpose());
	Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeThinU | Eigen::ComputeThinV);
	Eigen::Matrix3d U = svd.matrixU();
	Eigen::Matrix3d V = svd.matrixV();
	float det = (U * V.transpose()).determinant();
	Eigen::Vector3d diagVec(1.0, 1.0, det);
	Eigen::MatrixXd R = V * diagVec.asDiagonal() * U.transpose();
	//Eigen::MatrixXd R = H*((ReP*(ReP.transpose())).inverse());
	//求解平移向量
	Eigen::VectorXd T = MeanQ - R*MeanP;

	Eigen::MatrixXd Transmatrix = Eigen::Matrix4d::Identity();
	Transmatrix.block(0, 0, 3, 3) = R; 
	Transmatrix.block(0, 3, 3, 1) = T;
	cout << Transmatrix << endl;
	return Transmatrix;
}

/******************************************************
函数描述：点云变换
输入参数：ConP点云控制点3*N，transformation_matrix点云之间变换参数4*4
输出参数：NewP新的点云控制点3*N
********************************************************/
Eigen::MatrixXd Transform(const Eigen::MatrixXd ConP, const Eigen::MatrixXd Transmatrix)
{
	Eigen::initParallel();
	Eigen::MatrixXd R = Transmatrix.block(0, 0, 3, 3);
	Eigen::VectorXd T = Transmatrix.block(0, 3, 3, 1);
	
	Eigen::MatrixXd NewP = (R*ConP).colwise() + T;
	return NewP;
}

/*点云数据储存于数组中*/
Eigen::MatrixXd cloud2data(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
	int nsize = cloud->size();
	Eigen::MatrixXd Q(3, nsize);
	Eigen::initParallel();
	omp_set_num_threads(4);
#pragma omp parallel for
	for (int i = 0; i < nsize; i++) {
			Q(0, i) = cloud->points[i].x;
			Q(1, i) = cloud->points[i].y;
			Q(2, i) = cloud->points[i].z;
		}
	return Q;
}


/******************************************************
函数功能：ICP算法，计算两个点云之间的转换关系
输入参数：cloud_target目标点云，cloud_source原始点云
        Iter迭代参数
输出参数：transformation_matrix 转换参数
********************************************************/
void icp(const PointCloudT::Ptr cloud_target,
	const PointCloudT::Ptr cloud_source,
	const Iter_para Iter, Eigen::Matrix4d &transformation_matrix)
{
	//数组存储
	int nP = cloud_target->size();
	int nQ = cloud_source->size();
	Eigen::MatrixXd P = cloud2data(cloud_target);
	Eigen::MatrixXd Q = cloud2data(cloud_source);

	//寻找点云中心
	//迭代求解
	int i = 1;
	Eigen::MatrixXd ConP = P;
	Eigen::MatrixXd ConQ = Q;
	while(i<Iter.Maxiterate)
	{
		//1.寻找P中点在Q中距离最近的点
		double error=FindClosest(ConP, Q, ConQ);
		//2.求解对应的刚体变换
		transformation_matrix = GetTransform(ConP, ConQ);
		//3.对P做变换得到新的点云
		ConP = Transform(ConP, transformation_matrix);
		//4.迭代上述过程直至收敛
		if (abs(error)<Iter.ControlN*Iter.acceptrate*Iter.threshold)//80%点误差小于0.01
		{
			break;
		}
		i++;
	}
	//transformation_matrix = Transform(P, Q);
	transformation_matrix = GetTransform(P,ConP);
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

PointCloudT::Ptr ReadFile(std::string FileName)
{

	PointCloudT::Ptr cloud(new PointCloudT);
	if (pcl::io::loadPLYFile(FileName, *cloud) < 0)
	{
		PCL_ERROR("Error loading cloud %s.\n", FileName);
	}
	return cloud;
}