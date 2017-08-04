#include "helpers.cuh"
#include <host_defines.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

struct BSpline2d
{
	float * spline;
	int size;
	float sum;
//	BSpline2d(float * spline,int size) :spline(spline),size(size){}
//	BSpline2d() :spline(0), size(0) {}
};


struct BitpamData
{
	int bmpSize;
	int splineSize;
};
extern __constant__ BSpline2d dSplines;

float* generateBitmapRightSide(char* bpmPath, int elements);
