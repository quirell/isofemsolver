#include "helpers.cuh"
#include <host_defines.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <pplinterface.h>

struct BSpline2d
{
	float * spline;
	int size;
	float sum;
	double sump[3][3];
//	BSpline2d(float * spline,int size) :spline(spline),size(size){}
//	BSpline2d() :spline(0), size(0) {}
};


struct BitpamData
{
	int bmpSize;
	int splineSize;
};
extern __constant__ BSpline2d dSplines;

float* generateBitmapRightSide(char* bpmPath, int elements, BSpline2d * outBSpline = nullptr);

void measureGenBitmap(char* bmpPath, int elements, int iters);

BSpline2d generateTestBSplineIntegrals(int pixels, int elements);

BSpline2d generate2DSplineIntegrals(int pixels, int elements);

float* generateBitmapLeftSide(BSpline2d bSplines, int elements);

float * getBitmapApprox(float * solution, int elements, int resolution);