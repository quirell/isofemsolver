#include "helpers.cuh"
#include <host_defines.h>


struct BSpline2d
{
	float * spline;
	int size;
	BSpline2d(float * spline,int size) :spline(spline),size(size){}
};

extern __constant__ BSpline2d dSplines;
