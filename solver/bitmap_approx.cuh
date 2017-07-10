#include "helpers.cuh"
#include <host_defines.h>


struct BSplines
{
	//integrals
	float * left;
	float * middle;
	float * right;
	float * points;
	float elemSpan;
	float pointSpan;
	int count;
};

struct BSPline2d
{
	float * spline;
	int size;
	BSPline2d(float * spline,int size) :spline(spline),size(size){}
};

extern __constant__ BSplines dSplines;

BSplines generateBSplineIntegralsEvenly(int count, int elements);

struct BitmapStripes
{
	float * horizontalStripes;
	float * verticalStripes;
	int stripeSize;
	BitmapStripes(float *horizontalStripes,float *verticalStripes,int stripeSize)
	{
		this->horizontalStripes = horizontalStripes;
		this->verticalStripes = verticalStripes;
		this->stripeSize = stripeSize;
	}
};


BitmapStripes readBmp(char* filename, BSplines splines);