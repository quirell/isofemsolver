#include "bitmap_approx.cuh"
#include <device_launch_parameters.h>

#define THREADS 64
#define COLORS 3

BSpline2d generate2DSplineIntegrals(int pixels,int elements)
{
	int const pxPerElem = pixels / elements;
	int const pxPerSpline = 3 * pxPerElem;
	double const pxSpan = 1.0 / pixels;
	double const elemSpan = 1.0 / elements;
	double * spline = new double[pxPerSpline];
	double x = 0;
	for(int i = 0;i<pxPerElem;i++,x=x+pxSpan)
	{
		spline[i] = (x*x) / (2*elemSpan * elemSpan);
	}
	for (int i = pxPerElem; i<2*pxPerElem; i++, x = x + pxSpan)
	{
		spline[i] = (x*(2 * elemSpan - x) + (3 * elemSpan - x)*(x - elemSpan)) / (2 * elemSpan*elemSpan);
	}
	for (int i = 2*pxPerElem; i<3*pxPerElem; i++, x = x + pxSpan)
	{
		spline[i] = (3 * elemSpan - x) * (3 * elemSpan - x) / (2*elemSpan * elemSpan); 
	}
	float * spline2D = new float[pxPerSpline*pxPerSpline];
	for(int i = 0;i<pxPerSpline;i++)
	{
		for(int j = 0;j<pxPerSpline;j++)
		{
			spline[i*pxPerSpline + j] = spline[i] * spline[j];
		}
	}
	return BSpline2d(spline2D, pxPerSpline);
}

__global__ void computeRightSide(float* dRightSide)
{
	__shared__ float s[64];
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
}

float* generateBitmapRightSide(char* bpmPath, int elements, int integrationPoints)
{
	Bitmap bitmap = readBmp(bpmPath);
	float* dBitmap = nullptr;
	BSpline2d bSplines = generate2DSplineIntegrals(integrationPoints, elements);
	ERRCHECK(cudaMemcpyToSymbol(dSplines, &bSplines, sizeof(BSpline2d)));
	ERRCHECK(cudaMalloc(&dBitmap, sizeof(float)*bitmap.width*bitmap.height));
	ERRCHECK(cudaMemcpy(dBitmap, bitmap.bitmap, sizeof(float)*bitmap.width*bitmap.height, cudaMemcpyHostToDevice));
	float* dRightSide = nullptr;
	ERRCHECK(cudaMalloc(&dRightSide, sizeof(float)*elements*elements))

	return dRightSide;
}
