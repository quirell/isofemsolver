#include "bitmap_approx.cuh"
#include <device_launch_parameters.h>

#define THREADS 64
#define COLORS 3

BSPline2d generate2DSplineIntegrals(int pixels,int elements)
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
	return BSPline2d(spline2D, pxPerSpline);
}
//// integrals are equal in every element, therefore can be precomputed
//BSplines generateBSplineIntegralsEvenly(int count, int elements)
//{
//	BSplines splines;
//	float* memory = new float[count * 4];
//	splines.left = memory;
//	splines.middle = memory + count;
//	splines.right = splines.middle + count;
//	splines.points = splines.right + count;
//	splines.count = count;
//	double elemSpan = 1.0 / elements;
//	double pointSpan = elemSpan / (count + 1);//element split to count intervals
//	splines.elemSpan = elemSpan;
//	splines.pointSpan = pointSpan;
//	//Bspline is spread over 3 elements, so from 0 to 3elemSpan
//	//0--elemSpan--2elemSpan--3elemSpan
//	//evenly distribute points between 0 and elemSpan. 0 and elemSpan are roots of function so obviously = 0
//	for (int i = 0; i < count; i++)
//	{
//		splines.points[i] = pointSpan * (i + 1);
//	}
//	//following is derived from bspline2 formula for interval from 0 to elemSpan, so for size of 1 element
//	for (int i = 0; i < count; i++)
//	{
//		float point = splines.points[i];
//		splines.left[i] = (point * point) / (elemSpan * elemSpan);
//		//		splines.middle[i]= ((splines.points[i]- (-elemSpan))*(elemSpan-splines.points[i]) +
//		//			(2*elemSpan-splines.points[i])*splines.points[i]) / (elemSpan*elemSpan); //the same as below but shortened
//		splines.middle[i] = (elemSpan + point) * (elemSpan + point) - 3 * point * point;
//		splines.right[i] = (elemSpan - point) * (elemSpan - point) / (elemSpan * elemSpan);
//	}
//	return splines;
//}


//BitmapStripes readBmp(char* filename, int elements, BSplines splines)
//{
//	unsigned char* texels;
//	int width, height;
//	FILE* fd;
//	fd = fopen(filename, "rb");
//	if (fd == NULL)
//	{
//		printf("Error: fopen failed\n");
//		throw "Bitmap opening failed";
//	}
//
//	unsigned char header[54];
//
//	// Read header
//	fread(header, sizeof(unsigned char), 54, fd);
//
//	// Capture dimensions
//	width = *(int*)&header[18];
//	height = *(int*)&header[22];
//	if (width != height)
//		throw "Bitmap dimensions must be equal";
//	int padding = 0;
//
//	// Calculate padding
//	while ((width * 3 + padding) % 4 != 0)
//	{
//		padding++;
//	}
//
//	// Compute new width, which includes padding
//	int widthnew = width * 3 + padding;
//
//	//	// Allocate memory to store image data (non-padded)
//	//	texels = (unsigned char *)malloc(width * height * 3 * sizeof(unsigned char));
//	//	if (texels == NULL)
//	//	{
//	//		printf("Error: Malloc failed\n");
//	//		return;
//	//	}
//
//	// Allocate temporary memory to read widthnew size of data
//	int stripeElements = splines.count * elements;
//	float* horizontal = new float[stripeElements];
//	float* vertical = new float[stripeElements];
//
//	unsigned char* data = (unsigned char *)malloc(widthnew * sizeof(unsigned int));
//
//	int elemSpan = width / elements;
//	int* indexesToStore = new int[stripeElements];
//	int index = 0;
//	for (float e = 0; e <= 1; e += splines.elemSpan)
//	{
//		for (int p = 0; p < splines.count; p++)
//		{
//			int j = width * (e + splines.points[p]);
//			indexesToStore[index++] = j;
//		}
//	}
//	//input (bmp stores bitmap upside down)
//	// 3. 3 3 
//	// 2. 2 2
//	// 1. 1 1
//	//so output is upside down
//	// 1. 1 1
//	// 2. 2 2
//	// 3. 3 3
//	// Read row by row of data and skip padded data.
//	//output is vertical and horizontal stripes at the middle of each element
//	int nextHorizontalStripe = elemSpan / 2;
//	int nextHorizontalStripeIndex = 0;
//	int nextVerticalStripeIndex = 0;
//	int nextVerticalStripe = 0;
//	for (int i = height - 1; i > 0; i--)
//	{
//		// Read widthnew length of data
//		fread(data, sizeof(unsigned char), widthnew, fd);
//
//		if (i == nextHorizontalStripe)
//		{
//			nextHorizontalStripe += elemSpan;
//			for (int j = 0; j < stripeElements; j++)
//			{
//				int e = indexesToStore[j];//				R					   G				   B
//				horizontal[nextHorizontalStripeIndex++] = (0.299 * data[e + 2] + 0.587 * data[e + 1] + 0.114 * data[e]) / 255;
//			}
//		}
//		if (indexesToStore[nextVerticalStripe] == i)
//		{
//			for (int e = elemSpan / 2; e < width; e += elemSpan)
//			{
//				vertical[nextVerticalStripeIndex++] = (0.299 * data[e + 2] + 0.587 * data[e + 1] + 0.114 * data[e]) / 255;
//			}
//		}
//	}
//	free(data);
//	fclose(fd);
//	return BitmapStripes(horizontal, vertical, stripeElements);
//}


__device__ float getBitmapPixel(float* dBitmap, int size, float x, float y)
{
	int iy = y * size;
	int ix = x * size;
	return dBitmap[iy * size + ix];
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
	BSplines bSplines = generateBSplineIntegralsEvenly(integrationPoints, elements);
	ERRCHECK(cudaMemcpyToSymbol(dSplines, &bSplines, sizeof(BSplines)));
	ERRCHECK(cudaMalloc(&dBitmap, sizeof(float)*bitmap.width*bitmap.height));
	ERRCHECK(cudaMemcpy(dBitmap, bitmap.bitmap, sizeof(float)*bitmap.width*bitmap.height, cudaMemcpyHostToDevice));
	float* dRightSide = nullptr;
	ERRCHECK(cudaMalloc(&dRightSide, sizeof(float)*elements*elements))

	return dRightSide;
}
