#include "bitmap_approx.cuh"
#include <emmintrin.h>
#define THREADS 32 //THREADS MUST BE SET TO 32
__constant__ BSpline2d dSplines;

BSpline2d generateTestBSplineIntegrals(int pixels, int elements)
{
	int const pxPerElem = pixels / elements;
	int const pxPerSpline = 3 * pxPerElem;
	double* spline = new double[pxPerSpline];
	for (int i = 0; i < 3 * pxPerElem; i++)
	{
		spline[i] = 1.0L / pxPerElem;
	}

	float* spline2D = new float[pxPerSpline * pxPerSpline];
	float sum = 0;
	double sump[3][3] = {0};
	for (int i = 0; i < pxPerSpline; i++)
	{
		for (int j = 0; j < pxPerSpline; j++)
		{
			spline2D[i * pxPerSpline + j] = spline[i] * spline[j];
			sum += spline2D[i * pxPerSpline + j];
			sump[i / pxPerElem][j / pxPerElem] += spline2D[i * pxPerSpline + j];
		}
	}

	BSpline2d bs;
	bs.spline = spline2D;
	bs.size = pxPerSpline;
	bs.sum = sum;
	memcpy(&bs.sump, &sump, sizeof(sump));
	return bs;
}


double inline spline1(double x, double elemSpan)
{
	return (x * x) / (2* elemSpan * elemSpan);
}

double inline spline2(double x, double elemSpan)
{
	return (x * (2 * elemSpan - x) + (3 * elemSpan - x) * (x - elemSpan)) / (2* elemSpan * elemSpan);
}

double inline spline3(double x, double elemSpan)
{
	return (3 * elemSpan - x) * (3 * elemSpan - x) / (2* elemSpan * elemSpan);
}

BSpline2d generate2DSplineIntegrals(int pixels, int elements)
{
	int const pxPerElem = pixels / elements;
	int const pxPerSpline = 3 * pxPerElem;
	double const pxSpan = 1.0 / pixels;
	double const elemSpan = 1.0 / (elements);
	double* spline = new double[pxPerSpline];
	double x = pxSpan / 2;
	double sumf = 0;
	for (int i = 0; i < pxPerElem; i++ , x = x + pxSpan)
	{
		spline[i] = spline1(x, elemSpan)/pxPerElem;
		sumf += spline[i];
	}
	for (int i = pxPerElem; i < 2 * pxPerElem; i++ , x = x + pxSpan)
	{
		spline[i] = spline2(x, elemSpan)/pxPerElem;
		sumf += spline[i];
	}
	for (int i = 2 * pxPerElem; i < 3 * pxPerElem; i++ , x = x + pxSpan)
	{
		spline[i] = spline3(x, elemSpan)/pxPerElem;
		sumf += spline[i];
	}
	float* spline2D = new float[pxPerSpline * pxPerSpline];
	double sum = 0;
	double sump[3][3] = {{1.0L/20,13.0L/120,1.0L/120},
		{13.0L/120,45.0L/100,13.0L/120},
		{1.0L/120,13.0L/120,1.0L/20}};
//	double sump[3][3] = { 0 };
	for (int i = 0; i < pxPerSpline; i++)
	{
		for (int j = 0; j < pxPerSpline; j++)
		{
			spline2D[i * pxPerSpline + j] = spline[i] * spline[j];
			sum += spline2D[i * pxPerSpline + j];
//			sump[i / pxPerElem][j / pxPerElem] += spline2D[i * pxPerSpline + j];
		}
	}
	BSpline2d bs;
	bs.spline = spline2D;
	bs.size = pxPerSpline;
	bs.sum = sum;
	memcpy(&bs.sump, &sump, sizeof(sump));
	return bs;
}

///(ceil(pixPerElem/threadsPerBlock)+1)+2
__global__ void computeRightSide(float* dSplineArray, int toWriteSize, float* bitmap, float* dRightSide, float area, int pixPerElem, int bitmapSize, int memoryBlocks, int memBlockSize, int elements, int elements2, int idleThreads)
{
	__shared__ float toSum[THREADS];
	//__shared__ float toSum2[THREADS];//for reduction 2 times faster
	extern __shared__ float toWrite[];

	unsigned blockStart = blockIdx.x * blockDim.x;
	int idx = blockStart + threadIdx.x;
	int threadsfloat;
	if (idx >= bitmapSize * bitmapSize)
	{
		return;
	}
	if (idx >= (gridDim.x * blockDim.x - THREADS))
	{
		threadsfloat = THREADS - idleThreads;
	}
	else
	{
		threadsfloat = THREADS;
	}
	int row = idx / bitmapSize;
	int col = idx % bitmapSize;
	int pxIdx = idx; // = row*bitmapSize + col
	float pixel = bitmap[pxIdx];
	//compute summing indices
	int elemStart = threadIdx.x - idx % pixPerElem;
	int nextStart = elemStart + pixPerElem;
	elemStart -= (elemStart < 0) * elemStart; // elemStart < 0 ? elemStart : 0
	nextStart -= (nextStart > threadsfloat) * (nextStart - threadsfloat);// nextStart > THREADS ? THREADS : nextStart, moze >= ?
	int half = nextStart - elemStart;
	int threads = half / 2;
	half = (half + 1) / 2; //in case of odd float of elements - omit middle element, leave it for next iteration

	for (int splineRowDispl = pixPerElem * 2; splineRowDispl >= 0; splineRowDispl -= pixPerElem) //2ppe,1ppe,0ppe displacement
	{
		if (threadIdx.x < toWriteSize)
			toWrite[threadIdx.x] = 0;
		for (int splineColDispl = pixPerElem * 2; splineColDispl >= 0; splineColDispl -= pixPerElem)//2ppe + col
		{
			int splineCol = (col + splineColDispl) % dSplines.size;
			int splineIdx = (row % pixPerElem + splineRowDispl) * dSplines.size + splineCol; //spline row + spline col
			toSum[threadIdx.x] = pixel * dSplineArray[splineIdx];
			int t = threads;
			int h = half;
			while (threadIdx.x - elemStart < t) // size/=2
			{
				//				__syncthreads();
				toSum[threadIdx.x] += toSum[threadIdx.x + h];
				t = h / 2;
				h = (h + 1) / 2;
			}
			//			__syncthreads();
			if (threadIdx.x == elemStart)
			{
				int splinePart = splineCol / pixPerElem;//0,1,2	
				splinePart = 2 - splinePart; //revert spline to 2,1,0 to get shift
				int elem = idx / pixPerElem - blockStart / pixPerElem; // find elem to which thread belongs, 0,1,2,3,4..last in the block
				int rowShift = 2 * (idx / pixPerElem / elements - blockStart / pixPerElem / elements);//with each row shift gets greater by two, because row has elems + 2 elements
				elem = (elem + splinePart + rowShift);

				atomicAdd(toWrite + elem, toSum[threadIdx.x]); //mozna zastapic atomc add dodając 3 razy wiecej pamieci toWrite i watki sumuja modulo, a potem sumuje sie te 3 bloki pamiec
			}
		}
		//		__syncthreads();//?
		int blockToWrite = (blockIdx.x % memoryBlocks) * memBlockSize;
		int rowDisplacement = (pixPerElem * 2 - splineRowDispl);//bottom spline to first row, middle to middle, top spline to bottom row
		int indexToWrite = blockStart / pixPerElem + 2 * (blockStart / pixPerElem / elements) + threadIdx.x;//consecutiveElement
		indexToWrite += rowDisplacement * elements2 + blockToWrite;//plus elemsRowShift
		int restFromLast = pixPerElem - blockStart % pixPerElem; //when blockstart % ppe == 0 restFromLast is invalid (pixPerElem), but elemsInBlock is still being computed correctly
		int elemsInBlock = (restFromLast > 0) + (threadsfloat - restFromLast + pixPerElem - 1) / pixPerElem + 2 +
			2 * ((blockStart + threadsfloat - 1) / pixPerElem / elements - blockStart / pixPerElem / elements);//pixPerElem-1 = ceil(x) + 2*float of rows in this block
		if (threadIdx.x < elemsInBlock)
		{
			atomicAdd(dRightSide + indexToWrite, toWrite[threadIdx.x]);
		}
	}
}

//threads = elem2*elem2
__global__ void sumVerticalPxels(float* dRightSide, float* dOut, int elements2, int pixPerElem, float area)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= elements2 * elements2)
		return;
	float sum = 0;
	int elemStartRow = (idx / elements2) * pixPerElem * elements2;//skip all vertical elements which belong to previous row
	for (int i = 0; i < pixPerElem; i++)
	{
		int row = elements2 * i;
		int col = idx % elements2;
		sum += dRightSide[elemStartRow + col + row];
	}
	dOut[idx] = sum * area;
}

//threads = elements2*pixPerElem*elements2
__global__ void sumBlocks(float* dRightSide, int memoryBlocks, int memBlockSize)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= memBlockSize)
		return;
	float sum = 0;
	for (int block = 0; block < memoryBlocks; block++)
	{
		sum += dRightSide[memBlockSize * block + idx];
	}
	dRightSide[idx] = sum;
}

//width and height of bitmap must be equal, and moreover, size of bitmap must be divisible without remainder by float of elements
#define BLOCKS(N) (N+THREADS-1)/THREADS

number* generateBitmapRightSide(char* bpmPath, int elements, BSpline2d* outBSpline)
{
	int elements2 = elements + 2;
	Bitmap bitmap = readBmp(bpmPath);
	if (bitmap.height != bitmap.width || bitmap.width % elements != 0)
		throw "Bitmap dimensions must be equal. Bitmap size must be divisible by float of elements without remainder.";
	float* dBitmap = nullptr;
	BSpline2d bSplines = generate2DSplineIntegrals(bitmap.width, elements);
//		BSpline2d bSplines = generateTestBSplineIntegrals(bitmap.width, elements);
	if (outBSpline != nullptr)
		*outBSpline = bSplines;
	//	BSpline2d bSplines = generateTestBSplineIntegrals(bitmap.width, elements); //FOR TESTING
	//	for (int i = 0; i < bSplines.size; i++) {
	//		for (int j = 0; j < bSplines.size; j++)
	//			printf("%f ", bSplines.spline[i*bSplines.size+j]);
	//		printf("\n");
	//	}
	float* dSplineArray;
	ERRCHECK(cudaMemcpyToSymbol(dSplines, &bSplines, sizeof(BSpline2d)));
	ERRCHECK(cudaMalloc(&dSplineArray, sizeof(float)*bSplines.size*bSplines.size));
	ERRCHECK(cudaMemcpy(dSplineArray, bSplines.spline, sizeof(float)*bSplines.size*bSplines.size, cudaMemcpyHostToDevice));

	ERRCHECK(cudaMalloc(&dBitmap, sizeof(float)*bitmap.width*bitmap.width));
	ERRCHECK(cudaMemcpy(dBitmap, bitmap.bitmap, sizeof(float)*bitmap.width*bitmap.height, cudaMemcpyHostToDevice));
	delete[] bitmap.bitmap;

	float* dRightSide = nullptr;
	int pixPerElem = bitmap.width / elements;
	int floatOfMemoryBlocks = (3 * pixPerElem + THREADS - 1) / THREADS + (3 * pixPerElem < THREADS ? 1 : 0);//one element spans on 3 elements and corresponding float of blocks
	int blockSize = elements2 * elements2 * pixPerElem;
	int sharedMemorySize = 1 + (THREADS - 1 + pixPerElem - 1) / pixPerElem;//max float of elemens processed in one block
	sharedMemorySize += 2 + 2 * sharedMemorySize / elements + 4;//+ float of rows in elements, every row extends mem size by 2, +4 account for additional first and last element
	int totalThreads = BLOCKS(bitmap.width*bitmap.width) * THREADS;
	int idleThreads = totalThreads - bitmap.width * bitmap.width;
	float* rightSide = new float[elements2 * elements2];
//	double area = 1.0L / (elements * elements);
	double area = 1; //FOR TESTING

	ERRCHECK(cudaMalloc(&dRightSide, sizeof(float)*blockSize*floatOfMemoryBlocks))
	ERRCHECK(cudaMemset(dRightSide, 0, sizeof(float)*blockSize*floatOfMemoryBlocks));
	ERRCHECK(cudaDeviceSynchronize());
	//	ERRCHECK(cudaMemcpy(dRightSide, rightSide, sizeof(float)*blockSize*floatOfMemoryBlocks, cudaMemcpyHostToDevice));
	//	showMemoryConsumption();
	computeRightSide<<<BLOCKS(bitmap.width*bitmap.width), THREADS,sizeof(float) * sharedMemorySize >>>(dSplineArray, sharedMemorySize, dBitmap, dRightSide, area, pixPerElem, bitmap.width, floatOfMemoryBlocks, blockSize, elements, elements2, idleThreads);
	ERRCHECK(cudaGetLastError());
	ERRCHECK(cudaDeviceSynchronize());
	sumBlocks<<<BLOCKS(blockSize),THREADS>>>(dRightSide, floatOfMemoryBlocks, blockSize);
	ERRCHECK(cudaGetLastError());
	ERRCHECK(cudaDeviceSynchronize());
	sumVerticalPxels<<<BLOCKS(elements2*elements2),THREADS>>>(dRightSide, dRightSide + blockSize, elements2, pixPerElem, area);
	ERRCHECK(cudaGetLastError());
	ERRCHECK(cudaDeviceSynchronize());
	//	ERRCHECK(cudaMemcpy(rightSide,dRightSide, sizeof(float)*blockSize*floatOfMemoryBlocks, cudaMemcpyDeviceToHost)); FOR TESTING
	ERRCHECK(cudaMemcpy(rightSide,dRightSide + blockSize, sizeof(float)*elements2*elements2, cudaMemcpyDeviceToHost));
	ERRCHECK(cudaFree(dSplineArray));
	ERRCHECK(cudaFree(dBitmap));
	ERRCHECK(cudaFree(dRightSide));
	//	for (int i = 0; i<elements2*pixPerElem; i++)
	//	{
	//		for(int k = 0;k<floatOfMemoryBlocks;k++)
	//		{
	//			for (int j = 0; j<elements2; j++)
	//			{
	//				printf("%.2f ", *(rightSide+k*blockSize  + i*elements2 + j));
	//			}
	//			printf("\n");
	//		}
	//		printf("--\n");
	//	}
	//	for (int i = 0; i<elements2; i++) FOR TESTING
	//	{
	//
	//		for (int j = 0; j<elements2; j++)
	//		{
	//			printf("%.2f ", *(rightSide + blockSize + i*elements2 + j));
	//		}
	//		printf("\n");
	//	}
	//PRINT RESULT
	//		for (int i = 0; i<elements2; i++)
	//		{
	//	
	//			for (int j = 0; j<elements2; j++)
	//			{
	//				printf("%.2f ", *(rightSide + i*elements2 + j));
	//			}
	//			printf("\n");
	//		}
	//		printf("sum: %f\n", bSplines.sum*area);
	//	return dRightSide;
#ifdef DOUBLE_NUMBER
	double* rightSideD = new double[elements2 * elements2];
	for (int i = 0; i < elements2 * elements2; i++)
		rightSideD[i] = rightSide[i];
	delete[] rightSide;
	return rightSideD;
#endif
#ifdef FLOAT_NUMBER
	return rightSide;
#endif
}


void measureGenBitmap(char* bmpPath, int elements, int iters = 1)
{
	clock_t start, end;
	start = clock();
	for (int i = 0; i < iters; i++)
	{
		generateBitmapRightSide(bmpPath, elements);
	}
	end = clock();
	printf("time %f\n", ((float)(end - start) / CLOCKS_PER_SEC) / iters);
}

number* generateBitmapLeftSide(BSpline2d bSplines, int elements)
{
	int len = elements * 5;
	number* leftSide = new number[len];
	double a = bSplines.sump[2][0];
	double b = bSplines.sump[2][1] + bSplines.sump[1][0];
	double c = bSplines.sump[2][2] + bSplines.sump[1][1] + bSplines.sump[0][0];
	double d = bSplines.sump[1][2] + bSplines.sump[0][1];
	double e = bSplines.sump[0][2];
	for (int i = 0; i < len; i += 5)
	{
		leftSide[i] = a;
		leftSide[i + 1] = b;
		leftSide[i + 2] = c;
		leftSide[i + 3] = d;
		leftSide[i + 4] = e;
	}
	leftSide[0] = 0;
	leftSide[1] = 0;
//	leftSide[2] = bSplines.sump[0][0];
//	leftSide[3] = bSplines.sump[0][1];
//	leftSide[4] = bSplines.sump[0][2];
	leftSide[5] = 0;
//	leftSide[6] = bSplines.sump[1][0];
//	leftSide[7] = bSplines.sump[1][1] + bSplines.sump[0][0];
//	leftSide[8] = bSplines.sump[1][2] + bSplines.sump[0][1];
//	leftSide[9] = bSplines.sump[0][2];

//	leftSide[len - 10] = bSplines.sump[2][0];
//	leftSide[len - 9] = bSplines.sump[1][0] + bSplines.sump[2][1];
//	leftSide[len - 8] = bSplines.sump[1][1] + bSplines.sump[2][2];
//	leftSide[len - 7] = bSplines.sump[1][2];
	leftSide[len - 6] = 0;
//	leftSide[len - 5] = bSplines.sump[2][0];
//	leftSide[len - 4] = bSplines.sump[2][1];
//	leftSide[len - 3] = bSplines.sump[2][2];
	leftSide[len - 2] = 0;
	leftSide[len - 1] = 0;
	return leftSide;
}

number getApprox(number x, number y, number* solution, int elements)
{
	double const elemSpan = 1.0 / elements;
	int ex = x * elements;
	int ey = y * elements;
	double ly = y - elemSpan * ey;
	double ly1 = ly + elemSpan;
	double ly2 = ly1 + elemSpan;
	double lx = x - elemSpan * ex;
	double lx1 = lx + elemSpan;
	double lx2 = lx1 + elemSpan;

	int elements2 = elements + 2;
	double approx = 0;
	approx += spline1(lx, elemSpan) * spline1(ly, elemSpan) * solution[ex * elements2 + ey];
	approx += spline1(lx, elemSpan) * spline2(ly1, elemSpan) * solution[ex * elements2 + ey + 1];
	approx += spline1(lx, elemSpan) * spline3(ly2, elemSpan) * solution[ex * elements2 + ey + 2];
	ex++;
	approx += spline2(lx1, elemSpan) * spline1(ly, elemSpan) * solution[ex * elements2 + ey];
	approx += spline2(lx1, elemSpan) * spline2(ly1, elemSpan) * solution[ex * elements2 + ey + 1];
	approx += spline2(lx1, elemSpan) * spline3(ly2, elemSpan) * solution[ex * elements2 + ey + 2];
	ex++;
	approx += spline3(lx2, elemSpan) * spline1(ly, elemSpan) * solution[ex * elements2 + ey];
	approx += spline3(lx2, elemSpan) * spline2(ly1, elemSpan) * solution[ex * elements2 + ey + 1];
	approx += spline3(lx2, elemSpan) * spline3(ly2, elemSpan) * solution[ex * elements2 + ey + 2];
	return approx;
}

number* getBitmapApprox(number* solution, int elements, int resolution)
{
	number* approx = new number[resolution * resolution];
	double span = 1.0L / resolution;
	number x = -span/2;
	for (int i = 0; i < resolution; i++)
	{
		x += span;
		number y = -span/2;
		for (int j = 0; j < resolution; j++)
		{
			y += span;
			approx[i * resolution + j] = getApprox(x, y, solution, elements);
		}
	}
	printf("beginapprox\n");
	for (int i = 0; i < resolution; i++)
	{
		for (int j = 0; j < resolution; j++)
		{
			printf("%f ", approx[i * resolution + j]);
		}
		printf("\n");
	}
	printf("endapprox\n");
	return approx;
}
