#include "bitmap_approx.cuh"
#include <device_launch_parameters.h>
#include <pplinterface.h>

#define THREADS 32
__constant__ BSpline2d dSplines;
BSpline2d generate2DSplineIntegrals(int pixels, int elements)
{
	int const pxPerElem = pixels / elements;
	int const pxPerSpline = 3 * pxPerElem;
	double const pxSpan = 1.0 / pixels;
	double const elemSpan = 1.0 / elements;
	double* spline = new double[pxPerSpline];
	double x = 0;
	for (int i = 0; i < pxPerElem; i++ , x = x + pxSpan)
	{
		spline[i] = (x * x) / (2 * elemSpan * elemSpan);
	}
	for (int i = pxPerElem; i < 2 * pxPerElem; i++ , x = x + pxSpan)
	{
		spline[i] = (x * (2 * elemSpan - x) + (3 * elemSpan - x) * (x - elemSpan)) / (2 * elemSpan * elemSpan);
	}
	for (int i = 2 * pxPerElem; i < 3 * pxPerElem; i++ , x = x + pxSpan)
	{
		spline[i] = (3 * elemSpan - x) * (3 * elemSpan - x) / (2 * elemSpan * elemSpan);
	}
	float* spline2D = new float[pxPerSpline * pxPerSpline];
	float sum = 0;
	for (int i = 0; i < pxPerSpline; i++)
	{
		for (int j = 0; j < pxPerSpline; j++)
		{
			spline2D[i * pxPerSpline + j] = spline[i] * spline[j];
			sum += spline2D[i * pxPerSpline + j];
		}
	}

	BSpline2d bs;
	bs.spline = spline2D;
	bs.size = pxPerSpline;
	bs.sum = sum;
	return bs;
//	return BSpline2d(spline2D, pxPerSpline);
}

///(ceil(pixPerElem/threadsPerBlock)+1)+2
__global__ void computeRightSide(float* dSplineArray,int toWriteSize,float* bitmap, float* dRightSide, float area, int pixPerElem, int bitmapSize, int memoryBlocks, int memBlockSize,int elements,int elements2,int idleThreads)//=elements*elements (elements+=2)
{
	__shared__ float toSum[THREADS];
	//__shared__ float toSum2[THREADS];//for reduction 2 times faster
	extern __shared__ float toWrite[];//max_elems = size = 1+ceil(THREADS-1)/pixPerElem+2
	//account for more threads than pixels
	unsigned blockStart = blockIdx.x * blockDim.x;
	int idx = blockStart + threadIdx.x;
//	int threadsNumber;
	if (idx >= bitmapSize*bitmapSize)
	{
		printf("idx\n");
		return;
	}
//	if(idx >= (bitmapSize*bitmapSize-THREADS))
//	{
//		threadsNumber = idleThreads;
//	}else
//	{
//		threads = THREADS;
//	}
	int row = idx / bitmapSize;
	int col = idx % bitmapSize;
	int pxIdx = idx; // = row*bitmapSize + col
	float pixel = bitmap[pxIdx];
	//compute summing indices
	int elemStart = threadIdx.x - idx % pixPerElem;
	int nextStart = elemStart + pixPerElem;
	elemStart -= (elemStart < 0) * elemStart; // elemStart < 0 ? elemStart : 0
	nextStart -= (nextStart > THREADS) * (nextStart - THREADS);// nextStart > THREADS ? THREADS : nextStart, moze >= ?
	int half = nextStart - elemStart;
	int threads = half / 2;
	half = (half + 1) / 2; //in case of odd number of elements - omit middle element, leave it for next iteration

	for (int splineRowDispl = pixPerElem * 2; splineRowDispl >= 0; splineRowDispl -= pixPerElem) //2ppe,1ppe,0ppe displacement
	{
		if (threadIdx.x < toWriteSize)
			toWrite[threadIdx.x] = 0;
		for (int splineColDispl = pixPerElem * 2; splineColDispl >= 0; splineColDispl -= pixPerElem)//2ppe + col
		{
			int splineCol = (col + splineColDispl) % dSplines.size;
			int splineIdx = (row % pixPerElem + splineRowDispl) * dSplines.size + splineCol; //spline row + spline col
			toSum[threadIdx.x] = pixel * dSplineArray[splineIdx] * area;
//			printf("%f %f %f \n",pixel, dSplineArray[splineIdx], toSum[threadIdx.x]);
			int t = threads;
			int h = half;

			while (threadIdx.x - elemStart < t) // size/=2
			{
				toSum[threadIdx.x] += toSum[threadIdx.x + h];
				t = h / 2;
				h = (h + 1) / 2;
				__syncthreads();
			}

			if (threadIdx.x == elemStart)
			{
				int splinePart = splineCol / pixPerElem;//0,1,2	
				splinePart = 2 - splinePart; //revert spline to 2,1,0 to get shift
				int elem = idx / pixPerElem - blockStart / pixPerElem; // find elem to which thread belongs, 0,1,2,3,4..last in the block
				int rowShift = 2 * (idx / pixPerElem / elements - blockStart / pixPerElem / elements);//with each row shift gets greater by two, because row has elems + 2 elements
				elem = (elem + splinePart + rowShift);
				atomicAdd(toWrite + elem, toSum[threadIdx.x]);
			}
		__syncthreads();//?
		}
		int blockToWrite = (blockIdx.x % memoryBlocks) * memBlockSize;
//		int blockToWrite = 0;
		int rowDisplacement = (pixPerElem * 2 - splineRowDispl);//bottom spline to first row, middle to middle, top spline to bottom row
		int indexToWrite = blockStart / pixPerElem + 2 * (blockStart / pixPerElem / elements) + threadIdx.x;//consecutiveElement
		indexToWrite += rowDisplacement*elements2 + blockToWrite;//plus elemsRowShift
		int restFromLast = pixPerElem - blockStart % pixPerElem; //when blockstart % ppe == 0 restFromLast is invalid (pixPerElem), but elemsInBlock is still being computed correctly
		int elemsInBlock = (restFromLast > 0) + (THREADS - restFromLast + pixPerElem - 1) / pixPerElem + 2 +
			2 * ((blockStart + THREADS - 1) / pixPerElem / elements - blockStart / pixPerElem / elements);//pixPerElem-1 = ceil(x) + 2*number of rows in this block
		if (threadIdx.x < elemsInBlock)
						// dRightSide[indexToWrite] += toWrite[threadIdx.x];
//			atomicAdd(dRightSide + indexToWrite, 1);
			atomicAdd(dRightSide + indexToWrite, toWrite[threadIdx.x]);
//			dRightSide[indexToWrite] += 1;
		//TODO replace threads with something that accounts for last block which may contain less than THREADS pixels :'(
	}
}
//element musi sie skladac z wiecej niz 1 piksela
void sequentialComputeRightSideCopy(int blocks, int blockDim, int toWriteSize,BSpline2d dSplines, float* bitmap, float* dRightSide, float area, int pixPerElem, int bitmapSize, int memoryBlocks, int memBlockSize,int elements,int elements2)
{
//	for (int i = 0; i<elements2*pixPerElem; i++)
//	{
//		for (int j = 0; j<elements2; j++)
//		{
//			printf("%.2f ", *(dRightSide + i*elements2 + j));
//		}
//		printf("\n");
//	}
	for (int blockIdx = 0; blockIdx < blocks; blockIdx++)
	{
		float toSum[THREADS];
		float* toWrite = new float[toWriteSize];//max_elems = size = 1+ceil(THREADS-1)/pixPerElem+2
		for (int threadIdx = 0; threadIdx < blockDim; threadIdx++)
		{
			unsigned blockStart = blockIdx * blockDim;
			int idx = blockStart + threadIdx;
			int row = idx / bitmapSize;
			int col = idx % bitmapSize;
			int pxIdx = idx; // = row*bitmapSize + col
			float pixel = bitmap[pxIdx];
			//compute summing indices
			int elemStart = threadIdx - idx % pixPerElem;
			int nextStart = elemStart + pixPerElem;
			elemStart -= (elemStart < 0) * elemStart; // elemStart < 0 ? elemStart : 0
			nextStart -= (nextStart > THREADS) * (nextStart - THREADS);// nextStart > THREADS ? THREADS : nextStart, moze >= ?
			int half = nextStart - elemStart;
			int threads = half / 2;
			half = (half + 1) / 2; //in case of odd number of elements - omit middle element, leave it for next iteration

			for (int splineRowDispl = pixPerElem * 2; splineRowDispl >= 0; splineRowDispl -= pixPerElem) //2ppe,1ppe,0ppe displacement
			{
				for (int splineColDispl = pixPerElem * 2; splineColDispl >= 0; splineColDispl -= pixPerElem)//2ppe + col
				{
					int splineCol = (col + splineColDispl) % dSplines.size;
					int splineIdx = (row % pixPerElem + splineRowDispl) * dSplines.size + splineCol; //spline row + spline col
					toSum[threadIdx] = pixel * dSplines.spline[splineIdx] * area;

					int t = threads;
					int h = half;

					while (threadIdx - elemStart < t) // size/=2
					{
						toSum[threadIdx] += toSum[threadIdx + h];
						t = h / 2;
						h = (h + 1) / 2;
					}

					if (threadIdx == elemStart)
					{
						int splinePart = splineCol / pixPerElem;//0,1,2	
						splinePart = 2 - splinePart; //revert spline to 2,1,0 to get shift
						int elem = idx / pixPerElem - blockStart / pixPerElem; // find elem to which thread belongs, 0,1,2,3,4..last in the block
						int rowShift = 2*(idx / pixPerElem / elements - blockStart/pixPerElem/elements);//with each row shift gets greater by two, because row has elems + 2 elements
						elem = (elem + splinePart + rowShift);
						toWrite[elem] += toSum[threadIdx];
					}
				}
				int blockToWrite = (blockIdx % memoryBlocks) * memBlockSize;
//				int blockToWrite = 0;
				int rowDisplacement = (pixPerElem * 2 - splineRowDispl);//bottom spline to first row, middle to middle, top spline to bottom row
//				int indexToWrite = blockStart / pixPerElem + threadIdx;//consecutiveElement
//				indexToWrite += 2 * (indexToWrite / elements) + rowDisplacement*elements2 + blockToWrite;//plus elemsRowShift
				int indexToWrite = blockStart / pixPerElem + 2 * (blockStart / pixPerElem / elements) + threadIdx;//consecutiveElement
				indexToWrite += rowDisplacement*elements2 + blockToWrite;//plus elemsRowShift
				int restFromLast = pixPerElem - blockStart % pixPerElem; //when blockstart % ppe == 0 restFromLast is invalid (pixPerElem), but elemsInBlock is still being computed correctly
				int elemsInBlock = (restFromLast > 0) + (THREADS - restFromLast + pixPerElem - 1) / pixPerElem + 2 +
					2 * ((blockStart+THREADS-1) / pixPerElem / elements - blockStart / pixPerElem / elements);//pixPerElem-1 = ceil(x) + 2*number of rows in this block
				if (threadIdx < elemsInBlock) { 
//					dRightSide[indexToWrite] = toWrite[threadIdx];
					dRightSide[indexToWrite] += 1;
//					printf("i: %i, (x,y)%i,%i\n",indexToWrite-blockToWrite, (indexToWrite-blockToWrite) / (elements2), (indexToWrite-blockToWrite) %(elements2));
				}
				//TODO replace threads with something that accounts for last block which may contain less than THREADS pixels :'( or may it not?
			}
		}
		
	}
//	for (int i = 0; i<elements2*pixPerElem; i++)
//	{
//		for (int j = 0; j<elements2; j++)
//		{
//			printf("%.2f ", *(dRightSide + i*elements2 + j));
//		}
//		printf("\n");
//	}
}

//threads = elem2*elem2
__global__ void sumVerticalPxels(float* dRightSide, float* dOut, int elements2, int pixPerElem)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= elements2*elements2)
		return;
	float sum = 0;
	int elemStartRow = (idx / elements2) * pixPerElem * elements2;//skip all vertical elements which belong to previous row
	for (int i = 0; i < pixPerElem; i++)
	{
		int row = elements2 * i;
		int col = idx % elements2;
		sum += dRightSide[elemStartRow + col + row];
	}
	dOut[idx] = sum;
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

#define BLOCKS(N) (N+THREADS-1)/THREADS
float* generateBitmapRightSide(char* bpmPath, int elements)
{
	int elements2 = elements + 2;
	Bitmap bitmap = readBmp(bpmPath);
	float* dBitmap = nullptr;
	BSpline2d bSplines = generate2DSplineIntegrals(bitmap.width, elements);
//	for (int i = 0; i < bSplines.size; i++) {
//		for (int j = 0; j < bSplines.size; j++)
//			printf("%f ", bSplines.spline[i*bSplines.size+j]);
//		printf("\n");
//	}
	float * dSplineArray;
	ERRCHECK(cudaMemcpyToSymbol(dSplines, &bSplines, sizeof(BSpline2d)));
	ERRCHECK(cudaMalloc(&dSplineArray, sizeof(float)*bSplines.size*bSplines.size));
	ERRCHECK(cudaMemcpy(dSplineArray, bSplines.spline, sizeof(float)*bSplines.size*bSplines.size, cudaMemcpyHostToDevice));

	ERRCHECK(cudaMalloc(&dBitmap, sizeof(float)*bitmap.width*bitmap.width));
	ERRCHECK(cudaMemcpy(dBitmap, bitmap.bitmap, sizeof(float)*bitmap.width*bitmap.height, cudaMemcpyHostToDevice));

	float* dRightSide = nullptr;
	int pixPerElem = bitmap.width / elements;
	int numberOfMemoryBlocks = (3*pixPerElem + THREADS - 1) / THREADS + (3*pixPerElem < THREADS ? 1 : 0);//one element spans on 3 elements and corresponding number of blocks
	int blockSize = elements2 * elements2 * pixPerElem;
	int sharedMemorySize = 1 + (THREADS - 1 + pixPerElem - 1) / pixPerElem;//max number of elemens processed in one block
	sharedMemorySize += 2 + 2*sharedMemorySize/elements ;//+ number of rows in elements, every row extends mem size by 2
	int totalThreads = BLOCKS(bitmap.width*bitmap.width) * THREADS;
	int idleThreads = THREADS - ( totalThreads - bitmap.width * bitmap.width);
	float * rightSide = new float[blockSize*numberOfMemoryBlocks]{0};
	
//	sequentialComputeRightSideCopy(BLOCKS(bitmap.width*bitmap.width), THREADS, sharedMemorySize, bSplines, bitmap.bitmap, rightSide,
//	                               1.0 / (elements * elements), pixPerElem, bitmap.width, numberOfMemoryBlocks, blockSize,elements,elements2);

	ERRCHECK(cudaMalloc(&dRightSide, sizeof(float)*blockSize*numberOfMemoryBlocks))
	ERRCHECK(cudaMemset(dRightSide, 0, sizeof(float)*blockSize*numberOfMemoryBlocks));
//	ERRCHECK(cudaMemcpy(dRightSide, rightSide, sizeof(float)*blockSize*numberOfMemoryBlocks, cudaMemcpyHostToDevice));
	computeRightSide<<<BLOCKS(bitmap.width*bitmap.width), THREADS,sharedMemorySize >>>(dSplineArray,sharedMemorySize, dBitmap, dRightSide, 1.0L / (elements*elements), pixPerElem, bitmap.width, numberOfMemoryBlocks,blockSize, elements, elements2,idleThreads);
	ERRCHECK(cudaGetLastError());
	sumBlocks<<<BLOCKS(blockSize),THREADS>>>(dRightSide, numberOfMemoryBlocks, blockSize);
	ERRCHECK(cudaGetLastError());
	sumVerticalPxels<<<BLOCKS(elements2*elements2),THREADS>>>(dRightSide, dRightSide + blockSize, elements2, pixPerElem);
	ERRCHECK(cudaGetLastError());
	ERRCHECK(cudaDeviceSynchronize());
	ERRCHECK(cudaMemcpy(rightSide,dRightSide, sizeof(float)*blockSize*numberOfMemoryBlocks, cudaMemcpyDeviceToHost));
//	for (int i = 0; i<elements2*pixPerElem; i++)
//	{
//		for(int k = 0;k<numberOfMemoryBlocks;k++)
//		{
//			for (int j = 0; j<elements2; j++)
//			{
//				printf("%.2f ", *(rightSide+k*blockSize  + i*elements2 + j));
//			}
//			printf("\n");
//		}
//		printf("--\n");
//	}
	for (int i = 0; i<elements2; i++)
	{

		for (int j = 0; j<elements2; j++)
		{
			printf("%.2f ", *(rightSide+blockSize + i*elements2 + j));
		}
		printf("\n");
	}
	printf("sum: %f\n", bSplines.sum*1.0L / (elements*elements));
	return dRightSide;
}
