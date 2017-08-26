#include "helpers.cuh"
#include <cmath>

void printAllNodes(Node* nodes, int nodesStart, Properties props)
{
	int powerOfTwo = (int)log2(nodesStart + 1) + 1;
	for (int i = nodesStart; i < props.heapNodes; i++)
	{
		//		if (i == powerOfTwo)
		//		{
		//			printf("level %d\n", powerOfTwo);
		//			powerOfTwo <<= 1;
		//		}
		Node node = nodes[i];
		for (int j = i >= props.remainingNodes ? 1 : 0; j < 6; j++)
		{
			printf("%.2f %.2f %.2f %.2f %.2f %.2f | ", node.m[XY(j, 0)], node.m[XY(j, 1)], node.m[XY(j, 2)], node.m[XY(j, 3)], node.m[XY(j, 4)], node.m[XY(j, 5)]);
			for (int k = 0; k < props.rightCount; k++)
			{
				printf("%.2f ", node.x[j][k]);
			}
			printf("\n");
		}
		printf("\n");
	}
}

__device__ __host__ void printNode(Node node, int rightCount)
{
	for (int i = 0; i < 6; i++)
	{
		printf("%.2f %.2f %.2f %.2f %.2f %.2f | ", node.m[XY(i, 0)], node.m[XY(i, 1)], node.m[XY(i, 2)], node.m[XY(i, 3)], node.m[XY(i, 4)], node.m[XY(i, 5)]);
		for (int k = 0; k < rightCount; k++)
		{
			printf("%.2f ", node.x[i][k]);
		}
		printf("\n");
	}
	printf("\n");
}

void fillRightSide(number value, int row, number* rightSide, int rightCount)
{
	for (int i = 0; i < rightCount; i++)
	{
		rightSide[row * rightCount + i] = value*(i+1);
	}
}


void computeRightSide(int rightCount, number* leftSide, number* rightSide, int i, int offset)
{
	int rightSideVal = 0;
	for (int j = 0; j < 5; j++)
	{
		//		int solution = 1;
		int solution = i - 1 + j + offset;//solution is x(0)=1,x(1)=2,x(n-1)=n
		rightSideVal += leftSide[i * 5 + j] * solution;
	}
	fillRightSide(rightSideVal, i, rightSide, rightCount);
}

__device__ void printRow(number* m, int start, int count)
{
	for (int i = start; i < start + count; i++)
		printf("%.1f ", m[i]);
	printf("\n");
}

void generateTestEquation(int leftCount, int rightCount, number** leftSidePtr, number** rightSidePtr)
{
	number* leftSide = new number[leftCount * 5];
	number* rightSide = new number[rightCount * leftCount];
	for (int i = 0; i < leftCount * 5; i += 5)
	{
		leftSide[i] = 0.09766;
		leftSide[i + 1] = 0.85938;
		leftSide[i + 2] = 2.08594;
		leftSide[i + 3] = 0.85938;
		leftSide[i + 4] = 0.09766;
	}
	leftSide[0] = 0;
	leftSide[1] = 0;
	leftSide[5] = 0;
	leftSide[leftCount * 5 - 6] = 0;
	leftSide[leftCount * 5 - 2] = 0;
	leftSide[leftCount * 5 - 1] = 0;

	for (int i = 0; i < leftCount; i++)
	{
		computeRightSide(rightCount, leftSide, rightSide, i, 0);
	}
	//	fillRightSide(14, i, rightSide, rightCount);
	*leftSidePtr = leftSide;
	*rightSidePtr = rightSide;
	//	for (int i = 0; i < leftCount; i++)
	//	{
	//		printf("%d:", i + 1);
	//		for (int j = 0; j < 5; j++)
	//		{
	//			printf("%.0f ", leftSide[i * 5 + j]);
	//		}
	//		printf(" |  ");
	//		for (int j = 0; j < rightCount; j++)
	//		{
	//			printf("%.0f ", rightSide[i * rightCount + j]);
	//		}
	//		printf("\n");
	//	}
		int before = 0;
		int after = leftCount-1;
		for (int i = 0; i < leftCount; i++)
		{
			for (int i = 0; i < before; i++)
				printf("0 ");
			for (int j = 0; j < 5; j++)
			{
				printf("%.0f ", leftSide[i * 5 + j]);
			}
			for (int i = 0; i < after; i++)
				printf("0 ");
			printf(" |  ");
			for (int j = 0; j < rightCount; j++)
			{
				printf("%.0f ", rightSide[i * rightCount + j]);
			}
			printf("\n");
			before++;
			after--;
		}

	//	getch();
}

void printLeftAndRight(number * left,number * right,int size,int rsize)
{
	if (rsize == 0)
		rsize = size;
	printf("LEFT AND RIGHT\n");
	int before = 0;
	int after = size - 1;
	for (int i = 0; i < size; i++)
	{
		for (int i = 0; i < before; i++)
			printf("0 ");
		for (int j = 0; j < 5; j++)
		{
			printf("%.5f ", left[i * 5 + j]);
		}
		for (int i = 0; i < after; i++)
			printf("0 ");
		printf(" |  ");
		for (int j = 0; j < rsize; j++)
		{
			printf("%.5f ", right[i * rsize + j]);
		}
		printf("\n");
		before++;
		after--;
	}
	printf("END LEFT AND RIGHT\n");

}

void showMemoryConsumption()
{
	size_t free_byte;
	size_t total_byte;
	ERRCHECK(cudaMemGetInfo(&free_byte, &total_byte));
	double free_db = (double)free_byte;
	double total_db = (double)total_byte;
	double used_db = total_db - free_db;

	printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n", used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);
}

//inline
//cudaError_t checkCuda(cudaError_t result)
//{
//#if defined(DEBUG) || defined(_DEBUG)
//	if (result != cudaSuccess) {
//		fprintf(stderr, "CUDA Runtime Error: %sn", cudaGetErrorString(result));
//		assert(result == cudaSuccess);
//	}
//#endif
//	return result;
//}


//returns bitmap upside down and rotated right 90 
//TODO cuts bitmap edges
Bitmap readBmp(char* filename)
{
	unsigned char* texels;
	int width, height;
	FILE* fd;
	fd = fopen(filename, "rb");
	if (fd == NULL)
	{
		printf("Error: fopen failed\n");
		throw "Bitmap opening failed";
	}

	unsigned char header[54];

	// Read header
	fread(header, sizeof(unsigned char), 54, fd);

	// Capture dimensions
	width = *(int*)&header[18];
	height = *(int*)&header[22];
	if (width != height)
		throw "bitmap must be suqare size";
	int padding = 0;

	// Calculate padding
	while ((width * 3 + padding) % 4 != 0)
	{
		padding++;
	}

	// Compute new width, which includes padding
	int widthnew = width * 3 + padding;

	//	// Allocate memory to store image data (non-padded)
	//	texels = (unsigned char *)malloc(width * height * 3 * sizeof(unsigned char));
	//	if (texels == NULL)
	//	{
	//		printf("Error: Malloc failed\n");
	//		return;
	//	}

	number* bitmap = nullptr;
	bitmap = new number[width * height];

	// Allocate temporary memory to read widthnew size of data

	unsigned char* data = (unsigned char *)malloc(widthnew * sizeof(unsigned int));
	//input (bmp stores bitmap upside down)
	// 3. 3 3 x x x
	// 2. 2 2 y
	// 1. 1 1 y
	//output
	// 1. 2. 3. y y y 
	// 1  2  3 x 
	// 1  2  3 x
	// Read row by row of data and remove padded data.
	for (int i = 0; i <height; i++)
	{
		// Read widthnew length of data
		fread(data, sizeof(unsigned char), widthnew, fd);

		// Retain width length of data, and swizzle RB component.
		// BMP stores in BGR format, my usecase needs RGB format
		for (int j = 0; j < width * 3; j += 3)
		{
			//int index = j / 3 + width*i;// upside down
			int index = (j / 3) * width + i; //rotated 90 right
			bitmap[index] = (0.299 * data[j + 2] + 0.587 * data[j + 1] + 0.114 * data[j]);
		}
	}
	free(data);
	fclose(fd);
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			printf("%f ", bitmap[i*width + j]);
		}
		printf("\n");
	}
	printf("endbitmap\n");
	return Bitmap(bitmap, width, height);
}

number * cutSquare(number * input,int size,int targetCol)
{
	number * result = new number[size*targetCol];
	for(int i = 0;i<size;i++)
	{
		for(int j = 0;j<targetCol;j++)
		{
			result[i*targetCol + j] = input[i*size + j]*255;
		}
	}
	delete[] input;
	return result;
}