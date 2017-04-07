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
			printf("%.5f %.5f %.5f %.5f %.5f %.5f | ", node.m[XY(j, 0)], node.m[XY(j, 1)], node.m[XY(j, 2)], node.m[XY(j, 3)], node.m[XY(j, 4)], node.m[XY(j, 5)]);
			for (int k = 0; k < props.rightCount; k++)
			{
				printf("%.5f ", node.x[j][k]);
			}
			printf("\n");
		}
		printf("\n");
	}
}

__device__ __host__ void printNode(Node node,int rightCount)
{
	for (int i = 0; i < 6; i++) {
		printf("%.2f %.2f %.2f %.2f %.2f %.2f | ", node.m[XY(i, 0)], node.m[XY(i, 1)], node.m[XY(i, 2)], node.m[XY(i, 3)], node.m[XY(i, 4)], node.m[XY(i, 5)]);
		for (int k = 0; k < rightCount; k++)
		{
			printf("%.2f ", node.x[i][k]);
		}
		printf("\n");
	}
	printf("\n");
}

void fillRightSide(float value, int row, float* rightSide, int rightCount)
{
	for (int i = 0; i < rightCount; i++)
	{
		rightSide[row * rightCount + i] = value;
	}
}


void computeRightSide(int rightCount, float* leftSide, float* rightSide, int i,int offset)
{
	int rightSideVal = 0;
	for (int j = 0; j < 5; j++)
	{
//		int solution = 1;
		int solution = i - 1 + j + offset;//solution is x(0)=1,x(1)=2,x(n-1)=n
		rightSideVal += leftSide[i*5+j] * solution;
	}
	fillRightSide(rightSideVal, i, rightSide, rightCount);
}

__device__ void printRow(float * m, int start, int count)
{
	for (int i = start; i < start + count; i++)
		printf("%.1f ", m[i]);
	printf("\n");
}
void generateTestEquation(int leftCount, int rightCount, float** leftSidePtr, float** rightSidePtr)
{
	float* leftSide = new float[leftCount * 5];
	float* rightSide = new float[rightCount * leftCount];
	for (int i = 0; i < leftCount * 5; i+=5)
	{
		leftSide[i] = 1;
		leftSide[i+1] = 1;
		leftSide[i+2] = 5;
		leftSide[i+3] = 1;
		leftSide[i+4] = 1;
	}
	leftSide[0] = 0;
	leftSide[1] = 0;
	leftSide[5] = 0;
	leftSide[leftCount * 5 - 6] = 0;
	leftSide[leftCount * 5 - 2] = 0;
	leftSide[leftCount * 5 - 1] = 0;

	for (int i = 0; i < leftCount; i++)
	{
		computeRightSide(rightCount, leftSide, rightSide, i,0);
	}
//	fillRightSide(14, i, rightSide, rightCount);
	*leftSidePtr = leftSide;
	*rightSidePtr = rightSide;
	for (int i = 0; i < leftCount; i++)
	{
		printf("%d:", i + 1);
		for (int j = 0; j < 5; j++)
		{
			printf("%.0f ", leftSide[i * 5 + j]);
		}
		printf(" |  ");
		for (int j = 0; j < rightCount; j++)
		{
			printf("%.0f ", rightSide[i * rightCount + j]);
		}
		printf("\n");
	}
//	int before = 0;
//	int after = leftCount-1;
//	for (int i = 0; i < leftCount; i++)
//	{
//		for (int i = 0; i < before; i++)
//			printf("0 ");
//		for (int j = 0; j < 5; j++)
//		{
//			printf("%.0f ", leftSide[i * 5 + j]);
//		}
//		for (int i = 0; i < after; i++)
//			printf("0 ");
//		printf(" |  ");
//		for (int j = 0; j < rightCount; j++)
//		{
//			printf("%.0f ", rightSide[i * rightCount + j]);
//		}
//		printf("\n");
//		before++;
//		after--;
//	}

//	getch();
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