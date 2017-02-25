#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <conio.h>
#include <ctime>
#include <new>
#include <cmath>
#include <cstring>
#define ERRCHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true, bool wait = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (wait) getch();
		if (abort) exit(code);
	}
}

#define MSIZE 36
#define INPUT_SIZE(N) N*5 - 3*2
#define PARENT(i) (i-1)/2
#define LEFT(i) 2*i + 1
#define RIGHT(i) 2*i + 2
#define BOTTOM_HEAP_NODES_COUNT(N) (N-2)/3 //size of input must be 2+3n,n>1
#define HEAP_SIZE(N) 2*BOTTOM_HEAP_NODES_COUNT(N)-1 
#define FIRST_LEVEL_SIZE 19
#define ROW_LENGTH 5
#define FIRST_LVL_MAT_SIZE 5
#define XY(x,y) x*6+y
#define THREADS 512
#define BLOCKS(N) (N+THREADS)/THREADS
#define COLUMNS_PER_THREAD 1

const struct Properties
{
	int leftCount;
	int leftSize;
	int heapNodes;
	int bottomNodes;
	int remainingNodes;
	int lastLevelNodes;
	int beforeLastLevelNodes;
	int lastLevelStartIdx;
	int beforeLastLevelStartIdx;
	int rightCount;
	int beforeLastLevelNotBottomNodes;
};

Properties getProperities(int leftCount, int rightCount)
{
	Properties p;
	p.leftCount = leftCount;
	p.leftSize = leftCount * 5;
	p.heapNodes = HEAP_SIZE(leftCount);
	p.bottomNodes = BOTTOM_HEAP_NODES_COUNT(leftCount);
	p.remainingNodes = p.heapNodes - p.bottomNodes;
	p.beforeLastLevelStartIdx = (int)pow(2, (int)log2(p.remainingNodes)) - 1;
	p.beforeLastLevelNotBottomNodes = p.remainingNodes - p.beforeLastLevelStartIdx;
	p.beforeLastLevelNodes = pow(2, (int)log2(p.bottomNodes - 1)) - p.beforeLastLevelNotBottomNodes;// -1 is in case bottomNodes is power of two, then beforeLastLevelNodes should obviously be 0
	p.lastLevelNodes = p.bottomNodes - p.beforeLastLevelNodes;
	p.lastLevelStartIdx = p.heapNodes - p.lastLevelNodes;
	p.beforeLastLevelStartIdx = p.remainingNodes - p.lastLevelNodes; //account for idx value, undefined when beforeLastLevelNodes is = 0
	p.rightCount = rightCount;
	return p;
}


__constant__ Properties dProps;

struct Node
{
	float m[MSIZE];
	float* x[6];
};

__device__ __host__ void printNode(Node node);


__device__ __inline__ int rXY(int x, int y)
{
	return x * dProps.rightCount + y;
}

__global__ void backwardSubstitutionRight(Node* nodes, int startIdx, int nodesCount, int start, int elim)
{
	int idx = (blockIdx.x * blockDim.x + threadIdx.x) / (dProps.rightCount / COLUMNS_PER_THREAD);
	if (idx >= nodesCount)
		return;
	int colStart = ((blockIdx.x * blockDim.x + threadIdx.x) % (dProps.rightCount / COLUMNS_PER_THREAD))*COLUMNS_PER_THREAD;
//	printf("%d %d\n", idx, colStart);
	int nodeIdx = startIdx + idx;
	float* m = nodes[nodeIdx].m;
	float** x = nodes[nodeIdx].x;
	for (int rcol = colStart; rcol < colStart + COLUMNS_PER_THREAD; rcol++)
	{
		for (int row = elim; row >= 0; row--)//max elim == 4,5th is already done after elimination
		{
			for (int col = row + 1; col < 6; col++)
			{
				x[row][rcol] -= m[XY(row, col)] * x[col][rcol];
			}
		}
	}
}

__global__ void forwardEliminationLeft(Node* nodes, int startIdx, int nodesCount, int start, int elim)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= nodesCount)
		return;
	int nodeIdx = startIdx + idx;
	float* m = nodes[nodeIdx].m;
	for (int row = start; row < elim; row++)
	{
		for (int col = row + 1; col < 6; col++) //from element after diagonal
		{
			m[XY(row, col)] /= m[XY(row, row)];
		}
		for (int rowBelow = row + 1; rowBelow < 6; rowBelow++)
		{
			for (int col = row + 1; col < 6; col++)
			{
				m[XY(rowBelow, col)] -= m[XY(rowBelow, row)] * m[XY(row, col)];
			}
		}
	}
}


__global__ void forwardEliminationRight(Node* nodes, int startIdx, int nodesCount, int rowStart, int elim)
{
	int idx = (blockIdx.x * blockDim.x + threadIdx.x) / (dProps.rightCount / COLUMNS_PER_THREAD);
	if (idx >= nodesCount)
		return;
	int colStart = ((blockIdx.x * blockDim.x + threadIdx.x) % (dProps.rightCount / COLUMNS_PER_THREAD))*COLUMNS_PER_THREAD;
	int nodeIdx = startIdx + idx;
	float* m = nodes[nodeIdx].m;
	float** x = nodes[nodeIdx].x;
	for (int row = rowStart; row < elim; row++)
	{
		for (int col = colStart; col < colStart + COLUMNS_PER_THREAD; col++)
		{
			x[row][col] /= m[XY(row, row)];
		}
		for (int rowBelow = row + 1; rowBelow < 6; rowBelow++)
		{
			for (int col = colStart; col < colStart + COLUMNS_PER_THREAD; col++)
			{
				x[rowBelow][col] -= m[XY(rowBelow, row)] * x[row][col];
			}
		}
	}
}

__global__ void assignTestRightSize(Node* node, float* x)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= 1)
		return;
	node->x[0] = x;
	node->x[1] = x+4;
	node->x[2] = x+4*2;
	node->x[3] = x+4*3;
	node->x[4] = x+4*4;
	node->x[5] = x+4*5;
}

void testGaussianElimination()
{
	Properties props = getProperities(1, 4);
	ERRCHECK(cudaMemcpyToSymbol(dProps, &props, sizeof(Properties)));
	Node node;
	float m[] = {
		1, 1, -2, 1, 3, -1,
		2, -1, 1, 2, 1, -3,
		1, 3, -3, -1, 2, 1,
		5, 2, -1, -1, 2, 1,
		-3, -1, 2, 3, 1, 3,
		4, 3, 1, -6, -3, -2
	};
	memcpy(node.m, m, sizeof(float) * MSIZE);
	Node* dNode;
	printNode(node);
	ERRCHECK(cudaMalloc(&dNode, sizeof(Node)));
	ERRCHECK(cudaMemcpy(dNode, &node, sizeof(Node), cudaMemcpyHostToDevice));
	float x[] = {4,4,4,4,20,20,20,20,-15,-15,-15,-15,-3,-3,-3,-3,16,16,16,16,-27,-27,-27,-27};
	float* dX;
	ERRCHECK(cudaMalloc(&dX, sizeof(x)));
	ERRCHECK(cudaMemcpy(dX, &x, sizeof(x) , cudaMemcpyHostToDevice));
	assignTestRightSize << <1, 1 >> >(dNode, dX);
	ERRCHECK(cudaGetLastError());
	ERRCHECK(cudaDeviceSynchronize());
	forwardEliminationLeft << <1, 1 >> >(dNode, 0, 1, 0, 6);
	ERRCHECK(cudaGetLastError());
	ERRCHECK(cudaDeviceSynchronize());
	forwardEliminationRight << <1, 4/COLUMNS_PER_THREAD >> >(dNode, 0, 1, 0, 6);
	ERRCHECK(cudaGetLastError());
	ERRCHECK(cudaDeviceSynchronize());
	backwardSubstitutionRight << <1, 4/COLUMNS_PER_THREAD >> >(dNode, 0, 1, 0, 4);
	ERRCHECK(cudaGetLastError());
	ERRCHECK(cudaDeviceSynchronize());
	ERRCHECK(cudaMemcpy(&node, dNode, sizeof(Node), cudaMemcpyDeviceToHost));
	printNode(node);
	ERRCHECK(cudaMemcpy(x, dX, sizeof(x), cudaMemcpyDeviceToHost));
	float c[] = {1,-2,3,4,2,-1};
	for (int i = 0; i < 6; i++)
	{
		printf("%.1f ", c[i]);
	}
	printf("\n");
	for (int i = 0; i < props.rightCount; i++)
	{
		for (int j= 0; j < 6;j++)
			printf("%.1f ", x[j*props.rightCount+i]);
		printf("\n");
	}
}


__global__ void mergeLeftChild(Node* nodes, int startIdx, int nodesCount)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= nodesCount)
		return;
	int nodeIdx = startIdx + idx;
	Node* parent = &nodes[nodeIdx];

	Node* left = &nodes[LEFT(nodeIdx)];

	parent->m[XY(0, 0)] += left->m[XY(4, 4)];
	parent->m[XY(0, 1)] += left->m[XY(4, 5)];
	parent->m[XY(1, 0)] += left->m[XY(5, 4)];
	parent->m[XY(1, 1)] += left->m[XY(5, 5)];

	parent->m[XY(0, 2)] = left->m[XY(4, 2)];
	parent->m[XY(0, 3)] = left->m[XY(4, 3)];
	parent->m[XY(1, 2)] = left->m[XY(5, 2)];
	parent->m[XY(1, 3)] = left->m[XY(5, 3)];

	parent->m[XY(2, 0)] = left->m[XY(2, 4)];
	parent->m[XY(2, 1)] = left->m[XY(2, 5)];
	parent->m[XY(2, 2)] = left->m[XY(2, 2)];
	parent->m[XY(2, 3)] = left->m[XY(2, 3)];
	parent->m[XY(3, 0)] = left->m[XY(3, 4)];
	parent->m[XY(3, 1)] = left->m[XY(3, 5)];
	parent->m[XY(3, 2)] = left->m[XY(3, 2)];
	parent->m[XY(3, 3)] = left->m[XY(3, 3)];
}

__global__ void mergeRightChild(Node* nodes, int startIdx, int nodesCount)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= nodesCount)
		return;
	int nodeIdx = startIdx + idx;
	Node* parent = &nodes[nodeIdx];
	Node* right = &nodes[RIGHT(nodeIdx)];
	parent->m[XY(0, 0)] += right->m[XY(2, 2)];
	parent->m[XY(0, 1)] += right->m[XY(2, 3)];
	parent->m[XY(1, 0)] += right->m[XY(3, 2)];
	parent->m[XY(1, 1)] += right->m[XY(3, 3)];

	parent->m[XY(0, 4)] = right->m[XY(2, 4)];
	parent->m[XY(0, 5)] = right->m[XY(2, 5)];
	parent->m[XY(1, 4)] = right->m[XY(3, 4)];
	parent->m[XY(1, 5)] = right->m[XY(3, 5)];

	parent->m[XY(4, 0)] = right->m[XY(4, 2)];
	parent->m[XY(4, 1)] = right->m[XY(4, 3)];
	parent->m[XY(4, 4)] = right->m[XY(4, 4)];
	parent->m[XY(4, 5)] = right->m[XY(4, 5)];
	parent->m[XY(5, 0)] = right->m[XY(5, 2)];
	parent->m[XY(5, 1)] = right->m[XY(5, 3)];
	parent->m[XY(5, 4)] = right->m[XY(5, 4)];
	parent->m[XY(5, 5)] = right->m[XY(5, 5)];
}

__global__ void divideLeft(Node* nodes, float* leftSide)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= dProps.bottomNodes)
		return;
	int nodeIdx = (idx < dProps.lastLevelNodes) * (dProps.lastLevelStartIdx + idx) + (idx >= dProps.lastLevelNodes) * (dProps.beforeLastLevelStartIdx + idx);
	//printf("%d %d\n", idx, nodeIdx);
	Node node = nodes[nodeIdx];
	idx *= 5 * 3;
	node.m[XY(2, 3)] = leftSide[idx + 2] / 3;  //maybe swap 2 col with 3 col is also required
	node.m[XY(2, 2)] = leftSide[idx + 3] / 2;
	node.m[XY(2, 1)] = leftSide[idx + 4];

	node.m[XY(3, 3)] = leftSide[idx + 6] / 2;
	node.m[XY(3, 2)] = leftSide[idx + 7] * 2 / 3;
	node.m[XY(3, 1)] = leftSide[idx + 8];
	node.m[XY(3, 4)] = leftSide[idx + 9];

	node.m[XY(1, 3)] = leftSide[idx + 10];
	node.m[XY(1, 2)] = leftSide[idx + 11];
	node.m[XY(1, 1)] = leftSide[idx + 12];
	node.m[XY(1, 4)] = leftSide[idx + 13];
	node.m[XY(1, 5)] = leftSide[idx + 14];

	node.m[XY(4, 2)] = leftSide[idx + 15];
	node.m[XY(4, 1)] = leftSide[idx + 16];
	node.m[XY(4, 4)] = leftSide[idx + 17] * 2 / 3;
	node.m[XY(4, 5)] = leftSide[idx + 18] / 2;

	node.m[XY(5, 1)] = leftSide[idx + 20];
	node.m[XY(5, 4)] = leftSide[idx + 21] / 2;
	node.m[XY(5, 5)] = leftSide[idx + 22] / 3;
	nodes[nodeIdx] = node;
	//	printNode(node);
}

__global__ void divideFirstAndLast(Node* nodes, float* leftSide)
{
	int nodeIdx = dProps.lastLevelStartIdx;
	nodes[dProps.remainingNodes].m[XY(3, 3)] = leftSide[2];
	nodes[dProps.remainingNodes].m[XY(3, 2)] = leftSide[3];

	nodes[dProps.remainingNodes].m[XY(2, 3)] = leftSide[6];
	nodes[dProps.remainingNodes].m[XY(2, 2)] = leftSide[7];

	nodeIdx = (dProps.beforeLastLevelNodes == 0) * (dProps.heapNodes - 1) + (dProps.beforeLastLevelNodes != 0) * (dProps.heapNodes - dProps.lastLevelNodes - 1);
	nodes[nodeIdx].m[XY(4, 4)] = leftSide[dProps.leftSize - 25 + 17];
	nodes[nodeIdx].m[XY(4, 5)] = leftSide[dProps.leftSize - 25 + 18];

	nodes[nodeIdx].m[XY(5, 4)] = leftSide[dProps.leftSize - 25 + 21];
	nodes[nodeIdx].m[XY(5, 5)] = leftSide[dProps.leftSize - 25 + 22];
	//printf("|%d %d|\n", dProps.lastLevelStartIdx, nodeIdx);
}

__global__ void assignRightSideToNodes(Node* nodes, float* rightSide)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	Node node = nodes[idx];
	//	node.x = rightSide + idx*FIRST_LEVEL_SIZE*dRightCols;
}

void leftSideInit(float* leftSide, int size)
{
	for (int i = 0; i < size; i++)
	{
		leftSide[i] = 6;//(i+1)%26;
	}
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


void distributeInputAmongNodes(Node* dNodes, float* dLeftSide, Properties props)
{
	divideLeft << <(props.bottomNodes + THREADS) / THREADS, THREADS >> >(dNodes, dLeftSide);
	ERRCHECK(cudaGetLastError());
	ERRCHECK(cudaDeviceSynchronize());
	divideFirstAndLast << <1, 1 >> >(dNodes, dLeftSide);
	ERRCHECK(cudaGetLastError());
	ERRCHECK(cudaDeviceSynchronize());
	//divide right side somehow
}


int eliminateFirstRow(Node* dNodes, Properties props) //5x5 matrices
{
	forwardEliminationLeft << <BLOCKS(props.bottomNodes), THREADS >> >(dNodes, props.lastLevelStartIdx, props.bottomNodes, 1, 1);
	forwardEliminationRight << <BLOCKS(props.bottomNodes), THREADS >> >(dNodes, props.lastLevelStartIdx, props.bottomNodes, 1, 1);
	if (props.beforeLastLevelNodes > 0)
	{
		forwardEliminationLeft << <BLOCKS(props.beforeLastLevelNodes), THREADS >> >(dNodes, props.lastLevelStartIdx, props.bottomNodes, 1, 1); 
		forwardEliminationRight << <BLOCKS(props.beforeLastLevelNodes), THREADS >> >(dNodes, props.lastLevelStartIdx, props.bottomNodes, 1, 1);
	}
	return props.beforeLastLevelNotBottomNodes;
}

void run(Node* dNodes, float* dLeftSide, Properties props, float* leftSize)
{
	distributeInputAmongNodes(dNodes, dLeftSide, props);
	int nodesCount = eliminateFirstRow(dNodes, props);

	for (int start = PARENT(props.lastLevelStartIdx); start > 0; nodesCount = (start + 1) / 2 , start = PARENT(start))//order matters
	{
		mergeLeftChild << <BLOCKS(nodesCount), THREADS >> >(dNodes, start, nodesCount);
		mergeRightChild << <BLOCKS(nodesCount), THREADS >> >(dNodes, start, nodesCount);
		forwardEliminationLeft << <BLOCKS(nodesCount), THREADS >> >(dNodes, start, nodesCount, 0, 2);
		forwardEliminationRight << <BLOCKS(nodesCount*(props.rightCount / COLUMNS_PER_THREAD)), THREADS >> >(dNodes, start, nodesCount, 0, 2);
	}
//	nodesCount = 2;
//	for (int start = 1; start < ; nodesCount = (start + 1) / 2, start = PARENT(start))//order matters
//	{
//
//	}
}


int main()
{
	ERRCHECK(cudaSetDevice(0));
	testGaussianElimination();
	getch();
	return 0;
	clock_t start, end;
	//	int leftCount = (3*4+2)*10e5;
	int leftCount = 3 * 3 + 2;
	int rightCount = 1;
	const Properties props = getProperities(leftCount, rightCount);
	ERRCHECK(cudaMemcpyToSymbol(dProps, &props, sizeof(Properties)));

	float* leftSide = new float[props.leftSize];
	//	float * rightSide = new float[rightSize];
	Node* nodes = new Node[props.heapNodes];
	Node* dNodes = nullptr;
	float* dLeftSide = nullptr;
	leftSideInit(leftSide, props.leftSize);
	ERRCHECK(cudaMalloc(&dNodes, sizeof(Node)* props.heapNodes));
	ERRCHECK(cudaMemset(dNodes, 0, sizeof(Node)*props.heapNodes));
	ERRCHECK(cudaMalloc(&dLeftSide, sizeof(float)*props.leftSize));
	ERRCHECK(cudaMemcpy(dLeftSide, leftSide, sizeof(float)*props.leftSize, cudaMemcpyHostToDevice));
	showMemoryConsumption();
	start = clock();
	divideLeft << <(props.bottomNodes + 512) / 512, 512 >> >(dNodes, dLeftSide);
	ERRCHECK(cudaGetLastError());
	ERRCHECK(cudaDeviceSynchronize());
	end = clock();
	printf("time %f\n", (float)(end - start) / CLOCKS_PER_SEC);
	divideFirstAndLast << <1, 1 >> >(dNodes, dLeftSide);
	ERRCHECK(cudaGetLastError());
	ERRCHECK(cudaDeviceSynchronize());
	mergeLeftChild << <1, 1 >> >(dNodes, PARENT(props.lastLevelStartIdx), 1);
	ERRCHECK(cudaGetLastError());
	ERRCHECK(cudaDeviceSynchronize());
	mergeRightChild << <1, 1 >> >(dNodes, PARENT(props.lastLevelStartIdx), 1);
	ERRCHECK(cudaGetLastError());
	ERRCHECK(cudaDeviceSynchronize());
	ERRCHECK(cudaMemcpy(nodes, dNodes, sizeof(Node) * props.heapNodes, cudaMemcpyDeviceToHost));
	printNode(nodes[0]);
	printNode(nodes[1]);
	printNode(nodes[2]);
	printNode(nodes[3]);
	printNode(nodes[4]);
	delete[] nodes;
	cudaFree(dNodes);

	ERRCHECK(cudaDeviceReset());

	getch();
	return 0;
}


__device__ __host__ void printNode(Node node)
{
	for (int i = 0; i < 6; i++)
		printf("%.1f %.1f %.1f %.1f %.1f %.1f\n", node.m[XY(i, 0)], node.m[XY(i, 1)], node.m[XY(i, 2)], node.m[XY(i, 3)], node.m[XY(i, 4)], node.m[XY(i, 5)]);
	printf("\n");
}
