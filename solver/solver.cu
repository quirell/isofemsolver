﻿#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <conio.h>
#include <new>
#include "constants.cuh"
#include "helpers.cuh"
#include "solver.cuh"
#include "test_functions.cuh"
#include <cstring>
#include <ctime>
#include "bitmap_approx.cuh"
#include <string>

__constant__ Properties dProps;

__global__ void backwardSubstitutionRight(Node* nodes, int startIdx, int nodesCount, int end, int elim)
{
	int idx = (blockIdx.x * blockDim.x + threadIdx.x) / (dProps.rightCount / COLUMNS_PER_THREAD);
	if (idx >= nodesCount)
		return;
	int colStart = ((blockIdx.x * blockDim.x + threadIdx.x) % (dProps.rightCount / COLUMNS_PER_THREAD)) * COLUMNS_PER_THREAD;
	int nodeIdx = startIdx + idx;
	number* m = nodes[nodeIdx].m;
	number** x = nodes[nodeIdx].x;
	for (int rcol = colStart; rcol < colStart + COLUMNS_PER_THREAD; rcol++)
	{
		for (int row = elim; row >= end; row--)//max elim == 4,5th is already done after elimination
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
	number* m = nodes[nodeIdx].m;
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
	int colStart = ((blockIdx.x * blockDim.x + threadIdx.x) % (dProps.rightCount / COLUMNS_PER_THREAD)) * COLUMNS_PER_THREAD;
	int nodeIdx = startIdx + idx;
	number* m = nodes[nodeIdx].m;
	number** x = nodes[nodeIdx].x;
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

__global__ void divideLeft(Node* nodes, number* leftSide)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= dProps.bottomNodes)
		return;
	int nodeIdx = (idx < dProps.lastLevelNodes) * (dProps.lastLevelStartIdx + idx) + (idx >= dProps.lastLevelNodes) * (dProps.beforeLastLevelStartIdx + idx);

	Node node = nodes[nodeIdx];
	idx *= 5 * 3;

	node.m[XY(1, 1)] = leftSide[idx + 12];
	node.m[XY(1, 2)] = leftSide[idx + 10];
	node.m[XY(1, 3)] = leftSide[idx + 11];
	node.m[XY(1, 4)] = leftSide[idx + 13];
	node.m[XY(1, 5)] = leftSide[idx + 14];

	node.m[XY(2, 1)] = leftSide[idx + 4];
	node.m[XY(2, 2)] = leftSide[idx + 2] / 2.0;
	node.m[XY(2, 3)] = leftSide[idx + 3] / 2.0;

	node.m[XY(3, 1)] = leftSide[idx + 8];
	node.m[XY(3, 2)] = leftSide[idx + 6] / 2.0;
	node.m[XY(3, 3)] = leftSide[idx + 7] / 2.0;
	node.m[XY(3, 4)] = leftSide[idx + 9];

	node.m[XY(4, 1)] = leftSide[idx + 16];

	node.m[XY(4, 3)] = leftSide[idx + 15];
	node.m[XY(4, 4)] = leftSide[idx + 17] / 2.0;
	node.m[XY(4, 5)] = leftSide[idx + 18] / 2.0;

	node.m[XY(5, 1)] = leftSide[idx + 20];


	node.m[XY(5, 4)] = leftSide[idx + 21] / 2.0;
	node.m[XY(5, 5)] = leftSide[idx + 22] / 2.0;


	nodes[nodeIdx] = node;

}

__global__ void divideFirstAndLast(Node* nodes, number* leftSide)
{
	int nodeIdx = dProps.lastLevelStartIdx;
	nodes[nodeIdx].m[XY(2, 2)] = leftSide[2];
	nodes[nodeIdx].m[XY(2, 3)] = leftSide[3];

	nodes[nodeIdx].m[XY(3, 2)] = leftSide[6];
	nodes[nodeIdx].m[XY(3, 3)] = leftSide[7];

	nodeIdx = (dProps.beforeLastLevelNodes == 0) * (dProps.heapNodes - 1) + (dProps.beforeLastLevelNodes != 0) * (dProps.heapNodes - dProps.lastLevelNodes - 1);
	nodes[nodeIdx].m[XY(4, 4)] = leftSide[dProps.leftSize - 25 + 17];
	nodes[nodeIdx].m[XY(4, 5)] = leftSide[dProps.leftSize - 25 + 18];

	nodes[nodeIdx].m[XY(5, 4)] = leftSide[dProps.leftSize - 25 + 21];
	nodes[nodeIdx].m[XY(5, 5)] = leftSide[dProps.leftSize - 25 + 22];

}

inline __device__ void divideRightNode(Node* nodes, number* rightSide, int ord, int nodeIdx, int idx, int rightCount)
{
	Node* node = &nodes[nodeIdx];
	rightSide += ord * 3 * rightCount;
	idx %= rightCount;
	node->x[1][idx] = (rightSide + rightCount * 2)[idx]; //n+2  //swapped first and third row, and then second and third
	node->x[2][idx] = rightSide[idx] / (1 * (ord == 0) + 2 * (ord > 0));//n TODO extract expressions to different function
	node->x[3][idx] = (rightSide + rightCount)[idx] / (1 * (ord == 0) + 2 * (ord > 0)); //n+1
	node->x[4][idx] = (rightSide + rightCount * 3)[idx] / (1 * (ord == dProps.bottomNodes - 1) + 2 * (ord < dProps.bottomNodes - 1)); //n+3
	node->x[5][idx] = (rightSide + rightCount * 4)[idx] / (1 * (ord == dProps.bottomNodes - 1) + 2 * (ord < dProps.bottomNodes - 1)); //n+4	
}

__global__ void divideRight(Node* nodes, number* rightSide)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= dProps.bottomNodes * dProps.rightCount)
		return;
	int ord = idx / dProps.rightCount;
	int nodeIdx = (ord < dProps.lastLevelNodes) * (dProps.lastLevelStartIdx + ord) + (ord >= dProps.lastLevelNodes) * (dProps.beforeLastLevelStartIdx + ord);
	divideRightNode(nodes, rightSide, ord, nodeIdx, idx, dProps.rightCount);
}

inline __device__ void assignRightNodeMem(Node* nodes, number* rightSideMem, int nodeIdx, Properties props)
{
	Node* node = &nodes[nodeIdx];
	int start = nodeIdx * props.rightCount * 6;
	node->x[0] = rightSideMem + start;
	node->x[1] = rightSideMem + start + props.rightCount;
	node->x[2] = rightSideMem + start + props.rightCount * 2;
	node->x[3] = rightSideMem + start + props.rightCount * 3;
	node->x[4] = rightSideMem + start + props.rightCount * 4;
	node->x[5] = rightSideMem + start + props.rightCount * 5;
}

__global__ void assignRightSideMem(Node* nodes, number* rightSideMem)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= dProps.heapNodes)
		return;
	assignRightNodeMem(nodes, rightSideMem, idx, dProps);
}

inline __device__ void mergeRightSideNode(Node* nodes, int idx, int nodeIdx)
{
	Node* parent = &nodes[nodeIdx];
	Node* left = &nodes[LEFT(nodeIdx)];
	Node* right = &nodes[RIGHT(nodeIdx)];
	idx %= dProps.rightCount;
	parent->x[0][idx] = left->x[4][idx] + right->x[2][idx];
	parent->x[1][idx] = left->x[5][idx] + right->x[3][idx];
	parent->x[2][idx] = left->x[2][idx];
	parent->x[3][idx] = left->x[3][idx];
	parent->x[4][idx] = right->x[4][idx];
	parent->x[5][idx] = right->x[5][idx];
}

__global__ void mergeRightSideLayer(Node* nodes, int startNode, int rightSidesCount)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= rightSidesCount)
		return;
	int nodeIdx = startNode + idx / dProps.rightCount;
	mergeRightSideNode(nodes, idx, nodeIdx);
}

inline __device__ __host__ void assignParentToChildren(Node* nodes, int nodeIdx)
{
	Node* parent = &nodes[nodeIdx];
	Node* left = &nodes[LEFT(nodeIdx)];
	Node* right = &nodes[RIGHT(nodeIdx)];
	left->x[2] = parent->x[2]; //it's enough to assign pointers because those rows won't be modified 
	left->x[3] = parent->x[3];
	left->x[4] = parent->x[0];
	left->x[5] = parent->x[1];

	right->x[2] = parent->x[0];
	right->x[3] = parent->x[1];
	right->x[4] = parent->x[4];
	right->x[5] = parent->x[5];
}

__global__ void assignParentToChildrenLayer(Node* nodes, int startNode, int nodesCount)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= nodesCount)
		return;
	idx += startNode;
	assignParentToChildren(nodes, idx);
}

void distributeInputAmongNodes(Node* dNodes, number* dLeftSide, number* dRightSideMem, number* dRightSide, Properties props)
{
	divideLeft << <BLOCKS(props.bottomNodes), THREADS >> >(dNodes, dLeftSide);
	ERRCHECK(cudaGetLastError());
	ERRCHECK(cudaDeviceSynchronize());
	divideFirstAndLast << <1, 1 >> >(dNodes, dLeftSide);
	ERRCHECK(cudaGetLastError());
	ERRCHECK(cudaDeviceSynchronize());
	assignRightSideMem << <BLOCKS(props.heapNodes), THREADS >> >(dNodes, dRightSideMem);
	ERRCHECK(cudaGetLastError());
	ERRCHECK(cudaDeviceSynchronize());
	divideRight<<<BLOCKS(props.bottomNodes*props.rightCount),THREADS>>>(dNodes, dRightSide);
	ERRCHECK(cudaGetLastError());
	ERRCHECK(cudaDeviceSynchronize());
}

void eliminateFirstRow(Node* dNodes, Properties props) //5x5 matrices
{
	forwardEliminationLeft << <BLOCKS(props.lastLevelNodes), THREADS >> >(dNodes, props.lastLevelStartIdx, props.lastLevelNodes, 1, 2);
	ERRCHECK(cudaGetLastError());
	ERRCHECK(cudaDeviceSynchronize());
	forwardEliminationRight << <BLOCKS(props.lastLevelNodes*(props.rightCount / COLUMNS_PER_THREAD)), THREADS >> >(dNodes, props.lastLevelStartIdx, props.lastLevelNodes, 1, 2);
	ERRCHECK(cudaGetLastError());
	ERRCHECK(cudaDeviceSynchronize());
	if (props.beforeLastLevelNodes > 0)
	{
		forwardEliminationLeft << <BLOCKS(props.beforeLastLevelNodes), THREADS >> >(dNodes, props.remainingNodes, props.beforeLastLevelNodes, 1, 2);
		ERRCHECK(cudaGetLastError());
		ERRCHECK(cudaDeviceSynchronize());
		forwardEliminationRight << <BLOCKS(props.beforeLastLevelNodes*(props.rightCount / COLUMNS_PER_THREAD)), THREADS >> >(dNodes, props.remainingNodes, props.beforeLastLevelNodes, 1, 2);
		ERRCHECK(cudaGetLastError());
		ERRCHECK(cudaDeviceSynchronize());
	}
}

void eliminateRoot(Node* dNodes, Properties props)
{
	mergeRightSideLayer << <BLOCKS(props.rightCount), THREADS>> >(dNodes, 0, props.rightCount);
	ERRCHECK(cudaGetLastError());
	ERRCHECK(cudaDeviceSynchronize());
	mergeLeftChild << <1,1>> >(dNodes, 0, 1);
	ERRCHECK(cudaGetLastError());
	ERRCHECK(cudaDeviceSynchronize());
	mergeRightChild << <1,1>> >(dNodes, 0, 1);
	ERRCHECK(cudaGetLastError());
	ERRCHECK(cudaDeviceSynchronize());
	forwardEliminationLeft << <1,1>> >(dNodes, 0, 1, 0, 6);
	ERRCHECK(cudaGetLastError());
	ERRCHECK(cudaDeviceSynchronize());
	forwardEliminationRight << <BLOCKS((props.rightCount / COLUMNS_PER_THREAD)), THREADS >> >(dNodes, 0, 1, 0, 6);
	ERRCHECK(cudaGetLastError());
	ERRCHECK(cudaDeviceSynchronize());
	backwardSubstitutionRight<<<BLOCKS((props.rightCount / COLUMNS_PER_THREAD)), THREADS >>>(dNodes, 0, 1, 0, 4);
	ERRCHECK(cudaGetLastError());
	ERRCHECK(cudaDeviceSynchronize());
	assignParentToChildrenLayer<<<1,1>>>(dNodes, 0, 1);
	ERRCHECK(cudaGetLastError());
	ERRCHECK(cudaDeviceSynchronize());
}


void run(Node* dNodes, number* dLeftSide, Properties props, number* dRightSide, number* dRightSideMem)
{
	distributeInputAmongNodes(dNodes, dLeftSide, dRightSideMem, dRightSide, props);
	eliminateFirstRow(dNodes, props);
	int nodesCount = props.beforeLastLevelNotBottomNodes;

	for (int start = PARENT(props.lastLevelStartIdx); start > 0; nodesCount = (start + 1) / 2 , start = PARENT(start))//order matters
	{
		mergeRightSideLayer<<<BLOCKS(nodesCount*props.rightCount), THREADS>>>(dNodes, start, nodesCount * props.rightCount);
		ERRCHECK(cudaGetLastError());
		ERRCHECK(cudaDeviceSynchronize());
		mergeLeftChild << <BLOCKS(nodesCount), THREADS >> >(dNodes, start, nodesCount);
		ERRCHECK(cudaGetLastError());
		ERRCHECK(cudaDeviceSynchronize());
		mergeRightChild << <BLOCKS(nodesCount), THREADS >> >(dNodes, start, nodesCount);
		ERRCHECK(cudaGetLastError());
		ERRCHECK(cudaDeviceSynchronize());
		forwardEliminationLeft << <BLOCKS(nodesCount), THREADS >> >(dNodes, start, nodesCount, 0, 2);
		ERRCHECK(cudaGetLastError());
		ERRCHECK(cudaDeviceSynchronize());
		forwardEliminationRight << <BLOCKS(nodesCount*(props.rightCount / COLUMNS_PER_THREAD)), THREADS >> >(dNodes, start, nodesCount, 0, 2);
		ERRCHECK(cudaGetLastError());
		ERRCHECK(cudaDeviceSynchronize());
	}
	eliminateRoot(dNodes, props);
	nodesCount = props.heapNodes == 5 ? 1 : 2; //for smallest size tree is 1, otherwise 2
	for (int start = 1; start < PARENT(props.lastLevelStartIdx); start = LEFT(start) , nodesCount *= 2)
	{
		backwardSubstitutionRight<<<BLOCKS(nodesCount*(props.rightCount / COLUMNS_PER_THREAD)), THREADS >>>(dNodes, start, nodesCount, 0, 1);
		ERRCHECK(cudaGetLastError());
		ERRCHECK(cudaDeviceSynchronize());
		assignParentToChildrenLayer << <BLOCKS(nodesCount),THREADS>> >(dNodes, start, nodesCount);
		ERRCHECK(cudaGetLastError());
		ERRCHECK(cudaDeviceSynchronize());
	}

	backwardSubstitutionRight <<<BLOCKS(props.beforeLastLevelNotBottomNodes*(props.rightCount / COLUMNS_PER_THREAD)), THREADS >> >(dNodes, PARENT(props.lastLevelStartIdx), props.beforeLastLevelNotBottomNodes, 0, 1);
	ERRCHECK(cudaGetLastError());
	ERRCHECK(cudaDeviceSynchronize());
	if (props.beforeLastLevelNodes > 0)
	{
		backwardSubstitutionRight << <BLOCKS(props.beforeLastLevelNodes*(props.rightCount / COLUMNS_PER_THREAD)), THREADS >> >(dNodes, props.remainingNodes, props.beforeLastLevelNodes, 1, 1);
		ERRCHECK(cudaGetLastError());
		ERRCHECK(cudaDeviceSynchronize());
	}
	assignParentToChildrenLayer << <BLOCKS(props.beforeLastLevelNotBottomNodes), THREADS >> >(dNodes, PARENT(props.lastLevelStartIdx), props.beforeLastLevelNotBottomNodes);
	ERRCHECK(cudaGetLastError());
	ERRCHECK(cudaDeviceSynchronize());
	backwardSubstitutionRight << <BLOCKS(props.lastLevelNodes*(props.rightCount / COLUMNS_PER_THREAD)), THREADS >> >(dNodes, props.lastLevelStartIdx, props.lastLevelNodes, 1, 1);
	ERRCHECK(cudaGetLastError());
	ERRCHECK(cudaDeviceSynchronize());
}


const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;

__global__ void transpose32(number* out, const number* in, unsigned dim0, unsigned dim1)
{
	__shared__ number shrdMem[TILE_DIM][TILE_DIM + 1];

	unsigned lx = threadIdx.x;
	unsigned ly = threadIdx.y;

	unsigned gx = lx + blockDim.x * blockIdx.x;
	unsigned gy = ly + TILE_DIM * blockIdx.y;

#pragma unroll
	for (unsigned repeat = 0; repeat < TILE_DIM; repeat += blockDim.y)
	{
		unsigned gy_ = gy + repeat;
		if (gx < dim0 && gy_ < dim1)
			shrdMem[ly + repeat][lx] = in[gy_ * dim0 + gx];
	}
	__syncthreads();

	gx = lx + blockDim.x * blockIdx.y;
	gy = ly + TILE_DIM * blockIdx.x;

#pragma unroll
	for (unsigned repeat = 0; repeat < TILE_DIM; repeat += blockDim.y)
	{
		unsigned gy_ = gy + repeat;
		if (gx < dim1 && gy_ < dim0)
			out[gy_ * dim0 + gx] = shrdMem[lx][ly + repeat];
	}
}

__global__ void mergeLeavesKernel(Node* bottomNodes, number* rightSide)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= dProps.bottomNodes * dProps.rightCount)
		return;
	int ordIdx = idx / dProps.rightCount;
	int nodeIdx = (ordIdx < dProps.lastLevelNodes) * (ordIdx + dProps.beforeLastLevelNodes) + (ordIdx >= dProps.lastLevelNodes) * (ordIdx - dProps.lastLevelNodes);

	Node node = bottomNodes[nodeIdx];
	int i = idx % dProps.rightCount;
	rightSide[ordIdx * dProps.rightCount * 3 + i] = node.x[2][i];
	rightSide[ordIdx * dProps.rightCount * 3 + dProps.rightCount + i] = node.x[3][i];
	rightSide[ordIdx * dProps.rightCount * 3 + dProps.rightCount * 2 + i] = node.x[1][i];
}

void mergeLeavesIntoArray(Properties props, number* dRightSideCopy, number* dRightSideMem, Node* dNodes)
{
	mergeLeavesKernel << <BLOCKS(props.bottomNodes*props.rightCount), THREADS >> >(dNodes + props.remainingNodes, dRightSideCopy);
	ERRCHECK(cudaGetLastError());
	ERRCHECK(cudaDeviceSynchronize());//copy last two right side rows
	ERRCHECK(cudaMemcpy(dRightSideCopy + props.rightSize - 2 * props.rightCount, dRightSideMem + 4 * props.rightCount, sizeof(number) * 2 * props.rightCount, cudaMemcpyDeviceToDevice));
	ERRCHECK(cudaGetLastError());
	ERRCHECK(cudaDeviceSynchronize());
}

void runComputing(const int size, int iters, char* bitmapPath = nullptr, char* storePath = nullptr, float* colors = nullptr)
{
	if (size % 3 != 2)
		throw "size-2 must be divisible by 3 without a remainder";

	Properties props = getProperities(size, size);
	ERRCHECK(cudaMemcpyToSymbol(dProps, &props, sizeof(Properties)));
	number* leftSide = nullptr;
	number* rightSide = nullptr;
	number* dRightSideCopy = nullptr;
	number* dRightSideMem = nullptr;

	Node* dNodes = nullptr;
	number* dLeftSide = nullptr;
	number* dRightSide = nullptr;
	if (bitmapPath == nullptr)
		generateTestEquation(size, props.rightCount, &leftSide, &rightSide);
	ERRCHECK(cudaMalloc(&dNodes, sizeof(Node)* props.heapNodes));
	ERRCHECK(cudaMalloc(&dLeftSide, sizeof(number)*props.leftSize));
	ERRCHECK(cudaMalloc(&dRightSide, sizeof(number)*props.rightSize));
	ERRCHECK(cudaMalloc(&dRightSideCopy, sizeof(number)*props.rightSize));
	ERRCHECK(cudaMalloc(&dRightSideMem, sizeof(number)*props.rightSizeMem));
	clock_t start, end;
	start = clock();
	for (int i = 0; i < iters; i++)
	{
		if (bitmapPath != nullptr)
		{
			BSpline2d bSplines;
			rightSide = generateBitmapRightSide(bitmapPath, size - 2, &bSplines, colors);
			leftSide = generateBitmapLeftSide(bSplines, size);
			printLeftAndRight(leftSide, rightSide, size, props.rightCount);
		}

		ERRCHECK(cudaMemset(dNodes, 0, sizeof(Node)*props.heapNodes));
		ERRCHECK(cudaMemcpy(dLeftSide, leftSide, sizeof(number)*props.leftSize, cudaMemcpyHostToDevice));
		ERRCHECK(cudaMemcpy(dRightSide, rightSide, sizeof(number)*props.rightSize, cudaMemcpyHostToDevice));
		//		showMemoryConsumption();
		run(dNodes, dLeftSide, props, dRightSide, dRightSideMem);

		mergeLeavesIntoArray(props, dRightSideCopy, dRightSideMem, dNodes);

		transpose32 << <dim3((props.leftCount + TILE_DIM) / TILE_DIM, (props.rightCount + TILE_DIM) / TILE_DIM), dim3(TILE_DIM, BLOCK_ROWS) >> >(dRightSide, dRightSideCopy, props.leftCount, props.rightCount);
		ERRCHECK(cudaGetLastError());
		ERRCHECK(cudaDeviceSynchronize());


		//RUN FOR TRANSPOSED MATRIX
		printLeftAndRight(leftSide, rightSide, size, props.rightCount);
		ERRCHECK(cudaMemset(dNodes, 0, sizeof(Node)*props.heapNodes));
		run(dNodes, dLeftSide, props, dRightSide, dRightSideMem);
		//PUT RESULT INTO ONE ARRAY
		mergeLeavesIntoArray(props, dRightSideCopy, dRightSideMem, dNodes);

		if (bitmapPath != nullptr)
		{
			delete[] rightSide;
			delete[] leftSide;
		}
	}
	end = clock();
	printf("time %f\n", ((number)(end - start) / CLOCKS_PER_SEC) / iters);

	number* solution = new number[size * size];
	cudaMemcpy(solution, dRightSideCopy, sizeof(number) * props.rightSize, cudaMemcpyDeviceToHost);
	int resolution = (size - 2);
	getBitmapApprox(solution, size - 2, resolution, storePath);

	if (bitmapPath == nullptr)
	{
		delete[] leftSide;
		delete[] rightSide;
	}

	ERRCHECK(cudaFree(dRightSide));
	ERRCHECK(cudaFree(dRightSideCopy));
	ERRCHECK(cudaFree(dRightSideMem));
	ERRCHECK(cudaFree(dNodes));
	ERRCHECK(cudaFree(dLeftSide));
}

int main()
{
	//	generate2DSplineIntegrals(9, 3);
	//	readBmp("C:/Users/quirell/Pictures/Untitled.bmp");
	//	generateBitmapRightSide("C:/Users/quirell/Desktop/magisterka/bitmapy/e253ppe15white.bmp",253);
	//	measureGenBitmap("C:/Users/quirell/Desktop/magisterka/bitmapy/e510ppe20.bmp", 510, 1);
	//
	//	runComputing(146, 1, "C:/Users/quirell/Desktop/magisterka/bitmapy/test10px144.bmp", "C:/Users/quirell/Desktop/magisterka/bitmapy/outred.bmp",new float[3]{1,0,0});
	//	runComputing(146, 1, "C:/Users/quirell/Desktop/magisterka/bitmapy/test10px144.bmp", "C:/Users/quirell/Desktop/magisterka/bitmapy/outgreen.bmp", new float[3] {0, 1, 0});
	//	runComputing(146, 1, "C:/Users/quirell/Desktop/magisterka/bitmapy/test10px144.bmp", "C:/Users/quirell/Desktop/magisterka/bitmapy/outblue.bmp", new float[3] {0, 0, 1});
		runComputing(167, 1, "C:/Users/quirell/Desktop/magisterka/bitmapy/shibuya55px27.bmp", "C:/Users/quirell/Desktop/magisterka/util/outred.bmp",new float[3]{1,0,0});
		runComputing(167, 1, "C:/Users/quirell/Desktop/magisterka/bitmapy/shibuya55px27.bmp", "C:/Users/quirell/Desktop/magisterka/util/outgreen.bmp", new float[3] {0, 1, 0});
		runComputing(167, 1, "C:/Users/quirell/Desktop/magisterka/bitmapy/shibuya55px27.bmp", "C:/Users/quirell/Desktop/magisterka/util/outblue.bmp", new float[3] {0, 0, 1});
	//	runComputing(14, 1);
//	runComputing(8, 1, "C:/Users/quirell/Desktop/magisterka/bitmapy/4colors2px6.bmp");
	//	runComputing(255, 1, "C:/Users/quirell/Desktop/magisterka/bitmapy/e510ppe40.bmp");
	//		runComputing(65, 100);
	//		runComputing(128, 100);
	//		runComputing(255, 100);
	//		runComputing(512, 100);
	//		runComputing(1022, 100);
	//		runComputing(2048, 100);
	//		runComputing(4097, 100);
	//		runComputing(6146, 30);
	//		runComputing(8192, 30);
	//		runComputing(65, 1);
	//		runComputing(128, 1);
	//		runComputing(255, 1);
	//		runComputing(512, 1);
	//		runComputing(1022, 1);
	//		runComputing(2048, 1);
	//		runComputing(4097, 1);
	//		runComputing(6146, 1);
	//		runComputing(8192, 1);
	//		runComputing(65, 30, "C:/Users/quirell/Desktop/magisterka/bitmapy/e63ppe15.bmp");
	//		runComputing(128, 30, "C:/Users/quirell/Desktop/magisterka/bitmapy/e126ppe15.bmp");
	//		runComputing(255, 1, "C:/Users/quirell/Desktop/magisterka/bitmapy/e253ppe15white.bmp");
	//		runComputing(512, 15, "C:/Users/quirell/Desktop/magisterka/bitmapy/e510ppe15.bmp");
	//		runComputing(1022,1, "C:/Users/quirell/Desktop/magisterka/bitmapy/e1020ppe15.bmp");
	//			number * left;
	//			number * right;
	//			generateTestEquation(14, 1, &left, &right);
	//			testRun(44);
	//	testMultipleRun(1,1022);
	//	getch();
	//	testDistributeInputAmongNodes();
	//		getch();
	//			ERRCHECK(cudaSetDevice(0));
	//			testGaussianElimination();
	//	number * x, * y;
	//	int s = 8129;
	//	generateTestEquation(s,s, &x,&y);
	getch();
	return 0;
}
