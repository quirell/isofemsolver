#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <conio.h>
#include <new>
#include "constants.cuh"
#include "helpers.cuh"
#include "solver.cuh"
#include "test_functions.cuh"

__constant__ Properties dProps;

__global__ void backwardSubstitutionRight(Node* nodes, int startIdx, int nodesCount, int end, int elim)
{
	int idx = (blockIdx.x * blockDim.x + threadIdx.x) / (dProps.rightCount / COLUMNS_PER_THREAD);
	if (idx >= nodesCount)
		return;
	int colStart = ((blockIdx.x * blockDim.x + threadIdx.x) % (dProps.rightCount / COLUMNS_PER_THREAD)) * COLUMNS_PER_THREAD;
	//	printf("%d %d\n", idx, colStart);
	int nodeIdx = startIdx + idx;
	float* m = nodes[nodeIdx].m;
	float** x = nodes[nodeIdx].x;
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
	float* m = nodes[nodeIdx].m;
	for (int row = start; row < elim; row++)
	{
		for (int col = row + 1; col < 6; col++) //from element after diagonal
		{
			//printf("%.1f/%.1f = ", m[XY(row,col)], m[XY(row, row)]);
			m[XY(row, col)] /= m[XY(row, row)];
//			printf("%.2f  ", m[XY(row, col)]);
		}
//		printf("\n");
		for (int rowBelow = row + 1; rowBelow < 6; rowBelow++)
		{
			for (int col = row + 1; col < 6; col++)
			{
				//printf("%.1f-%.1f*%.1f = ", m[XY(rowBelow, col)], m[XY(rowBelow, row)], m[XY(row, col)]);
				m[XY(rowBelow, col)] -= m[XY(rowBelow, row)] * m[XY(row, col)];
//				printf("%.2f  ", m[XY(rowBelow, col)]);
			}
//			printf("\n");
		}
//		printf("\n\n");
	}
}

__global__ void forwardEliminationRight(Node* nodes, int startIdx, int nodesCount, int rowStart, int elim)
{
	int idx = (blockIdx.x * blockDim.x + threadIdx.x) / (dProps.rightCount / COLUMNS_PER_THREAD);
	if (idx >= nodesCount)
		return;
	int colStart = ((blockIdx.x * blockDim.x + threadIdx.x) % (dProps.rightCount / COLUMNS_PER_THREAD)) * COLUMNS_PER_THREAD;
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

	node.m[XY(1, 1)] = leftSide[idx + 12];
	node.m[XY(1, 2)] = leftSide[idx + 10];
	node.m[XY(1, 3)] = leftSide[idx + 11];
	node.m[XY(1, 4)] = leftSide[idx + 13];
	node.m[XY(1, 5)] = leftSide[idx + 14];

	node.m[XY(2, 1)] = leftSide[idx + 4];
	node.m[XY(2, 2)] = leftSide[idx + 2] / 2;
	node.m[XY(2, 3)] = leftSide[idx + 3] / 2;

	node.m[XY(3, 1)] = leftSide[idx + 8];
	node.m[XY(3, 2)] = leftSide[idx + 6] / 2;
	node.m[XY(3, 3)] = leftSide[idx + 7] / 2;
	node.m[XY(3, 4)] = leftSide[idx + 9];

	node.m[XY(4, 1)] = leftSide[idx + 16];

	node.m[XY(4, 3)] = leftSide[idx + 15];
	node.m[XY(4, 4)] = leftSide[idx + 17] / 2;
	node.m[XY(4, 5)] = leftSide[idx + 18] / 2;

	node.m[XY(5, 1)] = leftSide[idx + 20];


	node.m[XY(5, 4)] = leftSide[idx + 21] / 2;
	node.m[XY(5, 5)] = leftSide[idx + 22] / 2;


	nodes[nodeIdx] = node;
	//	printNode(node);
}

__global__ void divideFirstAndLast(Node* nodes, float* leftSide)
{
	int nodeIdx = dProps.lastLevelStartIdx;
	nodes[nodeIdx].m[XY(2, 3)] = leftSide[2];
	nodes[nodeIdx].m[XY(2, 2)] = leftSide[3];

	nodes[nodeIdx].m[XY(3, 3)] = leftSide[6];
	nodes[nodeIdx].m[XY(3, 2)] = leftSide[7];

	nodeIdx = (dProps.beforeLastLevelNodes == 0) * (dProps.heapNodes - 1) + (dProps.beforeLastLevelNodes != 0) * (dProps.heapNodes - dProps.lastLevelNodes - 1);
	nodes[nodeIdx].m[XY(4, 4)] = leftSide[dProps.leftSize - 25 + 17];
	nodes[nodeIdx].m[XY(4, 5)] = leftSide[dProps.leftSize - 25 + 18];

	nodes[nodeIdx].m[XY(5, 4)] = leftSide[dProps.leftSize - 25 + 21];
	nodes[nodeIdx].m[XY(5, 5)] = leftSide[dProps.leftSize - 25 + 22];
//	printf("|%d %d|\n", dProps.lastLevelStartIdx, nodeIdx);
}

inline __device__ void divideRightNode(Node* nodes, float* rightSide, int ord, int nodeIdx, int idx, int rightCount)
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

__global__ void divideRight(Node* nodes, float* rightSide)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= dProps.bottomNodes * dProps.rightCount)
		return;
	int ord = idx / dProps.rightCount;
	int nodeIdx = (ord < dProps.lastLevelNodes) * (dProps.lastLevelStartIdx + ord) + (ord >= dProps.lastLevelNodes) * (dProps.beforeLastLevelStartIdx + ord);
	divideRightNode(nodes, rightSide, ord, nodeIdx, idx, dProps.rightCount);
}

inline __device__ void assignRightNodeMem(Node* nodes, float* rightSideMem, int nodeIdx, Properties props)
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

__global__ void assignRightSideMem(Node* nodes, float* rightSideMem)
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
	left->x[2] = parent->x[2]; //it's enough to assign pointers because it won't be modified 
	left->x[3] = parent->x[3];
	left->x[4] = parent->x[0];
	left->x[5] = parent->x[1];

	right->x[2] = parent->x[0];
	right->x[3] = parent->x[1];
	right->x[4] = parent->x[4];
	right->x[5] = parent->x[5];
}

 __global__ void assignParentToChildrenLayer(Node* nodes,int startNode,int nodesCount)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= nodesCount)
		return;
	idx += startNode;
	assignParentToChildren(nodes, idx);
}

void distributeInputAmongNodes(Node* dNodes, float* dLeftSide, float* dRightSideMem, float* dRightSide, Properties props)
{
	divideLeft << <BLOCKS(props.bottomNodes), THREADS >> >(dNodes, dLeftSide);
	ERRCHECK(cudaGetLastError());
	ERRCHECK(cudaDeviceSynchronize());
	divideFirstAndLast << <1, 1 >> >(dNodes, dLeftSide);
	ERRCHECK(cudaGetLastError());
	ERRCHECK(cudaDeviceSynchronize());
	ERRCHECK(cudaFree(dLeftSide));
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
	mergeRightSideLayer << <1, props.rightCount>> >(dNodes, 0, props.rightCount);
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

void run(Node* dNodes, float* dLeftSide, Properties props, float* dRightSide, float* dRightSideMem)
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

int main()
{
//	float * left;
//	float * right;
//	generateTestEquation(14, 1, &left, &right);
		testRun();
		getch();
		return 0;
//	testDistributeInputAmongNodes();
//	getch();
//	return 0;
//		ERRCHECK(cudaSetDevice(0));
//		testGaussianElimination();
//		getch();
//		return 0;

}