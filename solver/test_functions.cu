#include "test_functions.cuh"
#include <cstring>
#include <device_launch_parameters.h>
#include "helpers.cuh"
#include "solver.cuh"
#include <ctime>
//copied from solver.cuh assignRightNodeMem
void assignRightNodeMemHost(Node* nodes, float* rightSideMem, int nodeIdx, Properties props)
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

void assignHostRightSide(Properties props, Node* nodes, float* rightSideMem)
{
	for (int i = 0; i < props.heapNodes; i++)
	{
		assignRightNodeMemHost(nodes, rightSideMem, i, props);
	}
}

void leftSideInit(float* leftSide, int size)
{
	for (int i = 0; i < size; i++)
	{
		leftSide[i] = 6;//(i+1)%26;
	}
}

void testDistributeInputAmongNodes()
{
	Properties props = getProperities(17, 1);
	ERRCHECK(cudaMemcpyToSymbol(dProps, &props, sizeof(Properties)));
	float* leftSide;
	float* rightSide;
	float* dRightSideMem;
	float* rightSideMem = new float[props.rightSizeMem];
	generateTestEquation(17, 1, &leftSide, &rightSide);
	Node* nodes = new Node[props.heapNodes];
	memset(nodes, 0, props.heapNodes * sizeof(Node));
	Node* dNodes = nullptr;
	float* dLeftSide = nullptr;
	float* dRightSide = nullptr;
	ERRCHECK(cudaMalloc(&dNodes, sizeof(Node)* props.heapNodes));
	ERRCHECK(cudaMemset(dNodes, 0, sizeof(Node)*props.heapNodes));
	ERRCHECK(cudaMalloc(&dLeftSide, sizeof(float)*props.leftSize));
	ERRCHECK(cudaMemcpy(dLeftSide, leftSide, sizeof(float)*props.leftSize, cudaMemcpyHostToDevice));
	ERRCHECK(cudaMalloc(&dRightSide, sizeof(float)*props.rightSize));
	ERRCHECK(cudaMemcpy(dRightSide, rightSide, sizeof(float)*props.rightSize, cudaMemcpyHostToDevice));
	ERRCHECK(cudaMalloc(&dRightSideMem, sizeof(float)*props.rightSizeMem));
	distributeInputAmongNodes(dNodes, dLeftSide, dRightSideMem, dRightSide, props);
	for (int start = PARENT(props.lastLevelStartIdx), nodesCount = props.beforeLastLevelNotBottomNodes; start > 0; nodesCount = (start + 1) / 2 , start = PARENT(start))//order matters
	{
		mergeRightSideLayer << <BLOCKS(nodesCount*props.rightCount), THREADS >> >(dNodes, start, nodesCount * props.rightCount);
		ERRCHECK(cudaGetLastError());
		ERRCHECK(cudaDeviceSynchronize());
		mergeLeftChild << <BLOCKS(nodesCount), THREADS >> >(dNodes, start, nodesCount);
		ERRCHECK(cudaGetLastError());
		ERRCHECK(cudaDeviceSynchronize());
		mergeRightChild << <BLOCKS(nodesCount), THREADS >> >(dNodes, start, nodesCount);
		ERRCHECK(cudaGetLastError());
		ERRCHECK(cudaDeviceSynchronize());
	}
	mergeRightSideLayer << <1, props.rightCount >> >(dNodes, 0, props.rightCount);
	ERRCHECK(cudaGetLastError());
	ERRCHECK(cudaDeviceSynchronize());
	mergeLeftChild << <1, 1 >> >(dNodes, 0, 1);
	ERRCHECK(cudaGetLastError());
	ERRCHECK(cudaDeviceSynchronize());
	mergeRightChild << <1, 1 >> >(dNodes, 0, 1);
	ERRCHECK(cudaGetLastError());
	ERRCHECK(cudaDeviceSynchronize());
	ERRCHECK(cudaMemcpy(nodes, dNodes, sizeof(Node) * props.heapNodes, cudaMemcpyDeviceToHost));
	ERRCHECK(cudaMemcpy(rightSideMem, dRightSideMem, sizeof(float)*props.rightSizeMem, cudaMemcpyDeviceToHost));
	assignHostRightSide(props, nodes, rightSideMem);
	printAllNodes(nodes, 0, props);
}

void testRun(const int size)
{
	const int rsize = 1;
	Properties props = getProperities(size, rsize);
	ERRCHECK(cudaMemcpyToSymbol(dProps, &props, sizeof(Properties)));
	float* leftSide = nullptr;
	float* rightSide = nullptr;
	float* dRightSideMem = nullptr;
	float* rightSideMem = new float[dProps.rightSizeMem];
	generateTestEquation(size, rsize, &leftSide, &rightSide);
	Node* nodes = new Node[props.heapNodes];
	memset(nodes, 0, props.heapNodes * sizeof(Node));
	Node* dNodes = nullptr;
	float* dLeftSide = nullptr;
	float* dRightSide = nullptr;
	ERRCHECK(cudaMalloc(&dNodes, sizeof(Node)* props.heapNodes));
	ERRCHECK(cudaMemset(dNodes, 0, sizeof(Node)*props.heapNodes));
	ERRCHECK(cudaMalloc(&dLeftSide, sizeof(float)*props.leftSize));
	ERRCHECK(cudaMemcpy(dLeftSide, leftSide, sizeof(float)*props.leftSize, cudaMemcpyHostToDevice));
	ERRCHECK(cudaMalloc(&dRightSide, sizeof(float)*props.rightSize));
	ERRCHECK(cudaMemcpy(dRightSide, rightSide, sizeof(float)*props.rightSize, cudaMemcpyHostToDevice));
	ERRCHECK(cudaMalloc(&dRightSideMem, sizeof(float)*props.rightSizeMem));
	run(dNodes, dLeftSide, props, dRightSide, dRightSideMem);
	ERRCHECK(cudaMemcpy(nodes, dNodes, sizeof(Node) * props.heapNodes, cudaMemcpyDeviceToHost));
	ERRCHECK(cudaMemcpy(rightSideMem, dRightSideMem, sizeof(float)*props.rightSizeMem, cudaMemcpyDeviceToHost));
	assignHostRightSide(props, nodes, rightSideMem);
	printAllNodes(nodes, 0, props);
//	delete [] leftSide;
//	delete [] rightSide;
//	delete [] rightSideMem;
//	delete [] nodes;
	ERRCHECK(cudaFree(dRightSide));
	ERRCHECK(cudaFree(dRightSideMem));
	ERRCHECK(cudaFree(dNodes));
	ERRCHECK(cudaFree(dLeftSide));
}


void testMultipleRun(int n, int size)
{
	clock_t start, end;
	start = clock();
	for (int i = 0; i < n; i++)
		testRun(size);
	end = clock();
	printf("time %f\n", (float)(end - start) / CLOCKS_PER_SEC);
}

__global__ void assignTestRightSide(Node* node, float* x)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= 1)
		return;
	node->x[0] = x;
	node->x[1] = x + 4;
	node->x[2] = x + 4 * 2;
	node->x[3] = x + 4 * 3;
	node->x[4] = x + 4 * 4;
	node->x[5] = x + 4 * 5;
}

void testGaussianElimination()
{
	Properties props = getProperities(6, 4);
	ERRCHECK(cudaMemcpyToSymbol(dProps, &props, sizeof(Properties)));
	Node node;
	//	float m[] = {
	//		1, 1, -2, 1, 3, -1,
	//		2, -1, 1, 2, 1, -3,
	//		1, 3, -3, -1, 2, 1,
	//		5, 2, -1, -1, 2, 1,
	//		-3, -1, 2, 3, 1, 3,
	//		4, 3, 1, -6, -3, -2
	//	};
	//	float x[] = {4,4,4,4,20,20,20,20,-15,-15,-15,-15,-3,-3,-3,-3,16,16,16,16,-27,-27,-27,-27};
	float m[] = {1.0 , 0.0 , -0.5 , 0.5 , -0.5 , -0.5 ,
		0.0 , 1.0 , -0.5 , 0.5 , 0.5 , 0.5 ,
		-0.5 , -0.5 , 0.5 , 3.5 , 0.0 , 0.0 ,
		0.5 , 0.5 , 3.5 , 0.5 , 0.0 , 0.0 ,
		-0.5 , 0.5 , 0.0 , 0.0 , 2.5 , 0.5 ,
		-0.5 , 0.5 , 0.0 , 0.0 , 0.5 , 2.5};
	float x[] = {0,0,0,0,0,0,0,0,2,2,2,2,4,4,4,4,4,4,4,4,2,2,2,2};
	memcpy(node.m, m, sizeof(float) * MSIZE);
	assignHostRightSide(props, &node, x);
	Node* dNode;
	printNode(node, props.rightCount);
	ERRCHECK(cudaMalloc(&dNode, sizeof(Node)));
	ERRCHECK(cudaMemcpy(dNode, &node, sizeof(Node), cudaMemcpyHostToDevice));
	float* dX;
	ERRCHECK(cudaMalloc(&dX, sizeof(x)));
	ERRCHECK(cudaMemcpy(dX, &x, sizeof(x), cudaMemcpyHostToDevice));
	assignTestRightSide << <1, 1 >> >(dNode, dX);
	ERRCHECK(cudaGetLastError());
	ERRCHECK(cudaDeviceSynchronize());
	forwardEliminationLeft << <1, 1 >> >(dNode, 0, 1, 0, 6);
	ERRCHECK(cudaGetLastError());
	ERRCHECK(cudaDeviceSynchronize());
	forwardEliminationRight << <1, 4 / COLUMNS_PER_THREAD >> >(dNode, 0, 1, 0, 6);
	//	ERRCHECK(cudaGetLastError());
	//	ERRCHECK(cudaDeviceSynchronize());
	//	backwardSubstitutionRight << <1, 4 / COLUMNS_PER_THREAD >> >(dNode, 0, 1, 0, 4);
	//	ERRCHECK(cudaGetLastError());
	//	ERRCHECK(cudaDeviceSynchronize());
	ERRCHECK(cudaMemcpy(&node, dNode, sizeof(Node), cudaMemcpyDeviceToHost));
	assignHostRightSide(props, &node, x);
	printNode(node, props.rightCount);
	ERRCHECK(cudaMemcpy(x, dX, sizeof(x), cudaMemcpyDeviceToHost));
	printNode(node, props.rightCount);
	float c[] = {1,-2,3,4,2,-1};
	for (int i = 0; i < 6; i++)
	{
		printf("%.1f ", c[i]);
	}
	printf("\n");
	for (int i = 0; i < props.rightCount; i++)
	{
		for (int j = 0; j < 6; j++)
			printf("%.1f ", x[j * props.rightCount + i]);
		printf("\n");
	}
}
