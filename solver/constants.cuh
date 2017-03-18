#include <host_defines.h>
#pragma once

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

struct Properties
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
	int rightSize;
	int rightSizeMem;
};

Properties getProperities(int leftCount, int rightCount);

extern __constant__ Properties dProps;

struct Node
{
	float m[MSIZE];
	float* x[6];
};
