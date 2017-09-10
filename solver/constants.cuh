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
#define THREADS 32
#define BLOCKS(N) (N+THREADS-1)/THREADS
#define COLUMNS_PER_THREAD 1

#define PRINT_EXPR "%.5f "
#define SUPRESS_PRINT

#define RED 0
#define GREEN 1
#define BLUE 2
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
#define FLOAT_NUMBER
extern __constant__ Properties dProps;
#ifdef DOUBLE_NUMBER //DO NOT USE calculating right side does not work with doubles, because there is no atomicAdd on doubles in CUDA
typedef double number;
#endif
#ifdef FLOAT_NUMBER
typedef float number;
#endif
struct Node
{
	number m[MSIZE];
	number* x[6];
};
