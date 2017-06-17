#include "cuda_runtime.h"
#include <stdio.h>
#include <conio.h>
#include <new>
#include "constants.cuh"
#pragma once
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



void printAllNodes(Node* nodes, int nodesStart, Properties props);
__device__ __host__ void printNode(Node node, int rightCount);

void fillRightSide(float value, int row, float* rightSide, int rightCount);
void generateTestEquation(int leftCount, int rightCount, float** leftSidePtr, float** rightSidePtr);

void showMemoryConsumption();

void printRow(float * m, int start, int count);

float * readBmpWithMargin(char* filename);