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

void fillRightSide(number value, int row, number* rightSide, int rightCount);
void generateTestEquation(int leftCount, int rightCount, number** leftSidePtr, number** rightSidePtr);

void showMemoryConsumption();

void printRow(number * m, int start, int count);

struct Bitmap
{
	number * bitmap;
	int width;
	int height;
	Bitmap(number * bmp, int w, int h) :bitmap(bmp), width(w), height(h) {}
};

Bitmap readBmp(char* filename);

void printLeftAndRight(number * left, number * right, int size, int rsize = 0);

number * cutSquare(number * input, int size, int targetCol);