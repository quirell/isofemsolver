#include "constants.cuh"
#pragma once

void testDistributeInputAmongNodes();
void testRun();
void testGaussianElimination();
__global__ void assignTestRightSide(Node* node, float* x);
void leftSideInit(float* leftSide, int size);
void assignHostRightSide(Properties props, Node* nodes, float* rightSideMem);
