#include "constants.cuh"
#pragma once

void testDistributeInputAmongNodes();
void testRun(const int size = 11);
void testMultipleRun(int n = 1, int size = 14);
void testGaussianElimination();
__global__ void assignTestRightSide(Node* node, number* x);
void leftSideInit(number* leftSide, int size);
void assignHostRightSide(Properties props, Node* nodes, number* rightSideMem);
