#include "constants.cuh"

__global__ void backwardSubstitutionRight(Node* nodes, int startIdx, int nodesCount, int end, int elim);
__global__ void forwardEliminationLeft(Node* nodes, int startIdx, int nodesCount, int start, int elim);
__global__ void forwardEliminationRight(Node* nodes, int startIdx, int nodesCount, int rowStart, int elim);
__global__ void mergeLeftChild(Node* nodes, int startIdx, int nodesCount);
__global__ void mergeRightChild(Node* nodes, int startIdx, int nodesCount);

__global__ void divideLeft(Node* nodes, float* leftSide);
__global__ void divideFirstAndLast(Node* nodes, float* leftSide);
inline __device__ void divideRightNode(Node* nodes, float* rightSide, int ord, int nodeIdx, int idx, int rightCount);
__global__ void divideRight(Node* nodes, float* rightSide);
inline __device__ void assignRightNodeMem(Node* nodes, float* rightSideMem, int nodeIdx, Properties props);
__global__ void assignRightSideMem(Node* nodes, float* rightSideMem);
inline __device__ void mergeRightSideNode(Node* nodes, int idx, int nodeIdx);
__global__ void mergeRightSideLayer(Node* nodes, int startNode, int rightSidesCount);
void distributeInputAmongNodes(Node* dNodes, float* dLeftSide, float* dRightSideMem, float* dRightSide, Properties props);
void eliminateFirstRow(Node* dNodes, Properties props);
void eliminateRoot(Node* dNodes, Properties props);
void run(Node* dNodes, float* dLeftSide, Properties props, float* dRightSide, float* dRightSideMem);
inline __device__ __host__ void assignParentToChildren(Node* nodes, int nodeIdx);
__global__ void assignParentToChildrenLayer(Node* nodes, int startNode, int nodesCount);
