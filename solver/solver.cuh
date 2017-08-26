#include "constants.cuh"

__global__ void backwardSubstitutionRight(Node* nodes, int startIdx, int nodesCount, int end, int elim);
__global__ void forwardEliminationLeft(Node* nodes, int startIdx, int nodesCount, int start, int elim);
__global__ void forwardEliminationRight(Node* nodes, int startIdx, int nodesCount, int rowStart, int elim);
__global__ void mergeLeftChild(Node* nodes, int startIdx, int nodesCount);
__global__ void mergeRightChild(Node* nodes, int startIdx, int nodesCount);

__global__ void divideLeft(Node* nodes, number* leftSide);
__global__ void divideFirstAndLast(Node* nodes, number* leftSide);
inline __device__ void divideRightNode(Node* nodes, number* rightSide, int ord, int nodeIdx, int idx, int rightCount);
__global__ void divideRight(Node* nodes, number* rightSide);
inline __device__ void assignRightNodeMem(Node* nodes, number* rightSideMem, int nodeIdx, Properties props);
__global__ void assignRightSideMem(Node* nodes, number* rightSideMem);
inline __device__ void mergeRightSideNode(Node* nodes, int idx, int nodeIdx);
__global__ void mergeRightSideLayer(Node* nodes, int startNode, int rightSidesCount);
void distributeInputAmongNodes(Node* dNodes, number* dLeftSide, number* dRightSideMem, number* dRightSide, Properties props);
void eliminateFirstRow(Node* dNodes, Properties props);
void eliminateRoot(Node* dNodes, Properties props);
void run(Node* dNodes, number* dLeftSide, Properties props, number* dRightSide, number* dRightSideMem);
inline __device__ __host__ void assignParentToChildren(Node* nodes, int nodeIdx);
__global__ void assignParentToChildrenLayer(Node* nodes, int startNode, int nodesCount);
