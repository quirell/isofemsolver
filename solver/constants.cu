#include "constants.cuh"
#include <cmath>


Properties getProperities(int leftCount, int rightCount)
{
	Properties p;
	p.leftCount = leftCount;
	p.leftSize = leftCount * 5;
	p.heapNodes = HEAP_SIZE(leftCount);
	p.bottomNodes = BOTTOM_HEAP_NODES_COUNT(leftCount);
	p.remainingNodes = p.heapNodes - p.bottomNodes;
	p.beforeLastLevelStartIdx = (int)pow(2, (int)log2(p.remainingNodes)) - 1;
	p.beforeLastLevelNotBottomNodes = p.remainingNodes - p.beforeLastLevelStartIdx;
	p.beforeLastLevelNodes = pow(2, (int)log2(p.bottomNodes - 1)) - p.beforeLastLevelNotBottomNodes;// -1 is in case bottomNodes is power of two, then beforeLastLevelNodes should obviously be 0
	p.lastLevelNodes = p.bottomNodes - p.beforeLastLevelNodes;
	p.lastLevelStartIdx = p.heapNodes - p.lastLevelNodes;
	p.beforeLastLevelStartIdx = p.remainingNodes - p.lastLevelNodes; //account for idx value, undefined when beforeLastLevelNodes is = 0
	p.rightCount = rightCount;
	p.rightSize = rightCount * leftCount;
	p.rightSizeMem = p.heapNodes * rightCount * 6;
	return p;
}
