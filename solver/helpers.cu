#include "helpers.cuh"
#include <cmath>

void printAllNodes(Node* nodes, int nodesStart, Properties props)
{
	int powerOfTwo = (int)log2(nodesStart + 1) + 1;
	for (int i = nodesStart; i < props.heapNodes; i++)
	{
		Node node = nodes[i];
		for (int j = i >= props.remainingNodes ? 1 : 0; j < 6; j++)
		{
			printf("%.2f %.2f %.2f %.2f %.2f %.2f | ", node.m[XY(j, 0)], node.m[XY(j, 1)], node.m[XY(j, 2)], node.m[XY(j, 3)], node.m[XY(j, 4)], node.m[XY(j, 5)]);
			for (int k = 0; k < props.rightCount; k++)
			{
				printf("%.2f ", node.x[j][k]);
			}
			printf("\n");
		}
		printf("\n");
	}
}

__device__ __host__ void printNode(Node node, int rightCount)
{
	for (int i = 0; i < 6; i++)
	{
		printf("%.2f %.2f %.2f %.2f %.2f %.2f | ", node.m[XY(i, 0)], node.m[XY(i, 1)], node.m[XY(i, 2)], node.m[XY(i, 3)], node.m[XY(i, 4)], node.m[XY(i, 5)]);
		for (int k = 0; k < rightCount; k++)
		{
			printf("%.2f ", node.x[i][k]);
		}
		printf("\n");
	}
	printf("\n");
}

void fillRightSide(number value, int row, number* rightSide, int rightCount)
{
	for (int i = 0; i < rightCount; i++)
	{
		rightSide[row * rightCount + i] = value * (i + 1);
	}
}


void computeRightSide(int rightCount, number* leftSide, number* rightSide, int i, int offset)
{
	int rightSideVal = 0;
	for (int j = 0; j < 5; j++)
	{
		//		int solution = 1;
		int solution = i - 1 + j + offset;//solution is x(0)=1,x(1)=2,x(n-1)=n
		rightSideVal += leftSide[i * 5 + j] * solution;
	}
	fillRightSide(rightSideVal, i, rightSide, rightCount);
}

__device__ void printRow(number* m, int start, int count)
{
	for (int i = start; i < start + count; i++)
		printf("%.1f ", m[i]);
	printf("\n");
}

void generateTestEquation(int leftCount, int rightCount, number** leftSidePtr, number** rightSidePtr)
{
	number* leftSide = new number[leftCount * 5];
	number* rightSide = new number[rightCount * leftCount];


	for (int i = 0; i < leftCount * 5; i += 5)
	{
		leftSide[i] = 1;
		leftSide[i + 1] = 2;
		leftSide[i + 2] = 6;
		leftSide[i + 3] = 2;
		leftSide[i + 4] = 1;
	}
	leftSide[0] = 0;
	leftSide[1] = 0;
	leftSide[2] = 6;
	leftSide[3] = 2;
	leftSide[4] = 1;
	leftSide[5] = 0;
	leftSide[6] = 2;
	leftSide[7] = 6;
	leftSide[8] = 2;
	leftSide[9] = 1;
	leftSide[leftCount * 5 - 10] = 1;
	leftSide[leftCount * 5 - 9] = 6;
	leftSide[leftCount * 5 - 8] = 2;
	leftSide[leftCount * 5 - 7] = 1;
	leftSide[leftCount * 5 - 6] = 0;
	leftSide[leftCount * 5 - 5] = 6;
	leftSide[leftCount * 5 - 4] = 2;
	leftSide[leftCount * 5 - 3] = 1;
	leftSide[leftCount * 5 - 2] = 0;
	leftSide[leftCount * 5 - 1] = 0;

	for (int i = 0; i < leftCount; i++)
	{
		computeRightSide(rightCount, leftSide, rightSide, i, 0);
	}
	//	fillRightSide(14, i, rightSide, rightCount);
	*leftSidePtr = leftSide;
	*rightSidePtr = rightSide;
	//	for (int i = 0; i < leftCount; i++)
	//	{
	//		printf("%d:", i + 1);
	//		for (int j = 0; j < 5; j++)
	//		{
	//			printf("%.0f ", leftSide[i * 5 + j]);
	//		}
	//		printf(" |  ");
	//		for (int j = 0; j < rightCount; j++)
	//		{
	//			printf("%.0f ", rightSide[i * rightCount + j]);
	//		}
	//		printf("\n");
	//	}
	printLeftAndRight(leftSide, rightSide, leftCount, rightCount);

	//	getch();
}

void printLeftAndRight(number* left, number* right, int size, int rsize)
{
#ifdef SUPRESS_PRINT
	return;
#endif
	if (rsize == 0)
		rsize = size;
	printf("LEFT AND RIGHT\n");
	int before = 0;
	int after = size - 1;
	for (int i = 0; i < size; i++)
	{
		for (int i = 0; i < before; i++)
			printf("0 ");
		for (int j = 0; j < 5; j++)
		{
			printf(PRINT_EXPR, left[i * 5 + j]);
		}
		for (int i = 0; i < after; i++)
			printf("0 ");
		printf(" |  ");
		for (int j = 0; j < rsize; j++)
		{
			printf(PRINT_EXPR, right[i * rsize + j]);
		}
		printf("\n");
		before++;
		after--;
	}
	printf("END LEFT AND RIGHT\n");
}

void showMemoryConsumption()
{
	size_t free_byte;
	size_t total_byte;
	ERRCHECK(cudaMemGetInfo(&free_byte, &total_byte));
	double free_db = (double)free_byte;
	double total_db = (double)total_byte;
	double used_db = total_db - free_db;

	printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n", used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);
}


Bitmap readBmp(char* filename, float const* colors)
{
	unsigned char* texels;
	int width, height;
	FILE* fd;
	fd = fopen(filename, "rb");
	if (fd == nullptr)
	{
		printf("Error: fopen failed\n");
		throw "Bitmap opening failed";
	}

	unsigned char header[54];

	fread(header, sizeof(unsigned char), 54, fd);

	width = *(int*)&header[18];
	height = *(int*)&header[22];
	if (width != height)
		throw "bitmap must be suqare";

	int padding = 0;

	// Calculate padding
	while ((width * 3 + padding) % 4 != 0)
	{
		padding++;
	}

	int paddedwidth = width * 3 + padding;

	number* bitmap = new number[width * height];

	unsigned char* bmprow = (unsigned char *)malloc(paddedwidth * sizeof(unsigned int));

	if (colors == nullptr)
	{
		colors = DEFAULT_COLORS;
	}
	//bitmap is stored upside down, so it must be read from bottom to top;
	for (int i = height - 1; i >= 0; i--)
	{
		fread(bmprow, sizeof(unsigned char), paddedwidth, fd);

		for (int j = 0; j < width * 3; j += 3)
		{
			int index = j / 3 + width * i;
			bitmap[index] = (colors[RED] * bmprow[j + 2] + colors[GREEN] * bmprow[j + 1] + colors[BLUE] * bmprow[j]) / 255.0L;
		}
	}
	free(bmprow);
	fclose(fd);
	return Bitmap(bitmap, width, height);
}

void saveArray(char* path, int size, float* data)
{
	FILE* fd;
	fd = fopen(path, "wb+");
	fwrite(data, sizeof(float), size * size, fd);
	fclose(fd);
}
