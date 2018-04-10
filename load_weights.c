#include <stdio.h>
#include <stdlib.h>
int main()
{	char filename[] = "darknet53.conv.74";
	FILE *fp = fopen(filename, "rb");
    int major;  // 0
    int minor;  // 2
    int revision;   // 0
    size_t seen;
    printf("sizeof(int): %lu\n", sizeof(int));	// 4
    printf("sizeof(size_t): %lu\n", sizeof(size_t));	// 8
    printf("sizeof(float): %lu\n", sizeof(float));	// 4
    // 调用格式：fread(buf,sizeof(buf),1,fp);
    // 读取成功时：当读取的数据量正好是sizeof(buf)个Byte时，返回值为1(即count)
    // 否则返回值为0(读取数据量小于sizeof(buf))
    float *biases = calloc(32, sizeof(float));
    float *scales = calloc(32, sizeof(float));
    float *rolling_mean = calloc(32, sizeof(float));
    float *rolling_variance = calloc(32, sizeof(float));
    float *weights = calloc(3*3*3*32, sizeof(float));

    fread(&major, sizeof(int), 1, fp);
    fread(&minor, sizeof(int), 1, fp);
    fread(&revision, sizeof(int), 1, fp);
    fread(&seen, sizeof(size_t), 1, fp);
    fread(biases, sizeof(float), 32, fp);
    fread(scales, sizeof(float), 32, fp);
    fread(rolling_mean, sizeof(float), 32, fp);
    fread(rolling_variance, sizeof(float), 32, fp);
    fread(weights, sizeof(float), 3*3*3*32, fp);

    printf("filename\n");
    printf("%d\n", major);
    printf("%d\n", minor);
    printf("%d\n", revision);
    printf("%lu\n", seen);
    printf("biases:\n");
    for(int i =0; i < 32; ++i)
    	printf("%f ",biases[i]);
    printf("\n");
    printf("scales:\n");
    for(int i =0; i < 32; ++i)
    	printf("%f ",scales[i]);
    printf("\n");
    printf("rolling_mean:\n");
    for(int i =0; i < 32; ++i)
    	printf("%f ",rolling_mean[i]);
    printf("\n");
    printf("rolling_variance:\n");
    for(int i =0; i < 32; ++i)
    	printf("%f ",rolling_variance[i]);
    printf("\n");
	fclose(fp);
	return 0;
}