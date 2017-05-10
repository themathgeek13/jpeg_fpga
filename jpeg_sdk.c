#include <stdio.h>
#include "platform.h"
#include "xil_printf.h"
#include "xrunloops.h"
#include "xtimctr.h"
#include <stdlib.h>

#define bufSize 50
#define PI 3.1415926535897932384626
#define rows 128
#define cols 128
#define QUALITY 50

float cosMatrix[8][8] = {
		{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
		{0.98079, 0.83147, 0.55557, 0.19509, -0.19509, -0.55557, -0.83147, -0.98079},
		{0.92388, 0.38268, -0.38268, -0.92388, -0.92388, -0.38268, 0.38268, 0.92388},
		{0.83147, -0.19509, -0.98079, -0.55557, 0.55557, 0.98079, 0.19509, -0.83147},
		{0.70711, -0.70711, -0.70711, 0.70711, 0.70711, -0.70711, -0.70711, 0.70711},
		{0.55557, -0.98079, 0.19509, 0.83147, -0.83147, -0.19509, 0.98079, -0.55557},
		{0.38268, -0.92388, 0.92388, -0.38268, -0.38268, 0.92388, -0.92388, 0.38268},
		{0.19509, -0.55557, 0.83147, -0.98079, 0.98079, -0.83147, 0.55557, -0.19509}
	};

float quant_matrix[8][8]= {
		{16,  11,  10,  16,  24, 40,  51,  61} ,
		{12,  12,  14,  19,  26, 58,  60,  55} ,
		{14,  13,  16,  24,  40, 57,  69,  56} ,
		{14,  17,  22,  29,  51, 87,  80,  62} ,
		{18,  22,  37,  56,  68, 109, 103, 77} ,
		{24,  35,  55,  64,  81, 104, 113,  92} ,
		{49,  64,  78,  87,  103, 121, 120, 101} ,
		{72,  92,  95,  98, 112, 100, 103,  99}
	};

//quantization function for a given quality requirement
//this simply evaluates the quantization matrix, and does not actually
//perform the quantization step in the algorithm
void calc_quant(float qm[8][8])
{	//http://stackoverflow.com/questions/29215879/how-can-i-generalize-the-quantization-matrix-in-jpeg-compression
	int i,j;
	float def[8][8] = {
		{16,  11,  10,  16,  24, 40,  51,  61} ,
		{12,  12,  14,  19,  26, 58,  60,  55} ,
		{14,  13,  16,  24,  40, 57,  69,  56} ,
		{14,  17,  22,  29,  51, 87,  80,  62} ,
		{18,  22,  37,  56,  68, 109, 103, 77} ,
		{24,  35,  55,  64,  81, 104, 113,  92} ,
		{49,  64,  78,  87,  103, 121, 120, 101} ,
		{72,  92,  95,  98, 112, 100, 103,  99}
	};
	float S;
	if(QUALITY < 50)
		S = (5000/(float)(QUALITY));
	else
		S = 200-2*QUALITY;
	for(i=0; i<8; i++)
	{
		for(j=0; j<8; j++)
		{
			qm[i][j]=(float)((S*def[i][j]+(float)50)/(float)100.0);
		}
	}
}

void quantize(float qm[8][8], float G[8][8])
{

	int i,j;
	for(i=0; i<8; i++)
	{
		for(j=0; j<8; j++)
		{
			G[i][j] = (G[i][j]/(float)qm[i][j])*qm[i][j];
		}
	}
}

float al(int x)
{
	if(x==0)
		return 1.0/sqrt(2.0);
	else
		return 1.0;
}

float singleGUVfwd(float subimg[8][8], int u, int v)
{
	float G=0;
	int x,y;

	for(x=0; x<8; x++)
	{
		for(y=0; y<8; y++)
		{
			G+=(float)0.25*al(u)*al(v)*subimg[x][y]*cosMatrix[u][x]*cosMatrix[v][y];
		}
	}
	return G;
}

float singleGUVinv(float subimg[8][8], int u, int v)
{

	float G=0;
	int x,y;

	for(x=0; x<8; x++)
	{
		for(y=0; y<8; y++)
		{
			G+=(float)0.25*al(x)*al(y)*subimg[x][y]*cosMatrix[x][u]*cosMatrix[y][v];
		}
	}
	return G;
}

void shift128(float subimg[8][8])
{

	int i,j;
	for(i=0; i<8; i++)
	{
		for(j=0; j<8; j++)
		{
			subimg[i][j]-=128;
		}
	}
}

void DCT_8x8_2D(float G[8][8], float subimg[8][8])
{

	shift128(subimg);
	int i,j;

	for(i=0; i<8; i++)
	{
		for(j=0; j<8; j++)
		{
			G[i][j]=singleGUVfwd(subimg, i, j);
		}
	}
}

void inv_DCT_8x8_2D(float G[8][8], float subimg[8][8])
{

	int i,j;

	for(i=0; i<8; i++)
	{
		for(j=0; j<8; j++)
		{
			G[i][j]=singleGUVinv(subimg, i, j);
		}
	}
}

float min(float a, float b)
{
	return (a>b)?b:a;
}

float max(float a, float b)
{
	return (a>b)?a:b;
}

//converts from RGB to YCbCr when fwd=1 and back when fwd=0
void colour_space_conversion(signed short int image[8][8][3])
{

	float val1, val2, val3;
	float ch1, ch2, ch3;
	int i,j,k;
	for(i=0; i<8; i++)
	{
		for(j=0; j<8; j++)
		{
			ch1 = (float)image[i][j][0];
			ch2 = (float)image[i][j][1];
			ch3 = (float)image[i][j][2];
			float delta = 128;

			val1 = (float)0.299*ch1+(float)0.587*ch2+(float)0.114*ch3;
			val2 = delta + (float)0.713*(ch1-val1);
			val3 = delta + (float)0.564*(ch3-val1);
			image[i][j][0]=min(max((int)(val1),0),255);
			image[i][j][1]=min(max((int)(val2),0),255);
			image[i][j][2]=min(max((int)(val3),0),255);
		}
	}
}

//converts from RGB to YCbCr when fwd=1 and back when fwd=0
void invcolour_space_conversion(signed short int image[8][8][3])
{

	float val1, val2, val3;
	float ch1, ch2, ch3;
	int i,j,k;
	for(i=0; i<8; i++)
	{
		for(j=0; j<8; j++)
		{
			ch1 = (float)image[i][j][0];
			ch2 = (float)image[i][j][1];
			ch3 = (float)image[i][j][2];
			float delta = 128;

			val1 = ch1+(float)1.403*(ch2-delta);
			val2 = ch1-(float)0.714*(ch2-delta)-(float)0.344*(ch3-delta);
			val3 = ch1+(float)1.733*(ch3-delta);
			image[i][j][0]=min(max((int)(val1),0),255);
			image[i][j][1]=min(max((int)(val2),0),255);
			image[i][j][2]=min(max((int)(val3),0),255);
		}
	}
}

void extract8x8subimg(signed short int subimg[8][8][3],signed short int image[128][128][3],int i,int j)
{

	int x,y;
	//extract the 8x8 sub-image for further processing of the image
	for(x=0; x<8; x++)
	{
		for(y=0; y<8; y++)
		{
			subimg[x][y][0]=image[x+i][y+j][0];
			subimg[x][y][1]=image[x+i][y+j][1];
			subimg[x][y][2]=image[x+i][y+j][2];
		}
	}
}

void saveback8x8subimg(signed short int subimg[8][8][3],signed short int image[128][128][3],int i, int j)
{

	int x,y;
	for(x=0; x<8; x++)
	{
		for(y=0; y<8; y++)
		{
			image[x+i][y+j][0]=subimg[x][y][0];
			image[x+i][y+j][1]=subimg[x][y][1];
			image[x+i][y+j][2]=subimg[x][y][2];
		}
	}
}

void extractYCrCb(signed short int subimg[8][8][3], float subimgY[8][8], float subimgC1[8][8], float subimgC2[8][8])
{
	int x,y;

	for(x=0; x<8; x++)
	{
		for(y=0; y<8; y++)
		{
			subimgY[x][y]=subimg[x][y][0];
			subimgC1[x][y]=subimg[x][y][1];
			subimgC2[x][y]=subimg[x][y][2];
		}
	}
}

void finalEval(signed short int subimg[8][8][3], float finalY[8][8], float finalC1[8][8], float finalC2[8][8])
{

	int x,y;
	for(x=0; x<8; x++)
	{
		for(y=0; y<8; y++)
			{
			subimg[x][y][0]=finalY[x][y]+128;
			subimg[x][y][1]=finalC1[x][y]+128;
			subimg[x][y][2]=finalC2[x][y]+128;
		}
	}
}

void JPEG_encode(signed short int subimg[8][8][3],float quant_matrix[8][8])
{

	float DCTY[8][8], DCTC1[8][8], DCTC2[8][8];
	float finalY[8][8], finalC1[8][8], finalC2[8][8];
	float subimgY[8][8], subimgC1[8][8], subimgC2[8][8];
	//Colour Space Conversion from RGB to YCrCb
	colour_space_conversion(subimg);
	//extract 3 components
	extractYCrCb(subimg,subimgY,subimgC1,subimgC2);

	//DCT Step
	DCT_8x8_2D(DCTY,subimgY);
	DCT_8x8_2D(DCTC1,subimgC1);
	DCT_8x8_2D(DCTC2,subimgC2);

	//Quantization Step
	quantize(quant_matrix,DCTY);
	quantize(quant_matrix,DCTC1);
	quantize(quant_matrix,DCTC2);

	//Inverse DCT Step
	inv_DCT_8x8_2D(finalY,DCTY);
	inv_DCT_8x8_2D(finalC1,DCTC1);
	inv_DCT_8x8_2D(finalC2,DCTC2);

	//evaluate the Y,Cr and Cb components of the sub-image
	finalEval(subimg,finalY,finalC1,finalC2);
	invcolour_space_conversion(subimg);
}

int runloops(signed short int image[128][128][3])
{
	int i,j,k;
	signed short int subimg[8][8][3];
	for(i=0; i<rows; i+=8)
	{
		for(j=0; j<cols; j+=8)
		{
			extract8x8subimg(subimg,image,i,j);
			JPEG_encode(subimg,quant_matrix);
			saveback8x8subimg(subimg,image,i,j);
		}
	}
	return 0;
}

int main()
{
	int i,j,k,ret;
	signed short int arr[128][128][3];
	int *arrptr = (int *) arr;


	for(i=0; i<128; i++)
	{
		for(j=0; j<128; j++)
		{
			for(k=0; k<3; k++)
				arr[i][j][k]=(i+j+k)%256;
		}
	}

    init_platform();
    int x1,y1,z1;
    int x2,y2,z2;
    int x3,y3,z3;
    int x4,y4,z4;

    x1=30; y1=25; z1=2;
    x2=92; y2=21; z2=0;
    x3=8; y3=77; z3=1;
    x4=12; y4=1; z4=0;
    XRunloops xrun;
	XRunloops* xrunptr = &xrun;
	XRunloops_Initialize(xrunptr,0);
	print("\r\nRunning HW-SW comparison\n\r");
	XTimctr xtimctr;
	XTimctr* xptr = &xtimctr;
	XTimctr_Initialize(xptr,0);
	XTimctr_EnableAutoRestart(xptr);
	XTimctr_Start(xptr);
	int start=XTimctr_Get_val_r(xptr);
	runloops(arr);
	int stop = XTimctr_Get_val_r(xptr);
	xil_printf("\r\n[sw] Took %d cycles\n\r",(stop-start));
	xil_printf("\r\nRandom values from software run: %d %d %d %d \r\n",arr[x1][y1][z1],arr[x2][y2][z2],arr[x3][y3][z3],arr[x4][y4][z4]);
	start = XTimctr_Get_val_r(xptr);
	ret = XRunloops_Write_image_r_Words(xrunptr,0,arrptr,128*128*3);
	int t1 = XTimctr_Get_val_r(xptr);
	XRunloops_Start(xrunptr);
	int ts = XTimctr_Get_val_r(xptr);
	while(!XRunloops_IsDone(xrunptr));
	int t2 = XTimctr_Get_val_r(xptr);
	xil_printf("\r\n[hw] took %d (%d + %d + %d) cycles \n\r", t2-start, t1-start, ts-t1, t2-ts);
	xil_printf("\r\nSame Random values from hardware run: %d %d %d %d \r\n",arr[x1][y1][z1],arr[x2][y2][z2],arr[x3][y3][z3],arr[x4][y4][z4]);

	cleanup_platform();
	return 0;
}

