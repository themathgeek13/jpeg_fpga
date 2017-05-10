#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <math.h>
#include <ap_fixed.h>
#include <complex>
#include <stdlib.h>
#include <fstream>

#define bufSize 50
#define PI 3.1415926535897932384626
#define rows 128
#define cols 128
#define QUALITY 50

using namespace std;

typedef ap_fixed <16,16> float1;

float1 cosMatrix[8][8] = {
		{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
		{0.98079, 0.83147, 0.55557, 0.19509, -0.19509, -0.55557, -0.83147, -0.98079},
		{0.92388, 0.38268, -0.38268, -0.92388, -0.92388, -0.38268, 0.38268, 0.92388},
		{0.83147, -0.19509, -0.98079, -0.55557, 0.55557, 0.98079, 0.19509, -0.83147},
		{0.70711, -0.70711, -0.70711, 0.70711, 0.70711, -0.70711, -0.70711, 0.70711},
		{0.55557, -0.98079, 0.19509, 0.83147, -0.83147, -0.19509, 0.98079, -0.55557},
		{0.38268, -0.92388, 0.92388, -0.38268, -0.38268, 0.92388, -0.92388, 0.38268},
		{0.19509, -0.55557, 0.83147, -0.98079, 0.98079, -0.83147, 0.55557, -0.19509}
	};

float1 quant_matrix[8][8]= {
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
void calc_quant(float1 qm[8][8])
{	//http://stackoverflow.com/questions/29215879/how-can-i-generalize-the-quantization-matrix-in-jpeg-compression
	int i,j;
	float1 def[8][8] = {
		{16,  11,  10,  16,  24, 40,  51,  61} ,
		{12,  12,  14,  19,  26, 58,  60,  55} ,
		{14,  13,  16,  24,  40, 57,  69,  56} ,
		{14,  17,  22,  29,  51, 87,  80,  62} ,
		{18,  22,  37,  56,  68, 109, 103, 77} ,
		{24,  35,  55,  64,  81, 104, 113,  92} ,
		{49,  64,  78,  87,  103, 121, 120, 101} ,
		{72,  92,  95,  98, 112, 100, 103,  99}
	};
	float1 S;
	if(QUALITY < 50)
		S = (5000/(float1)(QUALITY));
	else
		S = 200-2*QUALITY;
	for(i=0; i<8; i++)
	{
		for(j=0; j<8; j++)
		{
			qm[i][j]=(float1)((S*def[i][j]+(float1)50)/(float1)100.0);
		}
	}
}

void quantize(float1 qm[8][8], float1 G[8][8])
{
#pragma HLS PIPELINE
	int i,j;
	for(i=0; i<8; i++)
	{
		for(j=0; j<8; j++)
		{
			G[i][j] = (G[i][j]/(float1)qm[i][j])*qm[i][j];
		}
	}
}

float1 al(int x)
{
	if(x==0)
		return 1.0/sqrt(2.0);
	else
		return 1.0;
}

float1 singleGUVfwd(float1 subimg[8][8], int u, int v)
{
	float1 G=0;
	int x,y;

	for(x=0; x<8; x++)
	{
		for(y=0; y<8; y++)
		{
			G+=(float1)0.25*al(u)*al(v)*subimg[x][y]*cosMatrix[u][x]*cosMatrix[v][y];
		}
	}
	return G;
}

float1 singleGUVinv(float1 subimg[8][8], int u, int v)
{
#pragma HLS PIPELINE
	float1 G=0;
	int x,y;

	for(x=0; x<8; x++)
	{
		for(y=0; y<8; y++)
		{
			G+=(float1)0.25*al(x)*al(y)*subimg[x][y]*cosMatrix[x][u]*cosMatrix[y][v];
		}
	}
	return G;
}

void shift128(float1 subimg[8][8])
{
#pragma HLS PIPELINE
	int i,j;
	for(i=0; i<8; i++)
	{
		for(j=0; j<8; j++)
		{
			subimg[i][j]-=128;
		}
	}
}

void DCT_8x8_2D(float1 G[8][8], float1 subimg[8][8])
{
#pragma HLS PIPELINE
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

void inv_DCT_8x8_2D(float1 G[8][8], float1 subimg[8][8])
{
#pragma HLS PIPELINE
	int i,j;

	for(i=0; i<8; i++)
	{
		for(j=0; j<8; j++)
		{
			G[i][j]=singleGUVinv(subimg, i, j);
		}
	}
}

float1 min(float1 a, float1 b)
{
	return (a>b)?b:a;
}

float1 max(float1 a, float1 b)
{
	return (a>b)?a:b;
}

//converts from RGB to YCbCr when fwd=1 and back when fwd=0
void colour_space_conversion(signed short int image[8][8][3])
{
#pragma HLS PIPELINE
	float1 val1, val2, val3;
	float1 ch1, ch2, ch3;
	int i,j,k;
	for(i=0; i<8; i++)
	{
		for(j=0; j<8; j++)
		{
			ch1 = (float1)image[i][j][0];
			ch2 = (float1)image[i][j][1];
			ch3 = (float1)image[i][j][2];
			float1 delta = 128;

			val1 = (float1)0.299*ch1+(float1)0.587*ch2+(float1)0.114*ch3;
			val2 = delta + (float1)0.713*(ch1-val1);
			val3 = delta + (float1)0.564*(ch3-val1);
			image[i][j][0]=min(max((int)(val1),0),255);
			image[i][j][1]=min(max((int)(val2),0),255);
			image[i][j][2]=min(max((int)(val3),0),255);
		}
	}
}

//converts from RGB to YCbCr when fwd=1 and back when fwd=0
void invcolour_space_conversion(signed short int image[8][8][3])
{
#pragma HLS PIPELINE
	float1 val1, val2, val3;
	float1 ch1, ch2, ch3;
	int i,j,k;
	for(i=0; i<8; i++)
	{
		for(j=0; j<8; j++)
		{
			ch1 = (float1)image[i][j][0];
			ch2 = (float1)image[i][j][1];
			ch3 = (float1)image[i][j][2];
			float1 delta = 128;

			val1 = ch1+(float1)1.403*(ch2-delta);
			val2 = ch1-(float1)0.714*(ch2-delta)-(float1)0.344*(ch3-delta);
			val3 = ch1+(float1)1.733*(ch3-delta);
			image[i][j][0]=min(max((int)(val1),0),255);
			image[i][j][1]=min(max((int)(val2),0),255);
			image[i][j][2]=min(max((int)(val3),0),255);
		}
	}
}

void extract8x8subimg(signed short int subimg[8][8][3],signed short int image[128][128][3],int i,int j)
{
#pragma HLS PIPELINE
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
#pragma HLS PIPELINE
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

void extractYCrCb(signed short int subimg[8][8][3], float1 subimgY[8][8], float1 subimgC1[8][8], float1 subimgC2[8][8])
{
	int x,y;
#pragma HLS PIPELINE
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

void finalEval(signed short int subimg[8][8][3], float1 finalY[8][8], float1 finalC1[8][8], float1 finalC2[8][8])
{
#pragma HLS PIPELINE
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

void JPEG_encode(signed short int subimg[8][8][3],float1 quant_matrix[8][8])
{
#pragma HLS PIPELINE
	float1 DCTY[8][8], DCTC1[8][8], DCTC2[8][8];
	float1 finalY[8][8], finalC1[8][8], finalC2[8][8];
	float1 subimgY[8][8], subimgC1[8][8], subimgC2[8][8];
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
#pragma HLS INTERFACE s_axilite port=image bundle=a
#pragma HLS INTERFACE s_axilite port=return bundle=a
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
