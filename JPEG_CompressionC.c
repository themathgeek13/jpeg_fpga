#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>

#define bufSize 50
#define PI 3.1415926535897932384626

//quantization function for a given quality requirement
//this simply evaluates the quantization matrix, and does not actually
//perform the quantization step in the algorithm
void calc_quant(float qm[8][8], int quality)
{
	//http://stackoverflow.com/questions/29215879/how-can-i-generalize-the-quantization-matrix-in-jpeg-compression
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
	if(quality < 50)
		S = (5000/(float)(quality));
	else
		S = 200-2*quality;
	for(i=0; i<8; i++)
	{
		for(j=0; j<8; j++)
		{
			qm[i][j]=(float)((S*def[i][j]+50)/100.0);
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
			G[i][j] = (int)(G[i][j]/(float)qm[i][j])*qm[i][j];
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

float singleGUV(float subimg[8][8], int u, int v, int inv, float cosMat[8][8])
{
	float G=0;
	int x,y;
	
	for(x=0; x<8; x++)
	{
		for(y=0; y<8; y++)
		{
			if(inv==0)
				G+=0.25*al(u)*al(v)*subimg[x][y]*cosMat[u][x]*cosMat[v][y];
			else
				G+=0.25*al(x)*al(y)*subimg[x][y]*cosMat[x][u]*cosMat[y][v];
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

void DCT_8x8_2D(float G[8][8], float subimg[8][8], int shift, int inv, float cosMat[8][8])
{
	if(shift==1)
		shift128(subimg);
	int i,j;
	
	for(i=0; i<8; i++)
	{
		for(j=0; j<8; j++)
		{
			G[i][j]=singleGUV(subimg, i, j, inv, cosMat);
		}
	}
}

//pads the rows in the image and returns the new num_rows when ind=0
//pads the columns and returns new num_cols when ind=1
int imgPad(signed short int image[1024][1024][3], int rows, int cols, int ind)
{
	int i,j,k;
	if(ind==0)	//pad rows
	{
		if(rows>1016)	//pad to 1024
		{
			for(i=rows; i<1024; i++)
			{
				for(j=0; j<cols; j++)
				{
					for(k=0; k<3; k++)
					{
						image[i][j][k]=image[rows-1][j][k];
					}
				}
			}
			return 1024;
		}
		else
		{
			int val;
			if(rows%8 != 0)
				val=rows+8-rows%8;
			else
				val=rows;
			for(i=rows; i<val; i++)
			{
				for(j=0; j<cols; j++)
				{
					for(k=0; k<3; k++)
					{
						image[i][j][k]=image[rows-1][j][k];
					}
				}
			}
			return val;
		}
			
	}
	else
	{
		if(cols>1016)	//pad to 1024
		{
			for(i=0; i<rows; i++)
			{
				for(j=cols; j<1024; j++)
				{
					for(k=0; k<3; k++)
					{
						image[i][j][k]=image[i][cols-1][k];
					}
				}
			}
			return 1024;
		}
		else
		{
			int val;
			if(cols%8 != 0)
				val=cols+8-cols%8;
			else
				val=cols;
			for(i=0; i<rows; i++)
			{
				for(j=cols; j<val; j++)
				{
					for(k=0; k<3; k++)
					{
						image[i][j][k]=image[i][cols-1][k];
					}
				}
			}
			return val;
		}
	}
}

//function to evaluate the cosine matrix required for DCT calculations
//this will need to be hard coded as a look up table when working with the FPGA
//to avoid the use of the expensive CORDIC engine
void evalCos(float cosMatrix[8][8])
{
	int i,j;
	
	for(i=0; i<8; i++)
	{
		for(j=0; j<8; j++)
			cosMatrix[i][j]=cos(i*PI*(2*j+1)/16.0);
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
void colour_space_conversion(signed short int image[1024][1024][3], int rows, int cols, int fwd)
{
	float val1, val2, val3;
	float ch1, ch2, ch3;
	int i,j,k;
	
	for(i=0; i<rows; i++)
	{
		for(j=0; j<cols; j++)
		{
			ch1 = (float)image[i][j][0];
			ch2 = (float)image[i][j][1];
			ch3 = (float)image[i][j][2];
			if(fwd==1)		//RGB to YCrCb
			{
				val1 = 0.299*ch1+0.587*ch2+0.114*ch3;
				val2 = 128 + 0.713*(ch1-val1);
				val3 = 128 + 0.564*(ch3-val1);
				image[i][j][0]=min(max((int)(val1),0),255);
				image[i][j][1]=min(max((int)(val2),0),255);
				image[i][j][2]=min(max((int)(val3),0),255);
			}
			
			if(fwd==0)		//YCrCb to RGB
			{
				val1 = ch1+1.403*(ch2-128);
				val2 = ch1-0.714*(ch2-128)-0.344*(ch3-128);
				val3 = ch1+1.733*(ch3-128);
				image[i][j][0]=min(max((int)(val1),0),255);
				image[i][j][1]=min(max((int)(val2),0),255);
				image[i][j][2]=min(max((int)(val3),0),255);
			}
		}
	}
}

int main()
{
	FILE *fp;
	char buf[bufSize];
	int rows, cols;
	int i,j;
	signed short int image [1024][1024][3];
	float cosMatrix[8][8];
	float subimgY[8][8], subimgC1[8][8], subimgC2[8][8];
	float quant_matrix[8][8];

	fp = fopen("original.dat","r");
	
	//Successfully opened the file
	//line 1 has the rows and columns of the input matrix
	fgets(buf, sizeof(buf),fp);
	buf[strlen(buf)-1]='\0';	//removes the newline fgets stores
	sscanf(buf, "%d %d", &rows, &cols);
	
	//Code to read the whole file and obtain the image
	for(i=0; i<rows; i++)
	{
		for(j=0; j<cols; j++)
		{
			fgets(buf, sizeof(buf),fp);
			buf[strlen(buf)-1]='\0';	//removes the newline fgets stores
			sscanf(buf, "%hu %hu %hu", &image[i][j][0], &image[i][j][1], &image[i][j][2]);
		}
	}
	
	calc_quant(quant_matrix,10);	//create quantization matrix
	colour_space_conversion(image,rows,cols,1);		//RGB to YCrCb

	int rows1 = imgPad(image, rows, cols, 0);		//pads rows
	int cols1 = imgPad(image, rows, cols, 1);		//pads columns

	//printf("%d %d, %d %d\n", rows, cols, rows1, cols1);
	evalCos(cosMatrix);
	
	int rowval, colval;
	float DCTY[8][8], DCTC1[8][8], DCTC2[8][8];
	float finalY[8][8], finalC1[8][8], finalC2[8][8];
	for(i=0; i<rows1; i+=8)
	{
		for(j=0; j<cols1; j+=8)
		{
			//evaluate the Y,Cr and Cb components of the sub-image
			for(rowval=i; rowval<i+8; rowval++)
			{
				for(colval=j; colval<j+8; colval++)
				{
					subimgY[rowval-i][colval-j]=image[rowval][colval][0];
					subimgC1[rowval-i][colval-j]=image[rowval][colval][1];
					subimgC2[rowval-i][colval-j]=image[rowval][colval][2];
				}
			}
			//DCT Step
			DCT_8x8_2D(DCTY,subimgY,1,0,cosMatrix);		//shift is TRUE, invert is FALSE
			DCT_8x8_2D(DCTC1,subimgC1,1,0,cosMatrix);
			DCT_8x8_2D(DCTC2,subimgC2,1,0,cosMatrix);
			
			//Quantization Step
			quantize(quant_matrix,DCTY);
	 		quantize(quant_matrix,DCTC1);
			quantize(quant_matrix,DCTC2);

			//Inverse DCT Step
			DCT_8x8_2D(finalY,DCTY,0,1,cosMatrix);		//shift is FALSE, invert is TRUE
			DCT_8x8_2D(finalC1,DCTC1,0,1,cosMatrix);
			DCT_8x8_2D(finalC2,DCTC2,0,1,cosMatrix);
			
			//evaluate the Y,Cr and Cb components of the sub-image
			for(rowval=i; rowval<i+8; rowval++)
			{
				for(colval=j; colval<j+8; colval++)
				{
					image[rowval][colval][0]=finalY[rowval-i][colval-j]+128;
					image[rowval][colval][1]=finalC1[rowval-i][colval-j]+128;
					image[rowval][colval][2]=finalC2[rowval-i][colval-j]+128;
				}
			}
		}
	}
	
	colour_space_conversion(image,rows,cols,0);		//YCrCb to RGB
	
	for(i=0; i<rows; i++)
	{
		for(j=0; j<cols; j++)
			printf("%hu %hu %hu\n",image[i][j][0],image[i][j][1],image[i][j][2]);
	}
	return 0;
}
