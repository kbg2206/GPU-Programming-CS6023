#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <chrono>

using namespace std;

//******************************************

// Write down the kernels here

__global__ void kernel3(int *Health,int *ctr)
{
  //printf("%d\n",threadIdx.x);
  if(Health[threadIdx.x]>0)
  {
    atomicAdd(ctr,1);
  }

  
}

__global__ void kernel2(int *copy,int *score,int *Xcoor,int *Ycoor,int *Health,int T,long int k)
{
  //*ctr = 0;

  if(k%T == 0 || copy[blockIdx.x]<=0) return ;


  __shared__ long long MinDis;
  MinDis = LLONG_MAX;

  __syncthreads();

  int src= blockIdx.x;
  int des = threadIdx.x;
  int direction = (src+k)%T;

  int dirXcoor = Xcoor[direction] - Xcoor[src];
  int dirYcoor = Ycoor[direction] - Ycoor[src];
  int desXcoor = Xcoor[des] - Xcoor[src];
  int desYcoor = Ycoor[des] - Ycoor[src];

  //printf("%d %d %d %d %d %d %d\n",src,des,direction,dirXcoor,dirYcoor,desXcoor,desYcoor);

  int lhs = desXcoor*dirYcoor;
  int rhs = desYcoor*dirXcoor;
  long long distance =(long long) desYcoor * desYcoor + (long long) desXcoor * desXcoor;
  long int QuadX  = (long int)desXcoor * dirXcoor;
  long int QuadY  = (long int)desYcoor * dirYcoor;

  // if (dirXcoor>=0 && dirYcoor>=0) dirQuad=1;
  // else if (dirXcoor<0 && dirYcoor>=0) dirQuad=2;
  // else if (dirXcoor<0 && dirYcoor<0) dirQuad=3;
  // else dirQuad = 4;

  // if (desXcoor>=0 && desYcoor>=0) desQuad=1;
  // else if (desXcoor<0 && desYcoor>=0) desQuad=2;
  // else if (desXcoor<0 && desYcoor<0) desQuad=3;
  // else desQuad = 4;

  

  //printf("%d %d   ->   %d %d %ld %d %d\n",src,des,lhs,rhs,distance,dirQuad,desQuad);

  

  if(distance!=0  && copy[des]>0 && lhs == rhs && QuadX >=0 && QuadY>=0)
  {
    atomicMin(&MinDis,distance);
  }

  __syncthreads();

  //printf("%d %d  ->  %ld %ld %d %d\n",src,des,MinDis,distance,dirQuad,desQuad);

  //printf("Hello\n");
  if(MinDis == distance && QuadX >=0 && QuadY>=0 && lhs == rhs  && copy[des]>0)
  {
    
    atomicAdd(Health+des,-1);
    score[src]++;
    //atomicAdd(score+src,1);
  }

 // printf("%d %d  ->  %d %d %d\n",src,des,Health[des],score[src],MinDis);

  
}

__global__ void kernel1(int *Score,int *Health,int H)
{
   Health[threadIdx.x] = H;
   Score[threadIdx.x] = 0;
}



//***********************************************


int main(int argc,char **argv)
{
    // Variable declarations
    int M,N,T,H,*xcoord,*ycoord,*score;
    

    FILE *inputfilepointer;
    
    //File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer    = fopen( inputfilename , "r");

    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0; 
    }

    fscanf( inputfilepointer, "%d", &M );
    fscanf( inputfilepointer, "%d", &N );
    fscanf( inputfilepointer, "%d", &T ); // T is number of Tanks
    fscanf( inputfilepointer, "%d", &H ); // H is the starting Health point of each Tank
	
    // Allocate memory on CPU
    xcoord=(int*)malloc(T * sizeof (int));  // X coordinate of each tank
    ycoord=(int*)malloc(T * sizeof (int));  // Y coordinate of each tank
    score=(int*)malloc(T * sizeof (int));  // Score of each tank (ensure that at the end you have copied back the score calculations on the GPU back to this allocation)

    // Get the Input of Tank coordinates
    for(int i=0;i<T;i++)
    {
      fscanf( inputfilepointer, "%d", &xcoord[i] );
      fscanf( inputfilepointer, "%d", &ycoord[i] );
    }
		

    auto start = chrono::high_resolution_clock::now();

    //*********************************
    // Your Code begins here (Do not change anything in main() above this comment)
    //********************************



    


    int *GpuScore;
    cudaMalloc(&GpuScore,sizeof(int)*T);
    

    int *GpuHealth;
    cudaMalloc(&GpuHealth,sizeof(int)*T);

    // -------- Kernel Launch for Initialization -------------
    kernel1<<<1,T>>>(GpuScore,GpuHealth,H);

    int *GpuHealthcopy;
    cudaMalloc(&GpuHealthcopy,sizeof(int)*T);

    int *GpuXcoor;
    cudaMalloc(&GpuXcoor,sizeof(int)*T);
    cudaMemcpy(GpuXcoor,xcoord,sizeof(int)*T,cudaMemcpyHostToDevice);

    int *GpuYcoor;
    cudaMalloc(&GpuYcoor,sizeof(int)*T);
    cudaMemcpy(GpuYcoor,ycoord,sizeof(int)*T,cudaMemcpyHostToDevice);

    int *ctr;
    cudaHostAlloc(&ctr,sizeof(int),0);
    
    
    long int k=1;
    //printf("%ld\n",k);
  
    do
    { 
       *ctr = 0;
       
       cudaMemcpy(GpuHealthcopy,GpuHealth,sizeof(int)*T,cudaMemcpyDeviceToDevice);
       kernel2<<<T,T>>>(GpuHealthcopy,GpuScore,GpuXcoor,GpuYcoor,GpuHealth,T,k);
       kernel3<<<1,T>>>(GpuHealth,ctr);
       cudaDeviceSynchronize();
       //cout<<k<<" "<<*ctr<<"\n";
       //if(k==1000) break;
       
       k++;
    }while(*ctr>=2 );
    

    
  
    



    cudaMemcpy(score,GpuScore,sizeof(int)*T,cudaMemcpyDeviceToHost);

    cudaFree(GpuScore);
    cudaFree(GpuHealth);
    cudaFree(GpuHealthcopy);
    cudaFree(GpuXcoor);
    cudaFree(GpuYcoor);
    cudaFreeHost(ctr);
    // cudaFreeHost(k);

    //*********************************
    // Your Code ends here (Do not change anything in main() below this comment)
    //********************************

    auto end  = chrono::high_resolution_clock::now();

    chrono::duration<double, std::micro> timeTaken = end-start;

    printf("Execution time : %f\n", timeTaken.count());

    // Output
    char *outputfilename = argv[2];
    char *exectimefilename = argv[3]; 
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    for(int i=0;i<T;i++)
    {
        fprintf( outputfilepointer, "%d\n", score[i]);
    }
    fclose(inputfilepointer);
    fclose(outputfilepointer);

    outputfilepointer = fopen(exectimefilename,"w");
    fprintf(outputfilepointer,"%f", timeTaken.count());
    fclose(outputfilepointer);

    free(xcoord);
    free(ycoord);
    free(score);
    cudaDeviceSynchronize();
    return 0;
}