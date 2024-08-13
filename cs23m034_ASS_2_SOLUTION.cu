
/**
*   CS6023: GPU Programming
*   Assignment 2
*
*   Please don't change any existing code in this file.
*
*   Please add necessary memory APIs for your implementation. Use cudaFree()
*   to free up memory as soon as you're done with an allocation.
*   This will ensure that you don't run out of memory while running
*   large test cases. Use the minimum required memory for your
*   implementation. DO NOT change the kernel configuration parameters.
*/

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <cuda.h>

using namespace std;

using std::cin;
using std::cout;

typedef long long ll;

__global__ void dkernel(long int *matrix,long int *filter,long int *result,int m,int n,int k,int chunkSize)
{
  extern __shared__ long int arr[];
  long int id = blockIdx.x * blockDim.x + threadIdx.x;


  for(int i=0;i<chunkSize;++i)
  {
    if(threadIdx.x+i*n < k*k)
    {
       arr[threadIdx.x+i*n]=filter[threadIdx.x+i*n];
    }
  }

  __syncthreads();



  long int row=id/n;
  long int col=id%n;


  for(int i=row-k/2;i<=row+k/2;++i)
  {
    for(int j=col-k/2;j<=col+k/2;++j)
    {
      int filter_row=i-row+k/2;
      int filter_col=j-col+k/2;
      //printf("%d %d %d %d\n",i,j,filter_row,filter_col);
      if(i>=0 && i<m && j>=0 && j<n)
      {
        result[id]+=filter[filter_row*k+filter_col]*matrix[i*n+j];
      }
    }
  }


}



int main(int argc, char** argv) {

  //freopen("test2.txt","r",stdin);
  //freopen("output.txt","w",stdout);
    int m,n,k;
    cin>>m>>n>>k;


    long int* h_mat = new long int[m * n];
    long int* h_filter = new long int[k * k];


    long int* h_ans = new long int[m * n];


    for (long int i = 0; i < m * n; i++) {
        cin>>h_mat[i];
    }

    for (long int i = 0; i < k * k; i++) {
        cin>>h_filter[i];
    }

    /**
     *
     * DO NOT CHANGE ANYTHING ABOVE THIS LINE
     *
    **/

    long int *gmatrix;
    long int *gfilter;
    long int *result;
    int chunkSize;
    chunkSize=ceil((1.0*k*k)/n);
    cudaMalloc(&gmatrix,sizeof(long int)*m*n);
    cudaMemcpy(gmatrix,h_mat,sizeof(long int)*m*n,cudaMemcpyHostToDevice);



    cudaMalloc(&result,sizeof(long int)*m*n);
    cudaMemset(result,0,sizeof(long int)*m*n);


    cudaMalloc(&gfilter,sizeof(long int)*k*k);
    cudaMemcpy(gfilter,h_filter,sizeof(long int)*k*k,cudaMemcpyHostToDevice);
    /****************************************************Start Here***********************************************************/

    auto start = std::chrono::high_resolution_clock::now();//keep it just before the kernel launch

    dkernel<<<m,n,sizeof(long int)*k*k>>>(gmatrix,gfilter,result,m,n,k,chunkSize);
    auto end = std::chrono::high_resolution_clock::now();//keep it just after the kernel launch
    cudaMemcpy(h_ans,result,sizeof(long int)*m*n,cudaMemcpyDeviceToHost);


     cudaFree(gmatrix);
     cudaFree(gfilter);
     cudaFree(result);

     
    /*$$$$$$$$$$$$$$$$$$$$$$$$Make sure your final output from the device is stored in h_ans.$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
    std::chrono::duration<double> elapsed1 = end - start;
    /**
     *
     * DO NOT CHANGE ANYTHING BELOW THIS LINE
     *
    */



    std::ofstream file("cuda.out");
    if (file.is_open()) {
        for (long int i = 0; i < m; i++) {
            for (long int j = 0; j < n; j++) {
                file << h_ans[i * n + j] << " ";
            }
            file << "\n";
        }
        file.close();
    } else {
        std::cout << "Unable to open file";
    }

    std::ofstream file2("cuda_timing.out");
    if(file2.is_open()) {
        file2 << elapsed1.count() << "\n";
        file2.close();
    } else {
        std::cout << "Unable to open file";
    }

    return 0;
}
