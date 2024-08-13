
/*
	CS 6023 Assignment 3.
	Do not make any changes to the boiler plate code or the other files in the folder.
	Use cudaFree to deallocate any memory not in usage.
	Optimize as much as possible.
*/

#include "SceneNode.h"
#include <queue>
#include "Renderer.h"
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <chrono>


int ctr=0;
int dfs(int node,int *Offset,int *Csr,int *preorder,int *size_chart,int *start_chart)
{

	int start=Offset[node];
	int end=Offset[node+1];
  preorder[ctr++]=node;
  start_chart[node]=ctr-1;

  size_chart[node]=0;
	for(int i=start;i<end;++i)
	{
		int curr_node=Csr[i];
		size_chart[node]+=dfs(curr_node,Offset,Csr,preorder,size_chart,start_chart);

	}
  size_chart[node]+=(end-start);

  return size_chart[node];
}



__global__ void transaltion_kernel(int *preorder,int *start_chart,int *size_chart,int *transalation,int *CoordinateX,int *CoordinateY ,int n)
{
  long int id = blockIdx.x * blockDim.x + threadIdx.x;


	if(id<n)
	{

			int Mesh_id=transalation[id];
			int direction=transalation[id+n];
			int step=transalation[id+n*2];

			int size=size_chart[Mesh_id];
      int start=start_chart[Mesh_id];

      for(int i=start;i<=start+size;++i)
      {
        int *resultX=CoordinateX+preorder[i];
        int *resultY=CoordinateY+preorder[i];
        int dir[] = {-1,+1};

        if(direction/2==0)
        {
          atomicAdd(resultX,dir[direction]*step);
        }
        else
        {
          atomicAdd(resultY,dir[direction%2]*step);
        }
        
      }

	}
}


__global__ void map_create_kernel(int *map,int *opacity,int size)
{
	long int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id<size)
	{
		map[opacity[id]]=id;
	}
}

__global__ void opacity_kernel(int *Opacity,int *MeshSizeX,int *MeshSizeY,int *MeshStartX,int *MeshStartY,int *result,int V,int frameSizeX,int frameSizeY)
{
	long int id = ((long int)blockIdx.x * blockDim.x )+ threadIdx.x;
  int node = id / 10000;
  int idx  = id % 10000;


  if(node < V && idx < MeshSizeX[node]*MeshSizeY[node])
  {
      
      int row = idx / MeshSizeY[node];
      int col = idx % MeshSizeY[node];
      long int scene_idx = MeshStartX[node]+row;
      long int scene_idy = MeshStartY[node]+col;
      if(scene_idx>=0 && scene_idx<frameSizeX && scene_idy >=0 && scene_idy<frameSizeY)
      {
        int scene_id  = scene_idx * frameSizeY + scene_idy;
        atomicMax(&result[scene_id],Opacity[node]);
      }
  }


}




__global__ void scene_kernel(int *map,int *result,int **Mesh,int *MeshSizeX,int *MeshSizeY,int *MeshStartX,int *MeshStartY,int frameSizeX,int frameSizeY)
{
  	long int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id<frameSizeX*frameSizeY)
    {
      int row = id / frameSizeY;
      int col = id % frameSizeY;

      
      if(result[id]!=-1)
      {
        int node = map[result[id]];
        int Mesh_idx = row-MeshStartX[node];
        int Mesh_idy = col-MeshStartY[node];
        int Mesh_id = Mesh_idx*MeshSizeY[node]+Mesh_idy;
        result[id]=Mesh[node][Mesh_id];
      }
      else
      {
        result[id]=0;
      }
      
    }
}
void readFile (const char *fileName, std::vector<SceneNode*> &scenes, std::vector<std::vector<int> > &edges, std::vector<std::vector<int> > &translations, int &frameSizeX, int &frameSizeY) {
	/* Function for parsing input file*/

	FILE *inputFile = NULL;
	// Read the file for input.
	if ((inputFile = fopen (fileName, "r")) == NULL) {
		printf ("Failed at opening the file %s\n", fileName) ;
		return ;
	}

	// Input the header information.
	int numMeshes ;
	fscanf (inputFile, "%d", &numMeshes) ;
	fscanf (inputFile, "%d %d", &frameSizeX, &frameSizeY) ;


	// Input all meshes and store them inside a vector.
	int meshX, meshY ;
	int globalPositionX, globalPositionY; // top left corner of the matrix.
	int opacity ;
	int* currMesh ;
	for (int i=0; i<numMeshes; i++) {
		fscanf (inputFile, "%d %d", &meshX, &meshY) ;
		fscanf (inputFile, "%d %d", &globalPositionX, &globalPositionY) ;
		fscanf (inputFile, "%d", &opacity) ;
		currMesh = (int*) malloc (sizeof (int) * meshX * meshY) ;
		for (int j=0; j<meshX; j++) {
			for (int k=0; k<meshY; k++) {
				fscanf (inputFile, "%d", &currMesh[j*meshY+k]) ;
			}
		}
		//Create a Scene out of the mesh.
		SceneNode* scene = new SceneNode (i, currMesh, meshX, meshY, globalPositionX, globalPositionY, opacity) ;
		scenes.push_back (scene) ;
	}

	// Input all relations and store them in edges.
	int relations;
	fscanf (inputFile, "%d", &relations) ;
	int u, v ;
	for (int i=0; i<relations; i++) {
		fscanf (inputFile, "%d %d", &u, &v) ;
		edges.push_back ({u,v}) ;
	}

	// Input all translations.
	int numTranslations ;
	fscanf (inputFile, "%d", &numTranslations) ;
	std::vector<int> command (3, 0) ;
	for (int i=0; i<numTranslations; i++) {
		fscanf (inputFile, "%d %d %d", &command[0], &command[1], &command[2]) ;
		translations.push_back (command) ;
	}
}


void writeFile (const char* outputFileName, int *hFinalPng, int frameSizeX, int frameSizeY) {
	/* Function for writing the final png into a file.*/
	FILE *outputFile = NULL;
	if ((outputFile = fopen (outputFileName, "w")) == NULL) {
		printf ("Failed while opening output file\n") ;
	}

	for (int i=0; i<frameSizeX; i++) {
		for (int j=0; j<frameSizeY; j++) {
			fprintf (outputFile, "%d ", hFinalPng[i*frameSizeY+j]) ;
		}
		fprintf (outputFile, "\n") ;
	}
}


int main (int argc, char **argv) {

	// Read the scenes into memory from File.
	const char *inputFileName = argv[1] ;
	int* hFinalPng ;
	int frameSizeX, frameSizeY ;
	std::vector<SceneNode*> scenes ;
	std::vector<std::vector<int> > edges ;
	std::vector<std::vector<int> > translations ;
	readFile (inputFileName, scenes, edges, translations, frameSizeX, frameSizeY) ;
	hFinalPng = (int*) malloc (sizeof (int) * frameSizeX * frameSizeY) ;

	// Make the scene graph from the matrices.
    Renderer* scene = new Renderer(scenes, edges) ;

	// Basic information.
	int V = scenes.size () ;
	int E = edges.size () ;
	int numTranslations = translations.size () ;

	// Convert the scene graph into a csr.
	scene->make_csr () ; // Returns the Compressed Sparse Row representation for the graph.
	int *hOffset = scene->get_h_offset () ;
	int *hCsr = scene->get_h_csr () ;
	int *hOpacity = scene->get_opacity () ; // hOpacity[vertexNumber] contains opacity of vertex vertexNumber.
	int **hMesh = scene->get_mesh_csr () ; // hMesh[vertexNumber] contains the mesh attached to vertex vertexNumber.
	int *hGlobalCoordinatesX = scene->getGlobalCoordinatesX () ; // hGlobalCoordinatesX[vertexNumber] contains the X coordinate of the vertex vertexNumber.
	int *hGlobalCoordinatesY = scene->getGlobalCoordinatesY () ; // hGlobalCoordinatesY[vertexNumber] contains the Y coordinate of the vertex vertexNumber.
	int *hFrameSizeX = scene->getFrameSizeX () ; // hFrameSizeX[vertexNumber] contains the vertical size of the mesh attached to vertex vertexNumber.
	int *hFrameSizeY = scene->getFrameSizeY () ; // hFrameSizeY[vertexNumber] contains the horizontal size of the mesh attached to vertex vertexNumber.



auto start = std::chrono::high_resolution_clock::now () ;

  int *preorder;
  int *start_chart;
  int *size_chart;
  preorder=(int * )malloc(sizeof(int)*V);
  start_chart=(int * )malloc(sizeof(int)*V);
  size_chart=(int * )malloc(sizeof(int)*V);
  // ------- dfs call to store graph information -------------------------
  dfs(0,hOffset,hCsr,preorder,size_chart,start_chart);


// ---------------------------- GPU Allocation Start Here -------------------

	int *GpuPreOrder;
  cudaMalloc(&GpuPreOrder,sizeof(int)*V);
  cudaMemcpy(GpuPreOrder,preorder,sizeof(int)*V,cudaMemcpyHostToDevice);


  int *GpuStartChart;
  cudaMalloc(&GpuStartChart,sizeof(int)*V);
  cudaMemcpy(GpuStartChart,start_chart,sizeof(int)*V,cudaMemcpyHostToDevice);


  int *GpuSizeChart;
  cudaMalloc(&GpuSizeChart,sizeof(int)*V);
  cudaMemcpy(GpuSizeChart,size_chart,sizeof(int)*V,cudaMemcpyHostToDevice);


	int *GpuGlobalCoordinatesX;
	cudaMalloc(&GpuGlobalCoordinatesX,sizeof(int)*V);
	cudaMemcpy(GpuGlobalCoordinatesX,hGlobalCoordinatesX,sizeof(int)*V,cudaMemcpyHostToDevice);


	int *GpuGlobalCoordinatesY;
	cudaMalloc(&GpuGlobalCoordinatesY,sizeof(int)*V);
	cudaMemcpy(GpuGlobalCoordinatesY,hGlobalCoordinatesY,sizeof(int)*V,cudaMemcpyHostToDevice);


	int *GpuTranslations;
	int *dummyTranslations=(int *)malloc(sizeof(int)*numTranslations*3);
	for(int i=0;i<numTranslations;++i)
	{
		*(dummyTranslations+i)=translations[i][0];
		*(dummyTranslations+numTranslations+i)=translations[i][1];
		*(dummyTranslations+numTranslations*2+i)=translations[i][2];
    //cout<<*(dummyTranslations+i*numTranslations+0)<<" "<<*(dummyTranslations+i*numTranslations+1)<<" "<<*(dummyTranslations+i*numTranslations+2)<<"\n";
	}
	cudaMalloc(&GpuTranslations,sizeof(int)*numTranslations*3);
  cudaMemcpy(GpuTranslations,dummyTranslations,sizeof(int)*numTranslations*3,cudaMemcpyHostToDevice);



	// -------transalation Kernel launch ---------------
  long int block=ceil((1.0 * numTranslations)/1024);
  transaltion_kernel<<<block,1024>>>(GpuPreOrder,GpuStartChart,GpuSizeChart,GpuTranslations,GpuGlobalCoordinatesX,GpuGlobalCoordinatesY,numTranslations);
  // -------------------------------------------------------

	cudaFree(GpuPreOrder);
	cudaFree(GpuSizeChart);
	cudaFree(GpuStartChart);
	cudaFree(GpuTranslations);


	int *GpuOpacity;
	cudaMalloc(&GpuOpacity,sizeof(int)*V);
	cudaMemcpy(GpuOpacity,hOpacity,sizeof(int)*V,cudaMemcpyHostToDevice);


	int *GpuFrameSizeX;
	cudaMalloc(&GpuFrameSizeX,sizeof(int)*V);
	cudaMemcpy(GpuFrameSizeX,hFrameSizeX,sizeof(int)*V,cudaMemcpyHostToDevice);


  int *GpuFrameSizeY;
	cudaMalloc(&GpuFrameSizeY,sizeof(int)*V);
	cudaMemcpy(GpuFrameSizeY,hFrameSizeY,sizeof(int)*V,cudaMemcpyHostToDevice);


	int **GpuMesh;
	int **dummyMesh=(int **)malloc(sizeof(int *)*V);
	for(int i=0;i<V;++i)
	{
		int *address=*(hMesh+i);
		int m=*(hFrameSizeX+i);
		int n=*(hFrameSizeY+i);
		int *Mesh;

		cudaMalloc(&Mesh,sizeof(int)*m*n);
		cudaMemcpy(Mesh,address,sizeof(int)*m*n,cudaMemcpyHostToDevice);
		*(dummyMesh+i)=Mesh;
	}

	cudaMalloc(&GpuMesh,sizeof(int *)*V);
	cudaMemcpy(GpuMesh,dummyMesh,sizeof(int*)*V,cudaMemcpyHostToDevice);


  int *Opacity_matrix;
  cudaMalloc(&Opacity_matrix,sizeof(int)*frameSizeX*frameSizeY);
  cudaMemset(Opacity_matrix,-1,sizeof(int)*frameSizeX*frameSizeY);



  // ---------------------------- GPU Allocation End Here -------------------
  
  
  // -------------- opacity Matrix kernel Map ----------

	int *map;
	cudaMalloc(&map,sizeof(int)*3*100000000);

	int size_of_block =  ceil((1.0*V)/1024);
	map_create_kernel<<<size_of_block,1024>>>(map,GpuOpacity,V);
  //----------------------------------------------------------




	// ----------- kernel launch to create opacity Matrix ---------

  long int blockMeshes=ceil(((long int)V*100*100)/1024.0);
  opacity_kernel<<<blockMeshes,1024>>>(GpuOpacity,GpuFrameSizeX,GpuFrameSizeY,GpuGlobalCoordinatesX,GpuGlobalCoordinatesY,Opacity_matrix,V,frameSizeX,frameSizeY);
  //--------------------------------------------------------------

  // ------  kernel launch to create final matrix ----------

  long int  blockScene=ceil((1.0*frameSizeX*frameSizeY)/1024);
  scene_kernel<<<blockScene,1024>>>(map,Opacity_matrix,GpuMesh,GpuFrameSizeX,GpuFrameSizeY,GpuGlobalCoordinatesX,GpuGlobalCoordinatesY,frameSizeX,frameSizeY);
  // -----------------------------------------------------

	cudaFree(GpuMesh);
	cudaFree(GpuOpacity);
	cudaFree(GpuFrameSizeX);
	cudaFree(GpuFrameSizeY);
	cudaFree(GpuGlobalCoordinatesX);
	cudaFree(GpuGlobalCoordinatesY);

	

// Do not change anything below this comment.
// Code ends here.

  cudaMemcpy(hFinalPng,Opacity_matrix,sizeof(int)*frameSizeX*frameSizeY,cudaMemcpyDeviceToHost);


  cudaFree(map);
  cudaFree(Opacity_matrix);
	

	auto end  = std::chrono::high_resolution_clock::now () ;
	std::chrono::duration<double, std::micro> timeTaken = end-start;

	printf ("execution time : %f\n", timeTaken) ;
	//Write output matrix to file.
	const char *outputFileName = argv[2] ;
	writeFile (outputFileName, hFinalPng, frameSizeX, frameSizeY) ;

}
