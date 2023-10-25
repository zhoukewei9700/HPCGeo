#include <mpi.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <gdal/gdal.h>
#include <gdal/gdal_priv.h>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    double start_time,end_time;
    MPI_Init(&argc, &argv);
    GDALAllRegister();

    // 记录运行时间 
    MPI_Barrier(MPI_COMM_WORLD); /* IMPORTANT */
    start_time = MPI_Wtime();

    // 获取当前进程的秩和总进程数
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 根进程读取图像，获取行列号以及波段数
    int rows,cols,bands;
    uchar* imageData = nullptr;
    if (rank == 0)
    {
        GDALDataset* dataset = (GDALDataset*)GDALOpen("./a.tif",GA_ReadOnly);
        if (dataset == NULL)
        {
            // 处理打开文件失败的情况
            cerr << "Open file failed" << endl;
            return 1;
        }
        rows = dataset -> GetRasterYSize();
        cols = dataset -> GetRasterXSize();
        bands = dataset -> GetRasterCount();
        for(int i = 1; i <= bands; ++i){
            CPLErr res =  dataset -> GetRasterBand(i) -> RasterIO(
                GF_Read,
                0,0,
                cols,rows,
                imageData,
                cols,rows,
                GDT_Byte,0,0
            );
        }

        GDALClose(dataset);
        cout << "rows: " << rows << endl;
        cout << "cols: " << cols << endl;
        cout << "bands: " << bands <<endl;
    }

    // 广播行列数和波段数给其他进程
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bands, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 各进程根据自己的秩计算要处理的图像行号
    int kernelSize = 15;
    int startRow = (rank == 0) ? 0 : (rank * rows / size - (kernelSize - 1) / 2);
    int endRow = (rank == size - 1) ? rows : ((rank + 1) * rows / size + (kernelSize - 1) / 2);
    int rowsNeed2Process = endRow - startRow + 1;
    int displsPerProcess = rows / size;

    int* displs = nullptr;
    int* counts = nullptr;
    if (rank == 0)
    {
        int* displs = new int[size];
        int* counts = new int[size];
        for(int i = 0; i < size; ++i){
            counts[i] = displsPerProcess;
        }
    }

    MPI_Gather(&rowsNeed2Process,1,MPI_INT,counts+rank,1,MPI_INT,0,MPI_COMM_WORLD);
    
    


    int rowsPerProcess = rows / size;
    int remainder = rows % size;
    int* displs = new int[size];
    int* counts = new int[size];
    int displacement = 0;
    for(int i = 0; i < size; ++i)
    {
        int kernelSize = 15;
        int startRow = (i == 0) ? 0 : (i * rows / size - (kernelSize - 1) / 2);
        int endRow = (i == size - 1) ? rows : ((i + 1) * rows / size + (kernelSize - 1) / 2);
    }

    // 根进程读取数据并将每个进程负责的数据分发
    if(rank == 0){

    }



    MPI_Barrier(MPI_COMM_WORLD); /* IMPORTANT */
    end_time = MPI_Wtime();
    MPI_Finalize();
    
    if(rank == 0){
        cout << "RUNNING TIME :" << end_time-start_time << "s" << endl;
    }
    return 0;
}