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
    GDALAllRegister();
    GDALDataset* dataset = (GDALDataset*)GDALOpen("a.tif",GA_ReadOnly);
    rows = dataset -> GetRasterYSize();
    cols = dataset -> GetRasterXSize();
    
    cout << "rank: " << rank << endl;
    cout << "rows: " << rows << "  cols: " << cols << endl;

    MPI_Barrier(MPI_COMM_WORLD); /* IMPORTANT */
    end_time = MPI_Wtime();
    MPI_Finalize();
    
    if(rank == 0){
        cout << "RUNNING TIME :" << end_time-start_time << "s" << endl;
    }
    return 0;
}