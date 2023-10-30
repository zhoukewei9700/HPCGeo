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

    if(rank == 0){
        cout << "获取当前进程的秩和总进程数完成" << endl;
    }

    // 根进程读取图像，获取行列号以及波段数
    int rows,cols,bands;
    GDALDataset* dataset;
    uchar*  imageData;
    if (rank == 0)
    {
        dataset = (GDALDataset*)GDALOpen("./a.tif",GA_ReadOnly);
        if (dataset == NULL)
        {
            // 处理打开文件失败的情况
            cout << "Open file failed" << endl;
            GDALClose(dataset);
            MPI_Finalize();
            return 1;
        }
        rows = dataset -> GetRasterYSize();
        cols = dataset -> GetRasterXSize();
        bands = dataset -> GetRasterCount();
        cout << "打开文件成功" << endl;
        cout << dataset->GetRasterBand(1)->GetRasterDataType() << endl;

        imageData = new uchar[rows * cols * bands];
        for(int i = 0; i < bands; ++i){
            CPLErr res =  dataset -> GetRasterBand(i + 1) -> RasterIO(
                GF_Read,
                0,0,
                cols,rows,
                imageData + i*rows*cols,
                cols,rows,
                GDT_Byte,0,0
            );
            if(res != CE_None){
                cout << "Failed in RasterIO" << endl;
                GDALClose(dataset);
                MPI_Finalize();
                return 1;
            }
        }

        // GDALClose(dataset);
        cout << "rows: " << rows << endl;
        cout << "cols: " << cols << endl;
        cout << "bands: " << bands <<endl;
    }

    // 广播行列数和波段数给其他进程
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);  
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bands, 1, MPI_INT, 0, MPI_COMM_WORLD);

    //重新用常量定义行列数以及波段数
    const int totalRows = rows;
    const int totalCols = cols;
    const int nBand = bands;

    // 各进程根据自己的秩计算要处理的图像行号
    int kernelSize = 15;
    int startRow = (rank == 0) ? 0 : (rank * (totalRows / size) - (kernelSize - 1) / 2);
    cout << "rank: " << rank << " " << " startRow " << startRow << endl;
    int endRow = (rank == size - 1) ? totalRows : ((rank + 1) * (totalRows / size) + (kernelSize - 1) / 2);
    cout << "rank: " << rank << " " << " endRow " << endRow << endl;
    int dataNeed2Process = (endRow - startRow) * totalCols;
    cout << "rank: " << rank << " " << " dataNeed2Process " << dataNeed2Process << endl;
    int displsPerProcess = startRow * totalCols;
    cout << "rank: " << rank << " " << " displsPerProcess " << displsPerProcess << endl;

    // 根进程发送分组数据
    // 此处能否多线程？
    uchar* recvData = new uchar[dataNeed2Process * nBand];
    int* sendCounts = nullptr;
    int* sendDispls = nullptr;
    if(rank == 0 ){
        sendCounts = new int[size];
        sendDispls = new int[size];

        // for(int i=0;i<size;++i){
        //     int startRow = (i == 0) ? 0 : (i * (totalRows / size) - (kernelSize - 1) / 2);
        //     int endRow = (i == size - 1) ? totalRows : ((i + 1) * (totalRows / size) + (kernelSize - 1) / 2);
        //     sendCounts[i] = (endRow - startRow) * totalCols;
        //     sendDispls[i] = startRow * totalCols;
        // }
    }

    MPI_Gather(
        &dataNeed2Process,
        1,
        MPI_INT,
        sendCounts + rank,
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );
    MPI_Gather(
        &displsPerProcess,
        1,
        MPI_INT,
        sendDispls + rank,
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    // // uchar* recvStart;
    for(int i = 0; i < nBand; i++){
        MPI_Scatterv(
            imageData + i * totalRows * totalCols,
            sendCounts,
            sendDispls,
            MPI_UNSIGNED_CHAR,
            recvData + i * dataNeed2Process,
            dataNeed2Process,
            MPI_UNSIGNED_CHAR,
            0,
            MPI_COMM_WORLD
        );
    }
    if(rank == 0){
        cout << "Scatter完成" << endl;
    }

    // TODO:  处理数据
    // 生成OpenCV的Mat对象
    Mat* band2Process = new Mat[nBand];
    for(int i = 0; i < nBand; ++i){
        band2Process[i] = Mat(endRow-startRow,totalCols,CV_8UC1,recvData + dataNeed2Process * i);

        // 对每个波段进行高斯模糊
        GaussianBlur(band2Process[i], band2Process[i], Size(kernelSize, kernelSize), 0, 0);
    }

    if(rank == 0){
        cout << "GaussianBlur完成" << endl;
    }

    // 汇集数据

    uchar* proccessedData = nullptr;
    int dataNeed2Gather = (rank == size - 1) ? (totalRows / size + totalRows % size)*totalCols : (totalRows / size)*totalCols;
    cout << "rank: " << rank <<" dataNeed2Gather: " << dataNeed2Gather<<endl;
    int dataStartRowPos = (rank == 0) ? 0 : ((kernelSize-1)/2);
    cout << "rank: " << rank <<" dataStartRowPos: " << dataStartRowPos<<endl;

    int *recvcounts = nullptr;
    int *displs = nullptr;
    if(rank == 0){
        proccessedData = new uchar[totalRows*totalCols*nBand];
        recvcounts = new int[size];
        displs = new int[size];
        for(int i = 0; i < size; i++) {
            recvcounts[i] = (i == size - 1) ? (totalRows / size + totalRows % size)*totalCols : (totalRows / size)*totalCols;
            displs[i] = i * (totalRows / size) * totalCols ;
        }
    }
    for(int i=0; i<nBand; i++){
        MPI_Gatherv(
            band2Process[i].data + dataStartRowPos * totalCols,
            dataNeed2Gather,
            MPI_UNSIGNED_CHAR,
            proccessedData + i * totalCols * totalRows,
            recvcounts,
            displs,
            MPI_UNSIGNED_CHAR,
            0,
            MPI_COMM_WORLD);
    }


    delete[] recvcounts;
    delete[] displs;
    if(rank == 0){
        cout << "GatherV完成" << endl;
    }

    // 根进程保存成图像
    if(rank == 0){
        // 复制原图像信息生成新图像
        const char* outputFilePath = "output.tif";
        GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("GTiff");
        GDALDataset* outputDataset = driver->CreateCopy(outputFilePath, dataset, FALSE, NULL, NULL, NULL);

        // 输入新图像数据
        for (int i = 0; i < nBand; ++i)
        {
            CPLErr res =  outputDataset -> GetRasterBand(i + 1) -> RasterIO(
                GF_Write,
                0,0,
                totalCols,totalRows,
                proccessedData + i * totalCols * totalRows,
                totalCols,totalRows,
                GDT_Byte,0,0
            );
            if(res != CE_None){
                cerr << "Failed to write data" << endl;
                GDALClose(dataset);
                MPI_Finalize();
                return 1;
            }
        }

        GDALClose(outputDataset);
        GDALClose(dataset);
    }

    MPI_Barrier(MPI_COMM_WORLD); /* IMPORTANT */
    end_time = MPI_Wtime();
    MPI_Finalize();
    
    if(rank == 0){
        cout << "RUNNING TIME :" << end_time-start_time << "s" << endl;
    }
    return 0;
}