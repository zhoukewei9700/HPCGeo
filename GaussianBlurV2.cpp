#include <mpi.h>
// #include <omp.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <gdal/gdal.h>
#include <gdal/gdal_priv.h>
#include <iostream>
#include <string.h>


using namespace cv;
using namespace std;

#define KERNEL_SIZE 15
#define BUFFER_START_X 0
#define BUFFER_START_Y 1
#define BUFFER_ROWS 2
#define BUFFER_COLS 3


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
    // 每条边要分成几段
    const int numTilePerEdge = (int)sqrt(size);

    GDALDataset* dataset;
    uchar* imageData;
    int rows,cols,bands;

    // 根进程读取图片，并计算每个进程处理的范围
    if(rank == 0){
        dataset = (GDALDataset*)GDALOpen("./input/a.tif",GA_ReadOnly);
        if(dataset == NULL)
        {
            // 处理打开文件失败的情况
            cout << "Open file failed" << endl;
            GDALClose(dataset);
            MPI_Finalize();
            return 1;
        }
        cout << "打开文件成功" << endl;
        rows = dataset -> GetRasterYSize();
        cols = dataset -> GetRasterXSize();
        bands = dataset -> GetRasterCount();
        cout << "rows: " << rows << endl;
        cout << "cols: " << cols << endl;
        cout << "bands: " << bands <<endl;
    }
    // 广播行列数和波段数给其他进程
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);  
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bands, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 每个进程分别计算bufferInfo
    // 进程所负责的tile的行列号
    int tileY = rank / numTilePerEdge;
    int tileX = rank % numTilePerEdge;
    // 进程所负责的tile的起始坐标以及行列数
    int bufferWidth = (KERNEL_SIZE-1)/2;
    int startX = tileX * (cols / numTilePerEdge);
    int startY = tileY * (rows / numTilePerEdge);
    int tileRows = (tileY == numTilePerEdge-1) ? (rows / numTilePerEdge) : (rows / numTilePerEdge + rows % numTilePerEdge);
    int tileCols = (tileX == numTilePerEdge-1) ? (cols / numTilePerEdge) : (cols / numTilePerEdge + cols % numTilePerEdge);
    imageData = new uchar[tileRows*tileCols*bands];

    int* bufferInfo = new int[4];
    bufferInfo[0] = (tileX == 0) ? startX : (startX - bufferWidth); // bufferStartX
    bufferInfo[1] = (tileY == 0) ? startY : (startY - bufferWidth); // bufferStartY
    bufferInfo[2] = (tileY == 0 || tileY == numTilePerEdge-1) ? (tileRows + bufferWidth) : (tileRows + 2 * bufferWidth); // bufferRows
    bufferInfo[3] = (tileX == 0 || tileX == numTilePerEdge-1) ? (tileCols + bufferWidth) : (tileCols + 2 * bufferWidth); // bufferCols

    int* bufferInfoCollection;
    if (rank == 0)
    {
        bufferInfoCollection = new int[4 * size];
    }
    
    MPI_Gather(
        bufferInfo,
        4,
        MPI_INT,
        bufferInfoCollection + rank * 4,
        4,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    // 根进程分别读取图片，并转换成Mat
    if(rank == 0)
    {
        uchar* bufferData = new uchar[(rows * cols + KERNEL_SIZE * (numTilePerEdge - 1) * 2) * bands];  
        for (int i = 0; i < size; i++)
        {
            int dataSize = bufferInfoCollection[i*4+3] * bufferInfoCollection[i*4+2];
            uchar* tempData = new uchar[bands * dataSize];
            for (int band = 1; band <= bands; band++)
            {
                CPLErr res =  dataset -> GetRasterBand(band) -> RasterIO(
                    GF_Read,
                    bufferInfoCollection[i*4],bufferInfoCollection[i*4+1],
                    bufferInfoCollection[i*4+3],bufferInfoCollection[i*4+2],
                    tempData + (band-1)*dataSize,
                    bufferInfoCollection[i*4+3],bufferInfoCollection[i*4+2],
                    GDT_Byte,0,0
                );
                if(res != CE_None){
                    cout << "Failed in RasterIO" << endl;
                    GDALClose(dataset);
                    MPI_Finalize();
                    return 1;
                }

                // bufferData[i] = 
            }
            
        }
        
    }


    if(tileX == 0){}

    int tilerow = (rank == 0 || rank == 1) ? (rows/2) : (rows/2 + rows%2);
    int tilecol = (rank == 0 || rank == 2) ? (cols/2) : (cols/2 + cols%2);
    


    int startX,startY,bufferStartX,bufferStartY;
    if(rank == 0){
        startX = 0;
        startY = 0;
        bufferStartX = 0;
        bufferStartY = 0;
    }
    else if (rank == 1)
    {
        startX = cols/2;
        startY = 0;
        bufferStartX = startX - bufferWidth;
        bufferStartY = 0;
    }
    else if(rank == 2)
    {
        startX = 0;
        startY = rows/2;
        bufferStartX = 0;
        bufferStartY = startY - bufferWidth;
    }
    else if(rank == 3)
    {
        startX = cols/2;
        startY = rows/2;
        bufferStartX = startX - bufferWidth;
        bufferStartY = startY - bufferWidth;
    }
    
    cout << "rank:" << rank << " tilecol:" << tilecol << " tilerow:" << tilerow << " startX:" <<startX << " startY:" << startY << endl;

    CPLErr res =  dataset -> GetRasterBand(1) -> RasterIO(
        GF_Read,
        bufferStartX,bufferStartY,
        bufferCol, bufferRow,
        imageData,
        bufferCol, bufferRow,
        GDT_Byte,0,0
    );
    if(res != CE_None){
        cout << "Failed in RasterIO" << endl;
        GDALClose(dataset);
        MPI_Finalize();
        return 1;
    }

    ostringstream oss;
    oss << "./output/out.tif";
    string outpath = oss.str();
    
    if(rank == 0){
        GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("GTiff");
        GDALDataset* outDataset = driver->Create(outpath.c_str(),cols,rows,1,GDT_Byte,NULL);
        outDataset->SetProjection(dataset->GetProjectionRef());
        double* geoTransform = new double[6];
        dataset->GetGeoTransform(geoTransform);
        outDataset->SetGeoTransform(geoTransform);
        GDALClose(outDataset);
    }
    GDALClose(dataset);
    MPI_Barrier(MPI_COMM_WORLD);
    GDALDataset* outputDataset = (GDALDataset*)GDALOpen(outpath.c_str(),GA_Update);
    if(rank == 0 || rank == 2){
        CPLErr res2 =  outputDataset -> GetRasterBand(1) -> RasterIO(
            GF_Write,
            startX,startY,
            tilecol,tilerow,
            imageData,
            tilecol,tilerow,
            GDT_Byte,0,0
        );
        if(res2 != CE_None){
            cerr << "Failed to write data" << endl;
            GDALClose(dataset);
            MPI_Finalize();
            return 1;
        }
        GDALClose(outputDataset);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 1 || rank == 3){
        CPLErr res2 =  outputDataset -> GetRasterBand(1) -> RasterIO(
            GF_Write,
            startX,startY,
            tilecol,tilerow,
            imageData,
            tilecol,tilerow,
            GDT_Byte,0,0
        );
        if(res2 != CE_None){
            cerr << "Failed to write data" << endl;
            GDALClose(dataset);
            MPI_Finalize();
            return 1;
        }
        GDALClose(outputDataset);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    cout << "succeed" << endl;
    MPI_Finalize();
}