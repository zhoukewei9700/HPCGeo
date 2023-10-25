#include <mpi.h>
#include <time.h>
#include <opencv2/opencv.hpp>
// #include <gdal/gdal.h>
// #include <gdal/gdal_priv.h>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {

    double start_time,end_time;
    MPI_Init(&argc, &argv);

    // 记录运行时间 
    MPI_Barrier(MPI_COMM_WORLD); /* IMPORTANT */
    start_time = MPI_Wtime();

    // 获取当前进程的秩和总进程数
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 读取图像
    Mat image;
    if (rank == 0) {
        image = imread("test.png");
        cout << image.type() << endl;
        if (image.empty()) {
            cerr << "Failed to read the image." << endl;
            MPI_Finalize();
            return -1;
        }
    }

    // 广播图像尺寸
    int rows, cols,numChannels;
    if (rank == 0) {
        rows = image.rows;
        cols = image.cols;
        numChannels = image.channels();
    }
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numChannels, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // 计算每个进程的图像区域
    // 增加缓冲区
    // 缓冲区大小和卷积核大小有关
    int kernelSize = 15;
    int startRow,endRow;
    if(rank == 0)
    {
        startRow = 0;
        endRow = (rank + 1) * rows / size + (kernelSize - 1) / 2;
    }
    else if(rank == size - 1)
    {
        startRow = rank * rows / size - (kernelSize - 1) / 2;
        endRow = rows;
    }
    else
    {
        startRow = rank * rows / size - (kernelSize - 1) / 2;
        endRow = (rank + 1) * rows / size + (kernelSize - 1) / 2;
    }
    // int startRow = rank * rows / size;
    // int endRow = (rank + 1) * rows / size;

    // 分割图像
    Mat localImage(endRow-startRow,cols,CV_8UC(numChannels));
    uchar* localData = localImage.data;

    // 将图像数据转换为一维数组并按波段排列
    uchar* imageData = new uchar[rows*cols*numChannels];
    if (rank == 0) {
        imageData = image.data;
	if(imageData == nullptr){
	    cerr << "imageData is nullptr" << endl;
	    MPI_Finalize();
	    return -1;
	}
        //MPI_Bcast(imageData,rows*cols*numChannels,MPI_UNSIGNED_CHAR,0,MPI_COMM_WORLD);
    }

    // 广播图像数据
    MPI_Bcast(imageData, rows * cols * numChannels, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    // MPI_Bcast(&imageData, 1, MPI_A, 0, MPI_COMM_WORLD);

    // 每个进程从广播的数据中复制自己的图像区域
    // 图像在内存中是BGRBGRBGR这样排列的
    for (int i = startRow; i < endRow; ++i) {
        uchar* src = imageData + i * cols * numChannels;
        uchar* dst = localData + (i - startRow) * cols * numChannels;
        memcpy(dst, src, cols * numChannels);//内存拷贝
    }

    // 高斯模糊处理
    GaussianBlur(localImage, localImage, Size(kernelSize, kernelSize), 0, 0);
    cout << "GaussianBlur Finished" << endl;

    int dataSize = (rows / size) * cols * numChannels;
    uchar* processedData = (uchar*)malloc(dataSize);
    if(rank == 0){
        processedData = localImage.data;
    }
    else{
        processedData = localImage.data + (kernelSize - 1) * cols * numChannels / 2;
    }

    // 收集处理后的图像    
    Mat processedImage;
    if (rank == 0) {
        processedImage.create(rows, cols, image.type());
    }

    // 收集每个进程的数据
    MPI_Gather(
        processedData,
        dataSize,
        MPI_UNSIGNED_CHAR,
        processedImage.data + rank * dataSize,
        dataSize,
        MPI_UNSIGNED_CHAR,
        0,
        MPI_COMM_WORLD
    );

    // int dataSize = (rows/size)*cols*numChannels;
    // if(rank != 0)
    // {
    //     // 非根进程向根进程发送属于自己的数据，注意排除缓冲区
    //     MPI_Send(
    //         localImage.data + (kernelSize - 1) * cols * numChannels / 2, 
    //         rows * cols * numChannels / size,
    //         MPI_UNSIGNED_CHAR,
    //         0,
    //         0,
    //         MPI_COMM_WORLD
    //     );

    // }
    // else
    // {
    //     // 先排除自己部分的缓冲区
    //     uchar* src = localImage.data;
    //     uchar* dst = processedImage.data;
    //     memcpy(dst, src, (rows / size) * cols * numChannels);

    //     // 再接收其他进程的数据
    //     for (int rank = 1; rank < size; ++rank)
    //     {
    //         MPI_Recv(
    //             processedImage.data + rank * rows * cols * numChannels / size,
    //             rows * cols * numChannels / size,
    //             MPI_UNSIGNED_CHAR,
    //             rank,
    //             0,
    //             MPI_COMM_WORLD,
    //             MPI_STATUS_IGNORE
    //             );
    //     }
    // }
    
    // 合并图像
    if (rank == 0) {
        imwrite("output2.png", processedImage);
    }
    //delete[] imageData;

    MPI_Barrier(MPI_COMM_WORLD); /* IMPORTANT */
    end_time = MPI_Wtime();
    MPI_Finalize();
    
    if(rank == 0){
        cout << "RUNNING TIME :" << end_time-start_time << "s" << endl;
    }
    return 0;
}
