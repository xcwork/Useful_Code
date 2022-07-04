#include <LinearSolverConcrete_main.h>
#include <outerProduct_main.h>
//#include <cula_lapack_device.h>
#include <Utilities.h>
#include <cutil.h>
#include <gold/math/svd_gold.h>
#include <gold/Timers.h>
#include <outerProduct_gpu.h>
#include <Matrix_solve_main.h>
#include <CUDAConditionalMacros.h>
#include <gpuCompatibilityCheck.h>
/*
LinearSolverHandCraftedAndCula::LinearSolverHandCraftedAndCula(int rows, int columns, int blocks, int threads)
    :
    ATA_(columns*columns),ATx_(columns),
    rows_(rows), columns_(columns), blocks_(blocks), threads_(threads)
{
    if (threads==0)
        threads_ = 128;
    if( blocks == 0) 
        blocks_ = 128;


    workspace_.resize(blocks_*columns_*columns_);

    culaInitialize();

    timeSpent_=0.0;
}

LinearSolverHandCraftedAndCula::~LinearSolverHandCraftedAndCula()
{
    culaShutdown();
}



void LinearSolverHandCraftedAndCula::setDimensions(int rows, int columns)
{
    rows_ = rows;
    columns_ = columns;
    workspace_.resize(blocks_*columns_*columns_);
    ATA_.resize(columns*columns);
    ATx_.resize(columns);
}

void LinearSolverHandCraftedAndCula::solveSystem(thrust::device_vector<float>& AT, thrust::device_vector<float>& y, thrust::device_vector<float>& output )
{

    Timer hhand;

    reduceOuterProductSymmetric_gpu(rows_,
        columns_,
        threads_,
        blocks_,
        thrust::raw_pointer_cast(&AT[0]),
        thrust::raw_pointer_cast(&ATA_[0]),
        thrust::raw_pointer_cast(&workspace_[0]));




    reduceOuterProduct_gpu(rows_,
        columns_,
        1,
        false,
        threads_,
        blocks_,
        thrust::raw_pointer_cast(&AT[0]),
        thrust::raw_pointer_cast(&y[0]),
        thrust::raw_pointer_cast(&ATx_[0]),
        thrust::raw_pointer_cast(&workspace_[0]));

    cudaThreadSynchronize();

    double outerTime = hhand.timePassed();


    culaSolve(columns_,columns_, thrust::raw_pointer_cast(&ATA_[0]),thrust::raw_pointer_cast(&ATx_[0]),'N');


    cudaThreadSynchronize();

    output = ATx_;

    double handTime = hhand.timePassed();

    //	std::cout << "handTime total time " << handTime <<" " <<outerTime <<  " " << rows_ << " " << columns_ << " \n";

    timeSpent_+= handTime;

}
*/

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


LinearSolverHandCraftedAndGold::LinearSolverHandCraftedAndGold(int rows, int columns, int blocks, int threads)
    :
    ATA_(columns*columns),ATx_(columns),
    rows_(rows), columns_(columns), blocks_(blocks), threads_(threads),SVDsolver(columns)
{
    if (threads==0)
        threads_ = 128;
    if( blocks == 0) 
        blocks_ = 128;


    workspace_.resize(blocks_*columns_*columns_);


    timeSpent_=0.0;
}



void LinearSolverHandCraftedAndGold::setDimensions(int rows, int columns)
{
    rows_ = rows;
    columns_ = columns;
    workspace_.resize(blocks_*columns_*columns_);
    ATA_.resize(columns*columns);
    ATx_.resize(columns);
    SVDsolver = SVDSquareMatrix(columns);
}

void LinearSolverHandCraftedAndGold::solveSystem(thrust::device_vector<float>& AT, thrust::device_vector<float>& y, thrust::device_vector<float>& output )
{

    Timer hhand;

    reduceOuterProductSymmetric_gpu(rows_,
        columns_,
        threads_,
        blocks_,
        thrust::raw_pointer_cast(&AT[0]),
        thrust::raw_pointer_cast(&ATA_[0]),
        thrust::raw_pointer_cast(&workspace_[0]));

    ATA_host_ = ATA_;


    reduceOuterProduct_gpu(rows_,
        columns_,
        1,
        false,
        threads_,
        blocks_,
        thrust::raw_pointer_cast(&AT[0]),
        thrust::raw_pointer_cast(&y[0]),
        thrust::raw_pointer_cast(&ATx_[0]),
        thrust::raw_pointer_cast(&workspace_[0]));

    cudaThreadSynchronize();


    ATx_host_ = ATx_;

    double outerTime = hhand.timePassed();

    double tolerance = 1e-8;

    output_host_.resize(columns_);

    //culaSolve(columns_,columns_, thrust::raw_pointer_cast(&ATA_[0]),thrust::raw_pointer_cast(&ATx_[0]),'N');

    MatrixConstFacade<float> ATA_mat(&ATA_host_[0],columns_,columns_);

    {
        const float* const atx_ptr = &ATx_host_[0];

        SVDsolver.SVDSolve<float>(ATA_mat, 
            atx_ptr, 
            &output_host_[0],
            tolerance);
    }

    cudaThreadSynchronize();

    output = output_host_;

    double handTime = hhand.timePassed();

    //	std::cout << "handTime total time " << handTime <<" " <<outerTime <<  " " << rows_ << " " << columns_ << " \n";

    timeSpent_+= handTime;

}

