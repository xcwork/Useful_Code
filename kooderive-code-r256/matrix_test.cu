//
//
//       Matrix_Test.cu
//
//
// (c) Mark Joshi 2013,2014
// This code is released under the GNU public licence version 3

#include "matrix_test.h"
#include <vector>
#include <thrust/device_vector.h>
#include <Matrix_solve_main.h>
#include <curand.h>
#include <gold/math/basic_matrix_gold.h>
#include <Utilities.h>
#include <cutil.h>
#include <gold/math/svd_gold.h>
#include <gold/Timers.h>
//#include <cula_lapack_device.h>
#include <Utilities.h>

#include <outerProduct_gpu.h>
#include <cuBlas.h>
#include <LinearSolverConcrete_main.h>
#include <gold/math/transpose_gold.h>
#include <transpose_main.h>

#include <ComparisonCheck.h>

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

int Matrix_solving_test_routines(bool verbose,DeviceChooser& chooser)
{
    cublasInit();

    int result=2;
    double tolerance =1E-3;

    int rows = 327680;
    int columns = 10;
    int matrixDataSize = rows*columns;
    double noiseMultiplier = 0.0001;

    cudaSetDevice(chooser.WhichDevice());
    //	culaInitialize();

    thrust::device_vector<float> A_dev(rows*columns);
    thrust::device_vector<float> A_2_dev(rows*columns);
    thrust::device_vector<float> A_3_dev(rows*columns);
    thrust::device_vector<float> A_4_dev(rows*columns);
    thrust::device_vector<float> A_5_dev(rows*columns);
    thrust::device_vector<float> A_6_dev(rows*columns);
    thrust::device_vector<float> A_7_dev(rows*columns);


    thrust::device_vector<float> solution_dev(columns);

    thrust::device_vector<float> randomData_Dev(rows*columns);
    thrust::device_vector<float> randomTargetData_Dev(rows);

    thrust::device_vector<float> noisyTargetData_dev(rows);
    thrust::device_vector<float> noisyTarget2_dev(rows);
    thrust::device_vector<float> noisyTarget3_dev(rows);
    thrust::device_vector<float> noisyTarget4_dev(rows);
    thrust::device_vector<float> noisyTarget5_dev(rows);
    thrust::device_vector<float> noisyTarget6_dev(rows);
    thrust::device_vector<float> noisyTarget7_dev(rows);


    curandGenerator_t gen;

    CURAND_CALL(curandCreateGenerator(&gen, 
        CURAND_RNG_PSEUDO_MTGP32));

    /* Set seed */
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 
        1234ULL));

    /* Generate n floats on device */
    CURAND_CALL(curandGenerateUniform(gen, thrust::raw_pointer_cast(&randomData_Dev[0]), matrixDataSize));
    CURAND_CALL(curandGenerateUniform(gen, thrust::raw_pointer_cast(&randomTargetData_Dev[0]), rows));

    std::vector<double> workspace;
    std::vector<double> model_solution(columns);

    for (int i=0; i < columns; ++i)
        model_solution[i] = i;

    std::vector<float> stlvector(stlVecFromDevVec(randomData_Dev));

    Matrix_gold<double> A_cpu_mat(rows,columns,0.0);

    for (int i=0; i < rows; ++i)
        for (int j=0; j < columns; ++j)
            A_cpu_mat(i,j) = i ==j ? 5.0 : (noiseMultiplier*i+2*noiseMultiplier*j)*stlvector[j*rows+i];


    std::vector<double> matrixProduct(rows);
    MatrixArrayProductRobust(A_cpu_mat.ConstFacade(),model_solution,matrixProduct,workspace);

    std::vector<double> noisyTarget(matrixProduct);

    std::vector<float> targetNoise(stlVecFromDevVec(randomTargetData_Dev));

    for (int i=0; i < rows; ++i)
        noisyTarget[i] += (targetNoise[i]-0.5)*noiseMultiplier;

    std::vector<double> noisySolution_gold(columns);


    Matrix_gold<double> A_cpu_mat_transpose(columns,rows,0.0);
    Matrix_gold<double> ATA(columns,columns,0.0);
    std::vector<double> ATtarget(columns);


    MatrixTranspose(A_cpu_mat.Facade(),A_cpu_mat_transpose.Facade());

    Timer hgold;

    MatrixMatrixProductRobust(A_cpu_mat_transpose.ConstFacade(),A_cpu_mat.ConstFacade(),ATA.Facade(),workspace);
    MatrixArrayProductRobust(A_cpu_mat_transpose.ConstFacade(),noisyTarget,ATtarget,workspace);

    SVDSquareMatrix solver(columns);

    solver.SVDSolve(ATA.ConstFacade(),ATtarget,noisySolution_gold,1e-8);

    double time_gold=hgold.timePassed();


    thrust::host_vector<float> A_host(rows*columns);
    for (int i=0; i < rows*columns; ++i)
        A_host[i] = static_cast<float>(A_cpu_mat_transpose.getDataVector()[i]);

    A_dev = A_host;
    A_2_dev = A_dev;
    A_3_dev = A_dev;
    A_4_dev = A_dev;
    A_5_dev = A_dev;
    A_6_dev = A_dev;
    A_7_dev = A_dev;

    thrust::host_vector<float> noisyTarget_host(rows);
    for (int i=0; i < rows; ++i)
        noisyTarget_host[i] = static_cast<float>(noisyTarget[i]);

    noisyTargetData_dev=noisyTarget_host;

    noisyTarget2_dev= noisyTargetData_dev;
    noisyTarget3_dev= noisyTargetData_dev;
    noisyTarget4_dev= noisyTargetData_dev;
    noisyTarget5_dev= noisyTargetData_dev;
    noisyTarget6_dev= noisyTargetData_dev;
    noisyTarget7_dev= noisyTargetData_dev;

    //Timer hcula;

    //culaSolve(columns,rows, thrust::raw_pointer_cast(&A_dev[0]),thrust::raw_pointer_cast(&noisyTargetData_dev[0]),'N');
    //  cudaThreadSynchronize();

    //	double timeTakenCula = hcula.timePassed();

    thrust::copy(noisyTargetData_dev.begin(),noisyTargetData_dev.begin()+columns,solution_dev.begin());


    std::cout << "gold time was " << time_gold << "\n";
    /*
    bool testFailed = false;

    for (int i=0; i < columns; ++i)
    {
    if( (fabs(solution_dev[i] - noisySolution_gold[i])>tolerance && i >0) ||fabs(solution_dev[0] - noisySolution_gold[0]) > 0.01 )
    {
    std::cout << "test failed  , " << solution_dev[i] << "," <<  noisySolution_gold[i]		<< "\n";
    testFailed = true;
    }
    }

    if (!testFailed)
    {
    --result;
    std::cout << "pure cula test passed , speed ratio:" << time_gold/timeTakenCula << "\n";
    }

    std::cout << "timeTakenCula total time " << timeTakenCula << " \n";
    */

    Matrix_gold<float> AATthrust(columns,columns,0.0f);
    std::vector<float> output_vec(columns);
    std::vector<double> noisySolution2_gold(columns);

    Timer hthrust;

    thrustMultiplyByTransposeAAT(A_2_dev,rows,  columns, AATthrust.Facade());

    thrustMultiplyATx(A_2_dev, rows, columns,  noisyTarget2_dev,
        output_vec);

    Matrix_gold<double > AATdouble(MatrixCastMatrix<double,float>(AATthrust));
    std::vector<double> ATx(stlVecCastStlVec<double,float>(output_vec));


    solver.SVDSolve(AATdouble.ConstFacade(),ATx,noisySolution2_gold,1e-8);


    double time_thrust=hthrust.timePassed();

    bool test2Failed = false;

    for (int i=0; i < columns; ++i)
    {
        if( (fabs(noisySolution2_gold[i] - noisySolution_gold[i])>tolerance && i >0) ||fabs(noisySolution2_gold[0] - noisySolution_gold[0]) > 0.01 )
        {
            std::cout << "test failed  , " << noisySolution2_gold[i] << "," <<  noisySolution_gold[i]		<< "\n";
            test2Failed = true;
        }
    }

    if (!test2Failed)
    {
        --result;
        std::cout << "thrust and cula test passed ,ratio to gold:" <<time_gold/time_thrust<< "\n";
    }

    std::cout << "time_thrust total time " << time_thrust << " \n\n";


    thrust::device_vector<float> ATA_dev(columns*columns);
    thrust::device_vector<float> ATxCublas_dev(columns);
    /*
    Timer hcublas;

    reduceOuterProductSymmetric_cublas_initted(rows,
    columns,
    1.0f,
    0.0f,
    thrust::raw_pointer_cast(&A_3_dev[0]),
    thrust::raw_pointer_cast(&ATA_dev[0]));

    reduceOuterProduct_cublas_initted(rows,
    columns,
    1,
    1.0f, 
    0.0f, 
    thrust::raw_pointer_cast(&A_3_dev[0]),
    thrust::raw_pointer_cast(&noisyTarget3_dev[0]),
    thrust::raw_pointer_cast(&ATxCublas_dev[0]));

    cudaThreadSynchronize();

    culaSolve(columns,columns, thrust::raw_pointer_cast(&ATA_dev[0]),thrust::raw_pointer_cast(&ATxCublas_dev[0]),'N');


    cudaThreadSynchronize();

    double cublasTime = hcublas.timePassed();


    cublasShutdown();
    bool test3Failed = false;
    for (int i=0; i < columns; ++i)
    {
    if( (fabs(ATxCublas_dev[i] - noisySolution_gold[i])>tolerance && i >0) ||fabs(ATxCublas_dev[0] - noisySolution_gold[0]) > 0.01 )
    {
    std::cout << "test failed  , " << ATxCublas_dev[i] << "," <<  noisySolution_gold[i]		<< "\n";
    test3Failed = true;
    }
    }

    if (!test3Failed)
    {
    --result;
    std::cout << "cublas/cula test passed ,ratio: " <<time_gold/cublasTime<< "\n";
    }
    std::cout << "cublas plus cula total time " << cublasTime << " \n\n";
    */

    thrust::device_vector<float> ATA_hand_dev(columns*columns);
    thrust::device_vector<float> ATxhand_dev(columns);
    thrust::device_vector<float> ATxhand2_dev(columns);
    thrust::device_vector<float> ATxhand3_dev(columns);


    int threads = 128;
    int blocks = 128;
    thrust::device_vector<float> workspace_dev(columns*columns*blocks);
    /*
    Timer hhand;

    reduceOuterProductSymmetric_gpu(rows,
    columns,
    threads,
    blocks,
    thrust::raw_pointer_cast(&A_4_dev[0]),
    thrust::raw_pointer_cast(&ATA_hand_dev[0]),
    thrust::raw_pointer_cast(&workspace_dev[0]));

    reduceOuterProduct_gpu(rows,
    columns,
    1,
    false,
    threads,
    blocks,
    thrust::raw_pointer_cast(&A_4_dev[0]),
    thrust::raw_pointer_cast(&noisyTarget4_dev[0]),
    thrust::raw_pointer_cast(&ATxhand_dev[0]),
    thrust::raw_pointer_cast(&workspace_dev[0]));

    cudaThreadSynchronize();

    culaSolve(columns,columns, thrust::raw_pointer_cast(&ATA_hand_dev[0]),thrust::raw_pointer_cast(&ATxhand_dev[0]),'N');


    cudaThreadSynchronize();

    double handTime = hhand.timePassed();



    bool test4Failed = false;
    for (int i=0; i < columns; ++i)
    {
    if( (fabs(ATxhand_dev[i] - noisySolution_gold[i])>tolerance && i >0) ||fabs(ATxhand_dev[0] - noisySolution_gold[0]) > 0.01 )
    {
    std::cout << "hand test failed  , " << ATxhand_dev[i] << "," <<  noisySolution_gold[i]		<< "\n";
    test4Failed = true;
    }
    }

    if (!test4Failed)
    {
    --result;
    std::cout << "hand/cula test passed ," <<time_gold/handTime<< "\n";
    }

    std::cout << "hand/cula Time total time " << handTime << " \n\n";
    */
    {

        LinearSolverHandCraftedAndGold solverLSH(rows,  columns,  blocks,  threads);

        Timer LSH;

        solverLSH.solveSystem(A_5_dev, noisyTarget5_dev, ATxhand2_dev );

        cudaThreadSynchronize();

        double lshTime = LSH.timePassed();


        bool test5Failed = false;
        for (int i=0; i < columns; ++i)
        {
            if( (fabs(ATxhand2_dev[i] - noisySolution_gold[i])>tolerance && i >0) ||fabs(ATxhand2_dev[0] - noisySolution_gold[0]) > 0.01 )
            {
                std::cout << "hand test by class failed  , " << ATxhand2_dev[i] << "," <<  noisySolution_gold[i]		<< "\n";
                test5Failed = true;
            }
        }

        if (!test5Failed)
        {
            --result;
            std::cout << "LSH test passed ,ratio: " <<time_gold/lshTime<< "\n\n";
        }
    }


    /*
    std::cout << "got here\n";

    LinearSolverHandCraftedAndThrustAndCula solverLSTH(rows,  columns,  blocks,  threads);

    Timer LSTH;

    std::cout << "got here\n";
    solverLSTH.solveSystem(A_6_dev, noisyTarget6_dev, ATxhand2_dev );

    std::cout << "got here\n";
    cudaThreadSynchronize();

    std::cout << "got here\n";
    double lsthTime = LSTH.timePassed();

    std::cout << "got here\n";

    bool test6Failed = false;
    for (int i=0; i < columns; ++i)
    {
    if( (fabs(ATxhand2_dev[i] - noisySolution_gold[i])>tolerance && i >0) ||fabs(ATxhand2_dev[0] - noisySolution_gold[0]) > 0.01 )
    {
    std::cout << "hand, thrust test by class failed  , " << ATxhand2_dev[i] << "," <<  noisySolution_gold[i]		<< "\n";
    test6Failed = true;
    }
    }

    if (!test6Failed)
    {
    --result;
    std::cout << "LSTH test passed ," <<time_gold/lsthTime<< "\n";
    }
    }

    {
    */

    /*
    bool doallatonce = true;

    int ptthreads = 512;

    //	int ptblocks = 1+(rows-1)/ptthreads;
    int ptblocks = 2048;
    //	for (int threads =32; threads<=1024; threads+=32)
    {
    std::cout << "got here\n";

    LinearSolverHandCraftedAndThreadFenceAndCula solverLSTFH(rows,  columns,  blocks,  threads,doallatonce,ptblocks,ptthreads);
    std::cout << "got here\n";

    Timer LSTFH;
    std::cout << "got here\n";

    solverLSTFH.solveSystem(A_7_dev, noisyTarget7_dev, ATxhand3_dev );
    std::cout << "got here\n";

    cudaThreadSynchronize();
    std::cout << "got here\n";

    double lstfhTime = LSTFH.timePassed();


    bool test7Failed = false;
    for (int i=0; i < columns; ++i)
    {
    if( (fabs(ATxhand3_dev[i] - noisySolution_gold[i])>tolerance && i >0) ||fabs(ATxhand3_dev[0] - noisySolution_gold[0]) > 0.01 )
    {
    std::cout << "hand, thread fence test by class failed  , " << ATxhand3_dev[i] << "," <<  noisySolution_gold[i]		<< "\n";
    test7Failed = true;
    }
    }

    if (!test7Failed)
    {
    --result;
    std::cout << "LSTFH test passed ," <<time_gold/lstfhTime<< "," <<lstfhTime <<"," << ptblocks <<"\n";
    std::cout << "ptiwse time ," <<solverLSTFH.ptWiseTime()<< ", red time "<< solverLSTFH.justRedTimer() <<"\n";
    }
    }
    }
    */


    return result;
}


int MatrixTransposeTest(DeviceChooser& chooser)
{
    int result =2;

    int rows = 2048;
    int cols = 2048;



    cudaSetDevice(chooser.WhichDevice());
    {
        for (int i=0; i<2; ++i)
        {
            bool useTextures = i >0;
        thrust::host_vector<float> data_host(rows*cols);

        int l=0;

        for (int i=0; i < rows; ++i)
            for (int j=0; j < cols; ++j,++l)
                data_host[l] =static_cast<float>(l);

        thrust::device_vector<float> data_device(data_host);
        thrust::device_vector<float> data_transpose_device(data_host.size());

        std::vector<float> data_vec(data_host.begin(),data_host.end());
        std::vector<float> datatransposed_vec(data_vec.size());

        MatrixConstFacade<float> data_cmat(data_vec,rows,cols);
        MatrixFacade<float> data_transposed_mat(datatransposed_vec,cols,rows);

        Timer t_gold;


        TransposeMatrix(data_cmat, datatransposed_vec);

        double time_gold = t_gold.timePassed();
        cudaThreadSynchronize();

        Timer t_cuda1;

        transpose_using_gpu_main(data_device, rows, cols,
            data_transpose_device, useTextures);
	
        cudaThreadSynchronize();

	    double gpuTime = t_cuda1.timePassed();


        float tol = 1E-7;

        int errors =  numberMismatches(data_transposed_mat, data_transpose_device , tol, false);

        if (errors ==0)
        {
            --result;
              std::cout << "Transpose  test passed with " << useTextures << " textures.\n";
              std::cout << " Cpu time " << time_gold*1000 << "ms   GPU time " << gpuTime*1000 << "ms\n";
              std::cout << " ratio " << time_gold/gpuTime << "\n";
        }
        else
            std::cout << "Transpose using  test failed with " << errors << " errors" <<  useTextures << " textures\n";
    }
    }
    cudaDeviceReset();
    cudaThreadExit();


    return result;
}
