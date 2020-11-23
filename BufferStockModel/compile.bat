cd /d "C:/Program Files (x86)/Microsoft Visual Studio/2017/Community/VC/Auxiliary/Build/"
call vcvarsall.bat x64
cd /d "C:\Users\gmf123\Documents\repositories\test\ConsumptionSavingNotebooks\BufferStockModel"
cl /LD /EHsc /Ox /openmp cppfuncs/egm.cpp setup_omp.cpp  
