cd "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_2018.5.274/windows/bin/"
call ipsxe-comp-vars.bat intel64 vs2017
cd "C:\Users\gmf123\Dropbox\Repositories\ConsumptionSavingNotebooks\BufferStockModel"
icl /LD /O3 /arch:CORE-AVX512 /openmp cppfuncs//EGM.cpp
