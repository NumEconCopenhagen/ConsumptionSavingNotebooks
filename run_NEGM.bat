set root=C:\Users\gmf123.UNICPH\AppData\Local\Continuum\anaconda3
call %root%\Scripts\activate.bat %root%
call jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --inplace "DurableConsumptionModel\A Guide On Solving Non-Convex Consumption-Saving Models.ipynb"
call jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --inplace "DurableConsumptionModel\A Guide On Solving Non-Convex Consumption-Saving Models - 2D.ipynb"
call jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --inplace "G2EGM/G2EGM vs NEGM.ipynb"
pause
