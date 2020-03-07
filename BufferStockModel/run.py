from BufferStockModel import BufferStockModelClass
updpar = dict()
updpar["Np"] = 2000
updpar["Nm"] = 2000
updpar["Na"] = 2000
model = BufferStockModelClass(name="",solmethod="egm",**updpar)
model.test()
