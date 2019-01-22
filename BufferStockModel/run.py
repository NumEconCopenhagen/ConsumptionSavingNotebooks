from BufferStockModel import BufferStockModelClass
updpar = dict()
updpar["Np"] = 1500
updpar["Nm"] = 1500
updpar["Na"] = 1500
model = BufferStockModelClass(name="baseline",solmethod="egm",**updpar)
model.test()
