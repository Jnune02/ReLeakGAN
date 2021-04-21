import numpy

base = numpy.empty((6944, 1, 2000))
rebase = base.reshape((6944, 2000))

fd_out = open("fd_out", 'w')

print("New Base Vector: %s" % (base.tolist()), file=fd_out)
print("Rebased Shape: %s" % (list(rebase.shape)))
