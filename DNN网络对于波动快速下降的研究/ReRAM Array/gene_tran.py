output_file = open("gene_tran_file.sp", "w")

#for i in range(0, 512):
#    output_file.write(".PROBE tran v(l0bl" + str(i) + ")" + "\n")



for i in range(0, 512):
    output_file.write(".measure tran avgvall0bl%d AVG v(l0bl%d) FROM = 0ns TO =1ns\n" % (i, i))
