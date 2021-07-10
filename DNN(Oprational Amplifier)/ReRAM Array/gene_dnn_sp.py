import numpy as np

# neuron = np.sign(np.load('layer0_neuron0.npy'))
# neuron = np.random.randint(2,size=1024)

weight = np.load('weight_512.npy', allow_pickle=True, encoding="latin1")
neuron = []
for i in range(len(weight)):
    neuron.append(np.sign(weight[i]))
print(neuron)
# neuron = []
# for i in range(4):
#    neuron.append(-1*np.ones((2,2))

bld_len = 512
input_len = 784
## shape = (bl length, bl number)
# layer0_shape = (784,1024)
# layer0_shape = (784,2)
# layer1_shape = (2,2)
# layer2_shape = (2,2)
# layer3_shape = (2,10)
layer0_shape = neuron[0].shape
layer1_shape = neuron[2].shape
layer2_shape = neuron[4].shape
layer3_shape = neuron[6].shape
output_file = open("dnn1.sp", "w")
# output_file = open("dnn_test.sp","w")
print(neuron[0].shape)
print(neuron[2].shape)
print(neuron[4].shape)
print(neuron[6].shape)
header_file = open("../../DNN网络对于波动快速下降的研究/ReRAM Array/module_header.sp", "r")
line = header_file.readline()
while (not line == ""):
    output_file.write(line)
    line = header_file.readline()

for i in range(1600):
    output_file.write(".subckt SAVM%d bl blb dl vdd\n" % (i + 2))
    output_file.write("m1 dl net19 vdd vdd P L=180e-9 W=1e-6\n")
    output_file.write("m0 net19 net19 vdd vdd P L=180e-9 W=1e-6\n")
    output_file.write("m3 dl blb 0 0 N%d L=180e-9 W=1e-6\n" % (i + 1602))
    output_file.write("m2 net19 bl 0 0 N%d L=180e-9 W=1e-6\n" % (i + 2))
    output_file.write(".ends SAVM%d\n" % (i + 2))
    output_file.write("\n")

for i in range(3200):
    output_file.write(".subckt OA%d g in_n in_p out vdd\n" % (i + 2))
    output_file.write("m3 net18 in_p net08 net08 P L=360e-9 W=24e-6\n")
    output_file.write("m2 net9 in_n net08 net08 P L=360e-9 W=24e-6\n")
    output_file.write("m1 out net28 vdd vdd P L=360e-9 W=20e-6\n")
    output_file.write("m0 net08 net28 vdd vdd P L=360e-9 W=20e-6\n")
    output_file.write("v0 vdd net28 DC=500e-3\n")
    output_file.write("m7 out net18 g g N L=360e-9 W=40e-6\n")
    output_file.write("m5 net9 net9 g g N L=360e-9 W=2e-6\n")
    output_file.write("m4 net18 net9 g g N L=360e-9 W=2e-6\n")
    output_file.write("r0 net24 out r 15e3\n")
    output_file.write("c0 net24 net18 10e-12\n")
    output_file.write(".ends OA%d\n" % (i + 2))
    output_file.write("\n")

# ==================layer0===================
# layer0 bl cells ex:l0bl0
for j in range(0, layer0_shape[1]):
    for i in range(0, layer0_shape[0]):
        output_file.write("xl0b%dc%d l0bl%d vdd x%s x%sb CELLD r1=%se3 r0=%se3\n"
                          % (j, i, j, i, i, (100 if neuron[0][i][j] == 1 else 5.3),
                             (5.3 if neuron[0][i][j] == 1 else 100)))
#    for i in range(layer0_shape[0],bld_len):
#       output_file.write("xl0b%dc%d l0bl%d vdd vref vrefb CELLD r1=%de3 r0=%de3\n" %(j,i,j,(100 if i%2==0 else 1),(1 if i%2==0 else 10)))

output_file.write('\n')
for i in range(0, layer0_shape[1]):
    # 在这里添加Operational Amplifier
    #  output_file.write("xl0oa%d 0 in_n in_p out vdd OA%d", i, i, i)
    # in_p电压需要设置
    output_file.write("xl0oa%d 0 l0bl%d VOA l0oaout%d vdd OA%d\n"%( i, i, i, i + 2))
    # begin
    output_file.write("rl0bl%dr l0bl%d l0oaout%d blinresistor\n" % (i, i, i))
    #                          #IN/BL  DBL   OUT/DL vdd
    output_file.write("xl0sa%d l0oaout%d bldinoaout l0sa%da vdd SAVM%d\n" % (i, i, i, i + 2))
    output_file.write("xl0sa%dinva 0 l0sa%da l0sa%db vdd INV1\n" % (i, i, i))
    output_file.write("xl0sa%dinvb 0 l0sa%db l0dl%d vdd INV1\n" % (i, i, i))
    output_file.write("xl0dl%dinv 0 l0dl%d l0dl%db vdd INV1\n" % (i, i, i))

# ==================layer1=================
output_file.write('\n\n\n')

for j in range(0, layer1_shape[1]):
    for i in range(0, layer1_shape[0]):
        output_file.write("xl1b%dc%d l1bl%d vdd l0dl%s l0dl%sb CELLD r1=%se3 r0=%se3\n" % (
            j, i, j, i, i, (100 if neuron[2][i][j] == 1 else 5.3), (5.3 if neuron[2][i][j] == 1 else 100)))
#    for i in range(layer1_shape[0],bld_len):
#        output_file.write("xl1b%dc%d l1bl%d vdd vref vrefb CELLD r1=%de3 r0=%de3\n" %(j,i,j,(10 if i%2==0 else 1),(1 if i%2==0 else 10)))

output_file.write('\n')
for i in range(0, layer1_shape[1]):
    # 在这里添加Operational Amplifier
    #  output_file.write("xl0oa%d 0 in_n in_p out vdd OA%d", i, i, i)
    # in_p电压需要设置
    output_file.write("xl1oa%d 0 l1bl%d VOA l1oaout%d vdd OA%d\n" %( i, i, i, i + 514))
    # begin
    output_file.write("rl1bl%dr l1bl%d l1oaout%d blinresistor\n" % (i,i, i))

    output_file.write("xl1dl%dinv 0 l1dl%d l1dl%db vdd INV1\n" % (i, i, i))
    # ?
    output_file.write("xl1sa%d l1oaout%d bldoaout l1sa%da vdd SAVM%d\n" % (i, i, i, i + 514))
    output_file.write("xl1sa%dinvb 0 l1sa%db l1dl%d vdd INV1\n" % (i, i, i))
    output_file.write("xl1sa%dinva 0 l1sa%da l1sa%db vdd INV1\n" % (i, i, i))

# ==================layer2=================
output_file.write('\n\n\n')

for j in range(0, layer2_shape[1]):
    for i in range(0, layer2_shape[0]):
        output_file.write("xl2b%dc%d l2bl%d vdd l1dl%d l1dl%db CELLD r1=%se3 r0=%se3\n" % (
            j, i, j, i, i, (100 if neuron[4][i][j] == 1 else 5.3), (5.3 if neuron[4][i][j] == 1 else 100)))
    # for i in range(layer2_shape[0],bld_len):
    # output_file.write("xl2b%dc%d l2bl%d vdd vref vrefb CELLD r1=%de3 r0=%de3\n" %(j,i,j,(10 if i%2==0 else 1),(1 if i%2==0 else 10)))

output_file.write('\n')
for i in range(0, layer2_shape[1]):
    # 在这里添加Operational Amplifier
    #  output_file.write("xl0oa%d 0 in_n in_p out vdd OA%d", i, i, i)
    # in_p电压需要设置
    output_file.write("xl2oa%d 0 l2bl%d VOA l2oaout%d vdd OA%d\n" %( i, i, i, i + 1026))
    # begin
    output_file.write("rl2bl%dr l2bl%d l2oaout%d blinresistor\n" % (i, i, i))

    output_file.write("xl2dl%dinv 0 l2dl%d l2dl%db vdd INV1\n" % (i, i, i))
    output_file.write("xl2sa%d l2oaout%d bldoaout l2sa%da vdd SAVM%d\n" % (i, i, i, i + 1026))
    output_file.write("xl2sa%dinvb 0 l2sa%db l2dl%d vdd INV1\n" % (i, i, i))
    output_file.write("xl2sa%dinva 0 l2sa%da l2sa%db vdd INV1\n" % (i, i, i))

# ==================layer3=================
output_file.write('\n\n\n')

for j in range(0, layer3_shape[1]):
    for i in range(0, layer3_shape[0]):
        output_file.write("xl3b%dc%d l3bl%d vdd l2dl%d l2dl%db CELLD r1=%se3 r0=%se3\n" % (
            j, i, j, i, i, (100 if neuron[6][i][j] == 1 else 5.3), (5.3 if neuron[6][i][j] == 1 else 100)))
#   for i in range(layer3_shape[0],bld_len):
#      output_file.write("xl3b%dc%d l3bl%d vdd vref vrefb CELLD r1=%de3 r0=%de3\n" %(j,i,j,(10 if i%2==0 else 1),(1 if i%2==0 else 10)))

output_file.write('\n')
for i in range(0, layer3_shape[1]):
    # 在这里添加Operational Amplifier
    #  output_file.write("xl0oa%d 0 in_n in_p out vdd OA%d", i, i, i)
    # in_p电压需要设置
    output_file.write("xl3oa%d 0 l3bl%d VOA l3oaout%d vdd OA%d\n" % (i, i, i, i + 1538))
    # begin
    output_file.write("rl3bl%dr l3bl%d l3oaout%d blinresistor\n" % (i, i, i))

    output_file.write("xl3dl%dinv 0 l3dl%d l3dl%db vdd INV1\n" % (i, i, i))
    output_file.write("xl3sa%d l3oaout%d bldoaout l3sa%da vdd SAVM%d\n" % (i, i, i, i + 1538))
    output_file.write("xl3sa%dinvb 0 l3sa%db l3dl%d vdd INV1\n" % (i, i, i))
    output_file.write("xl3sa%dinva 0 l3sa%da l3sa%db vdd INV1\n" % (i, i, i))

##==================dummy bl==============
# dummy bl cells
output_file.write('\n')
output_file.write('xbdc0 bld vdd vref vrefb CELLDREF\n')

for i in range(1, bld_len):

    output_file.write("xbdc%d bld vdd vref vrefb CELLD r1=%se3 r0=%se3\n" % (
        i, (100 if i % 2 == 0 else 5.3), (5.3 if i % 2 == 0 else 100)))

# 在这里依然要添加运算放大器
#bld现在可以认为是输出的电压线上的一个值
#  output_file.write("xl0oa%d 0 in_n in_p out vdd OA%d", i, i, i)
output_file.write("xbdcoa 0 bld VOA bldoaout vdd OA%d\n" %( 3000))
output_file.write("rbldr bld bldoaout blresistor\n")

# for i in range(1,68):
#   output_file.write("xbdc%d bld vdd vref vrefb CELLD r1=10e3 r0=1e3\n")
# for i in range(1,60):
#  output_file.write("xbdc%d bld vdd vref vrefb CELLD r1=1e3 r0=10e3\n")
# output_file.write("rbldr bld 0 blresistor\n")

output_file.write('xbdinc0 bldin vdd vref vrefb CELLDREF\n')
for i in range(1, input_len):
    output_file.write("xbdinc%d bldin vdd vref vrefb CELLD r1=%se3 r0=%se3\n" % (
        i, (100 if i % 2 == 0 else 5.3), (5.3 if i % 2 == 0 else 100)))

# 这里同样也需要添加运算放大器
output_file.write("xbdincoaout 0 bld VOA bldinoaout vdd OA%d\n" %( 3100))
output_file.write("rbldinr bldin bldinoaout blinresistor\n")

output_file.write('\n')

# input invertor ex: x0inv
for i in range(layer0_shape[0]):
    output_file.write("x%dinv 0 x%d x%db vdd INV1\n" % (i, i, i))

# output_file.write("\n.include \"img1.sp\"\n")
# output_file.write(".END\n")

output_file.close()
