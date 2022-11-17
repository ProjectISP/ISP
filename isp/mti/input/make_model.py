#%% Model
f1 = open('AK135.txt', 'r')
lines = f1.readlines()[3:]

depth = np.zeros((len(lines)))
rho = np.zeros((len(lines)))
vp = np.zeros((len(lines)))
vs = np.zeros((len(lines)))
Qp = np.zeros((len(lines)), dtype=int)
Qs = np.zeros((len(lines)), dtype=int)
for i, line in enumerate(lines):
    depth[i] = 6371-float(line[0:7])/1e3
    rho[i] = round(float(line[8:17])/1e3,2)
    vp[i] = round(float(line[18:26])/1e3,2)
    vs[i] = round(float(line[27:35])/1e3,2)
    Qp[i] = int(round(float(line[36:44]),0))
    Qs[i] = int(round(float(line[45:53]),0))
f1.close()

depth = depth[::-1]
rho = rho[::-1]
vp = vp[::-1]
vs = vs[::-1]
Qp = Qp[::-1]
Qs = Qs[::-1]

duplicates = []
for i in range(len(lines)-1):
    if depth[i+1]==depth[i]:
        duplicates.append(i)

depth = np.delete(depth, duplicates)
rho = np.delete(rho, duplicates)
vp = np.delete(vp, duplicates)
vs = np.delete(vs, duplicates)
Qp = np.delete(Qp, duplicates)
Qs = np.delete(Qs, duplicates)

f2 = open('../MTI/AK135forISPtest.txt', 'w')
f2.write('AK135\nnumber of layers\n' + str(len(lines)) + '\nParameters of the layers\ndepth of layer top(km)   Vp(km/s)    Vs(km/s)    Rho(g/cm**3)    Qp     Qs\n')
for i in range(len(depth)):
    f2.write('    ' + str(depth[i]) + '        ' + str(vp[i]) + '        ' + str(vs[i]) + '        ' + str(rho[i]) + '        ' + str(Qp[i]) + '        ' + str(Qs[i]) + '    \n')
f2.write('*******************************************************************\n')
f2.close()