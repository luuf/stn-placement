import eval

iterative = False

if iterative:
    d = 'plankton/testingiterative/'
    depths = ['s22', 's44']
    extra = ['s0_', 's00_', 's000_', 's77loop_']
    architectures = ['loop_', 'deep_']
else:
    d = 'plankton/testing/'
    depths = ['s2', 's4', 's7', 's10']
    extra = ['s0_']
    architectures = ['_', 'loop_', 'deep_']
num = [0,1,2,3]

for depth in depths:
    for arch in architectures:
        print(depth + arch)
        res = [0,0,0,0]
        for n in num:
            di = d + depth + arch + str(n)
            res[n] = eval.plankton_rotation_statistics(di=di, samples=17, printprogress=False)
        print(sum(res)/4)

for ex in extra:
    print(ex)
    res = [0,0,0,0]
    for n in num:
        di = d + ex + str(n)
        res[n] = eval.plankton_rotation_statistics(di=di, samples=17, printprogress=False)
    print(sum(res)/4)
