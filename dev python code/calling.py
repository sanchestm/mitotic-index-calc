from subprocess import check_output
a = []

for i in range(50):
    a = eval(check_output(['python2', 'image_rec.py', 'singleCells/IF1i0.png'])[:-1])
