import shutil

for i in range(1, 101):

    src_path = "ObjectFolder1-100/"+str(i)+"/model.obj"
    dst_path = "input/MIT_CSAIL/complete/T"+str(i)+".obj"
    shutil.copy(src_path, dst_path)
print('Copied object', i)