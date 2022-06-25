import os

#dir = '/w/cnris/tartanair'
#dir = '/z/cnris/tartanair'
#dir = '/y/cnris/tartanair'
#dir = '/home/shared/tartanair'
dir = '/home/cnris/data/tartanair_test'
#dir = '/home/shared/tartanair'
#dir = '/scratch/justincj_root/justincj/cnris/data/tartanair'
count = 0 
for root, dnames, fnames in os.walk(dir):
    for fname in fnames:
        if fname.endswith('.png'):
            full_name = os.path.join(root, fname)
            value = os.system(f"mogrify -format jpg {full_name} -limit disk 50000")
            if value == 0:
                print(full_name)
                os.remove(full_name)
